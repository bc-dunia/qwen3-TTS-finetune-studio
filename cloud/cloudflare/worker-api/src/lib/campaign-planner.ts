/**
 * Campaign Planner v2 — autoresearch-inspired iterative training strategy engine.
 *
 * Core philosophy (from Karpathy's autoresearch):
 *   After EVERY attempt completes, review ALL history → generate next config →
 *   keep if improved, discard if not → repeat with full context.
 *
 * 3-lane strategy:
 *   - exploit: refine around proven-good zones
 *   - repair: fix the dominant failure mode
 *   - explore: try meaningfully different configs
 *
 * Direction mapping (user-facing):
 *   conservative = 70% exploit / 25% repair / 5% explore
 *   balanced     = 50% exploit / 30% repair / 20% explore
 *   exploratory  = 25% exploit / 35% repair / 40% explore
 */

import type { TrainingConfig, TrainingJob, Voice, TrainingCampaign, StrategyBrief, ResearchBottleneck } from '../types';
import { buildTrainingCheckoutSearch } from './training-checkout';
import { sanitizeConfig } from './training-advisor';
import { readNumber as readNum, clamp } from './training-domain';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type CampaignDirection = 'conservative' | 'balanced' | 'exploratory';
export type StrategyLane = 'exploit' | 'repair' | 'explore';
export type CampaignPhase = 'bootstrap' | 'searching' | 'exploiting' | 'infeasible';

export interface PlannerCandidate {
  lane: StrategyLane;
  config: TrainingConfig;
  reasoning: string;
  target_failure?: string;
}

export interface PlannerResult {
  phase: CampaignPhase;
  stop_recommendation: 'continue' | 'stop_model_unfit' | 'stop_diminishing_returns';
  candidates: PlannerCandidate[];
  /** Persisted across sweeps in campaign.planner_state */
  state_patch: Record<string, unknown>;
}

interface FailureCluster {
  reason: 'speed' | 'asr' | 'tone' | 'overall' | 'infra' | 'unknown';
  count: number;
  configs: Array<{
    lr: number;
    epochs: number;
    subtalker: number;
    seed: number;
  }>;
}

interface SuccessAnchor {
  score: number;
  lr: number;
  epochs: number;
  subtalker: number;
  seed: number;
  job_id: string;
}

interface NearMissSignal {
  lr: number;
  epochs: number;
  subtalker: number;
  seed: number;
  bestSubScore: string;
  bestSubValue: number;
  failedGate: string;
}

// ---------------------------------------------------------------------------
// Direction → Lane weight mapping
// ---------------------------------------------------------------------------

const LANE_WEIGHTS: Record<CampaignDirection, Record<StrategyLane, number>> = {
  conservative: { exploit: 0.7, repair: 0.25, explore: 0.05 },
  balanced: { exploit: 0.5, repair: 0.3, explore: 0.2 },
  exploratory: { exploit: 0.25, repair: 0.35, explore: 0.4 },
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getCheckout(job: TrainingJob) {
  if (!job.checkout_search) {
    job.checkout_search = buildTrainingCheckoutSearch(job);
  }
  return job.checkout_search;
}

function getScore(job: TrainingJob): number | null {
  const cs = getCheckout(job);
  return cs.selected?.score ?? cs.manual_promoted?.score ?? cs.champion?.score ?? null;
}

function getFailureReason(job: TrainingJob): FailureCluster['reason'] {
  const cs = getCheckout(job);
  const msg = (cs.message ?? job.error_message ?? '').toLowerCase();
  if (msg.includes('speed_score') || msg.includes('speed drift')) return 'speed';
  if (msg.includes('asr_score') || msg.includes('missing asr')) return 'asr';
  if (msg.includes('tone_score')) return 'tone';
  if (msg.includes('overall_score') || msg.includes('quality threshold')) return 'overall';
  if (
    msg.includes('no audio') ||
    msg.includes('stalled') ||
    msg.includes('supply_constraint') ||
    msg.includes('recovery')
  )
    return 'infra';
  return 'unknown';
}

function parseMessageKeyValues(message: string | null | undefined): Record<string, number> {
  if (!message) return {};
  const values: Record<string, number> = {};
  const kvPattern = /([a-zA-Z_][a-zA-Z0-9_]*)=([^\s]+)/g;
  for (const match of message.matchAll(kvPattern)) {
    const key = match[1].toLowerCase();
    const raw = match[2].replace(/[;,)]$/, '').trim();
    const num = Number(raw);
    if (Number.isFinite(num)) {
      values[key] = num;
    }
  }
  return values;
}

function metricGateThreshold(metric: string): number | null {
  const key = metric.toLowerCase();
  if (key === 'speed' || key === 'speed_score') return 0.2;
  if (key === 'asr' || key === 'asr_score') return 0.8;
  if (key === 'tone' || key === 'tone_score') return 0.4;
  if (key === 'overall' || key === 'overall_score' || key === 'score') return 0.85;
  if (key === 'speaker' || key === 'speaker_score') return 0.75;
  if (key === 'health' || key === 'health_score') return 0.72;
  if (key === 'duration' || key === 'duration_score') return 0.45;
  return null;
}

function normalizeMetricName(metric: string): string {
  const key = metric.toLowerCase();
  if (key.endsWith('_score')) return key.slice(0, -6);
  return key;
}

function detectFailedGate(
  message: string | null | undefined,
  parsedValues: Record<string, number>,
): string {
  const msg = (message ?? '').toLowerCase();
  const explicit = ['speed_score', 'asr_score', 'tone_score', 'overall_score'].find((k) =>
    msg.includes(k),
  );
  if (explicit) return normalizeMetricName(explicit);

  const failed = Object.entries(parsedValues)
    .map(([k, v]) => ({ key: normalizeMetricName(k), value: v, threshold: metricGateThreshold(k) }))
    .filter((item) => item.threshold !== null && item.value < (item.threshold as number))
    .sort((a, b) => a.value - b.value);
  if (failed.length > 0) {
    return failed[0].key;
  }

  if (msg.includes('speed')) return 'speed';
  if (msg.includes('asr')) return 'asr';
  if (msg.includes('tone')) return 'tone';
  if (msg.includes('overall') || msg.includes('quality')) return 'overall';
  return 'unknown';
}

function pickBestSubScore(
  parsedValues: Record<string, number>,
  failedGate: string,
): { metric: string; value: number } | null {
  let best: { metric: string; value: number; margin: number } | null = null;
  for (const [k, v] of Object.entries(parsedValues)) {
    const threshold = metricGateThreshold(k);
    if (threshold === null) continue;
    const metric = normalizeMetricName(k);
    if (metric === failedGate) continue;
    const margin = v - threshold;
    if (margin >= 0 && (!best || margin > best.margin)) {
      best = { metric, value: v, margin };
    }
  }
  return best ? { metric: best.metric, value: best.value } : null;
}

type TrajectoryTrend =
  | 'improving'
  | 'degrading'
  | 'peaked_then_regressed'
  | 'flat'
  | 'volatile'
  | 'unknown';

function computeTrajectoryTrend(
  evaluations: Array<{ epoch: number; ok: boolean; score: number }>,
): TrajectoryTrend {
  const sorted = [...evaluations].sort((a, b) => a.epoch - b.epoch);
  if (sorted.length < 3) return 'unknown';

  const scores = sorted.map((e) => e.score);
  const bestIdx = scores.indexOf(Math.max(...scores));
  const range = Math.max(...scores) - Math.min(...scores);

  if (range < 0.005) return 'flat';

  // Check if best is in first half and then regresses
  if (bestIdx <= scores.length * 0.5 && scores[scores.length - 1] < scores[bestIdx] - 0.01) {
    return 'peaked_then_regressed';
  }

  // Count direction changes
  let increases = 0;
  let decreases = 0;
  for (let i = 1; i < scores.length; i++) {
    if (scores[i] > scores[i - 1] + 0.003) increases++;
    else if (scores[i] < scores[i - 1] - 0.003) decreases++;
  }

  if (increases > 0 && decreases === 0) return 'improving';
  if (decreases > 0 && increases === 0) return 'degrading';
  if (increases >= 2 && decreases >= 2) return 'volatile';

  // Best score near end → improving tendency
  if (bestIdx >= scores.length * 0.7) return 'improving';
  // Best score near start → degrading tendency
  if (bestIdx <= scores.length * 0.3) return 'degrading';

  return 'volatile';
}

function parseFamilyKey(
  familyKey: string,
): { lr: number; epochs: number; subtalker: number } | null {
  const [lrRaw, epochsRaw, subtalkerRaw] = familyKey.split('|');
  const lr = Number(lrRaw);
  const epochs = Number(epochsRaw);
  const subtalker = Number(subtalkerRaw);
  if (!Number.isFinite(lr) || !Number.isFinite(epochs) || !Number.isFinite(subtalker)) return null;
  return { lr, epochs, subtalker };
}

function configDistance(
  a: { lr: number; epochs: number; subtalker: number },
  b: { lr: number; epochs: number; subtalker: number },
): { lr_pct: number; epochs_abs: number; subtalker_abs: number } {
  const lr_pct = (Math.abs(a.lr - b.lr) / Math.max(a.lr, b.lr, 1e-8)) * 100;
  return {
    lr_pct,
    epochs_abs: Math.abs(a.epochs - b.epochs),
    subtalker_abs: Math.abs(a.subtalker - b.subtalker),
  };
}

function configFamilyKey(lr: number, epochs: number, subtalker: number): string {
  const lrBucket = (Math.round(lr * 1e7) / 1e7).toFixed(7);
  const subBucket = (Math.round(subtalker * 100) / 100).toFixed(2);
  return `${lrBucket}|${epochs}|${subBucket}`;
}

function isInExclusionZone(
  candidate: { lr: number; epochs: number; subtalker: number },
  priorConfigs: Array<{ lr: number; epochs: number; subtalker: number }>,
): boolean {
  for (const prior of priorConfigs.slice(-10)) {
    const dist = configDistance(candidate, prior);
    if (dist.lr_pct < 12 && dist.epochs_abs <= 1 && dist.subtalker_abs < 0.02) {
      return true;
    }
  }
  return false;
}

function meetsMinDiversity(
  a: { lr: number; epochs: number; subtalker: number },
  b: { lr: number; epochs: number; subtalker: number },
): boolean {
  const dist = configDistance(a, b);
  let diffCount = 0;
  if (dist.lr_pct >= 20) diffCount++;
  if (dist.epochs_abs >= 2) diffCount++;
  if (dist.subtalker_abs >= 0.03) diffCount++;
  return diffCount >= 2;
}

function isFamilyBlocked(
  familyKey: string,
  familyStats: Map<string, { completed: number; active: number; passes: number; failures: number }>,
): boolean {
  const stats = familyStats.get(familyKey);
  if (!stats) return false;
  if (stats.failures >= 3 && stats.passes === 0) return true;
  if (stats.active >= 2 && stats.failures >= 2 && stats.passes === 0) return true;
  return false;
}

// ---------------------------------------------------------------------------
// Analysis: extract anchors and failure clusters from history
// ---------------------------------------------------------------------------

function analyzeHistory(
  voice: Voice,
  allVoiceJobs: TrainingJob[],
): {
  successAnchors: SuccessAnchor[];
  nearPassAnchors: SuccessAnchor[];
  failureClusters: Map<FailureCluster['reason'], FailureCluster>;
  dominantFailure: FailureCluster['reason'] | null;
  bestScore: number | null;
  phase: CampaignPhase;
  familyStats: Map<string, { completed: number; active: number; passes: number; failures: number }>;
  blockedFamilies: Set<string>;
  nearMissSignals: NearMissSignal[];
  allAttemptedConfigs: Array<{ lr: number; epochs: number; subtalker: number }>;
} {
  const is06b = voice.model_size.includes('0.6');
  const passThreshold = is06b ? 0.82 : 0.85;
  const nearPassDelta = 0.05;

  const completedJobs = allVoiceJobs
    .filter((j) => j.status === 'completed' || j.status === 'failed' || j.status === 'cancelled')
    .sort((a, b) => (a.created_at || 0) - (b.created_at || 0));

  const activeJobs = allVoiceJobs.filter(
    (j) => j.status !== 'completed' && j.status !== 'failed' && j.status !== 'cancelled',
  );

  const successAnchors: SuccessAnchor[] = [];
  const nearPassAnchors: SuccessAnchor[] = [];
  const nearMissSignals: NearMissSignal[] = [];
  const failureMap = new Map<FailureCluster['reason'], FailureCluster>();
  const familyStats = new Map<
    string,
    { completed: number; active: number; passes: number; failures: number }
  >();

  const ensureFamily = (key: string) => {
    if (!familyStats.has(key)) {
      familyStats.set(key, { completed: 0, active: 0, passes: 0, failures: 0 });
    }
    return familyStats.get(key)!;
  };

  for (const job of activeJobs) {
    const cfg = job.config;
    const lr = readNum(cfg.learning_rate) ?? 0;
    const epochs = readNum(cfg.num_epochs) ?? 0;
    const subtalker = readNum(cfg.subtalker_loss_weight) ?? 0;
    const fk = configFamilyKey(lr, epochs, subtalker);
    ensureFamily(fk).active++;
  }

  for (const job of completedJobs) {
    const score = getScore(job);
    const cfg = job.config;
    const lr = readNum(cfg.learning_rate) ?? 0;
    const epochs = readNum(cfg.num_epochs) ?? 0;
    const subtalker = readNum(cfg.subtalker_loss_weight) ?? 0;
    const seed = readNum(cfg.seed) ?? 0;
    const fk = configFamilyKey(lr, epochs, subtalker);
    const family = ensureFamily(fk);
    family.completed++;

    const cs = getCheckout(job);
    if (cs.validation_passed && score !== null) {
      successAnchors.push({ score, lr, epochs, subtalker, seed, job_id: job.job_id });
      family.passes++;
    } else if (score !== null && score > 0.01 && score >= passThreshold - nearPassDelta) {
      nearPassAnchors.push({ score, lr, epochs, subtalker, seed, job_id: job.job_id });
      family.failures++;
    } else if (cs.status === 'rejected' || job.status === 'failed') {
      const reason = getFailureReason(job);
      if (!failureMap.has(reason)) {
        failureMap.set(reason, { reason, count: 0, configs: [] });
      }
      const cluster = failureMap.get(reason)!;
      cluster.count++;
      cluster.configs.push({ lr, epochs, subtalker, seed });
      family.failures++;

      for (const evaluation of cs.evaluated) {
        if (evaluation.ok) continue;
        const parsed = parseMessageKeyValues(evaluation.message);
        const failedGate = detectFailedGate(evaluation.message, parsed);
        const bestSub = pickBestSubScore(parsed, failedGate);
        if (!bestSub) continue;
        nearMissSignals.push({
          lr,
          epochs,
          subtalker,
          seed,
          bestSubScore: bestSub.metric,
          bestSubValue: bestSub.value,
          failedGate,
        });
      }
    }
  }

  nearMissSignals.sort((a, b) => b.bestSubValue - a.bestSubValue);

  successAnchors.sort((a, b) => b.score - a.score);
  nearPassAnchors.sort((a, b) => b.score - a.score);

  const recentFailures = completedJobs
    .filter((j) => {
      const cs = getCheckout(j);
      return cs.status === 'rejected' || j.status === 'failed';
    })
    .slice(-10);
  let dominantFailure: FailureCluster['reason'] | null = null;
  let maxCount = 0;
  const recentReasonCounts = new Map<FailureCluster['reason'], number>();
  for (const job of recentFailures) {
    const reason = getFailureReason(job);
    if (reason !== 'infra') {
      const c = (recentReasonCounts.get(reason) ?? 0) + 1;
      recentReasonCounts.set(reason, c);
      if (c > maxCount) {
        maxCount = c;
        dominantFailure = reason;
      }
    }
  }

  const bestPassScore = successAnchors.length > 0 ? successAnchors[0].score : null;
  const bestNearScore = nearPassAnchors.length > 0 ? nearPassAnchors[0].score : null;
  const bestScore = bestPassScore ?? bestNearScore;

  const blockedFamilies = new Set<string>();
  for (const [key] of familyStats) {
    if (isFamilyBlocked(key, familyStats)) {
      blockedFamilies.add(key);
    }
  }

  let phase: CampaignPhase;
  if (is06b && completedJobs.length >= 8 && (bestScore === null || bestScore < 0.83)) {
    phase = 'infeasible';
  } else if (successAnchors.length >= 2) {
    phase = 'exploiting';
  } else if (successAnchors.length >= 1 || nearPassAnchors.length >= 2) {
    phase = 'searching';
  } else {
    phase = 'bootstrap';
  }

  const allAttemptedConfigs: Array<{ lr: number; epochs: number; subtalker: number }> = [];
  for (const job of [...completedJobs, ...activeJobs].sort(
    (a, b) => (a.created_at || 0) - (b.created_at || 0),
  )) {
    const cfg = job.config;
    const lr = readNum(cfg.learning_rate) ?? 0;
    const epochs = readNum(cfg.num_epochs) ?? 0;
    const subtalker = readNum(cfg.subtalker_loss_weight) ?? 0;
    if (lr > 0) {
      allAttemptedConfigs.push({ lr, epochs, subtalker });
    }
  }

  return {
    successAnchors,
    nearPassAnchors,
    failureClusters: failureMap,
    dominantFailure,
    bestScore,
    phase,
    familyStats,
    blockedFamilies,
    nearMissSignals,
    allAttemptedConfigs,
  };
}

// ---------------------------------------------------------------------------
// Lane generators
// ---------------------------------------------------------------------------

function generateExploitCandidate(
  anchors: SuccessAnchor[],
  nearPass: SuccessAnchor[],
  failureClusters: Map<FailureCluster['reason'], FailureCluster>,
  voice: Voice,
  attemptIndex: number,
  allAttemptedConfigs: Array<{ lr: number; epochs: number; subtalker: number }>,
): PlannerCandidate | null {
  const pool = [...anchors, ...nearPass];
  if (pool.length === 0) return null;

  // Pick the best anchor and perturb slightly
  const best = pool[0];
  const is06b = voice.model_size.includes('0.6');

  // Small perturbation around best config
  const lrJitter = best.lr * (1 + (((attemptIndex * 7) % 5) - 2) * 0.05); // +/-10%
  const epochJitter = best.epochs + ((attemptIndex % 3) - 1); // +/-1
  const subtalkerJitter = best.subtalker + (((attemptIndex * 3) % 5) - 2) * 0.01; // +/-0.02

  const config: TrainingConfig = {
    model_size: voice.model_size,
    batch_size: 2,
    num_epochs: clamp(Math.round(epochJitter), is06b ? 6 : 4, is06b ? 16 : 18),
    learning_rate: clamp(lrJitter, is06b ? 2e-6 : 4e-6, is06b ? 8e-6 : 8e-6),
    gradient_accumulation_steps: 4,
    subtalker_loss_weight: clamp(subtalkerJitter, is06b ? 0.15 : 0.14, is06b ? 0.35 : 0.32),
    save_every_n_epochs: 1,
    seed: best.seed + attemptIndex * 97,
    gpu_type_id: is06b ? 'NVIDIA L40S' : 'NVIDIA A100-SXM4-80GB',
  };

  for (let shift = 0; shift < 3; shift++) {
    if (
      !isInExclusionZone(
        {
          lr: Number(config.learning_rate),
          epochs: Number(config.num_epochs),
          subtalker: Number(config.subtalker_loss_weight),
        },
        allAttemptedConfigs,
      )
    )
      break;
    config.learning_rate = Number(config.learning_rate) * 1.15;
    config.num_epochs = Number(config.num_epochs) + 1;
  }

  return {
    lane: 'exploit',
    config: sanitizeConfig(config, voice.model_size, voice.labels?.language),
    reasoning: `Refining around best anchor (score=${best.score.toFixed(3)}) with small perturbation`,
  };
}

function generateRepairCandidate(
  dominantFailure: FailureCluster['reason'] | null,
  failureClusters: Map<FailureCluster['reason'], FailureCluster>,
  anchors: SuccessAnchor[],
  voice: Voice,
  attemptIndex: number,
  allAttemptedConfigs: Array<{ lr: number; epochs: number; subtalker: number }>,
): PlannerCandidate {
  const is06b = voice.model_size.includes('0.6');
  const bestAnchor = anchors[0] ?? null;

  // Start from best known config or default
  const baseLr = bestAnchor?.lr ?? (is06b ? 3e-6 : 5e-6);
  const baseEpochs = bestAnchor?.epochs ?? (is06b ? 10 : 8);
  const baseSubtalker = bestAnchor?.subtalker ?? (is06b ? 0.25 : 0.22);

  let lr = baseLr;
  let epochs = baseEpochs;
  let subtalker = baseSubtalker;
  let reasoning = '';

  switch (dominantFailure) {
    case 'speed': {
      const step = Math.min(attemptIndex, 4);
      const ladder = [
        { lrMul: 1.0, epochsDelta: -1, subDelta: 0.0 },
        { lrMul: 0.92, epochsDelta: -1, subDelta: -0.02 },
        { lrMul: 0.85, epochsDelta: -2, subDelta: -0.02 },
        { lrMul: 0.78, epochsDelta: -2, subDelta: -0.04 },
      ];
      const rung = ladder[Math.max(0, step - 1)];
      lr = baseLr * rung.lrMul;
      epochs = baseEpochs + rung.epochsDelta;
      subtalker = baseSubtalker + rung.subDelta;
      reasoning = `Speed repair step ${step}: shorter training (ep${rung.epochsDelta}) and LR×${rung.lrMul}`;
      break;
    }
    case 'tone': {
      lr = baseLr * 0.8;
      subtalker = Math.max(is06b ? 0.15 : 0.12, baseSubtalker - 0.03);
      epochs = baseEpochs + 2;
      reasoning = 'Tone repair: lower LR + more epochs to preserve speaker timbre';
      break;
    }
    case 'asr': {
      lr = baseLr * 0.7;
      subtalker = baseSubtalker + 0.02;
      epochs = Math.max(is06b ? 6 : 5, baseEpochs - 2);
      reasoning = 'ASR repair: conservative LR + fewer epochs to avoid transcript drift';
      break;
    }
    case 'overall': {
      lr = baseLr * 0.75;
      subtalker = baseSubtalker + 0.03;
      epochs = baseEpochs;
      reasoning = 'Overall quality repair: lower LR + higher subtalker for stability';
      break;
    }
    default: {
      lr = baseLr * 0.85;
      epochs = baseEpochs - 1;
      reasoning = 'General repair: slightly conservative config';
    }
  }

  for (let shift = 0; shift < 3; shift++) {
    if (!isInExclusionZone({ lr, epochs, subtalker }, allAttemptedConfigs)) break;
    lr *= 0.85;
    epochs = Math.max(is06b ? 5 : 4, epochs - 1);
    subtalker = Math.max(is06b ? 0.1 : 0.08, subtalker - 0.03);
    reasoning += ' (shifted away from exclusion zone)';
  }

  const config: TrainingConfig = {
    model_size: voice.model_size,
    batch_size: 2,
    num_epochs: clamp(Math.round(epochs), is06b ? 5 : 4, is06b ? 18 : 20),
    learning_rate: clamp(lr, is06b ? 1.5e-6 : 3e-6, is06b ? 1e-5 : 1e-5),
    gradient_accumulation_steps: 4,
    subtalker_loss_weight: clamp(subtalker, is06b ? 0.1 : 0.08, is06b ? 0.4 : 0.35),
    save_every_n_epochs: 1,
    seed: (bestAnchor?.seed ?? 303) + attemptIndex * 131,
    gpu_type_id: is06b ? 'NVIDIA L40S' : 'NVIDIA A100-SXM4-80GB',
  };

  return {
    lane: 'repair',
    config: sanitizeConfig(config, voice.model_size, voice.labels?.language),
    reasoning,
    target_failure: dominantFailure ?? undefined,
  };
}

function generateExploreCandidate(
  anchors: SuccessAnchor[],
  failureClusters: Map<FailureCluster['reason'], FailureCluster>,
  voice: Voice,
  attemptIndex: number,
  blockedFamilies: Set<string>,
  allAttemptedConfigs: Array<{ lr: number; epochs: number; subtalker: number }>,
): PlannerCandidate {
  const is06b = voice.model_size.includes('0.6');
  const anchor = anchors[0] ?? null;

  const centerLr = anchor?.lr ?? (is06b ? 3e-6 : 4.75e-6);
  const centerEp = anchor?.epochs ?? (is06b ? 10 : 6);
  const centerSub = anchor?.subtalker ?? (is06b ? 0.25 : 0.26);

  const lrMultipliers = [0.74, 0.84, 0.92, 1.0, 1.06, 1.15];
  const epochDeltas = [-2, -1, 0, 1];
  const subDeltas = [-0.04, -0.02, 0.0, 0.02];
  const seedOptions = [42, 77, 202, 303, 404, 505, 606, 707, 808, 909];

  const lrOptions = lrMultipliers.map((m) => centerLr * m);
  const epochOptions = epochDeltas.map((d) => centerEp + d);
  const subtalkerOptions = subDeltas.map((d) => centerSub + d);

  const lr = lrOptions[attemptIndex % lrOptions.length];
  const epochs = epochOptions[attemptIndex % epochOptions.length];
  const subtalker = subtalkerOptions[attemptIndex % subtalkerOptions.length];
  const seed = seedOptions[attemptIndex % seedOptions.length];

  let candidate = { lr, epochs, subtalker };
  let shifted = false;

  const candidateFk = configFamilyKey(
    candidate.lr,
    Math.round(candidate.epochs),
    Math.round(candidate.subtalker * 100) / 100,
  );
  if (blockedFamilies.has(candidateFk)) {
    candidate = {
      lr: candidate.lr * 0.82,
      epochs: Math.max(is06b ? 5 : 4, candidate.epochs - 1),
      subtalker: candidate.subtalker - 0.02,
    };
    shifted = true;
  }

  for (let shift = 0; shift < 3; shift++) {
    if (!isInExclusionZone(candidate, allAttemptedConfigs)) break;
    candidate = {
      lr: candidate.lr * 0.85,
      epochs: Math.max(is06b ? 5 : 4, candidate.epochs - 1),
      subtalker: Math.max(0.1, candidate.subtalker - 0.04),
    };
    shifted = true;
  }

  for (const a of anchors) {
    if (!meetsMinDiversity(candidate, a)) {
      candidate = {
        lr: candidate.lr * 0.78,
        epochs: Math.max(is06b ? 5 : 4, candidate.epochs - 2),
        subtalker: Math.max(0.1, candidate.subtalker - 0.05),
      };
      break;
    }
  }

  const config: TrainingConfig = {
    model_size: voice.model_size,
    batch_size: 2,
    num_epochs: clamp(Math.round(candidate.epochs), is06b ? 5 : 4, is06b ? 18 : 20),
    learning_rate: clamp(candidate.lr, is06b ? 1.5e-6 : 3e-6, is06b ? 1.5e-5 : 1.5e-5),
    gradient_accumulation_steps: 4,
    subtalker_loss_weight: clamp(candidate.subtalker, is06b ? 0.1 : 0.08, is06b ? 0.4 : 0.35),
    save_every_n_epochs: 1,
    seed,
    gpu_type_id: is06b ? 'NVIDIA L40S' : 'NVIDIA A100-SXM4-80GB',
  };

  return {
    lane: 'explore',
    config: sanitizeConfig(config, voice.model_size, voice.labels?.language),
    reasoning: `Exploring anchor-relative region: lr=${config.learning_rate} ep=${config.num_epochs} sub=${config.subtalker_loss_weight}${shifted ? ' (shifted from blocked/exclusion zone)' : ''}`,
  };
}

// ---------------------------------------------------------------------------
// Assign lanes based on direction and open slots
// ---------------------------------------------------------------------------

function applyBottleneckBias(
  weights: Record<StrategyLane, number>,
  bottleneck: ResearchBottleneck,
  calibrationAlpha?: number,
): Record<StrategyLane, number> {
  const w = { ...weights };
  if (!bottleneck) return w;

  // Dataset/ASR issues → repair lane knows how to address these
  if (bottleneck === 'dataset_quality' || bottleneck === 'asr') {
    w.repair = Math.min(w.repair + 0.15, 0.55);
    w.exploit = Math.max(w.exploit - 0.10, 0.15);
    w.explore = Math.max(w.explore - 0.05, 0.05);
  }
  // Specific metric bottleneck → exploit lane can target it with focused perturbation
  else if (bottleneck === 'tone' || bottleneck === 'speed' || bottleneck === 'speaker' || bottleneck === 'style') {
    w.exploit = Math.min(w.exploit + 0.10, 0.70);
    w.repair = Math.max(w.repair - 0.05, 0.10);
    w.explore = Math.max(w.explore - 0.05, 0.05);
  }
  // Overall weakness → explore different approaches
  else if (bottleneck === 'overall') {
    w.explore = Math.min(w.explore + 0.10, 0.50);
    w.exploit = Math.max(w.exploit - 0.10, 0.15);
  }
  // balanced = improving steadily, no special bias needed

  // High calibration alpha means good signal quality — lean into exploit
  if (typeof calibrationAlpha === 'number' && calibrationAlpha > 0.4) {
    const exploitBoost = Math.min(calibrationAlpha * 0.08, 0.06);
    w.exploit = Math.min(w.exploit + exploitBoost, 0.75);
    w.explore = Math.max(w.explore - exploitBoost, 0.05);
  }

  // Normalize so weights sum to ~1.0
  const total = w.exploit + w.repair + w.explore;
  if (total > 0) {
    w.exploit /= total;
    w.repair /= total;
    w.explore /= total;
  }

  return w;
}

function assignLanes(
  direction: CampaignDirection,
  slotsToFill: number,
  phase: CampaignPhase,
  bottleneckHint?: ResearchBottleneck,
  calibrationAlpha?: number,
): StrategyLane[] {
  if (slotsToFill === 0) return [];

  const weights = LANE_WEIGHTS[direction];

  // Adjust weights based on phase
  let effectiveWeights = { ...weights };
  if (phase === 'bootstrap') {
    // No proven zones yet — boost repair and explore
    effectiveWeights = { exploit: 0.2, repair: 0.4, explore: 0.4 };
  } else if (phase === 'exploiting') {
    // Proven zones exist — boost exploit
    effectiveWeights = {
      exploit: Math.max(weights.exploit, 0.6),
      repair: Math.min(weights.repair, 0.25),
      explore: Math.min(weights.explore, 0.15),
    };
  } else if (phase === 'infeasible') {
    // Stop wasting money
    return [];
  }

  // Apply research controller bottleneck bias when available
  if (bottleneckHint) {
    effectiveWeights = applyBottleneckBias(effectiveWeights, bottleneckHint, calibrationAlpha);
  }

  // For 1 slot: pick highest weight lane
  if (slotsToFill === 1) {
    const best = (Object.entries(effectiveWeights) as [StrategyLane, number][]).sort(
      ([, a], [, b]) => b - a,
    )[0][0];
    return [best];
  }

  // For 2 slots: top 2 weighted lanes
  if (slotsToFill === 2) {
    const sorted = (Object.entries(effectiveWeights) as [StrategyLane, number][]).sort(
      ([, a], [, b]) => b - a,
    );
    return [sorted[0][0], sorted[1][0]];
  }

  // For 3 slots: one of each
  return ['exploit', 'repair', 'explore'];
}

// ---------------------------------------------------------------------------
// Main planner entry point
// ---------------------------------------------------------------------------

/**
 * Generate campaign candidates using the 3-lane autoresearch strategy.
 *
 * Called by advanceTrainingCampaign() every time a slot opens.
 */
export function planCampaignAttempts(
  voice: Voice,
  campaign: TrainingCampaign,
  allVoiceJobs: TrainingJob[],
  slotsToFill: number,
  nextAttemptIndex: number,
  strategyBrief?: StrategyBrief,
): PlannerResult {
  const direction: CampaignDirection =
    (campaign.planner_state?.direction as CampaignDirection) ?? 'balanced';

  const {
    successAnchors,
    nearPassAnchors,
    failureClusters,
    dominantFailure,
    bestScore,
    phase,
    familyStats,
    blockedFamilies,
    nearMissSignals,
    allAttemptedConfigs,
  } = analyzeHistory(voice, allVoiceJobs);

  // Infeasibility stop for 0.6B
  if (phase === 'infeasible') {
    return {
      phase: 'infeasible',
      stop_recommendation: 'stop_model_unfit',
      candidates: [],
      state_patch: {
        phase: 'infeasible',
        stop_recommendation: 'stop_model_unfit',
        best_score: bestScore,
        dominant_failure: dominantFailure,
        updated_at: Date.now(),
      },
    };
  }

  // Diminishing returns check: if we have 5+ completed and improvement < 0.005 over last 4
  const completedScores = allVoiceJobs
    .filter((j) => getScore(j) !== null)
    .sort((a, b) => (a.created_at || 0) - (b.created_at || 0))
    .map((j) => getScore(j)!);
  let stopRec: PlannerResult['stop_recommendation'] = 'continue';
  if (completedScores.length >= 5) {
    const last4 = completedScores.slice(-4);
    const range = Math.max(...last4) - Math.min(...last4);
    if (range < 0.003 && phase === 'exploiting') {
      stopRec = 'stop_diminishing_returns';
    }
  }

  const lanes = assignLanes(
    direction,
    slotsToFill,
    phase,
    strategyBrief?.bottleneck,
    strategyBrief?.calibration_insights?.alpha,
  );
  const candidates: PlannerCandidate[] = [];
  const usedConfigs: Array<{ lr: number; epochs: number; subtalker: number }> = [];

  for (let i = 0; i < lanes.length; i++) {
    const lane = lanes[i];
    const idx = nextAttemptIndex + i;
    let candidate: PlannerCandidate | null = null;

    switch (lane) {
      case 'exploit':
        candidate = generateExploitCandidate(
          successAnchors,
          nearPassAnchors,
          failureClusters,
          voice,
          idx,
          allAttemptedConfigs,
        );
        break;
      case 'repair':
        candidate = generateRepairCandidate(
          dominantFailure,
          failureClusters,
          successAnchors,
          voice,
          idx,
          allAttemptedConfigs,
        );
        break;
      case 'explore':
        candidate = generateExploreCandidate(
          successAnchors,
          failureClusters,
          voice,
          idx,
          blockedFamilies,
          allAttemptedConfigs,
        );
        break;
    }

    if (!candidate) {
      candidate = generateRepairCandidate(
        dominantFailure,
        failureClusters,
        successAnchors,
        voice,
        idx,
        allAttemptedConfigs,
      );
    }

    const candLr = readNum(candidate.config.learning_rate) ?? 0;
    const candEp = readNum(candidate.config.num_epochs) ?? 0;
    const candSub = readNum(candidate.config.subtalker_loss_weight) ?? 0;
    const candFk = configFamilyKey(candLr, candEp, candSub);
    if (blockedFamilies.has(candFk)) {
      candidate = generateExploreCandidate(
        successAnchors,
        failureClusters,
        voice,
        idx + 10,
        blockedFamilies,
        allAttemptedConfigs,
      );
      candidate.reasoning += ' (original candidate was in blocked family)';
    }

    const candidateNums = {
      lr: readNum(candidate.config.learning_rate) ?? 0,
      epochs: readNum(candidate.config.num_epochs) ?? 0,
      subtalker: readNum(candidate.config.subtalker_loss_weight) ?? 0,
    };

    for (const used of usedConfigs) {
      if (!meetsMinDiversity(candidateNums, used)) {
        candidateNums.lr *= 1.25;
        candidateNums.epochs += 2;
        candidateNums.subtalker = Math.max(0.08, candidateNums.subtalker - 0.04);
        candidate.config = sanitizeConfig(
          {
            ...candidate.config,
            learning_rate: candidateNums.lr,
            num_epochs: Math.round(candidateNums.epochs),
            subtalker_loss_weight: candidateNums.subtalker,
          },
          voice.model_size,
          voice.labels?.language,
        );
        candidate.reasoning += ' (diversified from parallel run)';
      }
    }

    const finalFk = configFamilyKey(
      readNum(candidate.config.learning_rate) ?? 0,
      readNum(candidate.config.num_epochs) ?? 0,
      readNum(candidate.config.subtalker_loss_weight) ?? 0,
    );
    if (blockedFamilies.has(finalFk)) {
      const is06b = voice.model_size.includes('0.6');
      const escapeLr = (readNum(candidate.config.learning_rate) ?? (is06b ? 3e-6 : 5e-6)) * 0.7;
      const escapeEp = Math.max(is06b ? 5 : 4, (readNum(candidate.config.num_epochs) ?? 6) - 2);
      const escapeSub = Math.max(
        is06b ? 0.1 : 0.08,
        (readNum(candidate.config.subtalker_loss_weight) ?? 0.26) - 0.06,
      );
      candidate.config = sanitizeConfig(
        {
          ...candidate.config,
          learning_rate: escapeLr,
          num_epochs: escapeEp,
          subtalker_loss_weight: escapeSub,
        },
        voice.model_size,
        voice.labels?.language,
      );
      candidateNums.lr = escapeLr;
      candidateNums.epochs = escapeEp;
      candidateNums.subtalker = escapeSub;
      candidate.reasoning += ' (escape hatch: all nearby families blocked)';
    }

    usedConfigs.push(candidateNums);
    candidates.push(candidate);
  }

  return {
    phase,
    stop_recommendation: stopRec,
    candidates,
    state_patch: {
      phase,
      direction,
      stop_recommendation: stopRec,
      best_score: bestScore,
      success_anchor_count: successAnchors.length,
      near_pass_anchor_count: nearPassAnchors.length,
      dominant_failure: dominantFailure,
      failure_clusters: Object.fromEntries(
        [...failureClusters.entries()].map(([k, v]) => [k, { count: v.count }]),
      ),
      lanes_assigned: lanes,
      candidates_generated: candidates.length,
      blocked_families: [...blockedFamilies],
      family_stats: Object.fromEntries([...familyStats.entries()].map(([k, v]) => [k, v])),
      near_miss_signals: nearMissSignals,
      strategy_brief: strategyBrief ?? null,
      updated_at: Date.now(),
    },
  };
}

// ---------------------------------------------------------------------------
// LLM-enhanced planner (calls the LLM advisor for richer analysis)
// ---------------------------------------------------------------------------

export interface LLMPlannerInput {
  voice: Voice;
  campaign: TrainingCampaign;
  allVoiceJobs: TrainingJob[];
  slotsToFill: number;
  nextAttemptIndex: number;
  heuristicResult: PlannerResult;
}

/**
 * Build a rich prompt for the LLM planner that includes full history context,
 * success anchors, failure clusters, and the heuristic candidates as reference.
 */
export function buildLLMPlannerPrompt(input: LLMPlannerInput): string {
  const { voice, campaign, allVoiceJobs, heuristicResult } = input;
  const is06b = voice.model_size.includes('0.6');
  const direction = (campaign.planner_state?.direction as CampaignDirection) ?? 'balanced';

  const completedJobs = allVoiceJobs
    .filter((j) => j.status === 'completed' || j.status === 'failed' || j.status === 'cancelled')
    .sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
  const activeJobs = allVoiceJobs
    .filter((j) => j.status !== 'completed' && j.status !== 'failed' && j.status !== 'cancelled')
    .sort((a, b) => (b.created_at || 0) - (a.created_at || 0));

  const blockedFamilies = Array.isArray(heuristicResult.state_patch.blocked_families)
    ? heuristicResult.state_patch.blocked_families.filter(
        (value): value is string => typeof value === 'string',
      )
    : [];

  const lines: string[] = [];

  // Context
  lines.push('## Campaign Context');
  lines.push(
    `Voice: ${voice.name}, Model: ${voice.model_size}, Language: ${voice.labels?.language ?? 'ko'}`,
  );
  lines.push(`Direction: ${direction}`);
  lines.push(`Phase: ${heuristicResult.phase}`);
  lines.push(`Slots to fill: ${input.slotsToFill}`);
  lines.push(`Current champion score: ${voice.checkpoint_score?.toFixed(3) ?? 'none'}`);
  lines.push(`Validation threshold: ${is06b ? 0.82 : 0.85}`);
  lines.push('');

  // Success anchors (top 5)
  const anchors = heuristicResult.state_patch;
  lines.push('## Success Anchors (passing configs)');
  const passJobs = completedJobs
    .filter((j) => getCheckout(j).validation_passed)
    .sort((a, b) => (getScore(b) ?? 0) - (getScore(a) ?? 0))
    .slice(0, 5);
  if (passJobs.length === 0) {
    lines.push('No passing runs yet.');
  } else {
    for (const j of passJobs) {
      const c = j.config;
      lines.push(
        `  score=${getScore(j)?.toFixed(3)} lr=${c.learning_rate} ep=${c.num_epochs} sub=${c.subtalker_loss_weight} seed=${c.seed}`,
      );
    }
  }
  lines.push('');

  // Near-pass anchors
  lines.push('## Near-Pass Runs (within 0.02 of threshold)');
  const nearPass = completedJobs
    .filter((j) => {
      const s = getScore(j);
      return s !== null && !getCheckout(j).validation_passed && s >= (is06b ? 0.8 : 0.83);
    })
    .slice(0, 5);
  if (nearPass.length === 0) {
    lines.push('None.');
  } else {
    for (const j of nearPass) {
      const c = j.config;
      lines.push(
        `  score=${getScore(j)?.toFixed(3)} lr=${c.learning_rate} ep=${c.num_epochs} sub=${c.subtalker_loss_weight} seed=${c.seed} fail=${getFailureReason(j)}`,
      );
    }
  }
  lines.push('');

  lines.push('## Active Runs (do NOT suggest similar configs)');
  if (activeJobs.length === 0) {
    lines.push('None.');
  } else {
    for (const j of activeJobs.slice(0, 10)) {
      const c = j.config;
      lines.push(
        `  lr=${c.learning_rate} ep=${c.num_epochs} sub=${c.subtalker_loss_weight} seed=${c.seed} (${j.status})`,
      );
    }
  }
  lines.push('');

  // Failure clusters
  lines.push('## Failure Clusters (grouped by reason)');
  const failMap = new Map<string, number>();
  for (const j of completedJobs.filter(
    (j) => getCheckout(j).status === 'rejected' || j.status === 'failed',
  )) {
    const r = getFailureReason(j);
    failMap.set(r, (failMap.get(r) ?? 0) + 1);
  }
  for (const [reason, count] of [...failMap.entries()].sort(([, a], [, b]) => b - a)) {
    lines.push(`  ${reason}: ${count} failures`);
  }
  lines.push(`Dominant failure: ${anchors.dominant_failure ?? 'none'}`);
  lines.push('');

  lines.push('## Blocked Config Families (>=3 failures, 0 passes — DO NOT USE)');
  if (blockedFamilies.length === 0) {
    lines.push('None.');
  } else {
    for (const familyKey of blockedFamilies.slice(0, 5)) {
      const parsed = parseFamilyKey(familyKey);
      if (!parsed) continue;
      const dominantReason = (() => {
        const reasonCounts = new Map<string, number>();
        for (const j of completedJobs.filter(
          (job) => getCheckout(job).status === 'rejected' || job.status === 'failed',
        )) {
          const c = j.config;
          const fk = configFamilyKey(
            readNum(c.learning_rate) ?? 0,
            readNum(c.num_epochs) ?? 0,
            readNum(c.subtalker_loss_weight) ?? 0,
          );
          if (fk !== familyKey) continue;
          const r = getFailureReason(j);
          reasonCounts.set(r, (reasonCounts.get(r) ?? 0) + 1);
        }
        if (reasonCounts.size === 0) return 'unknown';
        return [...reasonCounts.entries()].sort(([, a], [, b]) => b - a)[0][0];
      })();
      lines.push(
        `  lr=${parsed.lr.toExponential(2)} ep=${parsed.epochs} sub=${parsed.subtalker.toFixed(2)} → blocked(${dominantReason})`,
      );
    }
  }
  lines.push('');

  lines.push('## Recent 10 Runs (most recent first)');
  for (const j of completedJobs.slice(0, 10)) {
    const c = j.config;
    const cs = getCheckout(j);
    const score = getScore(j);
    const reason = cs.validation_passed ? 'none' : getFailureReason(j);
    const bestEpoch = cs.selected?.epoch ?? cs.champion?.epoch ?? null;
    const maxEpoch = cs.evaluated.length > 0 ? Math.max(...cs.evaluated.map((e) => e.epoch)) : null;
    const peakTiming =
      bestEpoch !== null && maxEpoch !== null
        ? bestEpoch <= maxEpoch * 0.4
          ? 'early'
          : bestEpoch <= maxEpoch * 0.7
            ? 'mid'
            : 'late'
        : null;
    const trend = cs.evaluated.length >= 3 ? computeTrajectoryTrend(cs.evaluated) : null;

    lines.push(
      `  [${cs.validation_passed ? 'PASS' : cs.status === 'rejected' ? 'REJECT' : 'FAIL'}] score=${score?.toFixed(3) ?? 'n/a'} lr=${c.learning_rate} ep=${c.num_epochs} sub=${c.subtalker_loss_weight} seed=${c.seed} fail=${reason}${peakTiming ? ` peak=${peakTiming}` : ''}${trend && trend !== 'unknown' ? ` trend=${trend}` : ''}`,
    );
  }
  lines.push('');

  lines.push('## Heuristic Baseline (top 3)');
  for (const cand of heuristicResult.candidates.slice(0, 3)) {
    lines.push(
      `  ${cand.lane}: lr=${cand.config.learning_rate} ep=${cand.config.num_epochs} sub=${cand.config.subtalker_loss_weight} seed=${cand.config.seed}`,
    );
  }
  lines.push('');

  const brief = heuristicResult.state_patch.strategy_brief as StrategyBrief | null | undefined;
  if (brief) {
    lines.push('## Research Controller Context');
    if (brief.bottleneck) lines.push(`Bottleneck: ${brief.bottleneck}`);
    if (brief.calibration_insights) {
      lines.push(`Calibration: alpha=${brief.calibration_insights.alpha.toFixed(2)} state=${brief.calibration_insights.state} both_bad=${brief.calibration_insights.both_bad_rate.toFixed(2)}`);
    }
    if (brief.dataset_flags.length > 0) lines.push(`Flags: ${brief.dataset_flags.join(', ')}`);
    if (brief.arena_winner_patterns) lines.push(`Arena winners: ${brief.arena_winner_patterns}`);
    if (brief.lessons.length > 0) {
      lines.push('Lessons:');
      for (const lesson of brief.lessons.slice(0, 5)) {
        lines.push(`  [${lesson.confidence}] ${lesson.lesson} (evidence=${lesson.evidence_count})`);
      }
    }
    lines.push('');
  }

  lines.push(
    `Generate ${input.slotsToFill} candidate configs that are meaningfully different from each other.`,
  );

  return lines.join('\n');
}

export type LLMPlannerDecision = 'skip' | 'normal' | 'escalated';

const SPEED_REPAIR_LADDER = `
## Speed Failures
Speed failures mean the model drifts from the speaker's natural pace. Fix with:
- Step 1: epochs −1, LR ×1.00, subtalker unchanged (shorter training)
- Step 2: epochs −1, LR ×0.92, subtalker −0.02
- Step 3: epochs −2, LR ×0.85, subtalker −0.02
- Step 4: epochs −2, LR ×0.78, subtalker −0.04`;

const LLM_PLANNER_RESPONSE_FORMAT = `## Response Format (STRICT JSON)
{
  "phase": "bootstrap|searching|exploiting|infeasible",
  "stop_recommendation": "continue|stop_model_unfit|stop_diminishing_returns",
  "candidates": [
    {
      "lane": "exploit|repair|explore",
      "target_failure": "speed|tone|asr|overall|null",
      "config": {
        "learning_rate": 0.000005,
        "num_epochs": 8,
        "subtalker_loss_weight": 0.22,
        "seed": 808
      },
      "reasoning": "why this config"
    }
  ]
}`;

export function buildLLMPlannerSystemPrompt(dominantFailure: string): string {
  const lines: string[] = [
    'You are an autonomous TTS fine-tuning researcher.',
    'Analyze training history and generate the next batch of hyperparameter configurations.',
    '',
    '## Strategy Lanes',
    '- exploit: Small perturbations around proven-good configs (passing or near-pass)',
    '- repair: Targeted fixes for the dominant failure mode',
    '- explore: Meaningfully different configs to discover new regions',
    '',
    '## Constraints',
    '- Each candidate MUST specify: learning_rate, num_epochs, subtalker_loss_weight, seed',
    '- Parallel candidates MUST differ in ≥2 primary knobs (LR ≥20%, epochs ≥2, subtalker ≥0.03)',
    '- Do NOT suggest configs in the exclusion zone of recent failures (LR ±12%, epochs ±1, subtalker ±0.02)',
    '- Seed-only variation is FORBIDDEN (except to replicate passing configs)',
  ];

  if (dominantFailure === 'speed') {
    lines.push(SPEED_REPAIR_LADDER);
  } else if (dominantFailure && dominantFailure !== 'none' && dominantFailure !== 'unknown') {
    lines.push(
      '',
      `## Repair Focus`,
      `Focus repair candidates on fixing ${dominantFailure} failures.`,
    );
  }

  lines.push('', LLM_PLANNER_RESPONSE_FORMAT);
  return lines.join('\n');
}

export function buildLLMPlannerStateHash(
  voiceId: string,
  voiceJobs: TrainingJob[],
  slotsToFill: number,
  direction: string,
): string {
  const terminal = voiceJobs
    .filter((j) => j.status === 'completed' || j.status === 'failed' || j.status === 'cancelled')
    .sort((a, b) => a.job_id.localeCompare(b.job_id))
    .map((j) => {
      const score = getScore(j);
      const configKey = `${j.config.learning_rate}|${j.config.num_epochs}|${j.config.subtalker_loss_weight}`;
      return `${j.job_id}:${j.status}:${configKey}:${score?.toFixed(3) ?? 'x'}:${j.updated_at ?? 0}`;
    })
    .join(';');

  const active = voiceJobs
    .filter((j) => j.status !== 'completed' && j.status !== 'failed' && j.status !== 'cancelled')
    .sort((a, b) => a.job_id.localeCompare(b.job_id))
    .map((j) => `${j.job_id}:${j.status}`)
    .join(';');

  return `${voiceId}|${terminal}|${active}|${slotsToFill}|${direction}`;
}

/**
 * Parse a structured LLM planner response into PlannerResult candidates.
 */
export function parseLLMPlannerResponse(
  raw: unknown,
  voice: Voice,
  heuristicResult: PlannerResult,
): PlannerResult {
  if (!raw || typeof raw !== 'object') return heuristicResult;
  const data = raw as Record<string, unknown>;

  const VALID_PHASES: ReadonlySet<string> = new Set([
    'bootstrap',
    'searching',
    'exploiting',
    'infeasible',
  ]);
  const VALID_STOP_RECS: ReadonlySet<string> = new Set([
    'continue',
    'stop_model_unfit',
    'stop_diminishing_returns',
  ]);

  const phase =
    typeof data.phase === 'string' && VALID_PHASES.has(data.phase)
      ? (data.phase as CampaignPhase)
      : heuristicResult.phase;

  const stopRec =
    typeof data.stop_recommendation === 'string' && VALID_STOP_RECS.has(data.stop_recommendation)
      ? (data.stop_recommendation as PlannerResult['stop_recommendation'])
      : heuristicResult.stop_recommendation;

  const rawCandidates = Array.isArray(data.candidates) ? data.candidates : [];
  const candidates: PlannerCandidate[] = [];

  for (const rc of rawCandidates) {
    if (!rc || typeof rc !== 'object') continue;
    const entry = rc as Record<string, unknown>;

    const lane =
      typeof entry.lane === 'string' && ['exploit', 'repair', 'explore'].includes(entry.lane)
        ? (entry.lane as StrategyLane)
        : 'repair';

    const rawConfig = entry.config;
    if (!rawConfig || typeof rawConfig !== 'object') continue;
    const cfg = rawConfig as Record<string, unknown>;

    const config: TrainingConfig = sanitizeConfig(
      {
        model_size: voice.model_size,
        learning_rate: typeof cfg.learning_rate === 'number' ? cfg.learning_rate : undefined,
        num_epochs: typeof cfg.num_epochs === 'number' ? cfg.num_epochs : undefined,
        subtalker_loss_weight:
          typeof cfg.subtalker_loss_weight === 'number' ? cfg.subtalker_loss_weight : undefined,
        seed: typeof cfg.seed === 'number' ? cfg.seed : undefined,
        save_every_n_epochs: 1,
        batch_size: 2,
        gradient_accumulation_steps: 4,
      },
      voice.model_size,
      voice.labels?.language,
    );

    candidates.push({
      lane,
      config,
      reasoning: typeof entry.reasoning === 'string' ? entry.reasoning : '',
      target_failure: typeof entry.target_failure === 'string' ? entry.target_failure : undefined,
    });
  }

  // If LLM returned no valid candidates, fallback to heuristic
  if (candidates.length === 0) return heuristicResult;

  return {
    phase,
    stop_recommendation: stopRec,
    candidates,
    state_patch: {
      ...heuristicResult.state_patch,
      llm_enhanced: true,
      llm_phase: phase,
      llm_stop_recommendation: stopRec,
      llm_candidates_count: candidates.length,
    },
  };
}
