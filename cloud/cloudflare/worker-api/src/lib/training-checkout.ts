import type {
  TrainingCheckoutAdoptionMode,
  TrainingCheckoutEvaluation,
  TrainingCheckoutSearch,
  TrainingCheckoutTarget,
  TrainingJob,
} from "../types";
import { readText, readNumber, parseRunNameFromCheckpointPrefix } from "./training-domain";

export { parseRunNameFromCheckpointPrefix } from "./training-domain";

type RawCheckpointEvaluation = {
  epoch: number;
  prefix: string;
  ok: boolean;
  score: number;
  message: string;
  preset: string;
  passed_samples: number;
  total_samples: number;
};

function normalizeRawCheckpointEvaluation(value: unknown): RawCheckpointEvaluation | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const record = value as Record<string, unknown>;
  if (
    typeof record.epoch !== "number" ||
    typeof record.prefix !== "string" ||
    typeof record.ok !== "boolean" ||
    typeof record.score !== "number" ||
    typeof record.message !== "string" ||
    typeof record.preset !== "string" ||
    typeof record.passed_samples !== "number" ||
    typeof record.total_samples !== "number"
  ) {
    return null;
  }

  return {
    epoch: record.epoch,
    prefix: record.prefix,
    ok: record.ok,
    score: record.score,
    message: record.message,
    preset: record.preset,
    passed_samples: record.passed_samples,
    total_samples: record.total_samples,
  };
}

function buildTarget(input: {
  prefix: string | null;
  epoch: number | null;
  preset: string | null;
  score: number | null;
}): TrainingCheckoutTarget | null {
  if (!input.prefix) {
    return null;
  }

  return {
    prefix: input.prefix,
    epoch: input.epoch,
    preset: input.preset,
    score: input.score,
    run_name: parseRunNameFromCheckpointPrefix(input.prefix),
  };
}

function readTargetFromSummary(
  summary: Record<string, unknown>,
  keys: {
    prefix: string;
    epoch: string;
    preset: string;
    score: string;
  }
): TrainingCheckoutTarget | null {
  return buildTarget({
    prefix: readText(summary[keys.prefix]),
    epoch: readNumber(summary[keys.epoch]),
    preset: readText(summary[keys.preset]),
    score: readNumber(summary[keys.score]),
  });
}

function readAdoptionMode(summary: Record<string, unknown>): TrainingCheckoutAdoptionMode | null {
  const mode = readText(summary.candidate_promotion_mode);
  if (mode === "promote" || mode === "candidate" || mode === "keep_current") {
    return mode;
  }
  return null;
}

function resolveCheckoutSearchStatus(input: {
  job: TrainingJob;
  validationChecked: boolean;
  validationPassed: boolean;
  validationInProgress: boolean;
  hasCandidates: boolean;
  adoptionMode: TrainingCheckoutAdoptionMode | null;
  manualPromoted: TrainingCheckoutTarget | null;
}): TrainingCheckoutSearch["status"] {
  const {
    job,
    validationChecked,
    validationPassed,
    validationInProgress,
    hasCandidates,
    adoptionMode,
    manualPromoted,
  } = input;

  if (manualPromoted) {
    return "manual_promoted";
  }
  if (validationInProgress) {
    return "validating";
  }
  if (validationPassed) {
    if (adoptionMode === "candidate") {
      return "candidate_ready";
    }
    if (adoptionMode === "keep_current") {
      return "kept_current";
    }
    return "promoted";
  }
  if (validationChecked) {
    return hasCandidates ? "rejected" : "failed";
  }
  if (job.status === "failed" || job.status === "cancelled") {
    return "failed";
  }
  return "pending";
}

export function buildTrainingCheckoutSearch(job: TrainingJob): TrainingCheckoutSearch {
  const summary = (job.summary ?? {}) as Record<string, unknown>;
  const champion = readTargetFromSummary(summary, {
    prefix: "candidate_checkpoint_prefix",
    epoch: "candidate_checkpoint_epoch",
    preset: "candidate_preset",
    score: "candidate_score",
  });
  const selected = readTargetFromSummary(summary, {
    prefix: "selected_checkpoint_prefix",
    epoch: "selected_checkpoint_epoch",
    preset: "selected_preset",
    score: "selected_score",
  });
  const manualPromoted = readTargetFromSummary(summary, {
    prefix: "manual_promoted_checkpoint_prefix",
    epoch: "manual_promoted_checkpoint_epoch",
    preset: "manual_promoted_preset",
    score: "manual_promoted_score",
  });

  const validationChecked = summary.validation_checked === true;
  const validationPassed = summary.validation_passed === true;
  const validationInProgress =
    summary.validation_in_progress === true || (job.status === "completed" && !validationChecked);
  const adoptionMode = readAdoptionMode(summary);

  const evaluated = (Array.isArray(summary.evaluated_checkpoints) ? summary.evaluated_checkpoints : [])
    .map(normalizeRawCheckpointEvaluation)
    .filter((value): value is RawCheckpointEvaluation => value !== null)
    .map(
      (value): TrainingCheckoutEvaluation => ({
        ...value,
        run_name: parseRunNameFromCheckpointPrefix(value.prefix),
        is_champion: champion?.prefix === value.prefix,
        is_selected: selected?.prefix === value.prefix,
      })
    )
    .sort((a, b) => {
      if (a.is_champion !== b.is_champion) {
        return a.is_champion ? -1 : 1;
      }
      if (a.score !== b.score) {
        return b.score - a.score;
      }
      return b.epoch - a.epoch;
    });

  const hasCandidates = champion !== null || evaluated.length > 0;

  return {
    status: resolveCheckoutSearchStatus({
      job,
      validationChecked,
      validationPassed,
      validationInProgress,
      hasCandidates,
      adoptionMode,
      manualPromoted,
    }),
    validation_checked: validationChecked,
    validation_passed: validationPassed,
    validation_in_progress: validationInProgress,
    has_candidates: hasCandidates,
    compare_ready: hasCandidates || selected !== null || manualPromoted !== null,
    adoption_mode: adoptionMode,
    message: readText(summary.validation_message),
    last_message: readText(summary.last_message),
    champion,
    selected,
    manual_promoted: manualPromoted,
    evaluated,
  };
}
