import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router'
import {
  createArenaSession,
  completeArenaSession,
  generateArenaAudio,
  getArenaSession,
  promoteArenaWinner,
  submitArenaVote,
  type ArenaCandidate,
  type ArenaMatch,
  type ArenaSessionResponse,
  type ArenaVoteConfidence,
  type ArenaVoteWinner,
} from '../lib/api'

type ArenaStep = 'setup' | 'generating' | 'voting' | 'results'

const DEFAULT_TEST_TEXTS = [
  'Hello. The weather is great today.',
  'This product has excellent quality and high customer satisfaction.',
  'Please prepare for the meeting tomorrow at 10 a.m.',
]

function getSourceBadge(source: string): { label: string; className: string } {
  switch (source) {
    case 'champion_carry':
      return { label: 'champion', className: 'bg-accent-dim text-accent' }
    case 'second_carry':
      return { label: '2nd carry', className: 'bg-warning-dim text-warning' }
    case 'third_carry':
      return { label: '3rd carry', className: 'bg-raised text-muted' }
    default:
      return { label: 'new', className: 'bg-surface text-subtle' }
  }
}

function getRetentionBadge(rank: number | null): string {
  if (rank === 1) return '\u{1F3C6}'
  if (rank === 2) return '\u{1F948}'
  if (rank === 3) return '\u{1F949}'
  return ''
}

function getAutoScore(candidate: ArenaCandidate): string {
  const overall = candidate.auto_scores?.overall_score
  if (typeof overall === 'number' && Number.isFinite(overall)) {
    return overall.toFixed(3)
  }
  return 'n/a'
}

function getCandidateById(
  candidates: ArenaCandidate[],
  id: string,
): ArenaCandidate | undefined {
  return candidates.find((c) => c.candidate_id === id)
}

function getMatchAudioUrl(r2Key: string | null): string {
  if (!r2Key) return ''
  const apiUrl = import.meta.env.VITE_API_URL ?? ''
  return `${apiUrl}/v1/arena/audio/${encodeURIComponent(r2Key)}`
}

function getDisplaySamples(
  match: ArenaMatch,
): { sample1: { candidateId: string; audioUrl: string }; sample2: { candidateId: string; audioUrl: string } } {
  const isSwapped = match.display_order === 'ba'
  const firstId = isSwapped ? match.candidate_b_id : match.candidate_a_id
  const secondId = isSwapped ? match.candidate_a_id : match.candidate_b_id
  const firstAudio = isSwapped ? match.audio_b_r2_key : match.audio_a_r2_key
  const secondAudio = isSwapped ? match.audio_a_r2_key : match.audio_b_r2_key

  return {
    sample1: { candidateId: firstId, audioUrl: getMatchAudioUrl(firstAudio) },
    sample2: { candidateId: secondId, audioUrl: getMatchAudioUrl(secondAudio) },
  }
}

function mapVoteToWinner(
  vote: 'sample1' | 'sample2' | 'tie' | 'both_bad',
  displayOrder: 'ab' | 'ba',
): ArenaVoteWinner {
  if (vote === 'tie') return 'tie'
  if (vote === 'both_bad') return 'both_bad'
  if (vote === 'sample1') return displayOrder === 'ab' ? 'a' : 'b'
  return displayOrder === 'ab' ? 'b' : 'a'
}

function SimpleAudioPlayer({ src, label }: { src: string; label: string }) {
  const audioRef = useRef<HTMLAudioElement>(null)
  const [playing, setPlaying] = useState(false)
  const [progress, setProgress] = useState(0)
  const [duration, setDuration] = useState(0)

  function toggle() {
    const el = audioRef.current
    if (!el) return
    if (playing) {
      el.pause()
      setPlaying(false)
    } else {
      el.play().catch(() => {})
      setPlaying(true)
    }
  }

  return (
    <div className="rounded-xl border border-edge bg-surface p-4 flex-1 min-w-[200px]">
      <div className="text-heading text-sm font-semibold mb-3">{label}</div>
      {src ? (
        <>
          <div className="flex items-center gap-3">
            <button
              onClick={toggle}
              className="w-10 h-10 rounded-full bg-accent/20 text-accent flex items-center justify-center hover:bg-accent/30 transition-colors shrink-0"
              type="button"
              aria-label={playing ? 'Pause' : 'Play'}
            >
              {playing ? (
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                  <rect x="6" y="4" width="4" height="16" rx="1" />
                  <rect x="14" y="4" width="4" height="16" rx="1" />
                </svg>
              ) : (
                <svg className="w-4 h-4 ml-0.5" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M8 5.14v13.72a1 1 0 0 0 1.5.86l11.24-7.36a1 1 0 0 0 0-1.72L9.5 4.28A1 1 0 0 0 8 5.14z" />
                </svg>
              )}
            </button>
            <div className="flex-1 h-1.5 bg-edge rounded-full overflow-hidden">
              <div
                className="h-full bg-accent rounded-full transition-[width] duration-100"
                style={{ width: duration > 0 ? `${(progress / duration) * 100}%` : '0%' }}
              />
            </div>
            <span className="text-muted text-[10px] font-mono shrink-0">
              {duration > 0 ? `${Math.floor(progress)}/${Math.floor(duration)}s` : '--'}
            </span>
          </div>
          <audio
            ref={audioRef}
            src={src}
            preload="metadata"
            onTimeUpdate={() => setProgress(audioRef.current?.currentTime ?? 0)}
            onLoadedMetadata={() => setDuration(audioRef.current?.duration ?? 0)}
            onEnded={() => { setPlaying(false); setProgress(0) }}
          />
        </>
      ) : (
        <div className="text-muted text-xs py-4 text-center">No audio available</div>
      )}
    </div>
  )
}

export function VoiceArena() {
  const { voiceId = '' } = useParams()

  const [step, setStep] = useState<ArenaStep>('setup')
  const [session, setSession] = useState<ArenaSessionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [actionMessage, setActionMessage] = useState('')

  const [testTextsRaw, setTestTextsRaw] = useState(DEFAULT_TEST_TEXTS.join('\n'))
  const [seed, setSeed] = useState(42)

  const [confidence, setConfidence] = useState<ArenaVoteConfidence>('clear')
  const [votedMatchIds, setVotedMatchIds] = useState<Set<string>>(new Set())
  const [revealedMatchId, setRevealedMatchId] = useState<string | null>(null)
  const [voting, setVoting] = useState(false)
  const [promoting, setPromoting] = useState(false)
  const [completing, setCompleting] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const candidates = session?.candidates ?? []
  const allMatches = session?.matches ?? []
  const currentRound = session?.current_round ?? 0
  const totalRounds = session?.total_rounds ?? null

  const roundMatches = allMatches.filter((m) => m.round_number === currentRound)
  const unvotedMatches = roundMatches.filter((m) => m.winner === null && !votedMatchIds.has(m.match_id))
  const currentMatch = unvotedMatches[0] ?? null
  const votedInRound = roundMatches.filter((m) => m.winner !== null || votedMatchIds.has(m.match_id)).length
  const totalInRound = roundMatches.length

  const candidateCount = candidates.length
  const algorithm = session?.algorithm ?? (candidateCount <= 6 ? 'round_robin' : 'swiss')

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  useEffect(() => {
    return () => stopPolling()
  }, [stopPolling])

  async function handleStartArena() {
    if (!voiceId) return
    setLoading(true)
    setError('')
    try {
      const testTexts = testTextsRaw
        .split('\n')
        .map((t) => t.trim())
        .filter((t) => t.length > 0)

      if (testTexts.length === 0) {
        setError('At least one test text is required.')
        setLoading(false)
        return
      }

      const created = await createArenaSession(voiceId, testTexts, seed)
      setSession(created)
      setStep('generating')

      await generateArenaAudio(created.session_id)
      startPolling(created.session_id)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create arena session')
      setStep('setup')
    } finally {
      setLoading(false)
    }
  }

  function startPolling(sessionId: string) {
    stopPolling()
    let pollInFlight = false
    pollRef.current = setInterval(async () => {
      if (pollInFlight) return
      pollInFlight = true
      try {
        const updated = await getArenaSession(sessionId)
        setSession(updated)

        if (updated.status === 'active') {
          stopPolling()
          setStep('voting')
        } else if (updated.status === 'completed') {
          stopPolling()
          setStep('results')
        } else if (updated.status === 'cancelled') {
          stopPolling()
          setError('Arena session was cancelled.')
          setStep('setup')
        }
      } catch {
        // transient error, keep polling
      } finally {
        pollInFlight = false
      }
    }, 3000)
  }

  async function handleVote(vote: 'sample1' | 'sample2' | 'tie' | 'both_bad') {
    if (!currentMatch || !session) return
    setVoting(true)
    setError('')
    try {
      const winner = mapVoteToWinner(vote, currentMatch.display_order)
      const result = await submitArenaVote(currentMatch.match_id, winner, confidence)
      setRevealedMatchId(currentMatch.match_id)

      if (result.round_complete) {
        const updated = await getArenaSession(session.session_id)
        if (updated.status === 'completed') {
          setSession(updated)
          setTimeout(() => {
            setRevealedMatchId(null)
            setStep('results')
          }, 2000)
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit vote')
    } finally {
      setVoting(false)
    }
  }

  function handleNextMatch() {
    if (revealedMatchId) {
      setVotedMatchIds((prev) => new Set([...prev, revealedMatchId]))
    }
    setRevealedMatchId(null)

    if (session?.status === 'completed') {
      setStep('results')
      return
    }

    if (unvotedMatches.length <= 1 && session) {
      void getArenaSession(session.session_id).then((updated) => {
        setSession(updated)
        if (updated.status === 'completed') {
          setStep('results')
        }
      })
    }
  }

  async function handleComplete() {
    if (!session) return
    setCompleting(true)
    setError('')
    try {
      await completeArenaSession(session.session_id)
      const updated = await getArenaSession(session.session_id)
      setSession(updated)
      setStep('results')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to complete session')
    } finally {
      setCompleting(false)
    }
  }

  async function handlePromote() {
    if (!session) return
    setPromoting(true)
    setError('')
    try {
      await promoteArenaWinner(session.session_id)
      setActionMessage('Winner promoted successfully!')
      const updated = await getArenaSession(session.session_id)
      setSession(updated)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to promote winner')
    } finally {
      setPromoting(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-heading text-lg font-bold">TTS Arena</h2>
          <p className="text-subtle text-sm mt-1">
            Blind A/B evaluation of model checkpoints via Swiss tournament or round-robin.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {['setup', 'generating', 'voting', 'results'].map((s, i) => (
            <div
              key={s}
              className={`flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-wider ${
                s === step ? 'text-accent' : 'text-muted'
              }`}
            >
              {i > 0 && <span className="text-edge">/</span>}
              <span className={s === step ? 'border-b border-accent pb-0.5' : ''}>{s}</span>
            </div>
          ))}
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-error/20 bg-error-dim px-3 py-2 text-sm text-error">{error}</div>
      )}
      {actionMessage && (
        <div className="rounded-lg border border-accent/20 bg-accent-dim px-4 py-3 text-accent text-sm">{actionMessage}</div>
      )}

      {step === 'setup' && (
        <SetupStep
          candidates={candidates}
          algorithm={algorithm}
          candidateCount={candidateCount}
          testTextsRaw={testTextsRaw}
          onTestTextsChange={setTestTextsRaw}
          seed={seed}
          onSeedChange={setSeed}
          loading={loading}
          onStart={handleStartArena}
          voiceId={voiceId}
          session={session}
        />
      )}

      {step === 'generating' && (() => {
        const gp = session?.generation_progress
        const pct = gp && gp.total > 0 ? Math.round((gp.completed / gp.total) * 100) : 0
        return (
          <div className="rounded-xl border border-edge bg-raised p-8 text-center space-y-4">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-accent/10">
              <svg className="w-8 h-8 text-accent animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
              </svg>
            </div>
            <div className="text-heading text-lg font-semibold">Generating Audio</div>
            {gp ? (
              <>
                <div className="w-64 mx-auto h-2 bg-edge rounded-full overflow-hidden">
                  <div
                    className="h-full bg-accent rounded-full transition-[width] duration-500"
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <p className="text-subtle text-sm">
                  {gp.completed}/{gp.total} audio samples generated{gp.failed > 0 ? ` (${gp.failed} failed)` : ''}
                </p>
              </>
            ) : (
              <p className="text-subtle text-sm">
                Generating audio for all {candidateCount} candidates...
              </p>
            )}
            <p className="text-muted text-xs font-mono">
              session={session?.session_id.slice(0, 12)} status={session?.status ?? 'unknown'}
            </p>
          </div>
        )
      })()}

      {step === 'voting' && currentMatch && (
        <VotingStep
          key={currentMatch.match_id}
          match={currentMatch}
          candidates={candidates}
          currentRound={currentRound}
          totalRounds={totalRounds}
          votedInRound={votedInRound}
          totalInRound={totalInRound}
          confidence={confidence}
          onConfidenceChange={setConfidence}
          onVote={handleVote}
          voting={voting}
          revealedMatchId={revealedMatchId}
          onNextMatch={handleNextMatch}
        />
      )}

      {step === 'voting' && !currentMatch && session?.status !== 'completed' && (
        <div className="rounded-xl border border-edge bg-raised p-8 text-center space-y-4">
          <div className="text-heading text-lg font-semibold">Round Complete</div>
          <p className="text-subtle text-sm">
            All matches in the current round have been voted on.
            {session?.status === 'active' && ' Waiting for next round...'}
          </p>
          {session?.status === 'active' && (
            <button
              onClick={handleComplete}
              disabled={completing}
              className="bg-accent hover:bg-accent-light text-void font-semibold text-sm px-6 py-2.5 rounded-lg disabled:opacity-40 transition-colors"
              type="button"
            >
              {completing ? 'Completing...' : 'Finalize Arena'}
            </button>
          )}
        </div>
      )}

      {step === 'results' && session && (
        <ResultsStep
          session={session}
          candidates={candidates}
          onPromote={handlePromote}
          promoting={promoting}
          voiceId={voiceId}
        />
      )}
    </div>
  )
}

function SetupStep({
  candidates,
  algorithm,
  candidateCount,
  testTextsRaw,
  onTestTextsChange,
  seed,
  onSeedChange,
  loading,
  onStart,
  voiceId,
  session,
}: {
  candidates: ArenaCandidate[]
  algorithm: string
  candidateCount: number
  testTextsRaw: string
  onTestTextsChange: (v: string) => void
  seed: number
  onSeedChange: (v: number) => void
  loading: boolean
  onStart: () => void
  voiceId: string
  session: ArenaSessionResponse | null
}) {
  return (
    <div className="grid lg:grid-cols-[1fr_340px] gap-6">
      <div className="bg-raised border border-edge rounded-xl p-5 space-y-4">
        <div>
          <h3 className="text-heading text-sm font-semibold">Candidate Checkpoints</h3>
          <p className="text-subtle text-xs mt-1">
            {candidateCount > 0
              ? `${candidateCount} candidates assembled. Algorithm: ${algorithm === 'round_robin' ? 'Round Robin' : 'Swiss'}.`
              : 'Candidates will be auto-assembled when you start the arena.'}
          </p>
        </div>

        {candidates.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-muted text-[10px] font-mono uppercase tracking-wider border-b border-edge">
                  <th className="text-left py-2 pr-3">Rank</th>
                  <th className="text-left py-2 pr-3">Source</th>
                  <th className="text-left py-2 pr-3">Run</th>
                  <th className="text-left py-2 pr-3">Epoch</th>
                  <th className="text-left py-2">Auto Score</th>
                </tr>
              </thead>
              <tbody>
                {[...candidates]
                  .sort((a, b) => (a.seed_rank ?? 99) - (b.seed_rank ?? 99))
                  .map((c) => {
                    const badge = getSourceBadge(c.source)
                    return (
                      <tr key={c.candidate_id} className="border-b border-edge/50">
                        <td className="py-2 pr-3 text-primary font-mono">{c.seed_rank ?? '—'}</td>
                        <td className="py-2 pr-3">
                          <span className={`rounded-full px-2 py-0.5 text-[10px] font-mono ${badge.className}`}>
                            {badge.label}
                          </span>
                        </td>
                        <td className="py-2 pr-3 text-primary text-xs">{c.run_name ?? '—'}</td>
                        <td className="py-2 pr-3 text-muted font-mono text-xs">{c.epoch ?? '—'}</td>
                        <td className="py-2 text-muted font-mono text-xs">{getAutoScore(c)}</td>
                      </tr>
                    )
                  })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="rounded-xl border border-dashed border-edge bg-surface px-4 py-8 text-center">
            <div className="text-primary text-sm font-semibold">Ready to assemble</div>
            <p className="mt-2 text-subtle text-xs">
              Candidates will be auto-assembled from your training checkpoints when you start the arena.
              Requires at least 2 checkpoints that pass validation gates.
            </p>
          </div>
        )}

        {session && (
          <div className="text-muted text-[10px] font-mono">
            session={session.session_id.slice(0, 12)} algorithm={session.algorithm} rounds={session.total_rounds ?? 'auto'}
          </div>
        )}
      </div>

      <div className="space-y-4">
        <div className="bg-raised border border-edge rounded-xl p-5 space-y-4">
          <div>
            <h3 className="text-heading text-sm font-semibold">Arena Settings</h3>
          </div>

          <div>
            <label className="text-subtle text-xs font-medium mb-1.5 block">Test Texts (one per line)</label>
            <textarea
              value={testTextsRaw}
              onChange={(e) => onTestTextsChange(e.target.value)}
              rows={6}
              className="w-full bg-surface border border-edge rounded-lg px-3 py-2 text-sm text-primary placeholder:text-muted focus:border-accent transition-colors resize-none"
            />
            <p className="text-muted text-[10px] mt-1">
              {testTextsRaw.split('\n').filter((t) => t.trim()).length} text(s)
            </p>
          </div>

          <div>
            <label className="text-subtle text-xs font-medium mb-1.5 block">Seed</label>
            <input
              type="number"
              value={seed}
              onChange={(e) => onSeedChange(parseInt(e.target.value, 10) || 42)}
              className="w-full bg-surface border border-edge rounded-lg px-3 py-2 text-sm text-primary font-mono focus:border-accent transition-colors"
            />
          </div>

          <div className="text-muted text-[10px] font-mono space-y-1">
            <div>algorithm={algorithm === 'round_robin' ? 'round_robin' : 'swiss'}</div>
            <div>candidates={candidateCount}</div>
          </div>

          <button
            onClick={onStart}
            disabled={loading}
            className="w-full bg-accent hover:bg-accent-light text-void font-semibold text-sm py-2.5 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            type="button"
          >
            {loading ? 'Assembling candidates...' : 'Start Arena'}
          </button>
        </div>
      </div>
    </div>
  )
}

function VotingStep({
  match,
  candidates,
  currentRound,
  totalRounds,
  votedInRound,
  totalInRound,
  confidence,
  onConfidenceChange,
  onVote,
  voting,
  revealedMatchId,
  onNextMatch,
}: {
  match: ArenaMatch
  candidates: ArenaCandidate[]
  currentRound: number
  totalRounds: number | null
  votedInRound: number
  totalInRound: number
  confidence: ArenaVoteConfidence
  onConfidenceChange: (c: ArenaVoteConfidence) => void
  onVote: (vote: 'sample1' | 'sample2' | 'tie' | 'both_bad') => void
  voting: boolean
  revealedMatchId: string | null
  onNextMatch: () => void
}) {
  const { sample1, sample2 } = getDisplaySamples(match)
  const isRevealed = revealedMatchId === match.match_id

  const candidateA = getCandidateById(candidates, sample1.candidateId)
  const candidateB = getCandidateById(candidates, sample2.candidateId)

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div className="flex items-center justify-between text-xs">
        <div className="text-muted font-mono">
          Round {currentRound}{totalRounds ? `/${totalRounds}` : ''} — Match {votedInRound + 1}/{totalInRound}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-subtle">Confidence:</span>
          <button
            onClick={() => onConfidenceChange('clear')}
            className={`rounded-lg border px-2.5 py-1 text-[11px] font-semibold transition-colors ${
              confidence === 'clear'
                ? 'border-accent bg-accent-dim text-accent'
                : 'border-edge text-muted hover:border-accent/40'
            }`}
            type="button"
          >
            Clear win
          </button>
          <button
            onClick={() => onConfidenceChange('slight')}
            className={`rounded-lg border px-2.5 py-1 text-[11px] font-semibold transition-colors ${
              confidence === 'slight'
                ? 'border-accent bg-accent-dim text-accent'
                : 'border-edge text-muted hover:border-accent/40'
            }`}
            type="button"
          >
            Slight preference
          </button>
        </div>
      </div>

      <div className="flex gap-4">
        <SimpleAudioPlayer src={sample1.audioUrl} label="Sample 1" />
        <SimpleAudioPlayer src={sample2.audioUrl} label="Sample 2" />
      </div>

      {!isRevealed && (
        <div className="flex items-center justify-center gap-3">
          <button
            onClick={() => onVote('sample1')}
            disabled={voting}
            className="bg-accent hover:bg-accent-light text-void font-semibold text-sm px-5 py-2.5 rounded-lg disabled:opacity-40 transition-colors"
            type="button"
          >
            Sample 1
          </button>
          <button
            onClick={() => onVote('sample2')}
            disabled={voting}
            className="bg-accent hover:bg-accent-light text-void font-semibold text-sm px-5 py-2.5 rounded-lg disabled:opacity-40 transition-colors"
            type="button"
          >
            Sample 2
          </button>
          <button
            onClick={() => onVote('tie')}
            disabled={voting}
            className="border border-edge hover:border-accent text-primary hover:text-accent font-semibold text-sm px-5 py-2.5 rounded-lg disabled:opacity-40 transition-colors"
            type="button"
          >
            Both Good
          </button>
          <button
            onClick={() => onVote('both_bad')}
            disabled={voting}
            className="border border-error/30 hover:border-error text-error font-semibold text-sm px-5 py-2.5 rounded-lg disabled:opacity-40 transition-colors"
            type="button"
          >
            Both Bad
          </button>
        </div>
      )}

      {isRevealed && (
        <div className="rounded-xl border border-accent/30 bg-accent-dim/10 p-5 space-y-4 animate-fade-in">
          <div className="text-heading text-sm font-semibold text-center">Reveal</div>
          <div className="grid grid-cols-2 gap-4">
            <RevealCard label="Sample 1" candidate={candidateA} />
            <RevealCard label="Sample 2" candidate={candidateB} />
          </div>
          <div className="text-center">
            <button
              onClick={onNextMatch}
              className="bg-accent hover:bg-accent-light text-void font-semibold text-sm px-6 py-2.5 rounded-lg transition-colors"
              type="button"
            >
              Next Match
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

function RevealCard({ label, candidate }: { label: string; candidate: ArenaCandidate | undefined }) {
  if (!candidate) {
    return (
      <div className="rounded-lg border border-edge bg-surface p-3 text-muted text-xs text-center">
        Unknown candidate
      </div>
    )
  }

  return (
    <div className="rounded-lg border border-edge bg-surface p-3 space-y-1">
      <div className="text-subtle text-[10px] font-mono uppercase">{label}</div>
      <div className="text-primary text-sm font-semibold">{candidate.run_name ?? 'unknown'}</div>
      <div className="text-muted text-[10px] font-mono">
        epoch={candidate.epoch ?? 'n/a'} auto={getAutoScore(candidate)}
      </div>
      <div className="text-muted text-[10px] font-mono">
        source={candidate.source} rank={candidate.seed_rank ?? 'n/a'}
      </div>
    </div>
  )
}

function ResultsStep({
  session,
  candidates,
  onPromote,
  promoting,
  voiceId,
}: {
  session: ArenaSessionResponse
  candidates: ArenaCandidate[]
  onPromote: () => void
  promoting: boolean
  voiceId: string
}) {
  const ranked = [...candidates].sort((a, b) => {
    if (a.final_rank !== null && b.final_rank !== null) return a.final_rank - b.final_rank
    if (a.final_rank !== null) return -1
    if (b.final_rank !== null) return 1
    const aWins = a.wins - a.losses
    const bWins = b.wins - b.losses
    if (aWins !== bWins) return bWins - aWins
    return b.buchholz - a.buchholz
  })

  const winnerId = session.winner_candidate_id
  const isPromoted = session.promoted

  return (
    <div className="space-y-6">
      <div className="bg-raised border border-edge rounded-xl p-5">
        <h3 className="text-heading text-sm font-semibold mb-4">Final Rankings</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-muted text-[10px] font-mono uppercase tracking-wider border-b border-edge">
                <th className="text-left py-2 pr-3">Rank</th>
                <th className="text-left py-2 pr-3">Status</th>
                <th className="text-left py-2 pr-3">Run</th>
                <th className="text-left py-2 pr-3">Epoch</th>
                <th className="text-left py-2 pr-3">W-L-T</th>
                <th className="text-left py-2 pr-3">Buchholz</th>
                <th className="text-left py-2">Auto Score</th>
              </tr>
            </thead>
            <tbody>
              {ranked.map((c) => {
                const isWinner = c.candidate_id === winnerId
                const badge = getRetentionBadge(c.final_rank)
                return (
                  <tr
                    key={c.candidate_id}
                    className={`border-b border-edge/50 ${isWinner ? 'bg-accent-dim/10' : ''}`}
                  >
                    <td className="py-2.5 pr-3 text-primary font-mono font-bold">
                      {badge} {c.final_rank ?? '—'}
                    </td>
                    <td className="py-2.5 pr-3">
                      <span
                        className={`rounded-full px-2 py-0.5 text-[10px] font-mono ${
                          c.retention_status === 'champion'
                            ? 'bg-accent-dim text-accent'
                            : c.retention_status === 'second'
                              ? 'bg-warning-dim text-warning'
                              : c.retention_status === 'third'
                                ? 'bg-raised text-muted'
                                : c.retention_status === 'eliminated'
                                  ? 'bg-error-dim text-error'
                                  : 'bg-surface text-subtle'
                        }`}
                      >
                        {c.retention_status}
                      </span>
                    </td>
                    <td className="py-2.5 pr-3 text-primary text-xs">{c.run_name ?? '—'}</td>
                    <td className="py-2.5 pr-3 text-muted font-mono text-xs">{c.epoch ?? '—'}</td>
                    <td className="py-2.5 pr-3 text-muted font-mono text-xs">
                      {c.wins}-{c.losses}-{c.ties}
                    </td>
                    <td className="py-2.5 pr-3 text-muted font-mono text-xs">{c.buchholz.toFixed(1)}</td>
                    <td className="py-2.5 text-muted font-mono text-xs">{getAutoScore(c)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        {winnerId && !isPromoted && (
          <button
            onClick={onPromote}
            disabled={promoting}
            className="bg-accent hover:bg-accent-light text-void font-semibold text-sm px-6 py-2.5 rounded-lg disabled:opacity-40 transition-colors"
            type="button"
          >
            {promoting ? 'Promoting...' : 'Promote Winner'}
          </button>
        )}
        {isPromoted && (
          <span className="text-accent text-sm font-semibold">Winner has been promoted.</span>
        )}
        <Link
          to={`/voices/${voiceId}/arena/calibration`}
          className="inline-flex items-center rounded-lg border border-edge px-4 py-2.5 text-[11px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent"
        >
          View Calibration Insights
        </Link>
      </div>

      <div className="text-muted text-[10px] font-mono">
        session={session.session_id.slice(0, 12)} algorithm={session.algorithm} rounds={session.current_round}/{session.total_rounds ?? 'auto'} status={session.status}
      </div>
    </div>
  )
}
