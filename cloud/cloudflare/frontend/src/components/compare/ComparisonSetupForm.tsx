import { useState } from 'react'
import type { VoiceSettings } from '../../lib/api'

function SettingSlider({
  label,
  value,
  onChange,
  min,
  max,
  step,
}: {
  label: string
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step: number
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <label className="text-subtle text-xs font-medium">{label}</label>
        <span className="text-muted text-[10px] font-mono">{value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-accent"
      />
    </div>
  )
}

export function ComparisonSetupForm({
  text,
  onTextChange,
  seed,
  onSeedChange,
  settings,
  onSettingChange,
  stylePrompt,
  onStylePromptChange,
  instruct,
  onInstructChange,
  supportsPromptControls,
  trainingResetAt,
  archivedJobsCount,
}: {
  text: string
  onTextChange: (text: string) => void
  seed: number
  onSeedChange: (seed: number) => void
  settings: VoiceSettings
  onSettingChange: (key: keyof VoiceSettings, value: number) => void
  stylePrompt: string
  onStylePromptChange: (value: string) => void
  instruct: string
  onInstructChange: (value: string) => void
  supportsPromptControls: boolean
  trainingResetAt: number | null
  archivedJobsCount: number
}) {
  const [advancedOpen, setAdvancedOpen] = useState(false)

  return (
    <div className="rounded-xl border border-edge bg-raised p-5 space-y-4">
      <div>
        <h2 className="text-heading font-semibold text-sm">Comparison Setup</h2>
        <p className="text-subtle text-xs mt-1">
          Generate the same script across checkpoints to compare tone, pacing, and similarity.
        </p>
      </div>

      {/* Callouts */}
      {trainingResetAt !== null && (
        <div className="rounded-lg border border-warning/20 bg-warning-dim px-3 py-2 text-warning text-[11px]">
          Fresh-cycle mode is active. Older runs are hidden.
        </div>
      )}
      {archivedJobsCount > 0 && (
        <div className="text-muted text-[11px]">
          {archivedJobsCount} older run{archivedJobsCount !== 1 ? 's' : ''} hidden from this cycle.
        </div>
      )}

      {/* Always visible: Text + Seed */}
      <div className="grid sm:grid-cols-[1fr_120px] gap-3">
        <div>
          <label className="text-subtle text-xs font-medium mb-1.5 block">Comparison Text</label>
          <textarea
            value={text}
            onChange={(e) => onTextChange(e.target.value)}
            rows={4}
            className="w-full bg-surface border border-edge rounded-lg px-3 py-2 text-sm text-primary placeholder:text-muted focus:border-accent transition-colors resize-none"
          />
        </div>
        <div>
          <label className="text-subtle text-xs font-medium mb-1.5 block">Seed</label>
          <input
            type="number"
            value={seed}
            onChange={(e) => onSeedChange(parseInt(e.target.value, 10) || 1)}
            className="w-full bg-surface border border-edge rounded-lg px-3 py-2 text-sm text-primary font-mono focus:border-accent transition-colors"
          />
        </div>
      </div>

      {/* Accordion: Advanced Settings */}
      <div className="border-t border-edge pt-3">
        <button
          onClick={() => setAdvancedOpen((prev) => !prev)}
          className="flex items-center gap-2 text-xs font-semibold text-muted hover:text-primary transition-colors w-full"
          type="button"
        >
          <svg
            className={`w-3.5 h-3.5 transition-transform duration-200 ${advancedOpen ? 'rotate-90' : ''}`}
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="9 18 15 12 9 6" />
          </svg>
          Advanced Settings
        </button>

        {advancedOpen && (
          <div className="mt-3 space-y-4 animate-slide-up">
            <div className="space-y-4">
              <SettingSlider label="Stability" value={settings.stability} onChange={(v) => onSettingChange('stability', v)} min={0} max={1} step={0.01} />
              <SettingSlider label="Similarity Boost" value={settings.similarity_boost} onChange={(v) => onSettingChange('similarity_boost', v)} min={0} max={1} step={0.01} />
              <SettingSlider label="Style" value={settings.style} onChange={(v) => onSettingChange('style', v)} min={0} max={1} step={0.01} />
              <SettingSlider label="Speed" value={settings.speed} onChange={(v) => onSettingChange('speed', v)} min={0.5} max={2} step={0.05} />
            </div>

            <div className="space-y-3 pt-1 border-t border-edge">
              <div>
                <label className="text-subtle text-xs font-medium mb-1.5 block">Style Prompt</label>
                <textarea
                  value={stylePrompt}
                  onChange={(e) => onStylePromptChange(e.target.value)}
                  disabled={!supportsPromptControls}
                  rows={2}
                  placeholder="Preserve the speaker's distinctive phrasing, measured pauses, and sentence-ending emphasis."
                  className="w-full bg-surface border border-edge rounded-lg px-3 py-2 text-sm text-primary placeholder:text-muted disabled:opacity-50 disabled:cursor-not-allowed focus:border-accent transition-colors resize-none"
                />
              </div>
              <div>
                <label className="text-subtle text-xs font-medium mb-1.5 block">Instruct</label>
                <textarea
                  value={instruct}
                  onChange={(e) => onInstructChange(e.target.value)}
                  disabled={!supportsPromptControls}
                  rows={2}
                  placeholder="Keep the original speaking habit and intonation instead of smoothing it into a generic narration tone."
                  className="w-full bg-surface border border-edge rounded-lg px-3 py-2 text-sm text-primary placeholder:text-muted disabled:opacity-50 disabled:cursor-not-allowed focus:border-accent transition-colors resize-none"
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
