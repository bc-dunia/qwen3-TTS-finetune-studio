import { Routes, Route, Navigate } from 'react-router'
import { Layout } from './components/Layout'
import { Dashboard } from './pages/Dashboard'
import { Voices } from './pages/Voices'
import { VoiceDetail } from './pages/VoiceDetail'
import { VoiceCompare } from './pages/VoiceCompare'
import { VoiceDataset } from './pages/VoiceDataset'
import { Playground } from './pages/Playground'
import { VoiceWorkspace, VoiceDefaultRedirect } from './pages/VoiceWorkspace'
import { VoiceTrainingTab } from './pages/VoiceTrainingTab'
import { QueuePage } from './pages/QueuePage'

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="voices" element={<Voices />} />
        <Route path="voices/:voiceId" element={<VoiceWorkspace />}>
          <Route index element={<VoiceDefaultRedirect />} />
          <Route path="generate" element={<VoiceDetail />} />
          <Route path="training" element={<VoiceTrainingTab />} />
          <Route path="dataset" element={<VoiceDataset />} />
          <Route path="compare" element={<VoiceCompare />} />
        </Route>
        <Route path="playground" element={<Playground />} />
        <Route path="queue" element={<QueuePage />} />
        <Route path="training" element={<Navigate to="/queue" replace />} />
      </Route>
    </Routes>
  )
}
