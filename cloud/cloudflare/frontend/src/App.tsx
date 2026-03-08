import { Routes, Route } from 'react-router'
import { Layout } from './components/Layout'
import { Dashboard } from './pages/Dashboard'
import { Voices } from './pages/Voices'
import { VoiceDetail } from './pages/VoiceDetail'
import { VoiceCompare } from './pages/VoiceCompare'
import { VoiceDataset } from './pages/VoiceDataset'
import { Playground } from './pages/Playground'
import { Training } from './pages/Training'

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="voices" element={<Voices />} />
        <Route path="voices/:voiceId" element={<VoiceDetail />} />
        <Route path="voices/:voiceId/dataset" element={<VoiceDataset />} />
        <Route path="voices/:voiceId/compare" element={<VoiceCompare />} />
        <Route path="playground" element={<Playground />} />
        <Route path="training" element={<Training />} />
      </Route>
    </Routes>
  )
}
