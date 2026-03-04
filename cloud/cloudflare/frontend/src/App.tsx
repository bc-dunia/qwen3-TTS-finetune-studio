import { Routes, Route } from 'react-router'
import { Layout } from './components/Layout'
import { Dashboard } from './pages/Dashboard'
import { Voices } from './pages/Voices'
import { VoiceDetail } from './pages/VoiceDetail'
import { Playground } from './pages/Playground'
import { Training } from './pages/Training'

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="voices" element={<Voices />} />
        <Route path="voices/:voiceId" element={<VoiceDetail />} />
        <Route path="playground" element={<Playground />} />
        <Route path="training" element={<Training />} />
      </Route>
    </Routes>
  )
}
