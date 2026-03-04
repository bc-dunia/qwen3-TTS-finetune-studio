import { useState, useEffect } from 'react'
import { Outlet, NavLink, useLocation } from 'react-router'

interface NavItem {
  path: string
  label: string
  icon: string
  end?: boolean
}

const NAV_ITEMS: NavItem[] = [
  { path: '/', label: 'Dashboard', icon: 'dashboard', end: true },
  { path: '/voices', label: 'Voices', icon: 'mic' },
  { path: '/playground', label: 'Playground', icon: 'play' },
  { path: '/training', label: 'Training', icon: 'training' },
]

export function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [apiKey, setApiKey] = useState(() => localStorage.getItem('xi-api-key') ?? '')
  const [showKey, setShowKey] = useState(false)
  const location = useLocation()

  useEffect(() => {
    setSidebarOpen(false)
  }, [location.pathname])

  function saveApiKey(key: string) {
    setApiKey(key)
    localStorage.setItem('xi-api-key', key)
  }

  return (
    <div className="flex h-screen bg-void overflow-hidden">
      {/* Mobile backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30 lg:hidden"
          onClick={() => setSidebarOpen(false)}
          onKeyDown={(e) => { if (e.key === 'Escape') setSidebarOpen(false) }}
          role="button"
          tabIndex={-1}
          aria-label="Close sidebar"
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed inset-y-0 left-0 z-40 w-64 bg-surface border-r border-edge
          flex flex-col
          transform transition-transform duration-200 ease-out
          lg:relative lg:translate-x-0
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
        `}
      >
        {/* Logo */}
        <div className="px-5 py-5 border-b border-edge">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-accent-dim rounded-lg flex items-center justify-center shrink-0">
              <svg className="w-5 h-5 text-accent" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M2 12l10 5 10-5" />
                <path d="M2 17l10 5 10-5" />
                <path d="M12 2L2 7l10 5 10-5-10-5z" />
              </svg>
            </div>
            <div>
              <h1 className="text-heading font-bold text-sm tracking-wide leading-none">
                QwenTTS
              </h1>
              <p className="text-muted font-mono text-[10px] tracking-[0.25em] uppercase mt-0.5">
                Studio
              </p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.end}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors duration-150 ${
                  isActive
                    ? 'bg-accent-dim text-accent'
                    : 'text-subtle hover:text-primary hover:bg-raised'
                }`
              }
            >
              <NavIcon name={item.icon} />
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>

        {/* API Key Section */}
        <div className="px-4 py-4 border-t border-edge">
          <label className="text-muted text-[10px] font-mono uppercase tracking-widest mb-2 block">
            API Key
          </label>
          <div className="flex gap-1.5">
            <input
              type={showKey ? 'text' : 'password'}
              value={apiKey}
              onChange={(e) => saveApiKey(e.target.value)}
              placeholder="xi-api-key..."
              className="flex-1 min-w-0 bg-raised border border-edge rounded-md px-2.5 py-1.5 text-xs text-primary placeholder:text-muted font-mono focus:border-accent transition-colors"
            />
            <button
              onClick={() => setShowKey(!showKey)}
              className="px-2 text-muted hover:text-primary text-[10px] font-mono uppercase shrink-0"
              type="button"
            >
              {showKey ? 'Hide' : 'Show'}
            </button>
          </div>
          <div className="mt-2.5 flex items-center gap-2">
            <div
              className={`w-1.5 h-1.5 rounded-full ${
                apiKey
                  ? 'bg-accent shadow-[0_0_6px_rgba(16,185,129,0.5)]'
                  : 'bg-error shadow-[0_0_6px_rgba(239,68,68,0.4)]'
              }`}
            />
            <span className="text-muted text-[10px] font-mono tracking-wider">
              {apiKey ? 'CONNECTED' : 'NO KEY'}
            </span>
          </div>
        </div>
      </aside>

      {/* Main area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Mobile header */}
        <header className="lg:hidden flex items-center h-14 px-4 bg-surface border-b border-edge shrink-0">
          <button
            onClick={() => setSidebarOpen(true)}
            className="p-2 -ml-2 text-subtle hover:text-primary"
            type="button"
            aria-label="Open menu"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <span className="ml-3 text-heading font-bold text-sm tracking-wide">
            QwenTTS Studio
          </span>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-auto">
          <div className="p-6 lg:p-8 max-w-7xl mx-auto">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  )
}

// ── Nav Icons ──────────────────────────────────────────────────────────────────

function NavIcon({ name }: { name: string }) {
  const cls = "w-5 h-5 shrink-0"

  switch (name) {
    case 'dashboard':
      return (
        <svg className={cls} viewBox="0 0 24 24" fill="currentColor">
          <rect x="3" y="3" width="8" height="8" rx="1.5" />
          <rect x="13" y="3" width="8" height="8" rx="1.5" />
          <rect x="3" y="13" width="8" height="8" rx="1.5" />
          <rect x="13" y="13" width="8" height="8" rx="1.5" />
        </svg>
      )
    case 'mic':
      return (
        <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <rect x="9" y="2" width="6" height="11" rx="3" fill="currentColor" stroke="none" />
          <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
          <line x1="12" y1="19" x2="12" y2="22" />
          <line x1="8" y1="22" x2="16" y2="22" />
        </svg>
      )
    case 'play':
      return (
        <svg className={cls} viewBox="0 0 24 24" fill="currentColor">
          <path d="M8 5.14v14.72a1 1 0 0 0 1.5.86l11.24-7.36a1 1 0 0 0 0-1.72L9.5 4.28A1 1 0 0 0 8 5.14z" />
        </svg>
      )
    case 'training':
      return (
        <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="22,12 18,12 15,21 9,3 6,12 2,12" />
        </svg>
      )
    default:
      return <div className={cls} />
  }
}
