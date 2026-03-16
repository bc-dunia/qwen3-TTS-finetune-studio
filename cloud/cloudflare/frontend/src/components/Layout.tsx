import { useState, useEffect } from 'react'
import { Outlet, NavLink, useLocation } from 'react-router'
import { useTheme } from '../hooks/useTheme'

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
  { path: '/queue', label: 'Queue', icon: 'queue' },
  { path: '/statistics', label: 'Statistics', icon: 'chart' },
]

export function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const location = useLocation()
  const { theme, effectiveTheme, toggleTheme } = useTheme()

  useEffect(() => {
    setSidebarOpen(false)
  }, [location.pathname])

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

        {/* Deployment status */}
        <div className="px-4 py-4 border-t border-edge space-y-3">
          <button
            type="button"
            onClick={toggleTheme}
            className="w-full inline-flex items-center justify-between rounded-lg border border-edge bg-raised px-3 py-2 text-xs font-semibold text-primary hover:border-accent hover:text-accent"
            title={`Theme: ${theme}`}
          >
            <span className="inline-flex items-center gap-2">
              {effectiveTheme === 'light' ? (
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="4" />
                  <path d="M12 2v2m0 16v2m10-10h-2M4 12H2m16.95 6.95-1.4-1.4M6.45 6.45l-1.4-1.4m0 13.9 1.4-1.4m11.1-11.1 1.4-1.4" />
                </svg>
              ) : (
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 12.79A9 9 0 1 1 11.21 3c0 0 0 0 0 0a7 7 0 1 0 9.79 9.79z" />
                </svg>
              )}
              <span>Theme</span>
            </span>
            <span className="text-[10px] font-mono uppercase tracking-wider text-muted">{theme}</span>
          </button>

          <label className="text-muted text-[10px] font-mono uppercase tracking-widest mb-2 block">Access</label>
          <div className="mt-2.5 flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-accent" />
            <span className="text-muted text-[10px] font-mono tracking-wider">PUBLIC MODE</span>
          </div>
          <p className="mt-2 text-[11px] text-subtle">
            This deployment no longer requires entering an API key in the UI.
          </p>
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
    case 'queue':
      return (
        <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <line x1="8" y1="6" x2="21" y2="6" />
          <line x1="8" y1="12" x2="21" y2="12" />
          <line x1="8" y1="18" x2="21" y2="18" />
          <circle cx="4" cy="6" r="1" fill="currentColor" stroke="none" />
          <circle cx="4" cy="12" r="1" fill="currentColor" stroke="none" />
          <circle cx="4" cy="18" r="1" fill="currentColor" stroke="none" />
        </svg>
      )
    case 'chart':
      return (
        <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="3" y="12" width="4" height="9" rx="1" />
          <rect x="10" y="7" width="4" height="14" rx="1" />
          <rect x="17" y="3" width="4" height="18" rx="1" />
        </svg>
      )
    default:
      return <div className={cls} />
  }
}
