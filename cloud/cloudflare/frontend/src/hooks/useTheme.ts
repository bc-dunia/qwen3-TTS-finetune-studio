import { useEffect, useMemo, useState } from 'react'

export type ThemePreference = 'system' | 'light' | 'dark'
const THEME_STORAGE_KEY = 'theme-preference'

function getStoredTheme(): ThemePreference {
  const raw = localStorage.getItem(THEME_STORAGE_KEY)
  if (raw === 'light' || raw === 'dark' || raw === 'system') {
    return raw
  }
  return 'system'
}

function resolveTheme(theme: ThemePreference): 'light' | 'dark' {
  if (theme === 'light' || theme === 'dark') {
    return theme
  }
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
}

export function useTheme() {
  const [theme, setThemeState] = useState<ThemePreference>(() => getStoredTheme())

  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: light)')

    const apply = () => {
      document.documentElement.dataset.theme = resolveTheme(theme)
    }

    apply()
    const listener = () => {
      if (theme === 'system') {
        apply()
      }
    }

    media.addEventListener('change', listener)
    return () => {
      media.removeEventListener('change', listener)
    }
  }, [theme])

  const setTheme = (nextTheme: ThemePreference) => {
    setThemeState(nextTheme)
    localStorage.setItem(THEME_STORAGE_KEY, nextTheme)
  }

  const toggleTheme = () => {
    if (theme === 'system') {
      setTheme('light')
      return
    }
    if (theme === 'light') {
      setTheme('dark')
      return
    }
    setTheme('system')
  }

  const effectiveTheme = useMemo(() => resolveTheme(theme), [theme])

  return {
    theme,
    effectiveTheme,
    setTheme,
    toggleTheme,
  }
}
