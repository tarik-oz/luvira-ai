// Lightweight Google Analytics (GA4) helper with strict typing
import type { Router } from 'vue-router'

declare global {
  interface Window {
    dataLayer: unknown[]
    gtag?: (...args: unknown[]) => void
  }
}

export type AnalyticsParams = Record<string, string | number | boolean | null | undefined>

function pushGtag(...args: unknown[]): void {
  if (typeof window === 'undefined') return
  if (typeof window.gtag === 'function') {
    window.gtag(...args)
    return
  }
  window.dataLayer = window.dataLayer || []
  window.dataLayer.push(args)
}

export function initAnalytics(gaId: string, router: Router): void {
  // Inject GA script
  const existing = document.querySelector(`script[src*="googletagmanager.com/gtag/js?id=${gaId}"]`)
  if (!existing) {
    const s = document.createElement('script')
    s.async = true
    s.src = `https://www.googletagmanager.com/gtag/js?id=${gaId}`
    document.head.appendChild(s)
  }

  // Initialize dataLayer and gtag
  window.dataLayer = window.dataLayer || []
  window.gtag = (...gtagArgs: unknown[]) => {
    window.dataLayer.push(gtagArgs)
  }

  pushGtag('js', new Date())
  pushGtag('config', gaId)

  // Track SPA route changes as page views
  router.afterEach((to) => {
    pushGtag('event', 'page_view', {
      page_title: document.title,
      page_location: window.location.href,
      page_path: to.fullPath,
    } as AnalyticsParams)
  })
}

export function trackEvent(action: string, params?: AnalyticsParams): void {
  pushGtag('event', action, params || {})
}
