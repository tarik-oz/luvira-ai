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
  if (typeof window === 'undefined') return

  // Inject GA script (gtag.js)
  const existing = document.querySelector(`script[src*="googletagmanager.com/gtag/js?id=${gaId}"]`)
  if (!existing) {
    const s = document.createElement('script')
    s.async = true
    s.src = `https://www.googletagmanager.com/gtag/js?id=${gaId}`
    document.head.appendChild(s)
  }

  // Initialize dataLayer and official gtag shim
  window.dataLayer = window.dataLayer || []
  // Use function form to mirror Google's snippet precisely
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(window as any).gtag = function gtag() {
    // eslint-disable-next-line prefer-rest-params, @typescript-eslint/no-explicit-any
    ;(window.dataLayer as any[]).push(arguments as any)
  }

  // Default Consent (adjust as needed)
  // Grant analytics_storage so measurement works in regions requiring Consent Mode v2
  pushGtag('consent', 'default', {
    ad_storage: 'denied',
    ad_user_data: 'denied',
    ad_personalization: 'denied',
    analytics_storage: 'granted',
  })

  const isDebug = !!sessionStorage.getItem('GA_DEBUG')

  // Initialize GA and avoid auto page_view; we'll send page views via config on route changes
  pushGtag('js', new Date())
  pushGtag('config', gaId, { send_page_view: false, debug_mode: isDebug } as AnalyticsParams)

  // Helper to send SPA page views via config (recommended for SPAs)
  const sendPageView = () => {
    pushGtag('config', gaId, {
      page_title: document.title,
      page_location: window.location.href,
      page_path: window.location.pathname + window.location.search,
    } as AnalyticsParams)
  }

  // First view and subsequent navigations
  try {
    router.isReady().then(() => sendPageView())
  } catch {
    // Fallback if isReady is not available for any reason
    sendPageView()
  }

  router.afterEach(() => sendPageView())
}

export function trackEvent(action: string, params?: AnalyticsParams): void {
  const isDebug = !!sessionStorage.getItem('GA_DEBUG')
  pushGtag('event', action, {
    ...(params || {}),
    debug_mode: isDebug,
  } as AnalyticsParams)
}
