import './assets/css/main.css'
import { createApp } from 'vue'
import * as Sentry from '@sentry/vue'
import App from './App.vue'
import i18n from './i18n'
import router from './router'
import { initAnalytics } from './services/analytics'

// --- Google Analytics (optional) ---
const GA_ID = import.meta.env.VITE_GA_ID
if (GA_ID) initAnalytics(GA_ID, router)

// --- Tidy console output in production ---
if (import.meta.env.PROD) {
  const noop = () => {}
  console.log = noop
  console.debug = noop
  console.info = noop
  console.trace = noop
  console.warn = noop
  console.error = noop
}

// --- Language Preference ---
const savedLocale = localStorage.getItem('locale')
const browserLocale = navigator.language.startsWith('tr') ? 'tr' : 'en'
const locale =
  savedLocale === 'tr' || savedLocale === 'en' ? savedLocale : browserLocale === 'tr' ? 'tr' : 'en'

i18n.global.locale.value = locale
document.documentElement.setAttribute('lang', locale)

const app = createApp(App)

// --- Sentry (errors only, optional via env) ---
const SENTRY_DSN = import.meta.env.VITE_SENTRY_DSN
if (import.meta.env.PROD && SENTRY_DSN) {
  Sentry.init({
    app,
    dsn: SENTRY_DSN,
    environment: import.meta.env.MODE,
    enabled: true,
    tracesSampleRate: 0, // disable performance traces by default
    beforeBreadcrumb(breadcrumb) {
      if (breadcrumb && breadcrumb.category === 'console' && breadcrumb.level) {
        // keep only warning/error console breadcrumbs
        if (breadcrumb.level === 'warning' || breadcrumb.level === 'error') return breadcrumb
        return null
      }
      return breadcrumb
    },
  })
}

app.use(i18n)
app.use(router)
app.mount('#app')
