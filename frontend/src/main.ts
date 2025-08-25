import './assets/css/main.css'
import { createApp } from 'vue'
import App from './App.vue'
import i18n from './i18n'
import router from './router'
import { initAnalytics } from './services/analytics'

// --- Google Analytics (optional) ---
const GA_ID = import.meta.env.VITE_GA_ID
if (GA_ID) initAnalytics(GA_ID, router)

// --- Tidy console output in production (keep warnings/errors) ---
if (import.meta.env.PROD) {
  const noop = () => {}
  console.log = noop
  console.debug = noop
  console.info = noop
  console.trace = noop
}

// --- Language Preference ---
const savedLocale = localStorage.getItem('locale')
const browserLocale = navigator.language.startsWith('tr') ? 'tr' : 'en'
const locale =
  savedLocale === 'tr' || savedLocale === 'en' ? savedLocale : browserLocale === 'tr' ? 'tr' : 'en'

i18n.global.locale.value = locale
document.documentElement.setAttribute('lang', locale)

const app = createApp(App)
app.use(i18n)
app.use(router)
app.mount('#app')
