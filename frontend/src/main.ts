import './assets/css/main.css'
import { createApp } from 'vue'
import App from './App.vue'
import i18n from './i18n'
import router from './router'

// --- Theme Preference ---
const savedTheme = localStorage.getItem('theme')
const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
const theme = savedTheme || (prefersDark ? 'dark' : 'light')
document.documentElement.classList.remove('dark', 'light')
document.documentElement.classList.add(theme)

// --- Language Preference ---
const savedLocale = localStorage.getItem('locale')
const browserLocale = navigator.language.startsWith('tr') ? 'tr' : 'en'
const locale =
  savedLocale === 'tr' || savedLocale === 'en' ? savedLocale : browserLocale === 'tr' ? 'tr' : 'en'

i18n.global.locale.value = locale

const app = createApp(App)
app.use(i18n)
app.use(router)
app.mount('#app')
