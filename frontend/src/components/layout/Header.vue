<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount } from 'vue'
import { PhSun, PhMoon, PhGlobe, PhCaretDown, PhCheck } from '@phosphor-icons/vue'
import logo from '@/assets/logo/logo.png'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'

const { locale } = useI18n()

const langDropdownOpen = ref(false)
const dropdownRef = ref<HTMLElement | null>(null)
const theme = ref('light')

onMounted(() => {
  const savedTheme = localStorage.getItem('theme')
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
  theme.value = savedTheme || (prefersDark ? 'dark' : 'light')
  document.documentElement.classList.remove('dark', 'light')
  document.documentElement.classList.add(theme.value)
  document.documentElement.setAttribute('data-theme', theme.value)
  // Update checkbox state
  const input = document.querySelector('.theme-controller') as HTMLInputElement
  if (input) input.checked = theme.value === 'light'
})

function handleClickOutside(event: MouseEvent) {
  if (dropdownRef.value && !dropdownRef.value.contains(event.target as Node)) {
    closeLangDropdown()
  }
}

function openLangDropdown() {
  langDropdownOpen.value = true
  document.addEventListener('mousedown', handleClickOutside)
}

function closeLangDropdown() {
  langDropdownOpen.value = false
  document.removeEventListener('mousedown', handleClickOutside)
}

function toggleLangDropdown() {
  if (langDropdownOpen.value) {
    closeLangDropdown()
  } else {
    openLangDropdown()
  }
}

function setLanguage(lang: string) {
  locale.value = lang
  localStorage.setItem('locale', lang)
  closeLangDropdown()
}

function toggleTheme(e: Event) {
  const checked = (e.target as HTMLInputElement).checked
  theme.value = checked ? 'light' : 'dark'
  document.documentElement.classList.remove('dark', 'light')
  document.documentElement.classList.add(theme.value)
  document.documentElement.setAttribute('data-theme', theme.value)
  localStorage.setItem('theme', theme.value)
}
</script>

<template>
  <header
    class="fixed top-5 left-1/2 transform -translate-x-1/2 max-w-7xl w-full z-10 rounded-full bg-base-content"
  >
    <div class="flex items-center justify-between px-15 py-2">
      <!-- Logo and Title -->
      <RouterLink to="/" class="flex items-center gap-x-2">
        <img :src="logo" alt="LuviraAI Logo" class="w-12 h-12 object-contain" />
        <h1 class="text-2xl font-bold text-base-100">LuviraAI</h1>
      </RouterLink>

      <div class="flex items-center gap-x-4 px-6">
        <!-- Theme Toggle -->
        <label class="swap swap-rotate">
          <input
            type="checkbox"
            class="theme-controller"
            value="light"
            :checked="theme === 'light'"
            @change="toggleTheme"
          />
          <PhSun class="swap-on h-8 w-8 text-yellow-300" weight="bold" />
          <PhMoon class="swap-off h-8 w-8 text-base-100" weight="bold" />
        </label>

        <!-- Language Dropdown -->
        <div class="relative select-none" ref="dropdownRef">
          <button
            type="button"
            class="flex items-center cursor-pointer rounded"
            @click="toggleLangDropdown"
          >
            <PhGlobe class="h-8 w-8 text-base-100" weight="bold" />
            <PhCaretDown class="h-5 w-3 text-base-100" weight="bold" />
          </button>
          <ul
            v-if="langDropdownOpen"
            class="absolute right-0 mt-2 z-20 min-w-30 rounded shadow-lg bg-base-100 border border-gray-700"
          >
            <li>
              <button
                v-if="locale !== 'en'"
                class="w-full text-left px-4 py-2 hover:bg-base-300 cursor-pointer"
                @click="setLanguage('en')"
              >
                English
              </button>
              <span
                v-else
                class="w-full flex items-center gap-2 text-left px-4 py-2 bg-primary/10 text-primary font-semibold cursor-default select-none"
              >
                <PhCheck class="w-4 h-4 text-primary" /> English
              </span>
            </li>
            <li>
              <button
                v-if="locale !== 'tr'"
                class="w-full text-left px-4 py-2 hover:bg-base-300 cursor-pointer"
                @click="setLanguage('tr')"
              >
                Türkçe
              </button>
              <span
                v-else
                class="w-full flex items-center gap-2 text-left px-4 py-2 bg-primary/10 text-primary font-semibold cursor-default select-none"
              >
                <PhCheck class="w-4 h-4 text-primary" /> Türkçe
              </span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  </header>
</template>
