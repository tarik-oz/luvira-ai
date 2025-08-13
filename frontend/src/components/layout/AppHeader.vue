<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { PhSun, PhMoon, PhGlobe, PhCaretDown, PhCheck } from '@phosphor-icons/vue'
import logo from '@/assets/logo/logo.png'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import { useAppState } from '@/composables/useAppState'

const { locale } = useI18n()
const { resetState } = useAppState()

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

function handleHomeClick() {
  resetState()
}
</script>

<template>
  <header
    class="bg-base-content fixed top-5 left-1/2 z-10 w-full max-w-7xl -translate-x-1/2 transform rounded-full"
  >
    <div class="flex items-center justify-between px-15 py-2">
      <!-- Logo and Title -->
      <RouterLink to="/" class="flex items-center gap-x-2" @click="handleHomeClick">
        <img :src="logo" alt="LuviraAI Logo" class="h-12 w-12 object-contain" />
        <h1 class="text-base-100 text-2xl font-bold">LuviraAI</h1>
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
          <PhMoon class="swap-off text-base-100 h-8 w-8" weight="bold" />
        </label>

        <!-- Language Dropdown -->
        <div class="relative select-none" ref="dropdownRef">
          <button
            type="button"
            class="flex cursor-pointer items-center rounded"
            @click="toggleLangDropdown"
          >
            <PhGlobe class="text-base-100 h-8 w-8" weight="bold" />
            <PhCaretDown class="text-base-100 h-5 w-3" weight="bold" />
          </button>
          <ul
            v-if="langDropdownOpen"
            class="bg-base-100 absolute right-0 z-20 mt-2 min-w-30 rounded border border-gray-700 shadow-lg"
          >
            <li>
              <button
                v-if="locale !== 'en'"
                class="hover:bg-base-300 w-full cursor-pointer px-4 py-2 text-left"
                @click="setLanguage('en')"
              >
                English
              </button>
              <span
                v-else
                class="bg-primary/10 text-primary flex w-full cursor-default items-center gap-2 px-4 py-2 text-left font-semibold select-none"
              >
                <PhCheck class="text-primary h-4 w-4" /> English
              </span>
            </li>
            <li>
              <button
                v-if="locale !== 'tr'"
                class="hover:bg-base-300 w-full cursor-pointer px-4 py-2 text-left"
                @click="setLanguage('tr')"
              >
                Türkçe
              </button>
              <span
                v-else
                class="bg-primary/10 text-primary flex w-full cursor-default items-center gap-2 px-4 py-2 text-left font-semibold select-none"
              >
                <PhCheck class="text-primary h-4 w-4" /> Türkçe
              </span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  </header>
</template>
