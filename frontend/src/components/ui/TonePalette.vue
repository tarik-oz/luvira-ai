<script setup lang="ts">
import { computed, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useAppState } from '../../composables/useAppState'
import { TONE_DEFINITIONS } from '../../config/colorConfig'
import hairService from '../../services/hairService'

const { t } = useI18n()
const {
  isProcessing,
  currentColorResult,
  getColorToneState,
  setColorToneState,
  setProcessingError,
} = useAppState()

// Backend'deki CUSTOM_TONES ile eşleşen ton tanımları
const toneDefinitions = TONE_DEFINITIONS

// Mevcut rengin tonları
const availableTones = computed(() => {
  if (!currentColorResult.value) return []

  // Use originalColor instead of color (which contains cache key)
  const colorName = currentColorResult.value.originalColor
  const colorTones = toneDefinitions[colorName] || {}

  return Object.keys(colorTones).map((toneName) => ({
    name: toneName,
    displayName: toneName.charAt(0).toUpperCase() + toneName.slice(1),
    description: colorTones[toneName].description,
  }))
})

// Şu anki rengin seçili tonu - per-color state
const currentSelectedTone = computed(() => {
  if (!currentColorResult.value) return null
  // Use originalColor for tone state lookup
  return getColorToneState(currentColorResult.value.originalColor)
})

// Renk değiştiğinde: Eğer bu renk için daha önce ton seçilmemişse, base'i seç
watch(
  currentColorResult,
  (newResult) => {
    if (newResult && getColorToneState(newResult.originalColor) === undefined) {
      // İlk defa bu renk görünüyor, base'i seç
      setColorToneState(newResult.originalColor, null)
    }
  },
  { immediate: true },
)

// Ton seçimi
const selectTone = async (toneName: string) => {
  if (isProcessing.value || !currentColorResult.value) return

  // Clear previous processing error on new action
  setProcessingError(null)

  const colorName = currentColorResult.value.originalColor // Use originalColor

  // State'i güncelle
  setColorToneState(colorName, toneName)

  console.log('Selected tone:', toneName, 'for color:', colorName)

  try {
    await hairService.applyToneLocally(toneName)
    console.log('Tone switch completed locally')
  } catch (error) {
    console.error('Tone change failed:', error)
    setProcessingError(t('tonePalette.error') as string)
  }
}

// Base rengi seç (ton yok)
const selectBase = async () => {
  if (isProcessing.value || !currentColorResult.value) return

  const colorName = currentColorResult.value.originalColor // Use originalColor

  // State'i güncelle
  setColorToneState(colorName, null)

  console.log('Selected base color:', colorName)

  // Base rengini göstermek için local compose kullan
  await hairService.applyToneLocally(null)
}
</script>

<template>
  <!-- Sadece bir renk seçilmişse göster -->
  <div
    v-if="currentColorResult"
    class="bg-base-content/70 border-base-content/80 rounded-2xl border p-4 shadow-lg"
  >
    <!-- Header -->
    <div class="mb-4">
      <h3 class="text-base-100 mb-1 text-lg font-bold">
        {{ t('tonePalette.title') }} - {{ currentColorResult.originalColor }}
      </h3>
      <p class="text-base-100/70 text-xs">{{ t('tonePalette.instruction') }}</p>
    </div>

    <!-- Base Color + Tones Grid -->
    <div class="grid grid-cols-6 gap-2">
      <!-- Base Color (No Tone) -->
      <div
        @click="selectBase"
        :class="[
          'bg-base-content/80 rounded-xl border-2 p-2 transition-all duration-200',
          currentSelectedTone === null
            ? 'border-primary ring-primary/20 shadow-lg ring-2'
            : 'border-gray-200',
          isProcessing
            ? currentSelectedTone === null
              ? 'cursor-wait'
              : 'pointer-events-none cursor-not-allowed opacity-50'
            : 'cursor-pointer hover:scale-105 hover:border-gray-300 hover:shadow-md',
        ]"
      >
        <!-- Loading Spinner for base -->
        <div
          v-if="isProcessing && currentSelectedTone === null"
          class="bg-base-100 mx-auto mb-1 flex h-8 w-8 items-center justify-center rounded-lg"
        >
          <div class="border-accent h-6 w-6 animate-spin rounded-full border-t-4 border-b-4"></div>
        </div>
        <!-- Base Preview -->
        <div
          v-else
          class="bg-base-100 mx-auto mb-1 flex h-8 w-8 items-center justify-center rounded-lg"
        >
          <span class="text-base-300 text-sm">🏠</span>
        </div>
        <!-- Base Label -->
        <span
          :class="[
            'block text-center text-xs font-medium transition-colors duration-200',
            currentSelectedTone === null ? 'text-primary' : 'text-base-300',
            isProcessing && currentSelectedTone !== null ? 'opacity-50' : '',
          ]"
        >
          Base
        </span>
      </div>

      <!-- Tone Options -->
      <div
        v-for="tone in availableTones"
        :key="tone.name"
        @click="selectTone(tone.name)"
        :class="[
          'bg-base-content/80 rounded-xl border-2 p-2 transition-all duration-200',
          currentSelectedTone === tone.name
            ? 'border-primary ring-primary/20 shadow-lg ring-2'
            : 'border-gray-200',
          isProcessing
            ? currentSelectedTone === tone.name
              ? 'cursor-wait'
              : 'pointer-events-none cursor-not-allowed opacity-50'
            : 'cursor-pointer hover:scale-105 hover:border-gray-300 hover:shadow-md',
        ]"
      >
        <!-- Loading Spinner for selected tone -->
        <div
          v-if="isProcessing && currentSelectedTone === tone.name"
          class="bg-base-100 mx-auto mb-1 flex h-8 w-8 items-center justify-center rounded-lg"
        >
          <div class="border-accent h-4 w-4 animate-spin rounded-full border-t-2 border-b-2"></div>
        </div>
        <!-- Tone Preview -->
        <div
          v-else
          class="bg-base-100 mx-auto mb-1 flex h-8 w-8 items-center justify-center rounded-lg"
        >
          <span class="text-base-300 text-sm">✨</span>
        </div>
        <!-- Tone Name -->
        <span
          :class="[
            'block text-center text-xs font-medium transition-colors duration-200',
            currentSelectedTone === tone.name ? 'text-primary' : 'text-base-300',
            isProcessing && currentSelectedTone !== tone.name ? 'opacity-50' : '',
          ]"
        >
          {{ tone.displayName }}
        </span>
      </div>
    </div>

    <!-- Selected Tone Description -->
    <div v-if="currentSelectedTone" class="mt-3 text-center">
      <p class="text-base-100/80 text-xs italic">
        {{
          availableTones.find((t) => t.name === currentSelectedTone)?.description || 'Base color'
        }}
      </p>
    </div>
  </div>
</template>
