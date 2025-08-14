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
  setIsProcessing,
} = useAppState()

// Temporary pattern: use a single default preview for all tones
const defaultPatternUrl = new URL('../../assets/hair_patterns/default.webp', import.meta.url).href

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
  if (!currentColorResult.value) return

  // Clear previous processing error on new action
  setProcessingError(null)

  const colorName = currentColorResult.value.originalColor // Use originalColor

  // State'i hemen güncelle ki loading seçilen ton üzerinde görünsün
  setColorToneState(colorName, toneName)

  // Eğer tone henüz cache'te yoksa stream'i başlat (tek istek mantığı)
  const toneUrl = currentColorResult.value.tones[toneName]
  if (!toneUrl) {
    setIsProcessing(true)
    await hairService.ensureColorStream(colorName)
  }

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
  if (!currentColorResult.value) return

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
    class="bg-base-content/70 border-base-content/80 rounded-2xl border p-3 shadow-lg lg:p-4"
  >
    <!-- Header -->
    <div class="mb-3 lg:mb-4">
      <h3 class="text-base-100 mb-1 text-lg font-bold">
        {{ t('tonePalette.title') }} - {{ currentColorResult.originalColor }}
      </h3>
      <p class="text-base-100/70 text-xs">{{ t('tonePalette.instruction') }}</p>
    </div>

    <!-- Base Color + Tones Grid (slightly denser than color grid) -->
    <div class="grid grid-cols-6 gap-2 md:grid-cols-7 lg:grid-cols-8">
      <!-- Base Color (No Tone) -->
      <div
        @click="selectBase"
        :class="[
          'bg-base-content/80 border-base-100/20 rounded-xl border p-0.5 transition-all duration-200',
          currentSelectedTone === null ? 'border-primary ring-primary/20 shadow-lg ring-2' : '',
          isProcessing && currentSelectedTone === null
            ? 'cursor-wait'
            : 'cursor-pointer hover:scale-105 hover:border-gray-300 hover:shadow-md',
        ]"
      >
        <div
          class="bg-base-100 relative w-full overflow-hidden rounded-lg"
          style="aspect-ratio: 9 / 16"
        >
          <img :src="defaultPatternUrl" alt="Base pattern" class="h-full w-full object-cover" />
          <!-- Loading overlay on base -->
          <div
            v-if="isProcessing && currentSelectedTone === null"
            class="bg-base-content/30 absolute inset-0 flex items-center justify-center"
          >
            <div
              class="border-accent h-7 w-7 animate-spin rounded-full border-t-4 border-b-4"
            ></div>
          </div>
          <!-- Bottom label bar -->
          <div class="bg-base-100/95 absolute right-0 bottom-0 left-0" style="height: 20%">
            <div class="flex h-full w-full items-center justify-center">
              <span
                :class="[
                  'text-xs font-semibold',
                  currentSelectedTone === null ? 'text-primary' : 'text-base-content',
                ]"
                >Base</span
              >
            </div>
          </div>
        </div>
      </div>

      <!-- Tone Options -->
      <div
        v-for="tone in availableTones"
        :key="tone.name"
        @click="selectTone(tone.name)"
        :class="[
          'bg-base-content/80 border-base-100/20 rounded-xl border p-0.5 transition-all duration-200',
          currentSelectedTone === tone.name
            ? 'border-primary ring-primary/20 shadow-lg ring-2'
            : '',
          'cursor-pointer hover:scale-105 hover:border-gray-300 hover:shadow-md',
        ]"
      >
        <div
          class="bg-base-100 relative w-full overflow-hidden rounded-lg"
          style="aspect-ratio: 9 / 16"
        >
          <img
            :src="defaultPatternUrl"
            :alt="tone.displayName"
            class="h-full w-full object-cover"
          />
          <!-- Loading overlay on selected tone -->
          <div
            v-if="isProcessing && currentSelectedTone === tone.name"
            class="bg-base-content/30 absolute inset-0 flex items-center justify-center"
          >
            <div
              class="border-accent h-5 w-5 animate-spin rounded-full border-t-2 border-b-2"
            ></div>
          </div>
          <!-- Bottom label bar -->
          <div class="bg-base-100/95 absolute right-0 bottom-0 left-0" style="height: 20%">
            <div class="flex h-full w-full items-center justify-center">
              <span
                :class="[
                  'text-xs font-semibold',
                  currentSelectedTone === tone.name ? 'text-primary' : 'text-base-content',
                ]"
                >{{ tone.displayName }}</span
              >
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
