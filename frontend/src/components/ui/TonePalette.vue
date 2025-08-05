<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useAppState } from '../../composables/useAppState'
import hairService from '../../services/hairService'

const { t } = useI18n()
const {
  isProcessing,
  currentColorResult,
  getColorToneState,
  setColorToneState,
  setProcessedImageToTone,
} = useAppState()

// Backend'deki CUSTOM_TONES ile eşleşen ton tanımları
const toneDefinitions: Record<string, Record<string, { description: string }>> = {
  Black: {
    jet: { description: 'Pure jet black' },
    soft: { description: 'Soft, warm black' },
    onyx: { description: 'Deep, rich onyx black' },
    charcoal: { description: 'Charcoal gray-black' },
  },
  Blonde: {
    platinum: { description: 'Ultra-light platinum blonde' },
    golden: { description: 'Warm golden blonde' },
    ash: { description: 'Cool ash blonde' },
    honey: { description: 'Sweet honey blonde' },
    strawberry: { description: 'Strawberry blonde with red hints' },
  },
  Brown: {
    chestnut: { description: 'Rich chestnut brown' },
    chocolate: { description: 'Dark chocolate brown' },
    caramel: { description: 'Sweet caramel brown' },
    mahogany: { description: 'Reddish mahogany brown' },
    espresso: { description: 'Deep espresso brown' },
  },
  Auburn: {
    classic: { description: 'Classic auburn' },
    golden: { description: 'Golden auburn' },
    dark: { description: 'Dark auburn' },
    copper: { description: 'Copper auburn' },
  },
  Pink: {
    rose: { description: 'Romantic rose pink' },
    fuchsia: { description: 'Vibrant fuchsia pink' },
    blush: { description: 'Soft blush pink' },
    magenta: { description: 'Bold magenta pink' },
    coral: { description: 'Warm coral pink' },
  },
  Blue: {
    navy: { description: 'Deep navy blue' },
    electric: { description: 'Electric bright blue' },
    ice: { description: 'Ice blue' },
    midnight: { description: 'Midnight blue' },
    sky: { description: 'Sky blue' },
  },
  Purple: {
    violet: { description: 'Rich violet purple' },
    lavender: { description: 'Soft lavender purple' },
    plum: { description: 'Deep plum purple' },
    amethyst: { description: 'Gemstone amethyst purple' },
    orchid: { description: 'Delicate orchid purple' },
  },
  Gray: {
    silver: { description: 'Bright silver gray' },
    ash: { description: 'Cool ash gray' },
    charcoal: { description: 'Dark charcoal gray' },
    pearl: { description: 'Lustrous pearl gray' },
    steel: { description: 'Cool steel gray' },
  },
}

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

  const colorName = currentColorResult.value.originalColor // Use originalColor

  // State'i güncelle
  setColorToneState(colorName, toneName)

  console.log('Selected tone:', toneName, 'for color:', colorName)

  try {
    // Backend'e ton değişim isteği gönder
    await hairService.changeHairColor(colorName, toneName)
    console.log('Tone change completed successfully')
  } catch (error) {
    console.error('Tone change failed:', error)
    // TODO: Show error message to user
  }
}

// Base rengi seç (ton yok)
const selectBase = async () => {
  if (isProcessing.value || !currentColorResult.value) return

  const colorName = currentColorResult.value.originalColor // Use originalColor

  // State'i güncelle
  setColorToneState(colorName, null)

  console.log('Selected base color:', colorName)

  // Base rengini göstermek için setProcessedImageToTone kullan
  setProcessedImageToTone(null)
}
</script>

<template>
  <!-- Sadece bir renk seçilmişse göster -->
  <div
    v-if="currentColorResult"
    class="bg-base-content/70 rounded-2xl shadow-lg border border-base-content/80 p-4"
  >
    <!-- Header -->
    <div class="mb-4">
      <h3 class="text-lg font-bold text-base-100 mb-1">
        {{ t('tonePalette.title') }} - {{ currentColorResult.originalColor }}
      </h3>
      <p class="text-xs text-base-100/70">{{ t('tonePalette.instruction') }}</p>
    </div>

    <!-- Base Color + Tones Grid -->
    <div class="grid grid-cols-6 gap-2">
      <!-- Base Color (No Tone) -->
      <div
        @click="selectBase"
        :class="[
          'bg-base-content/80 rounded-xl border-2 p-2 transition-all duration-200',
          currentSelectedTone === null
            ? 'border-primary ring-2 ring-primary/20 shadow-lg'
            : 'border-gray-200',
          isProcessing
            ? currentSelectedTone === null
              ? 'cursor-wait'
              : 'opacity-50 cursor-not-allowed pointer-events-none'
            : 'cursor-pointer hover:border-gray-300 hover:scale-105 hover:shadow-md',
        ]"
      >
        <!-- Loading Spinner for base -->
        <div
          v-if="isProcessing && currentSelectedTone === null"
          class="w-8 h-8 rounded-lg mx-auto mb-1 bg-base-100 flex items-center justify-center"
        >
          <div
            class="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin"
          ></div>
        </div>
        <!-- Base Preview -->
        <div
          v-else
          class="w-8 h-8 rounded-lg mx-auto mb-1 bg-base-100 flex items-center justify-center"
        >
          <span class="text-base-300 text-sm">🏠</span>
        </div>
        <!-- Base Label -->
        <span
          :class="[
            'text-xs font-medium text-center block transition-colors duration-200',
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
            ? 'border-primary ring-2 ring-primary/20 shadow-lg'
            : 'border-gray-200',
          isProcessing
            ? currentSelectedTone === tone.name
              ? 'cursor-wait'
              : 'opacity-50 cursor-not-allowed pointer-events-none'
            : 'cursor-pointer hover:border-gray-300 hover:scale-105 hover:shadow-md',
        ]"
      >
        <!-- Loading Spinner for selected tone -->
        <div
          v-if="isProcessing && currentSelectedTone === tone.name"
          class="w-8 h-8 rounded-lg mx-auto mb-1 bg-base-100 flex items-center justify-center"
        >
          <div
            class="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin"
          ></div>
        </div>
        <!-- Tone Preview -->
        <div
          v-else
          class="w-8 h-8 rounded-lg mx-auto mb-1 bg-base-100 flex items-center justify-center"
        >
          <span class="text-base-300 text-sm">✨</span>
        </div>
        <!-- Tone Name -->
        <span
          :class="[
            'text-xs font-medium text-center block transition-colors duration-200',
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
      <p class="text-xs text-base-100/80 italic">
        {{
          availableTones.find((t) => t.name === currentSelectedTone)?.description || 'Base color'
        }}
      </p>
    </div>
  </div>
</template>
