<script setup lang="ts">
import { computed, watch, reactive, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useAppState } from '../../composables/useAppState'
import { TONE_DEFINITIONS } from '../../config/colorConfig'
import { getBasePreview, getTonePreview } from '@/data/hairAssets'
import { getToneSortOrder } from '@/data/colorMeta'
import hairService from '../../services/hairService'
import { trackEvent } from '../../services/analytics'

const { t } = useI18n()
const {
  isProcessing,
  currentColorResult,
  getColorToneState,
  setColorToneState,
  setProcessingError,
  setIsProcessing,
} = useAppState()

// Per-color base preview
const basePreview = computed(() =>
  currentColorResult.value ? getBasePreview(currentColorResult.value.originalColor) : '',
)

// Tone definitions matching CUSTOM_TONES in the backend
const toneDefinitions = TONE_DEFINITIONS

// Tones of the current color
const availableTones = computed(() => {
  if (!currentColorResult.value) return []

  // Use originalColor instead of color (which contains cache key)
  const colorName = currentColorResult.value.originalColor
  const colorTones = toneDefinitions[colorName] || {}
  const tones = Object.keys(colorTones).map((toneName) => ({
    name: toneName,
    displayName: toneName.charAt(0).toUpperCase() + toneName.slice(1),
    description: colorTones[toneName].description,
    preview: getTonePreview(colorName, toneName),
  }))
  return getToneSortOrder(colorName, tones)
})

// Track per-tone image load state to show skeletons until loaded
const toneLoaded: Record<string, boolean> = reactive({})
watch(
  () => currentColorResult.value?.originalColor,
  () => {
    // Reset tone loaded flags for new color
    Object.keys(toneLoaded).forEach((k) => delete toneLoaded[k])
  },
)
const onToneLoad = (toneName: string) => {
  toneLoaded[toneName] = true
}

// Local loading state for a tone that hasn't arrived via stream yet
const loadingToneName = ref<string | null>(null)

// Currently selected tone for the color - per-color state
const currentSelectedTone = computed(() => {
  if (!currentColorResult.value) return null
  // Use originalColor for tone state lookup
  return getColorToneState(currentColorResult.value.originalColor)
})

// When the color changes: If no tone has been selected for this color before, select base
watch(
  currentColorResult,
  (newResult) => {
    if (newResult && getColorToneState(newResult.originalColor) === undefined) {
      // This color appears for the first time, select base
      setColorToneState(newResult.originalColor, null)
    }
  },
  { immediate: true },
)

// Tone selection
const selectTone = async (toneName: string) => {
  if (!currentColorResult.value) return

  // Clear previous processing error on new action
  setProcessingError(null)

  const colorName = currentColorResult.value.originalColor // Use originalColor

  // Update the state immediately so loading appears on the selected tone
  setColorToneState(colorName, toneName)

  // If the tone is not in cache yet, start stream and wait only until this tone arrives
  const toneUrl = currentColorResult.value.tones[toneName]
  if (!toneUrl) {
    loadingToneName.value = toneName
    setIsProcessing(true)
    try {
      // Start/ensure stream, then wait until this tone becomes available or timeout
      hairService.ensureColorStream(colorName).catch((error) => {
        const message = error instanceof Error ? error.message : String(error)
        if (message === 'SESSION_EXPIRED') return
        if (message === 'Failed to fetch' || /network/i.test(message)) {
          setProcessingError(t('processing.networkError') as string)
        } else {
          setProcessingError(t('tonePalette.error') as string)
        }
      })

      await new Promise<void>((resolve) => {
        const TIMEOUT_MS = 10000
        const stop = watch(
          () => currentColorResult.value?.tones[toneName],
          (val) => {
            if (val) {
              stop()
              resolve()
            }
          },
          { immediate: false },
        )
        setTimeout(() => {
          try {
            stop()
          } catch {}
          resolve()
        }, TIMEOUT_MS)
      })
    } finally {
      loadingToneName.value = null
      setIsProcessing(false)
    }
  }

  console.log('Selected tone:', toneName, 'for color:', colorName)

  try {
    trackEvent('select_tone', { color: colorName, tone: toneName })
    await hairService.applyToneLocally(toneName)
    console.log('Tone switch completed locally')
  } catch (error) {
    console.error('Tone change failed:', error)
    setProcessingError(t('tonePalette.error') as string)
  }
}

// Select base color (no tone)
const selectBase = async () => {
  if (!currentColorResult.value) return

  const colorName = currentColorResult.value.originalColor // Use originalColor

  // Update the state
  setColorToneState(colorName, null)

  console.log('Selected base color:', colorName)

  // Use local compose to show the base color
  trackEvent('select_tone_base', { color: colorName })
  await hairService.applyToneLocally(null)
}
</script>

<template>
  <!-- Show only if a color is selected -->
  <div
    v-if="currentColorResult"
    class="bg-base-content/70 border-base-content/80 rounded-2xl border p-3 shadow-lg lg:p-4"
  >
    <!-- Header -->
    <div class="mb-3 lg:mb-4">
      <h3 class="text-base-100 mb-1 text-lg font-bold">
        {{ t('tonePalette.title') }} - {{ t('colors.' + currentColorResult.originalColor) }}
      </h3>
    </div>

    <!-- Base Color + Tones Grid -->
    <div class="grid grid-cols-6 gap-2 md:grid-cols-7 lg:grid-cols-8">
      <!-- Base Color (No Tone) -->
      <div
        @click="selectBase"
        :class="[
          'bg-base-content/80 border-base-100/20 rounded-xl border p-0.5 transition-all duration-200',
          currentSelectedTone === null ? 'border-primary ring-primary/20 shadow-lg ring-2' : '',
          isProcessing
            ? currentSelectedTone === null
              ? 'cursor-wait opacity-60'
              : 'pointer-events-none cursor-not-allowed opacity-50'
            : 'cursor-pointer hover:scale-105 hover:border-gray-300 hover:shadow-md',
        ]"
      >
        <div
          class="bg-base-100 relative w-full overflow-hidden rounded-lg"
          style="aspect-ratio: 9 / 16"
        >
          <img :src="basePreview" alt="Base pattern" class="h-full w-full object-cover" />
          <!-- Loading overlay on base -->
          <div
            v-if="isProcessing && currentSelectedTone === null"
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
          loadingToneName === tone.name
            ? 'cursor-wait opacity-60'
            : isProcessing
              ? currentSelectedTone === tone.name
                ? 'cursor-wait opacity-60'
                : 'pointer-events-none cursor-not-allowed opacity-50'
              : 'cursor-pointer hover:scale-105 hover:border-gray-300 hover:shadow-md',
        ]"
      >
        <div
          class="bg-base-100 relative w-full overflow-hidden rounded-lg"
          style="aspect-ratio: 9 / 16"
        >
          <div v-if="!toneLoaded[tone.name]" class="skeleton absolute inset-0"></div>
          <img
            :src="tone.preview"
            :alt="tone.displayName"
            class="h-full w-full object-cover"
            loading="lazy"
            @load="onToneLoad(tone.name)"
          />
          <!-- Loading overlay on selected tone or when waiting this tone to arrive -->
          <div
            v-if="
              (isProcessing && currentSelectedTone === tone.name) || loadingToneName === tone.name
            "
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
                  'px-[0.1px] text-xs font-semibold',
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
