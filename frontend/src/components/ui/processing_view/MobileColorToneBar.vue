<script setup lang="ts">
import { computed, watch, reactive, ref, onMounted, onUnmounted, nextTick } from 'vue'
import { useI18n } from 'vue-i18n'
import { useAppState } from '../../../composables/useAppState'
import { TONE_MAP } from '@/data/colorOptions'
import { getBasePreview, getTonePreview } from '@/data/hairAssets'
import { getToneSortOrder } from '@/data/colorMeta'
import hairService from '../../../services/hairService'
import { trackEvent } from '../../../services/analytics'
import { PhCaretLeft } from '@phosphor-icons/vue'

const { t } = useI18n()
const {
  isProcessing,
  currentColorResult,
  getColorToneState,
  setColorToneState,
  setProcessingError,
  setIsProcessing,
  setShowMobileToneBar,
} = useAppState()

// Per-color base preview
const basePreview = computed(() =>
  currentColorResult.value ? getBasePreview(currentColorResult.value.originalColor) : '',
)

// Tone keys by color
const toneMap = TONE_MAP

// Tones of the current color (including base as first tone)
const availableTones = computed(() => {
  if (!currentColorResult.value) return []

  const colorName = currentColorResult.value.originalColor
  const toneKeys = toneMap[colorName] || []
  const tones = toneKeys.map((toneName) => ({
    name: toneName,
    displayName: toneName.charAt(0).toUpperCase() + toneName.slice(1),
    preview: getTonePreview(colorName, toneName),
  }))

  // Add base tone as first option
  const baseTone = {
    name: 'base',
    displayName: 'Base',
    preview: getBasePreview(colorName),
  }

  return [baseTone, ...getToneSortOrder(colorName, tones)]
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

// Tone selection (including base)
const selectTone = async (toneName: string) => {
  if (!currentColorResult.value) return

  // Clear previous processing error on new action
  setProcessingError(null)

  const colorName = currentColorResult.value.originalColor

  // Handle base tone selection
  if (toneName === 'base') {
    setColorToneState(colorName, null)
    trackEvent('select_tone_base', { color: colorName })
    await hairService.applyToneLocally(null)
    return
  }

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

// Close the component and go back to ColorPalette
const closeComponent = () => {
  console.log('Closing MobileColorToneBar, returning to ColorPalette')
  // Hide mobile tone bar to show ColorPalette again (keep currentColorResult)
  setShowMobileToneBar(false)
}

// Horizontal scroll indicator (mobile) for tones
const scrollerRef = ref<HTMLElement | null>(null)
const thumbWidthPct = ref(0)
const thumbLeftPct = ref(0)
function updateScrollIndicator() {
  const el = scrollerRef.value
  if (!el) return
  const { scrollWidth, clientWidth, scrollLeft } = el
  if (scrollWidth <= clientWidth) {
    thumbWidthPct.value = 100
    thumbLeftPct.value = 0
    return
  }
  const widthPct = Math.max(15, Math.min(100, (clientWidth / scrollWidth) * 100))
  const maxLeftPct = 100 - widthPct
  const leftPct = (scrollLeft / (scrollWidth - clientWidth)) * maxLeftPct
  thumbWidthPct.value = widthPct
  thumbLeftPct.value = leftPct
}
onMounted(() => {
  updateScrollIndicator()
  window.addEventListener('resize', updateScrollIndicator)
})
onUnmounted(() => {
  window.removeEventListener('resize', updateScrollIndicator)
})

// Ensure the indicator initializes when tones first render
watch(
  [
    () => currentColorResult.value && currentColorResult.value.originalColor,
    () => availableTones.value.length,
    () => scrollerRef.value,
  ],
  async () => {
    await nextTick()
    updateScrollIndicator()
  },
  { immediate: true },
)

// Reset scroll position when color changes (new color selected)
watch(
  () => currentColorResult.value?.originalColor,
  async (newColor, oldColor) => {
    if (newColor && oldColor && newColor !== oldColor) {
      // Different color selected, reset scroll to start
      await nextTick()
      if (scrollerRef.value) {
        scrollerRef.value.scrollLeft = 0
        updateScrollIndicator()
      }
    }
  },
)
</script>

<template>
  <!-- Show only if a color is selected and only on mobile/tablet (not lg) -->
  <div
    v-if="currentColorResult"
    class="bg-base-content/70 border-base-content/80 h-full rounded-2xl border shadow-lg lg:hidden"
  >
    <!-- Main flex container: fixed box + tone content -->
    <div class="flex h-full gap-0">
      <!-- Fixed Color Box -->
      <div class="h-full w-[7.5rem] flex-shrink-0 md:w-[10rem]">
        <div class="bg-base-content/70 relative h-full rounded-xl rounded-r-none">
          <div class="flex h-full items-center justify-center">
            <!-- Back Button -->
            <button
              @click="closeComponent"
              class="text-base-100 z-10 h-full rounded-full p-1"
              :aria-label="'Close'"
            >
              <PhCaretLeft class="h-3 w-3" />
            </button>
            <!-- Selected Color Display (not clickable, no borders) -->
            <div
              class="bg-base-content/80 w-[5.25rem] flex-shrink-0 rounded-lg p-0.5 md:w-[6.25rem]"
            >
              <div class="bg-base-100 relative aspect-[3/4] w-full overflow-hidden rounded-md">
                <img :src="basePreview" alt="Selected Color" class="h-full w-full object-cover" />
                <!-- Bottom label bar -->
                <div class="bg-base-100/70 absolute right-0 bottom-0 left-0" style="height: 20%">
                  <div class="flex h-full w-full items-center justify-center">
                    <span class="text-base-content text-[10px] font-semibold">
                      {{ t('colors.' + currentColorResult.originalColor) }}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Tone Content Area -->
      <div class="flex-1 overflow-hidden py-3">
        <!-- Tones Grid / Horizontal scroller -->
        <div
          class="scrollbar-none grid scroll-px-2 auto-cols-[minmax(84px,1fr)] grid-flow-col gap-2 overflow-x-auto overflow-y-hidden px-2"
          ref="scrollerRef"
          @scroll.passive="updateScrollIndicator()"
        >
          <!-- Base + Tone Options -->
          <div
            v-for="tone in availableTones"
            :key="tone.name"
            @click="selectTone(tone.name)"
            @keydown.enter.prevent="selectTone(tone.name)"
            @keydown.space.prevent="selectTone(tone.name)"
            :class="[
              'bg-base-content/80 border-base-100/20 h-full rounded-xl border p-0.5 transition-all duration-200',
              (tone.name === 'base' && currentSelectedTone === null) ||
              (tone.name !== 'base' && currentSelectedTone === tone.name)
                ? 'border-primary ring-primary/20 shadow-lg ring-2'
                : '',
              loadingToneName === tone.name
                ? 'cursor-wait opacity-60'
                : isProcessing
                  ? (tone.name === 'base' && currentSelectedTone === null) ||
                    (tone.name !== 'base' && currentSelectedTone === tone.name)
                    ? 'cursor-wait opacity-60'
                    : 'pointer-events-none cursor-not-allowed opacity-50'
                  : 'cursor-pointer hover:scale-105 hover:border-gray-300 hover:shadow-md',
            ]"
            role="button"
            tabindex="0"
            :aria-current="
              (tone.name === 'base' && currentSelectedTone === null) ||
              (tone.name !== 'base' && currentSelectedTone === tone.name)
                ? 'true'
                : 'false'
            "
          >
            <div class="bg-base-100 relative aspect-[3/4] w-full overflow-hidden rounded-lg">
              <img
                :src="tone.preview"
                :alt="tone.displayName"
                class="h-full w-full object-cover"
                loading="lazy"
                @load="onToneLoad(tone.name)"
              />
              <div v-if="!toneLoaded[tone.name]" class="skeleton absolute inset-0"></div>
              <!-- Loading overlay on selected tone or when waiting this tone to arrive -->
              <div
                v-if="
                  (isProcessing &&
                    ((tone.name === 'base' && currentSelectedTone === null) ||
                      (tone.name !== 'base' && currentSelectedTone === tone.name))) ||
                  loadingToneName === tone.name
                "
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
                      'px-[0.1px] text-xs font-semibold',
                      (tone.name === 'base' && currentSelectedTone === null) ||
                      (tone.name !== 'base' && currentSelectedTone === tone.name)
                        ? 'text-primary'
                        : 'text-base-content',
                    ]"
                    >{{ tone.displayName }}</span
                  >
                </div>
              </div>
            </div>
          </div>
        </div>
        <!-- Scroll indicator -->
        <div class="relative mt-1 h-1">
          <div
            class="bg-base-100/40 absolute top-0 bottom-0 rounded-full shadow-sm"
            :style="{ width: thumbWidthPct + '%', left: thumbLeftPct + '%' }"
          ></div>
        </div>
      </div>
    </div>
  </div>
</template>
