<script setup lang="ts">
import { ref, reactive, onMounted, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useAppState } from '../../composables/useAppState'
import { AVAILABLE_COLORS } from '../../config/colorConfig'
import { getBasePreview } from '@/data/hairAssets'
import { getPreferredColorOrder } from '@/data/colorMeta'
import hairService from '../../services/hairService'
import { trackEvent } from '../../services/analytics'

const { t } = useI18n()
const { isProcessing, setProcessingError, setShowMobileToneBar } = useAppState()
const selectedColor = ref<string | null>(null)

const colors = getPreferredColorOrder(AVAILABLE_COLORS).map((name: string) => ({
  name,
  preview: getBasePreview(name),
}))
const imageLoaded: Record<string, boolean> = reactive(
  Object.fromEntries(colors.map((c) => [c.name, false])),
)
const onImageLoad = (name: string) => {
  imageLoaded[name] = true
}

const selectColor = async (colorName: string) => {
  if (isProcessing.value) return // Prevent multiple requests

  // Clear previous processing error on new action
  setProcessingError(null)

  const previous = selectedColor.value
  selectedColor.value = colorName
  console.log('Selected color:', colorName)

  // Show mobile tone bar immediately on mobile (don't wait for color change)
  setShowMobileToneBar(true)

  try {
    trackEvent('select_color', { color: colorName })
    await hairService.changeHairColorAllTones(colorName)
    console.log('Color change completed successfully')
  } catch (error) {
    console.error('Color change failed:', error)
    const message = error instanceof Error ? error.message : String(error)
    if (message === 'SESSION_EXPIRED') {
      // Do not revert UI; modal will handle session end
      return
    } else if (message === 'Failed to fetch' || /network/i.test(message)) {
      setProcessingError(t('processing.networkError') as string)
      // Revert selection on network error
      selectedColor.value = previous
      setShowMobileToneBar(false)
      return
    } else {
      setProcessingError(t('colorPalette.error') as string)
    }
    // Revert selection to previous on failure
    selectedColor.value = previous
    setShowMobileToneBar(false)
  }
}

// Horizontal scroll indicator for mobile
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
  const el = scrollerRef.value
  el?.addEventListener('scroll', updateScrollIndicator, {
    passive: true,
  } as AddEventListenerOptions)
  window.addEventListener('resize', updateScrollIndicator)
})
onUnmounted(() => {
  const el = scrollerRef.value
  el?.removeEventListener('scroll', updateScrollIndicator)
  window.removeEventListener('resize', updateScrollIndicator)
})
</script>

<template>
  <div class="bg-base-content/70 border-base-content/80 rounded-2xl border p-3 shadow-lg lg:p-4">
    <!-- Header -->
    <div class="mb-3 hidden lg:mb-4 lg:block">
      <h3 class="text-base-100 mb-1 text-lg font-bold">{{ t('colorPalette.title') }}</h3>
    </div>

    <!-- Color Grid / Horizontal scroller on mobile -->
    <div
      ref="scrollerRef"
      class="scrollbar-none grid auto-cols-[minmax(84px,1fr)] grid-flow-col gap-2 overflow-x-auto lg:auto-cols-auto lg:grid-flow-row lg:grid-cols-7 lg:overflow-visible"
    >
      <div
        v-for="color in colors"
        :key="color.name"
        @click="selectColor(color.name)"
        :class="[
          'bg-base-content/80 border-base-100/20 rounded-xl border p-0.5 transition-all duration-200',
          selectedColor === color.name ? 'border-primary ring-primary/20 shadow-lg ring-2' : '',
          isProcessing
            ? selectedColor === color.name
              ? 'cursor-wait opacity-60'
              : 'pointer-events-none cursor-not-allowed opacity-50'
            : 'cursor-pointer hover:scale-105 hover:border-gray-300 hover:shadow-md',
        ]"
      >
        <div
          class="bg-base-100 relative aspect-[3/4] w-full overflow-hidden rounded-lg lg:aspect-[9/16]"
        >
          <img
            :src="color.preview"
            :alt="t(`colors.${color.name}`) as string"
            class="h-full w-full object-cover"
            loading="lazy"
            @load="onImageLoad(color.name)"
          />
          <div v-if="!imageLoaded[color.name]" class="skeleton absolute inset-0"></div>
          <!-- Loading overlay on selected color -->
          <div
            v-if="isProcessing && selectedColor === color.name"
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
                  selectedColor === color.name ? 'text-primary' : 'text-base-content',
                ]"
                >{{ t(`colors.${color.name}`) }}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
    <!-- Scroll indicator (mobile) -->
    <div class="relative mt-1 h-1 lg:hidden">
      <div
        class="bg-base-100/40 absolute top-0 bottom-0 rounded-full shadow-sm"
        :style="{ width: thumbWidthPct + '%', left: thumbLeftPct + '%' }"
      ></div>
    </div>
  </div>
</template>
