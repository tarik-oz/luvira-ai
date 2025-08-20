<script setup lang="ts">
import { ref, watch, computed } from 'vue'
import { useAppState } from '../../composables/useAppState'
import { useI18n } from 'vue-i18n'

interface Props {
  compareMode?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  compareMode: false,
})

const { t } = useI18n()
const { uploadedImage, processedImage, isProcessing, currentColorResult, selectedTone } =
  useAppState()

// Keep container aspect ratio equal to the image to avoid cropping tall/portrait photos
const containerAspect = ref('1 / 1')
const imageNaturalWidth = ref(1)
const imageNaturalHeight = ref(1)
const updateAspectFromImage = () => {
  if (!uploadedImage.value) {
    containerAspect.value = '1 / 1'
    return
  }
  const img = new Image()
  img.onload = () => {
    const w = img.naturalWidth || img.width
    const h = img.naturalHeight || img.height
    if (w > 0 && h > 0) {
      containerAspect.value = `${w} / ${h}`
      imageNaturalWidth.value = w
      imageNaturalHeight.value = h
    }
  }
  img.src = uploadedImage.value
}

watch(uploadedImage, () => updateAspectFromImage(), { immediate: true })

// Compare mode functionality
const containerRef = ref<HTMLElement>()
const leftPaneRef = ref<HTMLElement>()
const rightPaneRef = ref<HTMLElement>()
const dividerRef = ref<HTMLElement>()
const sliderPosition = ref(50) // Percentage (source of truth only)
const isDragging = ref(false)
let activePointerId: number | null = null
let rafId: number | null = null
let pendingPercent = sliderPosition.value

function scheduleApply(percent: number) {
  pendingPercent = percent
  if (rafId == null) {
    rafId = requestAnimationFrame(() => {
      rafId = null
      const p = Math.max(0, Math.min(100, pendingPercent))
      const lp = leftPaneRef.value
      const rp = rightPaneRef.value
      const dv = dividerRef.value
      if (lp) lp.style.clipPath = `polygon(0 0, ${p}% 0, ${p}% 100%, 0 100%)`
      if (rp) rp.style.clipPath = `polygon(${p}% 0, 100% 0, 100% 100%, ${p}% 100%)`
      if (dv) dv.style.left = p + '%'
      sliderPosition.value = p
    })
  }
}

function updateFromClientX(clientX: number) {
  if (!containerRef.value) return
  const rect = containerRef.value.getBoundingClientRect()
  const x = clientX - rect.left
  const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100))
  scheduleApply(percentage)
}

function handlePointerDown(event: PointerEvent) {
  if (!props.compareMode || !containerRef.value) return
  isDragging.value = true
  activePointerId = event.pointerId
  try {
    containerRef.value.setPointerCapture(event.pointerId)
  } catch {}
  updateFromClientX(event.clientX)
  event.preventDefault()
}

function handlePointerMove(event: PointerEvent) {
  if (!isDragging.value || activePointerId == null) return
  updateFromClientX(event.clientX)
}

function handlePointerUp() {
  if (activePointerId !== null) {
    try {
      containerRef.value?.releasePointerCapture(activePointerId)
    } catch {}
  }
  activePointerId = null
  isDragging.value = false
}

function handleClick(event: MouseEvent) {
  if (!props.compareMode) return
  updateFromClientX(event.clientX)
}

// Ensure styles are applied once elements are ready or when compare toggles
watch(
  () => [containerRef.value, props.compareMode],
  () => scheduleApply(sliderPosition.value),
  { immediate: true },
)

// Label text for active color/tone
const labelText = computed(() => {
  if (!currentColorResult.value) return ''
  const colorKey = currentColorResult.value.originalColor
  const colorLabel = t('colors.' + colorKey) as string
  const tone = selectedTone.value
  if (tone) {
    const toneLabel = tone.charAt(0).toUpperCase() + tone.slice(1)
    return `${colorLabel} · ${toneLabel}`
  }
  return colorLabel
})

const isPortrait = computed(() => imageNaturalHeight.value > imageNaturalWidth.value)
// Larger on phones: allow taller viewport height when not lg
const maxHeightVh = computed(() => {
  // Heuristic using window width; Vue SSR safe guard
  const w = typeof window !== 'undefined' ? window.innerWidth : 1024
  const isLg = w >= 1024
  if (isLg) return isPortrait.value ? '60vh' : '90vh'
  // phone/tablet: give more height so square images feel bigger
  return isPortrait.value ? '72vh' : '92vh'
})
</script>

<template>
  <div class="mx-auto w-full max-w-4xl">
    <!-- Image Container -->
    <div class="border-base-content relative overflow-hidden rounded-2xl border-2 shadow-lg">
      <!-- Loading Overlay -->
      <div
        v-if="isProcessing"
        class="absolute inset-0 z-10 flex items-center justify-center bg-black/50"
      >
        <div class="flex flex-col items-center gap-3">
          <span
            class="border-accent h-12 w-12 animate-spin rounded-full border-t-4 border-b-4"
          ></span>
          <span class="font-medium text-white">{{ t('processing.working') }}</span>
        </div>
      </div>

      <!-- Image Display Area -->
      <div
        ref="containerRef"
        class="bg-base-content relative w-full touch-none"
        :style="{ aspectRatio: containerAspect, maxHeight: maxHeightVh }"
        :class="{ 'cursor-pointer select-none': compareMode }"
        @pointerdown="handlePointerDown"
        @pointermove="handlePointerMove"
        @pointerup="handlePointerUp"
        @pointercancel="handlePointerUp"
        @click="handleClick"
      >
        <!-- Normal Mode: layer original (bottom) + overlay (top) -->
        <template v-if="!compareMode">
          <img
            v-if="uploadedImage"
            :src="uploadedImage"
            alt="Original"
            class="h-full w-full object-contain"
            :class="{ 'opacity-75': isProcessing }"
          />
          <img
            v-if="processedImage"
            :src="processedImage"
            alt="Overlay"
            class="pointer-events-none absolute inset-0 z-0 h-full w-full object-contain"
            :class="{ 'opacity-75': isProcessing }"
          />
        </template>

        <!-- Compare Mode -->
        <template v-else>
          <!-- Original Image (Left Side) -->
          <div
            ref="leftPaneRef"
            class="absolute inset-0 h-full w-full overflow-hidden will-change-[clip-path]"
            :style="{
              clipPath: `polygon(0 0, ${sliderPosition}% 0, ${sliderPosition}% 100%, 0 100%)`,
            }"
          >
            <img
              v-if="uploadedImage"
              :src="uploadedImage"
              alt="Original"
              class="h-full w-full object-contain"
            />
            <!-- Before Label (Only visible in left part) -->
            <div
              class="bg-base-100 text-base-content absolute top-4 left-4 rounded-full px-3 py-1.5 text-sm font-semibold shadow-lg"
            >
              {{ t('processing.before') }}
            </div>
          </div>

          <!-- Processed Image (Right Side): original + overlay layered -->
          <div
            ref="rightPaneRef"
            class="absolute inset-0 h-full w-full overflow-hidden will-change-[clip-path]"
            :style="{
              clipPath: `polygon(${sliderPosition}% 0, 100% 0, 100% 100%, ${sliderPosition}% 100%)`,
            }"
          >
            <img
              v-if="uploadedImage"
              :src="uploadedImage"
              alt="Original"
              class="h-full w-full object-contain"
            />
            <img
              v-if="processedImage"
              :src="processedImage"
              alt="Overlay"
              class="pointer-events-none absolute inset-0 z-0 h-full w-full object-contain"
            />
            <!-- After Label (Only visible in right part) -->
            <div
              class="bg-base-100 text-base-content absolute top-4 right-4 rounded-full px-3 py-1.5 text-sm font-semibold shadow-lg"
            >
              {{ t('processing.after') }}
            </div>
          </div>

          <!-- Vertical Divider Line + Modern Handle -->
          <div
            ref="dividerRef"
            class="will-change-left pointer-events-none absolute top-0 bottom-0 z-10"
            :style="{ left: `${sliderPosition}%` }"
          >
            <!-- Line -->
            <div
              class="absolute top-0 bottom-0 left-1/2 w-[2px] -translate-x-1/2 bg-white/80 shadow-[0_0_8px_rgba(0,0,0,0.25)]"
            ></div>

            <!-- Handle: minimal, translucent, off-center to avoid face overlap -->
            <button
              type="button"
              class="focus:ring-accent/50 pointer-events-auto absolute top-[50%] left-1/2 flex h-7 w-7 -translate-x-1/2 -translate-y-1/2 transform cursor-grab items-center justify-center rounded-full bg-white/40 shadow ring-1 ring-white/50 backdrop-blur-sm transition-all duration-150 ease-out hover:scale-105 hover:bg-white/50 focus:ring-2 focus:outline-none"
              :class="{ 'scale-105 cursor-grabbing': isDragging }"
              @pointerdown.stop="handlePointerDown"
              aria-label="Compare slider"
            >
              <svg viewBox="0 0 24 24" class="h-3.5 w-3.5 text-white/90">
                <path fill="currentColor" d="M15.41 7.41 14 6l-6 6 6 6 1.41-1.41L10.83 12z" />
              </svg>
              <div class="mx-[3px] h-0.5 w-0.5 rounded-full bg-white/80"></div>
              <svg viewBox="0 0 24 24" class="h-3.5 w-3.5 text-white/90">
                <path fill="currentColor" d="M8.59 16.59 10 18l6-6-6-6-1.41 1.41L13.17 12z" />
              </svg>
            </button>
          </div>
        </template>

        <!-- Bottom centered color/tone label (only in normal mode) -->
        <div
          v-if="!compareMode && labelText"
          class="bg-base-100 text-base-content absolute bottom-4 left-1/2 z-30 w-fit max-w-[min(24ch,calc(100%-2rem))] -translate-x-1/2 transform rounded-full px-3 py-2 text-center text-sm leading-snug font-semibold break-words shadow-md transition duration-200 ease-out"
          style="text-wrap: balance"
          :class="[isProcessing ? 'opacity-75' : 'opacity-100', 'translate-y-0']"
        >
          {{ labelText }}
        </div>
      </div>
    </div>
  </div>
</template>
