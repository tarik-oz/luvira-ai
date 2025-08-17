<script setup lang="ts">
import { ref, watch, onUnmounted, computed } from 'vue'
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
const sliderPosition = ref(50) // Percentage
const isDragging = ref(false)

const handleMouseDown = (event: MouseEvent) => {
  if (!props.compareMode || !containerRef.value) return

  isDragging.value = true
  updateSliderPosition(event)
  event.preventDefault()
}

const handleMouseMove = (event: MouseEvent) => {
  if (!isDragging.value || !containerRef.value) return
  updateSliderPosition(event)
}

const handleMouseUp = () => {
  isDragging.value = false
}

const handleClick = (event: MouseEvent) => {
  if (!props.compareMode || !containerRef.value) return
  updateSliderPosition(event)
}

const updateSliderPosition = (event: MouseEvent) => {
  if (!containerRef.value) return

  const rect = containerRef.value.getBoundingClientRect()
  const x = event.clientX - rect.left
  const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100))
  sliderPosition.value = percentage
}

// Watch compareMode prop to add/remove event listeners
watch(
  () => props.compareMode,
  (newValue) => {
    if (newValue) {
      // Compare mode enabled - add event listeners
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
    } else {
      // Compare mode disabled - remove event listeners
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      isDragging.value = false
    }
  },
  { immediate: true },
)

onUnmounted(() => {
  // Component unmount olurken tüm listener'ları temizle
  document.removeEventListener('mousemove', handleMouseMove)
  document.removeEventListener('mouseup', handleMouseUp)
})

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
const maxHeightVh = computed(() => (isPortrait.value ? '60vh' : '90vh'))
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
        class="bg-base-content relative w-full"
        :style="{ aspectRatio: containerAspect, maxHeight: maxHeightVh }"
        :class="{ 'cursor-pointer select-none': compareMode }"
        @mousedown="handleMouseDown"
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
            class="absolute inset-0 h-full w-full overflow-hidden"
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
            class="absolute inset-0 h-full w-full overflow-hidden"
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
            class="pointer-events-none absolute top-0 bottom-0 z-10 transition-all duration-150 ease-out"
            :style="{ left: `${sliderPosition}%` }"
          >
            <!-- Line -->
            <div
              class="absolute top-0 bottom-0 left-1/2 w-[2px] -translate-x-1/2 bg-white/80 shadow-[0_0_8px_rgba(0,0,0,0.25)]"
            ></div>

            <!-- Handle -->
            <button
              type="button"
              class="focus:ring-accent/50 pointer-events-auto absolute top-1/2 left-1/2 flex h-10 w-10 -translate-x-1/2 -translate-y-1/2 transform cursor-grab items-center justify-center rounded-full bg-white/90 shadow-xl ring-1 ring-white/60 backdrop-blur-md transition-all duration-150 ease-out hover:scale-105 hover:bg-white focus:ring-2 focus:outline-none"
              :class="{ 'scale-110 cursor-grabbing': isDragging }"
              @mousedown.stop="handleMouseDown"
              aria-label="Compare slider"
            >
              <!-- Left chevron -->
              <svg viewBox="0 0 24 24" class="h-4 w-4 text-gray-600">
                <path fill="currentColor" d="M15.41 7.41L14 6l-6 6 6 6 1.41-1.41L10.83 12z" />
              </svg>
              <!-- Divider dot -->
              <div class="mx-1 h-1 w-1 rounded-full bg-gray-400/80"></div>
              <!-- Right chevron -->
              <svg viewBox="0 0 24 24" class="h-4 w-4 text-gray-600">
                <path fill="currentColor" d="M8.59 16.59L10 18l6-6-6-6-1.41 1.41L13.17 12z" />
              </svg>
            </button>
          </div>
        </template>

        <!-- Bottom centered color/tone label (only in normal mode) -->
        <div
          v-if="!compareMode && labelText"
          class="bg-base-100 text-base-content absolute bottom-4 left-1/2 z-30 -translate-x-1/2 transform rounded-full px-4 py-2 text-sm font-semibold shadow-md transition duration-200 ease-out"
          :class="[isProcessing ? 'opacity-75' : 'opacity-100', 'translate-y-0']"
        >
          {{ labelText }}
        </div>
      </div>
    </div>
  </div>
</template>
