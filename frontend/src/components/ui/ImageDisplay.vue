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
      // Compare mode açıldı - event listener'ları ekle
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
    } else {
      // Compare mode kapandı - event listener'ları kaldır
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
        :style="{ aspectRatio: containerAspect, maxHeight: '75vh' }"
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

          <!-- Vertical Divider Line -->
          <div
            class="pointer-events-none absolute top-0 bottom-0 z-10 w-0.5 bg-white shadow-lg transition-all duration-75 ease-out"
            :style="{ left: `${sliderPosition}%` }"
          >
            <!-- Slider Handle -->
            <div
              class="pointer-events-auto absolute top-1/2 left-1/2 h-8 w-8 -translate-x-1/2 -translate-y-1/2 transform cursor-grab rounded-full border-2 border-gray-300 bg-white shadow-lg transition-all duration-75 ease-out"
              :class="{ 'cursor-grabbing': isDragging, 'scale-110': isDragging }"
              @mousedown.stop="handleMouseDown"
            >
              <!-- Drag indicator dots -->
              <div class="flex h-full items-center justify-center">
                <div class="mx-0.5 h-1 w-1 rounded-full bg-gray-400"></div>
                <div class="mx-0.5 h-1 w-1 rounded-full bg-gray-400"></div>
              </div>
            </div>
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
