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
  <div class="w-full max-w-4xl mx-auto">
    <!-- Image Container -->
    <div class="relative rounded-2xl shadow-lg border-2 border-base-content overflow-hidden">
      <!-- Loading Overlay -->
      <div
        v-if="isProcessing"
        class="absolute inset-0 bg-black/50 flex items-center justify-center z-20"
      >
        <div class="flex flex-col items-center gap-3">
          <span
            class="animate-spin rounded-full h-12 w-12 border-t-4 border-b-4 border-accent"
          ></span>
          <span class="text-white font-medium">Processing hair color...</span>
        </div>
      </div>

      <!-- Image Display Area -->
      <div
        ref="containerRef"
        class="aspect-square w-full bg-gradient-to-br from-gray-50 to-gray-100 relative"
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
            class="w-full h-full object-cover"
            :class="{ 'opacity-75': isProcessing }"
          />
          <img
            v-if="processedImage"
            :src="processedImage"
            alt="Overlay"
            class="w-full h-full object-cover absolute inset-0 pointer-events-none z-10"
            :class="{ 'opacity-75': isProcessing }"
          />
        </template>

        <!-- Compare Mode -->
        <template v-else>
          <!-- Original Image (Left Side) -->
          <div
            class="absolute inset-0 w-full h-full overflow-hidden"
            :style="{
              clipPath: `polygon(0 0, ${sliderPosition}% 0, ${sliderPosition}% 100%, 0 100%)`,
            }"
          >
            <img
              v-if="uploadedImage"
              :src="uploadedImage"
              alt="Original"
              class="w-full h-full object-cover"
            />
            <!-- Before Label (Only visible in left part) -->
            <div
              class="absolute top-4 left-4 px-3 py-1.5 rounded-full bg-base-100 text-base-content text-sm font-semibold shadow-lg"
            >
              {{ t('processing.before') }}
            </div>
          </div>

          <!-- Processed Image (Right Side): original + overlay layered -->
          <div
            class="absolute inset-0 w-full h-full overflow-hidden"
            :style="{
              clipPath: `polygon(${sliderPosition}% 0, 100% 0, 100% 100%, ${sliderPosition}% 100%)`,
            }"
          >
            <img
              v-if="uploadedImage"
              :src="uploadedImage"
              alt="Original"
              class="w-full h-full object-cover"
            />
            <img
              v-if="processedImage"
              :src="processedImage"
              alt="Overlay"
              class="w-full h-full object-cover absolute inset-0 pointer-events-none z-10"
            />
            <!-- After Label (Only visible in right part) -->
            <div
              class="absolute top-4 right-4 px-3 py-1.5 rounded-full bg-base-100 text-base-content text-sm font-semibold shadow-lg"
            >
              {{ t('processing.after') }}
            </div>
          </div>

          <!-- Vertical Divider Line -->
          <div
            class="absolute top-0 bottom-0 w-0.5 bg-white shadow-lg pointer-events-none z-10 transition-all duration-75 ease-out"
            :style="{ left: `${sliderPosition}%` }"
          >
            <!-- Slider Handle -->
            <div
              class="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-white rounded-full shadow-lg border-2 border-gray-300 cursor-grab pointer-events-auto transition-all duration-75 ease-out"
              :class="{ 'cursor-grabbing': isDragging, 'scale-110': isDragging }"
              @mousedown.stop="handleMouseDown"
            >
              <!-- Drag indicator dots -->
              <div class="flex items-center justify-center h-full">
                <div class="w-1 h-1 bg-gray-400 rounded-full mx-0.5"></div>
                <div class="w-1 h-1 bg-gray-400 rounded-full mx-0.5"></div>
              </div>
            </div>
          </div>
        </template>

        <!-- Bottom centered color/tone label (only in normal mode) -->
        <div
          v-if="!compareMode && labelText"
          class="absolute left-1/2 bottom-4 -translate-x-1/2 px-4 py-2 rounded-full bg-base-100 text-base-content text-sm font-semibold shadow-md z-30 transition ease-out duration-200 transform"
          :class="[isProcessing ? 'opacity-75' : 'opacity-100', 'translate-y-0']"
        >
          {{ labelText }}
        </div>
      </div>
    </div>
  </div>
</template>
