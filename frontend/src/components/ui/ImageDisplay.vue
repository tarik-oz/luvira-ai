<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useAppState } from '../../composables/useAppState'

interface Props {
  uploadedImage: string | null
  sessionId: string | null
}

const props = defineProps<Props>()
const { processedImage, currentColorResult, selectedTone } = useAppState()

const isCompareMode = ref(false)
const sliderPosition = ref(50) // Percentage for slider position
const isDragging = ref(false)
const containerRef = ref<HTMLElement>()

const toggleCompareMode = () => {
  isCompareMode.value = !isCompareMode.value
}

const handleMouseDown = (event: MouseEvent) => {
  isDragging.value = true
  handleSliderMove(event)
}

const handleMouseUp = () => {
  isDragging.value = false
}

const handleSliderMove = (event: MouseEvent) => {
  if (!isDragging.value || !containerRef.value) return
  const rect = containerRef.value.getBoundingClientRect()
  const x = event.clientX - rect.left
  const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100))
  sliderPosition.value = percentage
}

const handleGlobalMouseMove = (event: MouseEvent) => {
  if (isDragging.value) {
    handleSliderMove(event)
  }
}

const handleGlobalMouseUp = () => {
  isDragging.value = false
}

onMounted(() => {
  document.addEventListener('mousemove', handleGlobalMouseMove)
  document.addEventListener('mouseup', handleGlobalMouseUp)
})

onUnmounted(() => {
  document.removeEventListener('mousemove', handleGlobalMouseMove)
  document.removeEventListener('mouseup', handleGlobalMouseUp)
})

// Determine which image to show in single mode
const displayImage = computed(() => processedImage.value || props.uploadedImage)
const imageTitle = computed(() => (processedImage.value ? 'Processed Image' : 'Original Image'))

// Download functionality
const downloadImage = () => {
  if (!processedImage.value || !currentColorResult.value) return

  const link = document.createElement('a')
  link.href = processedImage.value

  // Create filename with color and tone
  const colorName = currentColorResult.value.color.toLowerCase().replace(/\s+/g, '_')
  const toneName = selectedTone.value
    ? selectedTone.value.toLowerCase().replace(/\s+/g, '_')
    : 'base'
  const filename = `hair_${colorName}_${toneName}.png`

  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}
</script>

<template>
  <div class="space-y-4">
    <!-- Image Display Header -->
    <div class="flex justify-between items-center">
      <h3 class="text-lg font-semibold text-gray-800">
        {{ isCompareMode ? 'Compare Images' : imageTitle }}
        <span v-if="currentColorResult && !isCompareMode" class="text-sm font-normal text-gray-600">
          ({{ currentColorResult.color }}{{ selectedTone ? ` - ${selectedTone}` : '' }})
        </span>
      </h3>

      <div class="flex items-center space-x-3">
        <!-- Download Button -->
        <button
          v-if="processedImage"
          @click="downloadImage"
          class="inline-flex items-center px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors"
          title="Download processed image"
        >
          <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            ></path>
          </svg>
          Download
        </button>

        <!-- Compare Mode Toggle -->
        <div v-if="processedImage && props.uploadedImage" class="flex items-center space-x-2">
          <span class="text-sm text-gray-600">Compare</span>
          <button
            @click="toggleCompareMode"
            :class="[
              'relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2',
              isCompareMode ? 'bg-green-600' : 'bg-gray-200',
            ]"
          >
            <span
              :class="[
                'inline-block h-4 w-4 transform rounded-full bg-white transition-transform',
                isCompareMode ? 'translate-x-6' : 'translate-x-1',
              ]"
            />
          </button>
        </div>
      </div>
    </div>

    <!-- Image Container -->
    <div class="bg-white rounded-lg border-2 border-gray-200 p-4">
      <!-- Single Image Mode -->
      <div
        v-if="!isCompareMode"
        class="aspect-square w-full overflow-hidden rounded-lg bg-gray-100"
      >
        <img
          v-if="displayImage"
          :src="displayImage"
          :alt="imageTitle"
          class="w-full h-full object-cover"
        />
        <div v-else class="w-full h-full flex items-center justify-center">
          <p class="text-gray-500">No image uploaded</p>
        </div>
      </div>

      <!-- Compare Mode -->
      <div
        v-else
        ref="containerRef"
        class="relative aspect-square w-full overflow-hidden rounded-lg bg-gray-100 cursor-col-resize select-none"
        @mousedown="handleMouseDown"
      >
        <!-- Original Image (Right side) -->
        <div class="absolute inset-0">
          <img
            :src="props.uploadedImage!"
            alt="Original image"
            class="w-full h-full object-cover"
          />
        </div>

        <!-- Processed Image (Left side) -->
        <div
          class="absolute inset-0 overflow-hidden"
          :style="{
            clipPath: `polygon(0 0, ${sliderPosition}% 0, ${sliderPosition}% 100%, 0 100%)`,
          }"
        >
          <img :src="processedImage!" alt="Processed image" class="w-full h-full object-cover" />
        </div>

        <!-- Slider Line -->
        <div
          class="absolute top-0 bottom-0 w-0.5 bg-white shadow-lg pointer-events-none"
          :style="{ left: `${sliderPosition}%` }"
        >
          <!-- Slider Handle -->
          <div
            :class="[
              'absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-white rounded-full shadow-lg border-2 flex items-center justify-center transition-colors',
              isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300',
            ]"
          >
            <svg
              class="w-4 h-4 text-gray-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M8 9l4-4 4 4m0 6l-4 4-4-4"
              ></path>
            </svg>
          </div>
        </div>
      </div>

      <!-- Compare Mode Labels -->
      <div v-if="isCompareMode" class="flex justify-between mt-2 text-xs text-gray-500">
        <span>← Processed</span>
        <span>Original →</span>
      </div>
    </div>

    <!-- Session Info -->
    <div v-if="props.sessionId" class="bg-blue-50 border border-blue-200 rounded-lg p-3">
      <p class="text-sm text-blue-800">
        <span class="font-semibold">Session ID:</span> {{ props.sessionId }}
      </p>
      <p class="text-xs text-blue-600 mt-1">
        Hair mask has been generated and cached for fast color changes
      </p>
    </div>
  </div>
</template>
