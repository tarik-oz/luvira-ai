<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { apiService } from '../../services/apiService'
import ToneOptionsPanel from './ToneOptionsPanel.vue'
import { useAppState } from '../../composables/useAppState'

type ColorItem = { name: string; rgb: [number, number, number] }
const colors = ref<ColorItem[]>([])
const isLoading = ref(true)
const error = ref<string | null>(null)

const selectedColor = ref<ColorItem | null>(null)
const isProcessingColor = ref(false)
const { sessionId, setCurrentColorResult, getCachedColorResult } = useAppState()

onMounted(async () => {
  try {
    isLoading.value = true
    const fetchedColors = await apiService.getAvailableColors()
    colors.value = fetchedColors.sort((a, b) => a.name.localeCompare(b.name))
  } catch (e: any) {
    error.value = e?.message || 'Failed to load colors'
  } finally {
    isLoading.value = false
  }
})

const handleColorSelect = async (color: ColorItem) => {
  if (selectedColor.value?.name === color.name) return
  if (!sessionId.value) return

  selectedColor.value = color

  // Check if we have this color cached
  const cachedResult = getCachedColorResult(color.name)
  if (cachedResult) {
    console.log(`Using cached result for color: ${color.name}`)
    setCurrentColorResult(cachedResult)
    return
  }

  // Process new color
  isProcessingColor.value = true

  try {
    console.log(`Processing new color: ${color.name}`)
    const result = await apiService.changeHairColorAllTones(sessionId.value, color.name)
    setCurrentColorResult({
      color: result.color,
      baseResult: result.base_result,
      tones: result.tones,
    })
  } catch (e: any) {
    error.value = e?.message || 'Failed to process color change'
    console.error('Color processing error:', e)
  } finally {
    isProcessingColor.value = false
  }
}
</script>

<template>
  <div class="space-y-4">
    <h3 class="text-lg font-semibold text-gray-800">Choose Hair Color</h3>
    <div class="bg-white rounded-lg border-2 border-gray-200 p-6">
      <div v-if="isLoading" class="text-center py-8">
        <span class="text-gray-500">Loading colors...</span>
      </div>
      <div v-else-if="error" class="text-center py-8">
        <span class="text-red-500">{{ error }}</span>
      </div>
      <div v-else>
        <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
          <button
            v-for="color in colors"
            :key="color.name"
            @click="handleColorSelect(color)"
            :disabled="isProcessingColor"
            :class="[
              'relative w-full aspect-square rounded-lg border-2 flex flex-col items-center justify-center cursor-pointer shadow-sm p-2 transition-all',
              selectedColor?.name === color.name
                ? 'border-blue-500 ring-2 ring-blue-300 scale-105'
                : 'border-gray-200',
              isProcessingColor ? 'opacity-50 cursor-not-allowed' : 'hover:border-gray-300',
            ]"
            :style="{ backgroundColor: `rgb(${color.rgb[0]},${color.rgb[1]},${color.rgb[2]})` }"
          >
            <div
              v-if="isProcessingColor && selectedColor?.name === color.name"
              class="absolute inset-0 flex items-center justify-center bg-white/80 rounded-lg"
            >
              <svg class="animate-spin h-4 w-4 text-blue-600" fill="none" viewBox="0 0 24 24">
                <circle
                  class="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  stroke-width="4"
                ></circle>
                <path
                  class="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                ></path>
              </svg>
            </div>
            <span
              class="block text-xs font-semibold text-gray-900 bg-white/80 rounded px-2 py-1 mt-auto mb-1 shadow"
            >
              {{ color.name }}
            </span>
          </button>
        </div>
      </div>
    </div>

    <!-- Tone Section -->
    <ToneOptionsPanel :color="selectedColor" :is-loading="isProcessingColor" v-if="selectedColor" />
  </div>
</template>
