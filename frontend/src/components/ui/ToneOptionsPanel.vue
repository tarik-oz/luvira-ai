<script setup lang="ts">
import { ref, watch, computed } from 'vue'
import { useAppState } from '../../composables/useAppState'

const props = defineProps<{
  color: { name: string; rgb: [number, number, number] } | null
  isLoading?: boolean
}>()

const { currentColorResult, setProcessedImageToTone } = useAppState()
const selectedTone = ref<string | null>(null)

// Get available tones from the current color result
const availableTones = computed(() => {
  if (!currentColorResult.value || !props.color) return []
  return Object.keys(currentColorResult.value.tones).filter(
    (tone) => currentColorResult.value!.tones[tone] !== null,
  )
})

// Reset selected tone when color changes
watch(
  () => props.color,
  () => {
    selectedTone.value = null
  },
)

const handleToneSelect = (tone: string) => {
  selectedTone.value = tone
  setProcessedImageToTone(tone)
}

const handleBaseSelect = () => {
  selectedTone.value = null
  setProcessedImageToTone(null)
}
</script>

<template>
  <div v-if="color && currentColorResult" class="mt-6">
    <h4 class="text-md font-semibold text-gray-700 mb-2">Choose Tone</h4>
    <div class="bg-white rounded-lg border border-gray-200 p-4">
      <!-- Loading State -->
      <div v-if="props.isLoading" class="text-center py-8">
        <div class="flex items-center justify-center space-x-2">
          <svg class="animate-spin h-5 w-5 text-blue-600" fill="none" viewBox="0 0 24 24">
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
          <span class="text-gray-500">Loading tones...</span>
        </div>
      </div>

      <!-- Tone Options -->
      <div v-else class="grid grid-cols-2 sm:grid-cols-3 gap-3">
        <!-- Base Color Option -->
        <button
          @click="handleBaseSelect"
          :class="[
            'w-full aspect-square rounded-lg border flex flex-col items-center justify-center cursor-pointer shadow-sm p-2 bg-gray-50 transition-all',
            selectedTone === null
              ? 'border-blue-500 ring-2 ring-blue-300 scale-105'
              : 'border-gray-200',
          ]"
        >
          <span
            class="block text-xs font-semibold text-gray-900 bg-white/80 rounded px-2 py-1 mt-auto mb-1 shadow"
          >
            Base
          </span>
        </button>

        <!-- Tone Options -->
        <button
          v-for="tone in availableTones"
          :key="tone"
          @click="handleToneSelect(tone)"
          :class="[
            'w-full aspect-square rounded-lg border flex flex-col items-center justify-center cursor-pointer shadow-sm p-2 bg-gray-50 transition-all',
            selectedTone === tone
              ? 'border-blue-500 ring-2 ring-blue-300 scale-105'
              : 'border-gray-200',
          ]"
        >
          <span
            class="block text-xs font-semibold text-gray-900 bg-white/80 rounded px-2 py-1 mt-auto mb-1 shadow"
          >
            {{ tone }}
          </span>
        </button>

        <!-- No tones available message -->
        <div v-if="availableTones.length === 0" class="col-span-full text-center py-4">
          <span class="text-gray-400">No tones available for this color.</span>
        </div>
      </div>
    </div>
  </div>
</template>
