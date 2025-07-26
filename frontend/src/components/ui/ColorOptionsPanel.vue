<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { apiService } from '../../services/apiService'

type ColorItem = { name: string; rgb: [number, number, number] }
const colors = ref<ColorItem[]>([])
const isLoading = ref(true)
const error = ref<string | null>(null)

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
            class="w-full aspect-square rounded-lg border-2 border-gray-200 flex flex-col items-center justify-center cursor-pointer shadow-sm p-2"
            :style="{ backgroundColor: `rgb(${color.rgb[0]},${color.rgb[1]},${color.rgb[2]})` }"
          >
            <span class="block text-xs font-semibold text-gray-900 bg-white/80 rounded px-2 py-1 mt-auto mb-1 shadow">
              {{ color.name }}
            </span>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
