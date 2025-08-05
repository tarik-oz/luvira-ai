<script setup lang="ts">
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useAppState } from '../../composables/useAppState'
import hairService from '../../services/hairService'

const { t } = useI18n()
const { isProcessing } = useAppState()
const selectedColor = ref<string | null>(null)

const colors = [
  { name: 'Black' },
  { name: 'Blonde' },
  { name: 'Brown' },
  { name: 'Auburn' },
  { name: 'Pink' },
  { name: 'Blue' },
  { name: 'Purple' },
  { name: 'Gray' },
]

const selectColor = async (colorName: string) => {
  if (isProcessing.value) return // Prevent multiple requests

  selectedColor.value = colorName
  console.log('Selected color:', colorName)

  try {
    await hairService.changeHairColor(colorName)
    console.log('Color change completed successfully')
  } catch (error) {
    console.error('Color change failed:', error)
    // TODO: Show error message to user
  }
}
</script>

<template>
  <div class="bg-base-content/70 rounded-2xl shadow-lg border border-base-content/80 p-4">
    <!-- Header -->
    <div class="mb-4">
      <h3 class="text-lg font-bold text-base-100 mb-1">{{ t('colorPalette.title') }}</h3>
      <p class="text-xs text-base-100/70">{{ t('colorPalette.instruction') }}</p>
    </div>

    <!-- Color Grid -->
    <div class="grid grid-cols-4 gap-3">
      <div
        v-for="color in colors"
        :key="color.name"
        @click="selectColor(color.name)"
        :class="[
          'bg-base-content/80 rounded-xl border-2 p-3 transition-all duration-200',
          selectedColor === color.name
            ? 'border-primary ring-2 ring-primary/20 shadow-lg'
            : 'border-gray-200',
          isProcessing
            ? selectedColor === color.name
              ? 'cursor-wait'
              : 'opacity-50 cursor-not-allowed pointer-events-none'
            : 'cursor-pointer hover:border-gray-300 hover:scale-105 hover:shadow-md',
        ]"
      >
        <!-- Loading Spinner for selected color -->
        <div
          v-if="isProcessing && selectedColor === color.name"
          class="w-12 h-12 rounded-lg mx-auto mb-2 bg-base-100 flex items-center justify-center"
        >
          <div class="animate-spin rounded-full h-6 w-6 border-t-4 border-b-4 border-accent"></div>
        </div>
        <!-- Placeholder Image -->
        <div
          v-else
          class="w-12 h-12 rounded-lg mx-auto mb-2 bg-base-100 flex items-center justify-center"
        >
          <span class="text-base-300 text-xl">🖼️</span>
        </div>
        <!-- Color Name -->
        <span
          :class="[
            'text-xs font-medium text-center block transition-colors duration-200',
            selectedColor === color.name ? 'text-primary' : 'text-base-300',
            isProcessing && selectedColor !== color.name ? 'opacity-50' : '',
          ]"
        >
          {{ t(`colors.${color.name}`) }}
        </span>
      </div>
    </div>
  </div>
</template>
