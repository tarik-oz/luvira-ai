<script setup lang="ts">
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useAppState } from '../../composables/useAppState'
import { AVAILABLE_COLORS } from '../../config/colorConfig'
import hairService from '../../services/hairService'

const { t } = useI18n()
const { isProcessing, setProcessingError } = useAppState()
const selectedColor = ref<string | null>(null)

const colors = AVAILABLE_COLORS.map((name) => ({ name }))

const selectColor = async (colorName: string) => {
  if (isProcessing.value) return // Prevent multiple requests

  // Clear previous processing error on new action
  setProcessingError(null)

  const previous = selectedColor.value
  selectedColor.value = colorName
  console.log('Selected color:', colorName)

  try {
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
    } else {
      setProcessingError(t('colorPalette.error') as string)
    }
    // Revert selection to previous on failure
    selectedColor.value = previous
  }
}
</script>

<template>
  <div class="bg-base-content/70 border-base-content/80 rounded-2xl border p-4 shadow-lg">
    <!-- Header -->
    <div class="mb-4">
      <h3 class="text-base-100 mb-1 text-lg font-bold">{{ t('colorPalette.title') }}</h3>
      <p class="text-base-100/70 text-xs">{{ t('colorPalette.instruction') }}</p>
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
            ? 'border-primary ring-primary/20 shadow-lg ring-2'
            : 'border-gray-200',
          isProcessing
            ? selectedColor === color.name
              ? 'cursor-wait'
              : 'pointer-events-none cursor-not-allowed opacity-50'
            : 'cursor-pointer hover:scale-105 hover:border-gray-300 hover:shadow-md',
        ]"
      >
        <!-- Loading Spinner for selected color -->
        <div
          v-if="isProcessing && selectedColor === color.name"
          class="bg-base-100 mx-auto mb-2 flex h-12 w-12 items-center justify-center rounded-lg"
        >
          <div class="border-accent h-6 w-6 animate-spin rounded-full border-t-4 border-b-4"></div>
        </div>
        <!-- Placeholder Image -->
        <div
          v-else
          class="bg-base-100 mx-auto mb-2 flex h-12 w-12 items-center justify-center rounded-lg"
        >
          <span class="text-base-300 text-xl">🖼️</span>
        </div>
        <!-- Color Name -->
        <span
          :class="[
            'block text-center text-xs font-medium transition-colors duration-200',
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
