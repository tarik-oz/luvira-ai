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

// Default vertical pattern preview
const defaultPatternUrl = new URL('../../assets/hair_patterns/default.webp', import.meta.url).href

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
    <div class="grid grid-cols-6 gap-2 md:grid-cols-6 lg:grid-cols-7">
      <div
        v-for="color in colors"
        :key="color.name"
        @click="selectColor(color.name)"
        :class="[
          'bg-base-content/80 border-base-100/20 rounded-xl border p-0.5 transition-all duration-200',
          selectedColor === color.name ? 'border-primary ring-primary/20 shadow-lg ring-2' : '',
          isProcessing
            ? selectedColor === color.name
              ? 'cursor-wait opacity-60'
              : 'pointer-events-none cursor-not-allowed opacity-50'
            : 'cursor-pointer hover:scale-105 hover:border-gray-300 hover:shadow-md',
        ]"
      >
        <div
          class="bg-base-100 relative w-full overflow-hidden rounded-lg"
          style="aspect-ratio: 9 / 16"
        >
          <img
            :src="defaultPatternUrl"
            :alt="t(`colors.${color.name}`) as string"
            class="h-full w-full object-cover"
          />
          <!-- Loading overlay on selected color -->
          <div
            v-if="isProcessing && selectedColor === color.name"
            class="bg-base-content/30 absolute inset-0 flex items-center justify-center"
          >
            <div
              class="border-accent h-7 w-7 animate-spin rounded-full border-t-4 border-b-4"
            ></div>
          </div>
          <!-- Bottom label bar -->
          <div class="bg-base-100/95 absolute right-0 bottom-0 left-0" style="height: 20%">
            <div class="flex h-full w-full items-center justify-center">
              <span
                :class="[
                  'text-xs font-semibold',
                  selectedColor === color.name ? 'text-primary' : 'text-base-content',
                ]"
                >{{ t(`colors.${color.name}`) }}</span
              >
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
