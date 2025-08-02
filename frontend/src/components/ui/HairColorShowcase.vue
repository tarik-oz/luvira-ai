<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { showcaseModels } from '@/data/showcaseModels'
const { t } = useI18n()

const models = ref(showcaseModels)

// --- Animation Timing ---
const TIMINGS = {
  HOLD_DURATION: 2000,
  TEXT_FADE_DURATION: 1000,
  COLOR_TRANSITION: 2000,
  STEP_INTERVAL: 50,
  MODEL_FADE_DURATION: 300, // Faster model transition - 300ms instead of 500ms
} as const

const COLOR_STEP_SIZE = 100 / (TIMINGS.COLOR_TRANSITION / TIMINGS.STEP_INTERVAL)

// --- Reactive State ---
const activeModelIndex = ref(0)
const currentColorIndex = ref(0)
const animationProgress = ref(0)
const textOpacity = ref(1)
const showcaseOpacity = ref(1) // For smooth model transitions

// --- Computed Properties for Dynamic Data ---
const activeModel = computed(() => models.value[activeModelIndex.value])
const activeColors = computed(() => activeModel.value.colors)
const currentColor = computed(() => activeColors.value[currentColorIndex.value])
const nextColor = computed(() => {
  const nextIndex = currentColorIndex.value + 1
  // If we're at the last color, show the same color as next (no wrap around)
  if (nextIndex >= activeColors.value.length) {
    return activeColors.value[currentColorIndex.value]
  }
  return activeColors.value[nextIndex]
})

// --- Animation Control ---
let colorInterval: number | null = null
let timeouts: number[] = []

const clearTimers = () => {
  if (colorInterval) clearInterval(colorInterval)
  timeouts.forEach(clearTimeout)
  colorInterval = null
  timeouts = []
}

const runCycle = () => {
  // This is the start of a hold period for the CURRENT color.
  textOpacity.value = 1
  animationProgress.value = 0

  const holdTimeout = setTimeout(() => {
    // Check if we're at the last color - if so, switch to next model directly
    if (currentColorIndex.value === activeColors.value.length - 1) {
      // We're at the last color, switch to the next model immediately
      switchToNextModel()
      return
    }

    // Hold is over. Start the transition to the NEXT color.
    textOpacity.value = 0 // Fade out text during transition

    // Fade the text back in to show the name of the NEXT color, timed with the transition.
    const textTimeout = setTimeout(() => {
      textOpacity.value = 1
    }, TIMINGS.TEXT_FADE_DURATION)
    timeouts.push(textTimeout)

    // Start the visual transition (opacity mask)
    colorInterval = setInterval(() => {
      animationProgress.value += COLOR_STEP_SIZE
      if (animationProgress.value >= 100) {
        animationProgress.value = 100
        completeTransition()
      }
    }, TIMINGS.STEP_INTERVAL)
  }, TIMINGS.HOLD_DURATION)
  timeouts.push(holdTimeout)
}

const completeTransition = () => {
  // The visual transition to the next color is now complete.
  if (colorInterval) clearInterval(colorInterval)
  colorInterval = null

  // Now, we HOLD the newly displayed color.
  const holdNewColorTimeout = setTimeout(() => {
    // Move to the next color in the same model (we already checked it's not the last color)
    currentColorIndex.value = currentColorIndex.value + 1
    runCycle() // Start the next cycle (hold, then transition).
  }, TIMINGS.HOLD_DURATION)
  timeouts.push(holdNewColorTimeout)
}

const switchToNextModel = () => {
  // Start fade out
  showcaseOpacity.value = 0

  // After fade out completes, switch model and start fade in
  const switchTimeout = setTimeout(() => {
    // Update model data
    activeModelIndex.value = (activeModelIndex.value + 1) % models.value.length
    currentColorIndex.value = 0

    // Reset animation state for new model
    animationProgress.value = 0
    textOpacity.value = 1

    // Start fade in immediately
    showcaseOpacity.value = 1

    // Wait for fade in to COMPLETELY finish before starting animation cycle
    const startCycleTimeout = setTimeout(() => {
      // Fade in tamamlandıktan sonra, ilk renk tam 2 saniye gözüksün
      setTimeout(() => {
        runCycle()
      }, TIMINGS.HOLD_DURATION - TIMINGS.MODEL_FADE_DURATION)
    }, TIMINGS.MODEL_FADE_DURATION) // Wait for fade in to complete
    timeouts.push(startCycleTimeout)
  }, TIMINGS.MODEL_FADE_DURATION) // Wait for fade out to complete
  timeouts.push(switchTimeout)
}

const selectModel = (index: number) => {
  if (activeModelIndex.value === index) return
  clearTimers() // Stop any ongoing animation immediately

  // Start fade out
  showcaseOpacity.value = 0

  // After fade out completes, switch model and start fade in
  const switchTimeout = setTimeout(() => {
    // Update model data
    activeModelIndex.value = index
    currentColorIndex.value = 0

    // Reset animation state for new model
    animationProgress.value = 0
    textOpacity.value = 1

    // Start fade in immediately
    showcaseOpacity.value = 1

    // Wait for fade in to COMPLETELY finish before starting animation cycle
    const startCycleTimeout = setTimeout(() => {
      // Fade in tamamlandıktan sonra, ilk renk tam 2 saniye gözüksün
      setTimeout(() => {
        runCycle()
      }, TIMINGS.HOLD_DURATION - TIMINGS.MODEL_FADE_DURATION)
    }, TIMINGS.MODEL_FADE_DURATION) // Wait for fade in to complete
    timeouts.push(startCycleTimeout)
  }, TIMINGS.MODEL_FADE_DURATION) // Wait for fade out to complete
  timeouts.push(switchTimeout)
}

onMounted(runCycle)
onUnmounted(clearTimers)

// --- Style Generation ---
const getTransitionStyle = () => ({
  opacity: animationProgress.value / 100,
  mask: `url(${activeModel.value.mask})`,
  WebkitMask: `url(${activeModel.value.mask})`,
  maskSize: 'cover',
  WebkitMaskSize: 'cover',
  maskRepeat: 'no-repeat',
  WebkitMaskRepeat: 'no-repeat',
  maskPosition: 'center',
  WebkitMaskPosition: 'center',
})
</script>

<template>
  <div class="flex flex-col items-center justify-center gap-4 h-full">
    <!-- Main Showcase -->
    <div
      class="relative overflow-hidden rounded-xl w-96 h-[28rem] transition-opacity duration-300"
      :style="{ opacity: showcaseOpacity }"
    >
      <!-- Base Image (Current Color) -->
      <img
        :src="currentColor.src"
        :alt="currentColor.alt"
        class="absolute inset-0 w-full h-full object-cover"
      />
      <!-- Overlay Image (Next Color) -->
      <img
        :src="nextColor.src"
        :alt="nextColor.alt"
        class="absolute inset-0 w-full h-full object-cover"
        :style="getTransitionStyle()"
      />
      <!-- Color Label -->
      <div
        class="absolute left-1/2 bottom-4 -translate-x-1/2 px-4 py-2 rounded-full bg-base-100 text-base-content text-base font-semibold transition-opacity duration-1000"
        :style="{ opacity: textOpacity }"
      >
        {{
          animationProgress > 50 ? t('colors.' + nextColor.name) : t('colors.' + currentColor.name)
        }}
      </div>
    </div>

    <!-- Model Selection Thumbnails -->
    <div class="flex gap-4">
      <div
        v-for="(model, index) in models"
        :key="model.id"
        @click="selectModel(index)"
        class="overflow-hidden rounded-lg w-24 h-24 cursor-pointer transition-all duration-500"
        :class="{
          'border-2 border-primary shadow-lg scale-110': activeModelIndex === index,
          'border-2 border-transparent hover:border-base-content/50': activeModelIndex !== index,
        }"
      >
        <img :src="model.thumbnail" :alt="`Model ${model.id}`" class="w-full h-full object-cover" />
      </div>
    </div>
  </div>
</template>
