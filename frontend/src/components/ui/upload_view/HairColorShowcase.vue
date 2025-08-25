<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue'
import { useI18n } from 'vue-i18n'
import { showcaseModels } from '@/data/showcaseModels'
const { t } = useI18n()

const models = ref(showcaseModels)

// --- Animation Timing ---
const TIMINGS = {
  HOLD_DURATION: 1500,
  TEXT_FADE_DURATION: 1000,
  COLOR_TRANSITION: 2000,
  STEP_INTERVAL: 50,
} as const

const COLOR_STEP_SIZE = 100 / (TIMINGS.COLOR_TRANSITION / TIMINGS.STEP_INTERVAL)

// --- Reactive State ---
const activeModelIndex = ref(0)
const currentColorIndex = ref(0)
const animationProgress = ref(0)
const textOpacity = ref(1)
const isTransitioning = ref(false)
const pendingModelIndex = ref<number | null>(null)

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
  if (isTransitioning.value) return
  // Switch model index; Transition handles crossfade
  isTransitioning.value = true
  activeModelIndex.value = (activeModelIndex.value + 1) % models.value.length
  currentColorIndex.value = 0
  animationProgress.value = 0
  textOpacity.value = 1
}

const selectModel = (index: number) => {
  if (activeModelIndex.value === index) return
  if (isTransitioning.value) {
    pendingModelIndex.value = index
    return
  }
  clearTimers() // Stop any ongoing animation immediately
  isTransitioning.value = true
  // Update model; Transition will crossfade
  activeModelIndex.value = index
  currentColorIndex.value = 0
  animationProgress.value = 0
  textOpacity.value = 1
}

function onAfterEnter() {
  // Crossfade completed
  isTransitioning.value = false
  runCycle()
  if (pendingModelIndex.value !== null && pendingModelIndex.value !== activeModelIndex.value) {
    const nextIndex = pendingModelIndex.value
    pendingModelIndex.value = null
    // defer to next tick to avoid interrupting current cycle start
    nextTick(() => selectModel(nextIndex))
  } else {
    pendingModelIndex.value = null
  }
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
  <div class="flex h-full flex-col items-center justify-center gap-4">
    <!-- Main Showcase -->
    <Transition name="fade" mode="out-in" @after-enter="onAfterEnter">
      <div
        :key="activeModelIndex"
        class="relative mb-4 aspect-[3/4] w-full max-w-[22rem] overflow-hidden rounded-xl sm:max-w-[24rem] md:aspect-auto md:h-[31rem] md:w-[29rem] md:max-w-none lg:h-[28rem] lg:w-[24rem]"
        role="img"
        aria-label="Hair color showcase"
      >
        <!-- Base Image (Current Color) -->
        <img
          :src="currentColor.src"
          :alt="currentColor.alt"
          class="absolute inset-0 h-full w-full object-cover"
        />
        <!-- Overlay Image (Next Color) -->
        <img
          :src="nextColor.src"
          :alt="nextColor.alt"
          class="absolute inset-0 h-full w-full object-cover"
          :style="getTransitionStyle()"
          width="480"
          height="640"
        />
        <!-- Color Label -->
        <div
          class="bg-base-100 text-base-content absolute bottom-4 left-1/2 -translate-x-1/2 rounded-full px-4 py-2 text-base font-semibold transition-opacity duration-1000"
          :style="{ opacity: textOpacity }"
        >
          {{
            animationProgress > 50
              ? `${t('colors.' + nextColor.colorName)} · ${nextColor.toneName}`
              : `${t('colors.' + currentColor.colorName)} · ${currentColor.toneName}`
          }}
        </div>
      </div>
    </Transition>
  </div>

  <!-- Model Selection Thumbnails -->
  <div
    class="mx-auto flex w-full max-w-[22rem] justify-center gap-3 sm:max-w-[24rem] md:w-[29rem] md:max-w-none md:gap-4 lg:w-96"
  >
    <div
      v-for="(model, index) in models"
      :key="model.id"
      @click="selectModel(index)"
      class="h-20 w-20 cursor-pointer overflow-hidden rounded-lg transition-all duration-500 sm:h-24 sm:w-24"
      :class="{
        'border-primary scale-110 border-2 shadow-lg': activeModelIndex === index,
        'hover:border-base-content/50 border-2 border-transparent': activeModelIndex !== index,
      }"
    >
      <img :src="model.thumbnail" :alt="model.thumbnailAlt" class="h-full w-full object-cover" />
    </div>
  </div>
</template>
