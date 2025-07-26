<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import demoImage1 from '../../assets/samples/sample1.jpg'
import demoImage2 from '../../assets/samples/sample2.jpg'
import demoImage3 from '../../assets/samples/sample3.jpg'

const currentColorIndex = ref(0)
const intervalId = ref<number | null>(null)

const demoImages = [demoImage1, demoImage2, demoImage3]

// Different hair colors to cycle through
const hairColors = [
  { name: 'Original', class: '' },
  { name: 'Blonde', class: 'filter hue-rotate-30 brightness-110' },
  { name: 'Red', class: 'filter hue-rotate-15 saturate-150' },
  { name: 'Brown', class: 'filter hue-rotate-45 saturate-80 brightness-90' },
  { name: 'Black', class: 'filter brightness-50 contrast-120' },
  { name: 'Purple', class: 'filter hue-rotate-270 saturate-150' },
  { name: 'Pink', class: 'filter hue-rotate-320 saturate-200' },
  { name: 'Blue', class: 'filter hue-rotate-200 saturate-150' },
]

const startAnimation = () => {
  intervalId.value = window.setInterval(() => {
    currentColorIndex.value = (currentColorIndex.value + 1) % hairColors.length
  }, 2000) // Change color every 2 seconds
}

const stopAnimation = () => {
  if (intervalId.value) {
    clearInterval(intervalId.value)
    intervalId.value = null
  }
}

onMounted(() => {
  startAnimation()
})

onUnmounted(() => {
  stopAnimation()
})
</script>

<template>
  <div class="text-center mb-6">
    <div class="flex justify-center space-x-4">
      <!-- Three demo images with animated color changes -->
      <div v-for="(image, imageIndex) in demoImages" :key="imageIndex" class="relative">
        <div class="relative overflow-hidden rounded-xl shadow-lg">
          <img
            :src="image"
            :alt="`Hair color demo ${imageIndex + 1}`"
            class="w-40 h-40 object-cover transition-all duration-1000 ease-in-out"
            :class="hairColors[currentColorIndex].class"
          />

          <!-- Color name overlay -->
          <div
            class="absolute bottom-2 left-2 bg-black bg-opacity-70 text-white px-2 py-1 rounded-full text-xs font-medium"
          >
            {{ hairColors[currentColorIndex].name }}
          </div>
        </div>
      </div>
    </div>

    <!-- Color indicators -->
    <div class="flex justify-center mt-3 space-x-1">
      <div
        v-for="(color, index) in hairColors"
        :key="index"
        class="w-2 h-2 rounded-full transition-all duration-300"
        :class="[index === currentColorIndex ? 'bg-blue-500 scale-125' : 'bg-gray-300']"
      ></div>
    </div>
  </div>
</template>
