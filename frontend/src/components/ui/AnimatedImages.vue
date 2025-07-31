<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'

import imgBlack from '@/assets/samples/1_black.png'
import imgBrown from '@/assets/samples/1_brown.png'
import imgBlonde from '@/assets/samples/1_blonde.png'
import imgRed from '@/assets/samples/1_copper.png'
import imgGray from '@/assets/samples/1_gray.png'
import imgPurple from '@/assets/samples/1_purple.png'
import imgBlue from '@/assets/samples/1_blue.png'
import imgPink from '@/assets/samples/1_pink.png'

const images = [
  { src: imgBlack, name: 'Black' },
  { src: imgBrown, name: 'Brown' },
  { src: imgBlonde, name: 'Blonde' },
  { src: imgRed, name: 'Copper' },
  { src: imgGray, name: 'Gray' },
  { src: imgPurple, name: 'Purple' },
  { src: imgBlue, name: 'Blue' },
  { src: imgPink, name: 'Pink' },
]

const currentPair = ref(0)

const reveal = ref(0) // 0-100
const center = ref({ x: 50, y: 50 }) // clip-path merkez yüzdesi

// Her seferinde 4'lü (2x2) göster
const getQuad = () => {
  const i = currentPair.value
  return [
    images[i % images.length],
    images[(i + 1) % images.length],
    images[(i + 2) % images.length],
    images[(i + 3) % images.length],
  ]
}

const animateReveal = () => {
  reveal.value = 0
  // Merkez rastgele (40-60 arası)
  center.value = {
    x: 40 + Math.random() * 20,
    y: 40 + Math.random() * 20,
  }
  const step = 0.5
  const interval = setInterval(() => {
    if (reveal.value < 100) {
      reveal.value += step
    } else {
      clearInterval(interval)
    }
  }, 20) // çok daha yavaş ve yumuşak
}

const nextQuad = () => {
  currentPair.value = (currentPair.value + 4) % images.length
  animateReveal()
}

onMounted(() => {
  animateReveal()
  setInterval(nextQuad, 2500)
})
</script>

<template>
  <div class="flex flex-col items-center gap-6">
    <div class="grid grid-cols-2 gap-8">
      <div v-for="(img, idx) in getQuad()" :key="img.name" class="relative w-56 h-56">
        <!-- Alttaki eski renk -->
        <img
          :src="getQuad()[0].src"
          :alt="getQuad()[0].name"
          class="w-56 h-56 object-cover rounded-xl shadow-lg absolute inset-0"
        />
        <!-- Üstteki yeni renk, yuvarlak reveal -->
        <img
          v-if="idx !== 0"
          :src="img.src"
          :alt="img.name"
          class="w-56 h-56 object-cover rounded-xl shadow-lg absolute inset-0 transition-all duration-700"
          :style="{
            clipPath: `circle(${reveal}% at ${center.x}% ${center.y}%)`,
          }"
        />
        <div
          class="absolute bottom-3 left-1/2 -translate-x-1/2 bg-black bg-opacity-70 text-white px-3 py-1 rounded-full text-base font-semibold mt-2"
        >
          {{ img.name }}
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.7s;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
