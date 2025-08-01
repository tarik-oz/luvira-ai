<script setup lang="ts">
import { ref, onMounted } from 'vue'
import imgBlack from '@/assets/samples/1_black.png'
import imgBrown from '@/assets/samples/1_brown.png'
import hairMask from '@/assets/samples/1_prob_mask.png'

const currentTransition = ref(1) // 1-2 arası geçiş türü
const animationProgress = ref(0) // 0-100 renk geçişi
const currentColor = ref('Black')
const textOpacity = ref(1) // Text opacity kontrolü

// Geçiş türleri
const transitions = [{ name: 'Hair-Only Gradient', id: 1 }]

// Animasyon cycle
const startAnimation = () => {
  const runCycle = () => {
    // Phase 1: Hold current color (2 saniye)
    textOpacity.value = 1
    animationProgress.value = 0

    setTimeout(() => {
      // Phase 2: Text fade out (1 saniye)
      textOpacity.value = 0

      setTimeout(() => {
        // Phase 3: Color transition + Text fade in (aynı anda, 2 saniye)
        const colorInterval = setInterval(() => {
          animationProgress.value += 4 // 2 saniye için

          // Text fade in color transition ile birlikte başlasın
          if (animationProgress.value >= 20) {
            // %20'de text fade in başla
            textOpacity.value = 1
            currentColor.value = currentColor.value === 'Black' ? 'Brown' : 'Black'
          }

          if (animationProgress.value >= 100) {
            clearInterval(colorInterval)

            // Phase 4: Hold new color (2 saniye)
            setTimeout(() => {
              runCycle() // Yeni döngü başlat (tek geçiş türü olduğu için transition değiştirmeye gerek yok)
            }, 2000)
          }
        }, 50)
      }, 1000)
    }, 2000)
  }

  runCycle()
}

onMounted(() => {
  startAnimation()
})

// Geçiş stillerini döndür
const getTransitionStyle = () => {
  const progress = animationProgress.value

  return {
    opacity: progress / 100,
    mask: `url(${hairMask})`,
    WebkitMask: `url(${hairMask})`,
    maskSize: 'cover',
    WebkitMaskSize: 'cover',
    maskRepeat: 'no-repeat',
    WebkitMaskRepeat: 'no-repeat',
    maskPosition: 'center',
    WebkitMaskPosition: 'center',
  }
}
</script>

<template>
  <div class="flex flex-col items-center justify-center h-full">
    <!-- Tek model ortalı -->
    <div class="relative w-96 h-[28rem] overflow-hidden">
      <!-- Base image (Black) -->
      <img
        :src="imgBlack"
        alt="Hair Color Base"
        class="w-full h-full object-cover rounded-xl shadow-lg absolute inset-0"
      />

      <!-- Overlay image (Brown) with different transitions -->
      <img
        :src="imgBrown"
        alt="Hair Color Transition"
        class="w-full h-full object-cover rounded-xl shadow-lg absolute inset-0 transition-all duration-100"
        :style="getTransitionStyle()"
      />

      <!-- Color label with smooth text transition -->
      <div
        class="absolute bottom-4 left-1/2 -translate-x-1/2 bg-base-100 bg-opacity-70 text-base-content px-4 py-2 rounded-full text-base font-semibold transition-opacity duration-1000"
        :style="{ opacity: textOpacity }"
      >
        {{ currentColor }}
      </div>
    </div>
  </div>
</template>
