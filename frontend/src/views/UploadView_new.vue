<script setup lang="ts">
import UploadSection from '../components/ui/UploadSection.vue'
import AppButton from '../components/ui/AppButton.vue'
import { PhImage, PhCamera } from '@phosphor-icons/vue'

const handleFileSelect = (file: File) => {
  console.log('Selected file:', file.name)
}

const handleCamera = () => {
  // Camera logic
  console.log('Camera clicked')
}
</script>

<template>
  <div class="grid grid-cols-1 md:grid-cols-2">
    <!-- Left column -->
    <div class="flex flex-col justify-center items-start px-4 md:px-0">
      <!-- Title -->
      <h1 class="text-5xl md:text-6xl font-extrabold text-base-content leading-tight mb-6">
        Online Hair Color & Tone Changer
      </h1>

      <!-- Description -->
      <p class="text-lg md:text-xl text-base-content/70 mb-8 max-w-xl">
        Easily try a wide selection of hair colors and natural tones. Instantly preview different
        looks and find the style that suits you best. Fast, realistic, and free. Give it a try now!
      </p>

      <!-- UploadSection -->
      <UploadSection @file-select="handleFileSelect" class="w-full max-w-xl" />

      <!-- Divider -->
      <div class="divider w-full max-w-xl my-6 text-base-content/70 font-semibold select-none">
        or
      </div>

      <!-- Buttons -->
      <div class="flex gap-4 w-full max-w-xl">
        <!-- Sample Images Button -->
        <AppButton class="flex-1" @click="() => console.log('Sample image selected')">
          <template #icon>
            <PhImage class="w-5 h-5" />
          </template>
          Use Sample Images
        </AppButton>

        <!-- Camera Button -->
        <AppButton class="flex-1" @click="handleCamera">
          <template #icon>
            <PhCamera class="w-5 h-5" />
          </template>
          Use Camera
        </AppButton>
      </div>
    </div>

    <!-- Right column -->
    <div>
      <AnimatedImages />
    </div>
  </div>
</template>
