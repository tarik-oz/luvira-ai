<script setup lang="ts">
import { useAppState } from '../../composables/useAppState'

// Global state'ten uploadedImage'i alıyoruz
const { uploadedImage } = useAppState()
</script>

<template>
  <div class="w-full max-w-4xl mx-auto">
    <!-- Image Container -->
    <div class="relative rounded-2xl shadow-lg border-2 border-base-content overflow-hidden">
      <!-- Image Display Area -->
      <div class="aspect-square w-full bg-gradient-to-br from-gray-50 to-gray-100">
        <!-- Image Display -->
        <img :src="uploadedImage!" alt="Uploaded image" class="w-full h-full object-cover" />
      </div>
    </div>
  </div>
</template>
