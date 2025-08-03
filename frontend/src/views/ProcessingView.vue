<script setup lang="ts">
import { useAppState } from '../composables/useAppState'
import { useRouter } from 'vue-router'
import { onMounted } from 'vue'

const { uploadedImage, sessionId } = useAppState()
const router = useRouter()

// Check if user has valid session on page load
onMounted(() => {
  if (!sessionId.value || !uploadedImage.value) {
    // No session or image, redirect to upload page
    router.push('/')
  }
})
</script>

<template>
  <div class="min-h-screen bg-base-300 p-8">
    <div class="max-w-4xl mx-auto">
      <h1 class="text-3xl font-bold text-center mb-8">Hair Color Processing</h1>

      <!-- Debug Info -->
      <div class="bg-base-100 p-4 rounded-lg mb-6">
        <h2 class="text-xl font-semibold mb-4">Debug Info:</h2>
        <p><strong>Session ID:</strong> {{ sessionId || 'No session' }}</p>
        <p><strong>Image URL:</strong> {{ uploadedImage || 'No image' }}</p>
      </div>

      <!-- Original Image Display -->
      <div v-if="uploadedImage" class="bg-base-100 p-6 rounded-lg">
        <h2 class="text-xl font-semibold mb-4">Original Image (Client Cache):</h2>
        <div class="flex justify-center">
          <img
            :src="uploadedImage"
            alt="Original uploaded image"
            class="max-w-md max-h-96 object-contain rounded-lg shadow-lg"
          />
        </div>
      </div>

      <!-- No Image State -->
      <div v-else class="bg-base-100 p-6 rounded-lg text-center">
        <h2 class="text-xl font-semibold mb-4">No Image Found</h2>
        <p class="text-base-content/60">Please upload an image first</p>
      </div>
    </div>
  </div>
</template>
