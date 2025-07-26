<script setup lang="ts">
import OriginalImageDisplay from '../components/ui/OriginalImageDisplay.vue'
import ColorOptionsPanel from '../components/ui/ColorOptionsPanel.vue'
import { useAppState } from '../composables/useAppState'

const { uploadedImage, sessionId, resetState, setCurrentView } = useAppState()

const handleBackToUpload = () => {
  resetState()
  setCurrentView('upload')
  console.log('Returned to upload view')
}
</script>

<template>
  <div class="w-full max-w-4xl mx-auto">
    <!-- Back Button -->
    <div class="mb-6">
      <button
        @click="handleBackToUpload"
        class="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
      >
        <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M15 19l-7-7 7-7"
          />
        </svg>
        Upload New Image
      </button>
    </div>

    <!-- Page Title -->
    <div class="text-center mb-8">
      <h1 class="text-3xl font-bold text-gray-800 mb-2">Transform Your Hair Color</h1>
      <p class="text-base text-gray-600">
        Your image has been processed successfully. Now you can experiment with different hair
        colors.
      </p>
    </div>

    <!-- Main Content Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
      <!-- Original Image Display -->
      <OriginalImageDisplay :uploaded-image="uploadedImage" :session-id="sessionId" />

      <!-- Color Options Panel -->
      <ColorOptionsPanel />
    </div>
  </div>
</template>
