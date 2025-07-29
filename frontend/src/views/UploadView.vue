<script setup lang="ts">
import UploadSection from '../components/ui/UploadSection.vue'
import SampleImages from '../components/ui/SampleImages.vue'
import AnimatedDemo from '../components/ui/AnimatedDemo.vue'
import CameraCapture from '../components/ui/CameraCapture.vue'
import { useAppState } from '../composables/useAppState'
import { apiService } from '../services/apiService'
import type { StaticSampleImage } from '../data/sampleImages'

const { isUploading, setIsUploading, setSessionId, setCurrentView, createImageUrl, resetState } =
  useAppState()

const handleFileSelect = async (file: File) => {
  console.log('Selected file:', file.name)

  // Store image in browser memory for display
  createImageUrl(file)

  // Upload to API
  await uploadImageToAPI(file)
}

const handleSampleImageSelect = async (sampleImage: StaticSampleImage) => {
  try {
    console.log('Selected sample image:', sampleImage.name)

    // Fetch the image from public folder and convert to File
    const response = await fetch(sampleImage.url)
    const blob = await response.blob()

    const file = new File([blob], sampleImage.filename, {
      type: blob.type || 'image/jpeg',
    })

    console.log('Created file from sample image:', file.name, file.size)

    // Handle the file the same way as uploaded files
    await handleFileSelect(file)
  } catch (error) {
    console.error('Error loading sample image:', error)
    alert('Error loading sample image. Please try again.')
  }
}

const uploadImageToAPI = async (file: File) => {
  setIsUploading(true)

  try {
    const response = await apiService.uploadAndPrepare(file)
    setSessionId(response.session_id)

    console.log('Upload successful. Session ID:', response.session_id)

    // Switch to processing view
    setCurrentView('processing')
  } catch (error) {
    console.error('Upload failed:', error)
    alert(`Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`)

    // Cleanup on error
    resetState()
  } finally {
    setIsUploading(false)
  }
}
</script>

<template>
  <div class="space-y-6">
    <div class="text-center mb-6">
      <h1 class="text-3xl font-bold text-gray-800 mb-2">Try on different hair colors</h1>
      <p class="text-base text-gray-600">Upload your photo and experiment with new hair colors</p>
    </div>

    <!-- Animated Demo Section -->
    <AnimatedDemo />

    <!-- Upload Component -->
    <UploadSection @file-select="handleFileSelect" />

    <!-- Loading State -->
    <div v-if="isUploading" class="text-center py-8">
      <div class="inline-flex items-center">
        <svg
          class="animate-spin -ml-1 mr-3 h-5 w-5 text-blue-600"
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
        >
          <circle
            class="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            stroke-width="4"
          ></circle>
          <path
            class="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          ></path>
        </svg>
        <span class="text-blue-600 font-medium">
          Processing image and generating hair mask...
        </span>
      </div>
    </div>

    <!-- Simple divider -->
    <div v-if="!isUploading" class="text-center my-4">
      <span class="text-gray-400 text-sm">or</span>
    </div>

    <!-- Sample images section -->
    <div v-if="!isUploading" class="text-center">
      <SampleImages @select-image="handleSampleImageSelect" />
    </div>

    <!-- Camera capture section -->
    <div v-if="!isUploading" class="text-center">
      <CameraCapture @capture="handleFileSelect" />
    </div>
  </div>
</template>
