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
  <div class="w-full min-h-[70vh] flex flex-col justify-center items-center">
    <div class="w-full max-w-7xl grid grid-cols-1 md:grid-cols-2 gap-10 items-center">
      <!-- Left: Title, Description, Upload -->
      <div class="flex flex-col justify-center items-start px-4 md:px-0">
        <h1 class="text-5xl md:text-6xl font-extrabold text-base-content mb-6 leading-tight">
          Free Online Hair Color & Tone Changer
        </h1>
        <p class="text-lg md:text-xl text-base-content/70 mb-8 max-w-xl">
          Easily try a wide selection of hair colors and natural tones. Instantly preview different
          looks and find the style that suits you best. Fast, realistic, and free—give it a try now!
        </p>
        <div class="w-full max-w-md">
          <UploadSection @file-select="handleFileSelect" />
        </div>
        <div v-if="isUploading" class="w-full text-center py-8">
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
        <div v-if="!isUploading" class="w-full flex flex-col items-center mt-4">
          <!-- Divider with 'or' -->
          <div class="flex items-center w-full my-4">
            <div class="flex-grow border-t border-base-content/20"></div>
            <span class="mx-4 text-base-content/40 text-sm font-medium select-none">or</span>
            <div class="flex-grow border-t border-base-content/20"></div>
          </div>
          <!-- Two options side by side -->
          <div class="flex flex-row gap-4 w-full justify-center">
            <button
              class="btn btn-outline btn-sm md:btn-md rounded-lg"
              @click="$refs.sampleImages && $refs.sampleImages.open && $refs.sampleImages.open()"
            >
              Choose from sample images
            </button>
            <button
              class="btn btn-outline btn-sm md:btn-md rounded-lg"
              @click="$refs.cameraCapture && $refs.cameraCapture.open && $refs.cameraCapture.open()"
            >
              Take a photo
            </button>
          </div>
          <!-- Hidden components for programmatic open -->
          <div class="hidden">
            <SampleImages ref="sampleImages" @select-image="handleSampleImageSelect" />
            <CameraCapture ref="cameraCapture" @capture="handleFileSelect" />
          </div>
        </div>
      </div>
      <!-- Right: Animated Demo -->
      <div class="flex justify-center items-center w-full">
        <AnimatedDemo />
      </div>
    </div>
  </div>
</template>
