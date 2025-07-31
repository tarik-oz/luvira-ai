<script setup lang="ts">
import { ref } from 'vue'
import { sampleImages, type StaticSampleImage } from '../../data/sampleImages'

const emit = defineEmits<{
  selectImage: [image: StaticSampleImage]
}>()

// Popup state
const showPopup = ref(false)

const togglePopup = () => {
  showPopup.value = !showPopup.value
}

const closePopup = () => {
  showPopup.value = false
}

const open = () => {
  showPopup.value = true
}

import { defineExpose } from 'vue'
defineExpose({ open })

const selectSampleImage = async (image: StaticSampleImage) => {
  try {
    console.log('Selected sample image:', image.name)

    // Fetch the image from the imported URL and convert to File
    const response = await fetch(image.url)
    const blob = await response.blob()

    const file = new File([blob], image.filename, {
      type: blob.type || 'image/jpeg',
    })

    console.log('Created file from sample image:', file.name, file.size)

    // Emit the image data for parent component
    emit('selectImage', image)

    // Close popup after selection
    closePopup()
  } catch (error) {
    console.error('Error loading sample image:', error)
  }
}
</script>

<template>
  <div class="sample-images-section">
    <!-- Sample images button and popup -->
    <div class="relative">
      <!-- Button to open popup -->
      <button @click="togglePopup" class="btn btn-primary btn-outline gap-2">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          class="h-5 w-5"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
          />
        </svg>
        Choose from sample images
      </button>

      <!-- Popup with sample images -->
      <div
        v-if="showPopup"
        class="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 bg-white rounded-lg shadow-xl border border-gray-200 p-3 z-50 w-[600px]"
      >
        <!-- Popup header -->
        <div class="flex justify-end mb-2">
          <button @click="togglePopup" class="text-gray-300 hover:text-gray-500 transition-colors">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              class="h-5 w-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        <!-- Images grid -->
        <div class="flex gap-2 justify-center">
          <div
            v-for="image in sampleImages"
            :key="image.id"
            class="cursor-pointer border-2 border-gray-200 rounded-lg hover:border-primary transition-all duration-200 flex-shrink-0"
            @click="selectSampleImage(image)"
          >
            <img :src="image.url" :alt="image.name" class="w-20 h-24 object-cover rounded-lg" />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.sample-image-card {
  transition: all 0.2s ease-in-out;
}

.sample-image-card:hover {
  transform: translateY(-2px);
}
</style>
