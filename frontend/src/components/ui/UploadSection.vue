<script setup lang="ts">
import { ref } from 'vue'

// Define emits
const emit = defineEmits<{
  'file-select': [file: File]
}>()

const fileInput = ref<HTMLInputElement>()

const openFileDialog = () => {
  fileInput.value?.click()
}

const handleFileSelect = (event: Event) => {
  const target = event.target as HTMLInputElement
  if (target.files && target.files[0]) {
    validateAndUpload(target.files[0])
  }
}

const handleDrop = (event: DragEvent) => {
  event.preventDefault()
  if (event.dataTransfer?.files && event.dataTransfer.files[0]) {
    validateAndUpload(event.dataTransfer.files[0])
  }
}

const handleDragOver = (event: DragEvent) => {
  event.preventDefault()
}

const handleDragLeave = (event: DragEvent) => {
  event.preventDefault()
}

const validateAndUpload = (file: File) => {
  // File validation
  const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png']
  const maxSize = 10 * 1024 * 1024 // 10MB

  if (!allowedTypes.includes(file.type)) {
    alert('Please upload a JPG, JPEG, or PNG file.')
    return
  }

  if (file.size > maxSize) {
    alert('File size must be less than 10MB.')
    return
  }

  emit('file-select', file)
}
</script>

<template>
  <div class="w-full max-w-2xl mx-auto">
    <!-- Upload Box -->
    <div
      class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-gray-400 transition-colors cursor-pointer"
      @click="openFileDialog"
      @drop="handleDrop"
      @dragover="handleDragOver"
      @dragleave="handleDragLeave"
    >
      <!-- Upload Icon -->
      <div class="mb-4">
        <svg
          class="w-12 h-12 text-gray-400 mx-auto"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
          />
        </svg>
      </div>

      <!-- Upload Text -->
      <div class="space-y-2">
        <h3 class="text-lg font-medium text-gray-700">Click to upload or drag and drop</h3>
        <p class="text-sm text-gray-500">JPG, JPEG, PNG up to 10MB</p>
      </div>

      <!-- Tips -->
      <div class="mt-4">
        <p class="text-xs text-gray-400">
          💡 Tips: Use clear photos with visible hair, avoid objects in front of hair, and ensure
          good lighting
        </p>
      </div>

      <!-- Hidden File Input -->
      <input
        ref="fileInput"
        type="file"
        accept=".jpg,.jpeg,.png"
        class="hidden"
        @change="handleFileSelect"
      />
    </div>
  </div>
</template>
