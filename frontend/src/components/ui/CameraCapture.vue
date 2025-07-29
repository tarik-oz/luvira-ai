<script setup lang="ts">
import { ref, nextTick } from 'vue'

const emit = defineEmits(['capture'])

const videoRef = ref<HTMLVideoElement | null>(null)
const stream = ref<MediaStream | null>(null)
const isCameraOpen = ref(false)
const isCapturing = ref(false)
const error = ref<string | null>(null)

const openCamera = async () => {
  error.value = null
  try {
    stream.value = await navigator.mediaDevices.getUserMedia({ video: true })
    isCameraOpen.value = true
    await nextTick()
    if (videoRef.value && stream.value) {
      videoRef.value.srcObject = stream.value
      videoRef.value.play()
    }
  } catch (e: any) {
    error.value = 'Unable to access camera.'
  }
}

const closeCamera = () => {
  if (stream.value) {
    stream.value.getTracks().forEach((track) => track.stop())
    stream.value = null
  }
  isCameraOpen.value = false
}

const capturePhoto = () => {
  if (!videoRef.value) return
  isCapturing.value = true
  const canvas = document.createElement('canvas')
  canvas.width = videoRef.value.videoWidth
  canvas.height = videoRef.value.videoHeight
  const ctx = canvas.getContext('2d')
  if (ctx) {
    ctx.drawImage(videoRef.value, 0, 0, canvas.width, canvas.height)
    canvas.toBlob(
      (blob) => {
        if (blob) {
          const file = new File([blob], 'captured_photo.jpg', { type: 'image/jpeg' })
          emit('capture', file)
        }
        isCapturing.value = false
        closeCamera()
      },
      'image/jpeg',
      0.95,
    )
  }
}
</script>

<template>
  <div>
    <button
      v-if="!isCameraOpen"
      @click="openCamera"
      class="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 mb-2"
    >
      Take a Photo
    </button>

    <div v-if="isCameraOpen" class="flex flex-col items-center space-y-2">
      <video
        ref="videoRef"
        autoplay
        playsinline
        class="rounded-lg border w-full max-w-xs aspect-video bg-black"
      ></video>
      <div class="flex space-x-2">
        <button
          @click="capturePhoto"
          :disabled="isCapturing"
          class="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-green-600 rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2"
        >
          Capture
        </button>
        <button
          @click="closeCamera"
          :disabled="isCapturing"
          class="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-700 bg-gray-200 rounded-md hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2"
        >
          Cancel
        </button>
      </div>
      <div v-if="error" class="text-red-500 text-sm mt-2">{{ error }}</div>
    </div>
  </div>
</template>
