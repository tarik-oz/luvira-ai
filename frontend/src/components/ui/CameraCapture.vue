<script setup lang="ts">
import { ref, onUnmounted } from 'vue'
import { defineExpose } from 'vue'
import { PhCamera, PhCheck, PhX, PhArrowClockwise, PhWarning } from '@phosphor-icons/vue'
import AppButton from './AppButton.vue'
import { useI18n } from 'vue-i18n'
import { useAppState } from '../../composables/useAppState'
import { useRouter } from 'vue-router'

const { t } = useI18n()
const { uploadFileAndSetSession } = useAppState()
const router = useRouter()

const showModal = ref(false)
const videoRef = ref<HTMLVideoElement | null>(null)
const stream = ref<MediaStream | null>(null)
let pendingStreamPromise: Promise<MediaStream> | null = null
const isLoading = ref(false)
const isStreamReady = ref(false)
const capturedImage = ref<string | null>(null)
const isUploading = ref(false)
const errorMessage = ref<string | null>(null)

const open = async () => {
  if (stream.value) {
    stream.value.getTracks().forEach((track: MediaStreamTrack) => {
      track.stop()
    })
    stream.value = null
  }

  // Reset states
  showModal.value = true
  isLoading.value = true
  isStreamReady.value = false
  capturedImage.value = null
  isUploading.value = false
  errorMessage.value = null

  try {
    pendingStreamPromise = navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    const newStream = await pendingStreamPromise
    stream.value = newStream
    if (videoRef.value) {
      videoRef.value.srcObject = newStream
      await videoRef.value.play()
      isStreamReady.value = true
    }
  } catch (err) {
    console.error('Camera error:', err)
    if (err instanceof DOMException) {
      if (err.name === 'NotFoundError') {
        errorMessage.value = t('camera.notFound')
      } else if (err.name === 'NotAllowedError') {
        errorMessage.value = t('camera.noPermission')
      } else {
        errorMessage.value = t('camera.genericError')
      }
    } else {
      // Fallback for other error types
      errorMessage.value = t('camera.genericError')
    }
    stream.value = null
    isStreamReady.value = false
  } finally {
    isLoading.value = false
    pendingStreamPromise = null
  }
}
const close = () => {
  showModal.value = false
  isLoading.value = false
  isStreamReady.value = false
  capturedImage.value = null
  isUploading.value = false

  if (videoRef.value) {
    videoRef.value.pause()
    videoRef.value.srcObject = null
    videoRef.value.load()
  }

  if (stream.value) {
    stream.value.getTracks().forEach((track: MediaStreamTrack) => {
      track.stop()
    })
    stream.value = null
  }
  // If getUserMedia is still pending, try to stop it
  if (pendingStreamPromise) {
    pendingStreamPromise
      .then((pendingStream) => {
        pendingStream.getTracks().forEach((track: MediaStreamTrack) => {
          track.stop()
        })
      })
      .catch(() => {})
    pendingStreamPromise = null
  }
}

const takePhoto = async () => {
  if (!videoRef.value || !isStreamReady.value) return

  try {
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const video = videoRef.value
    const videoWidth = video.videoWidth
    const videoHeight = video.videoHeight

    // --- START: object-cover CALCULATION ---
    const containerWidth = video.clientWidth
    const containerHeight = video.clientHeight

    const videoRatio = videoWidth / videoHeight
    const containerRatio = containerWidth / containerHeight

    let sourceX = 0,
      sourceY = 0,
      sourceWidth = videoWidth,
      sourceHeight = videoHeight

    // Simulate object-cover cropping behavior
    if (videoRatio > containerRatio) {
      // Video is wider, crop from sides
      sourceWidth = videoHeight * containerRatio
      sourceX = (videoWidth - sourceWidth) / 2
    } else {
      // Video is taller, crop from top/bottom
      sourceHeight = videoWidth / containerRatio
      sourceY = (videoHeight - sourceHeight) / 2
    }
    // --- END: object-cover CALCULATION ---

    // --- START: 40% AREA CALCULATION ---
    // Now we'll take 40% from the cropped (visible) area
    const finalCropWidth = Math.floor(sourceWidth * 0.4)
    const finalCropHeight = sourceHeight
    const finalCropX = sourceX + Math.floor((sourceWidth - finalCropWidth) / 2)
    const finalCropY = sourceY
    // --- END: 40% AREA CALCULATION ---

    // Set canvas size to final cropped dimensions
    canvas.width = finalCropWidth
    canvas.height = finalCropHeight

    // Draw only the calculated area
    ctx.drawImage(
      video,
      finalCropX,
      finalCropY,
      finalCropWidth,
      finalCropHeight,
      0,
      0,
      finalCropWidth,
      finalCropHeight,
    )

    capturedImage.value = canvas.toDataURL('image/jpeg', 0.8)

    // Stop camera stream after photo is taken
    if (stream.value) {
      stream.value.getTracks().forEach((track: MediaStreamTrack) => {
        track.stop()
      })
      stream.value = null
    }
    isStreamReady.value = false
  } catch (error) {
    console.error('Photo capture error:', error)
  }
}

const retakePhoto = async () => {
  capturedImage.value = null
  errorMessage.value = null

  // Restart camera
  isLoading.value = true
  try {
    pendingStreamPromise = navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    const newStream = await pendingStreamPromise
    stream.value = newStream
    if (videoRef.value) {
      videoRef.value.srcObject = newStream
      await videoRef.value.play()
      isStreamReady.value = true
    }
  } catch (err) {
    console.error('Camera error during retake:', err)
    if (err instanceof DOMException) {
      if (err.name === 'NotFoundError') {
        errorMessage.value = t('camera.notFound')
      } else if (err.name === 'NotAllowedError') {
        errorMessage.value = t('camera.noPermission')
      } else {
        errorMessage.value = t('camera.genericError')
      }
    } else {
      errorMessage.value = t('camera.genericError')
    }
    stream.value = null
    isStreamReady.value = false
  } finally {
    isLoading.value = false
    pendingStreamPromise = null
  }
}

const submitPhoto = async () => {
  if (!capturedImage.value || isUploading.value) return

  isUploading.value = true
  errorMessage.value = null

  try {
    // Convert base64 to File
    const response = await fetch(capturedImage.value)
    const blob = await response.blob()
    const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' })

    await uploadFileAndSetSession(file, capturedImage.value)

    // Close modal and navigate
    close()
    router.push('/color-tone-changer')
  } catch (error) {
    console.error('Camera upload failed:', error)
    errorMessage.value = t('sampleImages.errorMessage')
  } finally {
    isUploading.value = false
  }
}

onUnmounted(() => {
  if (videoRef.value) {
    videoRef.value.pause()
    videoRef.value.srcObject = null
    videoRef.value.load()
  }
  if (stream.value) {
    stream.value.getTracks().forEach((track: MediaStreamTrack) => {
      track.stop()
    })
    stream.value = null
  }
  if (pendingStreamPromise) {
    pendingStreamPromise
      .then((pendingStream) => {
        pendingStream.getTracks().forEach((track: MediaStreamTrack) => {
          track.stop()
        })
      })
      .catch(() => {})
    pendingStreamPromise = null
  }
})
defineExpose({ open })
</script>

<template>
  <!-- Full Screen Modal -->
  <div
    v-if="showModal"
    class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-xs"
    @click.self="close"
  >
    <div
      class="relative flex flex-col items-center w-full p-8 rounded-xl bg-base-content/80 shadow-2xl"
      :class="[capturedImage ? 'max-w-md' : 'max-w-2xl']"
    >
      <button
        @click="close"
        class="absolute top-4 right-4 btn btn-sm btn-circle border-none btn-ghost text-base-100 hover:bg-accent/80 shadow-none"
      >
        <PhX class="w-4 h-4" weight="bold" />
      </button>
      <div class="text-2xl font-bold text-center text-base-100 mb-6">
        {{ capturedImage ? t('camera.preview') : t('camera.title') }}
      </div>
      <!-- Camera Box with overlay -->
      <div
        class="relative flex items-center justify-center w-full mb-1 overflow-hidden rounded-lg transition-all duration-500 ease-in-out"
        :class="[capturedImage ? 'h-96' : 'h-72', capturedImage ? '' : 'bg-base-200']"
      >
        <!-- Video Stream -->
        <video
          v-show="!capturedImage"
          ref="videoRef"
          class="w-full h-full object-cover rounded-lg transition-all duration-500 ease-in-out"
          autoplay
          playsinline
        />

        <!-- Captured Image Preview (only cropped area, centered) -->
        <div
          v-if="capturedImage"
          class="flex items-center justify-center w-full h-full transition-all duration-500 ease-in-out"
        >
          <img
            :src="capturedImage"
            class="h-full object-contain rounded-lg transition-all duration-500 ease-in-out"
            alt="Captured photo"
          />
        </div>

        <!-- Loading Spinner -->
        <div v-if="isLoading" class="absolute inset-0 flex items-center justify-center z-10">
          <div
            class="animate-spin rounded-full h-12 w-12 border-t-4 border-b-4 border-accent"
          ></div>
        </div>

        <!-- Portrait overlay: only visible when video is showing -->
        <div
          v-show="!capturedImage"
          class="absolute top-0 left-1/2 hidden h-full md:flex flex-col items-center justify-center -translate-x-1/2"
          style="width: 40%; pointer-events: none"
        >
          <div class="h-full w-full border-4 border-accent"></div>
        </div>

        <!-- Darkening: left (only when video is showing) -->
        <div
          v-show="!capturedImage"
          class="absolute top-0 left-0 hidden h-full md:block"
          style="width: 30%; background: rgba(0, 0, 0, 0.5); pointer-events: none"
        ></div>

        <!-- Darkening: right (only when video is showing) -->
        <div
          v-show="!capturedImage"
          class="absolute top-0 right-0 hidden h-full md:block"
          style="width: 30%; background: rgba(0, 0, 0, 0.5); pointer-events: none"
        ></div>
      </div>
      <!-- Guidance text: only when video is showing -->
      <div v-show="!capturedImage" class="hidden md:flex items-center justify-center w-full mb-1">
        <span class="text-sm text-center italic drop-shadow-md text-base-100/70">
          {{ t('camera.guidance') }}
        </span>
      </div>

      <!-- Error Section -->
      <div v-if="errorMessage" class="text-center my-4 flex items-center justify-center gap-2">
        <PhWarning class="w-5 h-5 text-red-600 shrink-0" />
        <span class="text-red-600 text-sm font-medium">{{ errorMessage }}</span>
      </div>

      <!-- Buttons Row -->
      <div class="flex w-full gap-4 mt-2 md:mt-4">
        <!-- Only take photo button when video is showing -->
        <template v-if="!capturedImage">
          <AppButton
            class="w-full"
            type="button"
            :disabled="!isStreamReady || isUploading"
            @click="takePhoto"
          >
            <template #icon>
              <PhCamera class="w-5 h-5" />
            </template>
            {{ t('camera.takePhoto') }}
          </AppButton>
        </template>

        <!-- 2 buttons when photo is taken -->
        <template v-else>
          <div class="flex justify-center w-full gap-2">
            <AppButton class="px-4 py-2" type="button" :disabled="isUploading" @click="retakePhoto">
              <template #icon>
                <PhArrowClockwise class="w-4 h-4" />
              </template>
              {{ t('camera.retake') }}
            </AppButton>

            <AppButton class="px-4 py-2" type="button" :disabled="isUploading" @click="submitPhoto">
              <template #icon>
                <PhCheck class="w-4 h-4" />
              </template>
              {{ isUploading ? t('camera.uploading') : t('camera.submit') }}
            </AppButton>
          </div>
        </template>
      </div>
    </div>
  </div>
</template>
