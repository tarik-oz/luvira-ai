<script setup lang="ts">
import { ref, onUnmounted } from 'vue'
import { defineExpose } from 'vue'
import { PhCamera, PhCheck, PhX, PhArrowClockwise, PhWarning } from '@phosphor-icons/vue'
import AppButton from './AppButton.vue'
import { useI18n } from 'vue-i18n'
import { useAppState } from '../../composables/useAppState'
import { useRouter } from 'vue-router'

const { t } = useI18n()
const { isUploading } = useAppState()
const router = useRouter()

// Import hairService for upload
import hairService from '../../services/hairService'

const showModal = ref(false)
const videoRef = ref<HTMLVideoElement | null>(null)
const stream = ref<MediaStream | null>(null)
let pendingStreamPromise: Promise<MediaStream> | null = null
const isLoading = ref(false)
const isStreamReady = ref(false)
const capturedImage = ref<string | null>(null)
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
  errorMessage.value = null

  try {
    if (!isSecureContext) {
      throw new DOMException('HTTPS is required for camera access', 'SecurityError')
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new DOMException('Camera not supported', 'NotSupportedError')
    }
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
      } else if (err.name === 'NotReadableError') {
        errorMessage.value = t('camera.notReadable')
      } else if (err.name === 'OverconstrainedError') {
        errorMessage.value = t('camera.overconstrained')
      } else if (err.name === 'SecurityError') {
        errorMessage.value = isSecureContext ? t('camera.genericError') : t('camera.httpsRequired')
      } else if (err.name === 'NotSupportedError') {
        errorMessage.value = t('camera.notSupported')
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

  errorMessage.value = null

  try {
    // Convert base64 to File
    const response = await fetch(capturedImage.value)
    const blob = await response.blob()
    const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' })

    await hairService.uploadImage(file, capturedImage.value)

    // Close modal and navigate
    close()
    router.push('/color-tone-changer')
  } catch (error) {
    console.error('Camera upload failed:', error)
    const message = error instanceof Error ? error.message : String(error)
    if (!navigator.onLine || message === 'Failed to fetch' || /network/i.test(message)) {
      errorMessage.value = t('processing.networkError')
    } else if (message === 'TIMEOUT') {
      errorMessage.value = t('camera.uploading')
    } else {
      errorMessage.value = t('sampleImages.errorMessage')
    }
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
      class="bg-base-content/80 relative flex w-full flex-col items-center rounded-xl p-8 shadow-2xl"
      :class="[capturedImage ? 'max-w-md' : 'max-w-2xl']"
    >
      <button
        @click="close"
        class="btn btn-sm btn-circle btn-ghost text-base-100 hover:bg-accent/80 absolute top-4 right-4 border-none shadow-none"
      >
        <PhX class="h-4 w-4" weight="bold" />
      </button>
      <div class="text-base-100 mb-6 text-center text-2xl font-bold">
        {{ capturedImage ? t('camera.preview') : t('camera.title') }}
      </div>
      <!-- Camera Box with overlay -->
      <div
        class="relative mb-1 flex w-full items-center justify-center overflow-hidden rounded-lg transition-all duration-500 ease-in-out"
        :class="[capturedImage ? 'h-96' : 'h-72', capturedImage ? '' : 'bg-base-200']"
      >
        <!-- Video Stream -->
        <video
          v-show="!capturedImage"
          ref="videoRef"
          class="h-full w-full rounded-lg object-cover transition-all duration-500 ease-in-out"
          autoplay
          playsinline
        />

        <!-- Captured Image Preview (only cropped area, centered) -->
        <div
          v-if="capturedImage"
          class="flex h-full w-full items-center justify-center transition-all duration-500 ease-in-out"
        >
          <img
            :src="capturedImage"
            class="h-full rounded-lg object-contain transition-all duration-500 ease-in-out"
            alt="Captured photo"
          />
        </div>

        <!-- Loading Spinner -->
        <div v-if="isLoading" class="absolute inset-0 z-10 flex items-center justify-center">
          <div
            class="border-accent h-12 w-12 animate-spin rounded-full border-t-4 border-b-4"
          ></div>
        </div>

        <!-- Portrait overlay: only visible when video is showing -->
        <div
          v-show="!capturedImage"
          class="absolute top-0 left-1/2 hidden h-full -translate-x-1/2 flex-col items-center justify-center md:flex"
          style="width: 40%; pointer-events: none"
        >
          <div class="border-accent h-full w-full border-4"></div>
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
      <div v-show="!capturedImage" class="mb-1 hidden w-full items-center justify-center md:flex">
        <span class="text-base-100/70 text-center text-sm italic drop-shadow-md">
          {{ t('camera.guidance') }}
        </span>
      </div>

      <!-- Error Section -->
      <div v-if="errorMessage" class="my-4 flex items-center justify-center gap-2 text-center">
        <PhWarning class="h-5 w-5 shrink-0 text-red-600" />
        <span class="text-sm font-medium text-red-600">{{ errorMessage }}</span>
      </div>

      <!-- Buttons Row -->
      <div class="mt-2 flex w-full gap-4 md:mt-4">
        <!-- Only take photo button when video is showing -->
        <template v-if="!capturedImage">
          <AppButton
            class="w-full"
            type="button"
            :disabled="!isStreamReady || isUploading"
            @click="takePhoto"
          >
            <template #icon>
              <PhCamera class="h-5 w-5" />
            </template>
            {{ t('camera.takePhoto') }}
          </AppButton>
        </template>

        <!-- 2 buttons when photo is taken -->
        <template v-else>
          <div class="flex w-full justify-center gap-2">
            <AppButton class="px-4 py-2" type="button" :disabled="isUploading" @click="retakePhoto">
              <template #icon>
                <PhArrowClockwise class="h-4 w-4" />
              </template>
              {{ t('camera.retake') }}
            </AppButton>

            <AppButton class="px-4 py-2" type="button" :disabled="isUploading" @click="submitPhoto">
              <template #icon>
                <PhCheck class="h-4 w-4" />
              </template>
              {{ isUploading ? t('camera.uploading') : t('camera.submit') }}
            </AppButton>
          </div>
        </template>
      </div>
    </div>
  </div>
</template>
