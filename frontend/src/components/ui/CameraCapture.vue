<script setup lang="ts">
import { ref, onUnmounted } from 'vue'
import { defineExpose } from 'vue'
import {
  PhCamera,
  PhCheck,
  PhX,
  PhArrowClockwise,
  PhWarning,
  PhCameraRotate,
} from '@phosphor-icons/vue'
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

// Camera preferences and orientation handling
const preferredFacing = ref<'user' | 'environment'>('user')
const isPortraitVideo = ref(false)
const useCroppingOverlay = ref(false)
const hasMultipleCameras = ref(false)
const MAX_CAPTURE_DIMENSION = 1600

const updateOrientationFromVideo = () => {
  if (!videoRef.value) return
  const vw = videoRef.value.videoWidth
  const vh = videoRef.value.videoHeight
  if (!vw || !vh) return
  isPortraitVideo.value = vh >= vw
  // Show overlay only for landscape videos
  useCroppingOverlay.value = !isPortraitVideo.value
}

const updateHasMultipleCameras = async () => {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    hasMultipleCameras.value = false
    return
  }
  try {
    const devices = await navigator.mediaDevices.enumerateDevices()
    const videoInputs = devices.filter((d) => d.kind === 'videoinput')
    hasMultipleCameras.value = videoInputs.length >= 2
  } catch {
    hasMultipleCameras.value = false
  }
}

// Prefer front (user-facing) camera when available, with safe fallbacks
const getPreferredCameraStream = async (
  prefer: 'user' | 'environment' = 'user',
): Promise<MediaStream> => {
  const attempts: MediaStreamConstraints[] = [
    { video: { facingMode: { exact: prefer } }, audio: false },
    { video: { facingMode: { ideal: prefer } }, audio: false },
    // Some browsers accept string form as well
    { video: { facingMode: prefer }, audio: false },
    // Try the opposite as a fallback (in case front cam is unavailable)
    { video: { facingMode: { ideal: prefer === 'user' ? 'environment' : 'user' } }, audio: false },
    // Final fallback: any camera
    { video: true, audio: false },
  ]

  let lastError: unknown = null
  for (const constraints of attempts) {
    try {
      return await navigator.mediaDevices.getUserMedia(constraints)
    } catch (err) {
      lastError = err
    }
  }
  throw lastError instanceof Error ? lastError : new Error('Camera not available')
}

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
    // Always start with front camera on open
    preferredFacing.value = 'user'
    pendingStreamPromise = getPreferredCameraStream(preferredFacing.value)
    const newStream = await pendingStreamPromise
    stream.value = newStream
    if (videoRef.value) {
      videoRef.value.srcObject = newStream
      await videoRef.value.play()
      isStreamReady.value = true
      updateOrientationFromVideo()
      // Update orientation once metadata is loaded too
      videoRef.value.onloadedmetadata = () => updateOrientationFromVideo()
      await updateHasMultipleCameras()
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

    // --- START: FINAL CROP DECISION ---
    // If video is portrait, send exactly what user sees (full visible area)
    // If landscape, keep center 40% width crop
    const portrait = videoHeight >= videoWidth
    const finalCropWidth = portrait ? Math.floor(sourceWidth) : Math.floor(sourceWidth * 0.4)
    const finalCropHeight = Math.floor(sourceHeight)
    const finalCropX = portrait
      ? Math.floor(sourceX)
      : Math.floor(sourceX + (sourceWidth - finalCropWidth) / 2)
    const finalCropY = Math.floor(sourceY)
    // --- END: FINAL CROP DECISION ---

    // Clamp output to MAX_CAPTURE_DIMENSION while preserving AR
    const maxDim = MAX_CAPTURE_DIMENSION
    const scale = Math.min(1, maxDim / Math.max(finalCropWidth, finalCropHeight))
    const outputWidth = Math.floor(finalCropWidth * scale)
    const outputHeight = Math.floor(finalCropHeight * scale)

    // Set canvas size to output dimensions
    canvas.width = outputWidth
    canvas.height = outputHeight

    // Draw only the calculated area
    ctx.drawImage(
      video,
      finalCropX,
      finalCropY,
      finalCropWidth,
      finalCropHeight,
      0,
      0,
      outputWidth,
      outputHeight,
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
    pendingStreamPromise = getPreferredCameraStream(preferredFacing.value)
    const newStream = await pendingStreamPromise
    stream.value = newStream
    if (videoRef.value) {
      videoRef.value.srcObject = newStream
      await videoRef.value.play()
      isStreamReady.value = true
      updateOrientationFromVideo()
      videoRef.value.onloadedmetadata = () => updateOrientationFromVideo()
      await updateHasMultipleCameras()
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
    router.push({ name: 'HairColorEditor' })
  } catch (error) {
    console.error('Camera upload failed:', error)
    const message = error instanceof Error ? error.message : String(error)
    const errorWithCode = error as Error & { error_code?: string }
    const errorCode =
      error && typeof error === 'object' && 'error_code' in errorWithCode
        ? errorWithCode.error_code
        : undefined
    if (!navigator.onLine || message === 'Failed to fetch' || /network/i.test(message)) {
      errorMessage.value = t('processing.networkError')
    } else if (message === 'TIMEOUT') {
      errorMessage.value = t('camera.uploading')
    } else if (errorCode === 'NO_HAIR_DETECTED') {
      errorMessage.value = t('uploadSection.noHairDetected')
    } else {
      errorMessage.value = t('sampleImages.errorMessage')
    }
  }
}

const switchCamera = async () => {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return
  if (isLoading.value) return
  const previous = preferredFacing.value
  preferredFacing.value = previous === 'user' ? 'environment' : 'user'
  isLoading.value = true
  errorMessage.value = null
  try {
    const newStream = await getPreferredCameraStream(preferredFacing.value)
    // Stop old stream
    if (stream.value) {
      stream.value.getTracks().forEach((track: MediaStreamTrack) => track.stop())
    }
    stream.value = newStream
    if (videoRef.value) {
      videoRef.value.srcObject = newStream
      await videoRef.value.play()
      isStreamReady.value = true
      updateOrientationFromVideo()
      videoRef.value.onloadedmetadata = () => updateOrientationFromVideo()
      await updateHasMultipleCameras()
    }
  } catch (err) {
    console.error('Camera switch error:', err)
    // Revert preference on failure
    preferredFacing.value = previous
    if (err instanceof DOMException) {
      errorMessage.value = t('camera.genericError')
    }
  } finally {
    isLoading.value = false
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
    class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-[clamp(12px,5vw,24px)] backdrop-blur-xs"
    @click.self="close"
  >
    <div
      class="bg-base-content/80 relative flex w-full flex-col items-center rounded-xl p-[clamp(12px,3vw,24px)] shadow-2xl"
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
        :class="[isPortraitVideo ? 'h-96' : 'h-72', capturedImage ? '' : 'bg-base-200']"
      >
        <!-- Camera switch button (top-right, round) -->
        <button
          v-show="!capturedImage && hasMultipleCameras"
          @click="switchCamera"
          :disabled="isLoading || isUploading"
          class="btn btn-circle btn-ghost bg-base-100/90 text-base-content hover:bg-accent/80 absolute top-3 right-3 z-20 border-none shadow-md"
          title="Switch camera"
        >
          <PhCameraRotate class="h-5 w-5" />
        </button>
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
            class="h-full rounded-lg object-cover transition-all duration-500 ease-in-out"
            alt="Captured photo"
          />
        </div>

        <!-- Loading Spinner -->
        <div v-if="isLoading" class="absolute inset-0 z-10 flex items-center justify-center">
          <div
            class="border-accent h-12 w-12 animate-spin rounded-full border-t-4 border-b-4"
          ></div>
        </div>

        <!-- Landscape overlay: only when streaming and landscape -->
        <div
          v-show="!capturedImage && useCroppingOverlay"
          class="absolute top-0 left-1/2 hidden h-full -translate-x-1/2 flex-col items-center justify-center md:flex"
          style="width: 40%; pointer-events: none"
        >
          <div class="border-accent h-full w-full border-4"></div>
        </div>

        <!-- Darkening: left (only when streaming and landscape) -->
        <div
          v-show="!capturedImage && useCroppingOverlay"
          class="absolute top-0 left-0 hidden h-full md:block"
          style="width: 30%; background: rgba(0, 0, 0, 0.5); pointer-events: none"
        ></div>

        <!-- Darkening: right (only when streaming and landscape) -->
        <div
          v-show="!capturedImage && useCroppingOverlay"
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
        <span
          class="text-sm font-medium text-red-600 md:max-w-[420px] md:truncate md:whitespace-nowrap"
          >{{ errorMessage }}</span
        >
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
