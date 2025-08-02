<script setup lang="ts">
import { ref, onUnmounted } from 'vue'
import { defineExpose } from 'vue'
import { PhCamera, PhCheck, PhX } from '@phosphor-icons/vue'
import AppButton from './AppButton.vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

const showModal = ref(false)
const videoRef = ref<HTMLVideoElement | null>(null)
const stream = ref<MediaStream | null>(null)
let pendingStreamPromise: Promise<MediaStream> | null = null
const isLoading = ref(false)

const open = async () => {
  if (stream.value) {
    stream.value.getTracks().forEach((track: MediaStreamTrack) => {
      track.stop()
    })
    stream.value = null
  }
  showModal.value = true
  isLoading.value = true
  try {
    pendingStreamPromise = navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    const newStream = await pendingStreamPromise
    stream.value = newStream
    if (videoRef.value) {
      videoRef.value.srcObject = newStream
      await videoRef.value.play()
    }
    isLoading.value = false
  } catch (err) {
    console.error('Camera error:', err)
    stream.value = null
    isLoading.value = false
  } finally {
    pendingStreamPromise = null
  }
}
const close = () => {
  showModal.value = false
  isLoading.value = false

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
      class="relative flex flex-col items-center w-full max-w-2xl p-8 rounded-xl bg-base-content/80 shadow-2xl"
    >
      <button
        @click="close"
        class="absolute top-4 right-4 btn btn-sm btn-circle border-none btn-ghost text-base-100 hover:bg-accent/80 shadow-none"
      >
        <PhX class="w-4 h-4" weight="bold" />
      </button>
      <div class="text-2xl font-bold text-center text-base-100 mb-6">{{ t('camera.title') }}</div>
      <!-- Camera Box with overlay -->
      <div
        class="relative flex items-center justify-center w-full h-72 mb-1 overflow-hidden rounded-lg bg-base-200"
      >
        <video ref="videoRef" class="w-full h-full object-cover rounded-lg" autoplay playsinline />

        <!-- Loading Spinner -->
        <div v-if="isLoading" class="absolute inset-0 flex items-center justify-center z-10">
          <div
            class="animate-spin rounded-full h-12 w-12 border-t-4 border-b-4 border-accent"
          ></div>
        </div>
        <!-- Portrait overlay: only visible on md and up -->
        <div
          class="absolute top-0 left-1/2 hidden h-full md:flex flex-col items-center justify-center -translate-x-1/2"
          style="width: 40%; pointer-events: none"
        >
          <div class="h-full w-full border-4 border-accent"></div>
        </div>
        <!-- Darkening: left -->
        <div
          class="absolute top-0 left-0 hidden h-full md:block"
          style="width: 30%; background: rgba(0, 0, 0, 0.5); pointer-events: none"
        ></div>
        <!-- Darkening: right -->
        <div
          class="absolute top-0 right-0 hidden h-full md:block"
          style="width: 30%; background: rgba(0, 0, 0, 0.5); pointer-events: none"
        ></div>
      </div>
      <!-- Guidance text: only when portrait overlay is visible (md and up) -->
      <div class="hidden md:flex items-center justify-center w-full mb-1">
        <span class="text-sm text-center italic drop-shadow-md text-base-100/70">
          {{ t('camera.guidance') }}
        </span>
      </div>
      <!-- Buttons Row -->
      <div class="flex w-full gap-4 mt-2 md:mt-4">
        <AppButton class="flex-1" type="button">
          <template #icon>
            <PhCamera class="w-5 h-5" />
          </template>
          {{ t('camera.takePhoto') }}
        </AppButton>
        <AppButton class="flex-1" type="button">
          <template #icon>
            <PhCheck class="w-5 h-5" />
          </template>
          {{ t('camera.submit') }}
        </AppButton>
      </div>
    </div>
  </div>
</template>
