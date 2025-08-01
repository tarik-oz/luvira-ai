<script setup lang="ts">
import { ref, onUnmounted } from 'vue'
import { defineExpose } from 'vue'
import { PhCamera, PhCheck, PhX } from '@phosphor-icons/vue'
import AppButton from './AppButton.vue'

const showModal = ref(false)
const videoRef = ref<HTMLVideoElement | null>(null)
const stream = ref<MediaStream | null>(null)

const open = async () => {
  if (stream.value) {
    stream.value.getTracks().forEach((track: MediaStreamTrack) => {
      track.stop()
    })
    stream.value = null
  }

  showModal.value = true

  try {
    const newStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    stream.value = newStream

    if (videoRef.value) {
      videoRef.value.srcObject = newStream
      await videoRef.value.play()
    }
  } catch (err) {
    console.error('Camera error:', err)
    stream.value = null
  }
}
const close = () => {
  showModal.value = false

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
      class="bg-base-content/80 rounded-xl shadow-2xl p-8 w-full max-w-2xl relative flex flex-col items-center"
    >
      <button
        @click="close"
        class="absolute top-4 right-4 btn btn-sm btn-circle border-none btn-ghost text-base-100 hover:bg-accent/80 shadow-none"
      >
        <PhX class="w-4 h-4" weight="bold" />
      </button>
      <div class="text-2xl font-bold mb-6 text-center text-base-100">Take a Photo</div>
      <!-- Camera Box with overlay -->
      <div
        class="w-full h-72 bg-base-200 rounded-lg flex items-center justify-center mb-1 overflow-hidden relative"
      >
        <video ref="videoRef" class="w-full h-full object-cover rounded-lg" autoplay playsinline />
        <!-- Portrait overlay: only visible on md and up -->
        <div
          class="hidden md:flex flex-col items-center justify-center absolute top-0 left-1/2 -translate-x-1/2 h-full"
          style="width: 40%; pointer-events: none"
        >
          <div class="h-full w-full border-4 border-accent"></div>
        </div>
        <!-- Darkening: left -->
        <div
          class="hidden md:block absolute top-0 left-0 h-full"
          style="width: 30%; background: rgba(0, 0, 0, 0.5); pointer-events: none"
        ></div>
        <!-- Darkening: right -->
        <div
          class="hidden md:block absolute top-0 right-0 h-full"
          style="width: 30%; background: rgba(0, 0, 0, 0.5); pointer-events: none"
        ></div>
      </div>
      <!-- Guidance text: only when portrait overlay is visible (md and up) -->
      <div class="hidden md:flex w-full mb-1 items-center justify-center">
        <span class="italic text-base-100/70 text-sm text-center drop-shadow-md">
          Please align your face within the center area.
        </span>
      </div>
      <!-- Buttons Row -->
      <div class="flex w-full gap-4 mt-2 md:mt-4">
        <AppButton class="flex-1" type="button">
          <template #icon>
            <PhCamera class="w-5 h-5" />
          </template>
          Take Photo
        </AppButton>
        <AppButton class="flex-1" type="button">
          <template #icon>
            <PhCheck class="w-5 h-5" />
          </template>
          Submit
        </AppButton>
      </div>
    </div>
  </div>
</template>
