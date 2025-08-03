<script setup lang="ts">
import { ref } from 'vue'
import { sampleImages } from '../../data/sampleImages'
import { defineExpose } from 'vue'
import { PhX } from '@phosphor-icons/vue'
import { useI18n } from 'vue-i18n'
import { useAppState } from '../../composables/useAppState'
import { useRouter } from 'vue-router'
import { PhWarning } from '@phosphor-icons/vue'

const { t } = useI18n()
const { isUploading, uploadFileAndSetSession } = useAppState()
const router = useRouter()

const showModal = ref(false)
const selectedImageIndex = ref<number | null>(null)
const errorMessage = ref<string | null>(null)

const open = () => {
  showModal.value = true
  selectedImageIndex.value = null
  errorMessage.value = null
}

const close = () => {
  if (!isUploading.value) {
    showModal.value = false
    selectedImageIndex.value = null
    errorMessage.value = null
  }
}

const handleImageSelect = async (index: number, imageUrl: string) => {
  if (isUploading.value) return

  selectedImageIndex.value = index
  errorMessage.value = null // Clear previous errors

  try {
    // Convert image URL to File object
    const response = await fetch(imageUrl)
    const blob = await response.blob()
    const file = new File([blob], `sample-image-${index + 1}.jpg`, { type: 'image/jpeg' })

    await uploadFileAndSetSession(file, imageUrl)

    // Close modal and navigate to processing page
    showModal.value = false
    selectedImageIndex.value = null
    router.push('/color-tone-changer')
  } catch (error) {
    console.error('Sample image upload failed:', error)
    errorMessage.value = t('sampleImages.errorMessage')
    selectedImageIndex.value = null
  }
}

defineExpose({ open })
</script>

<template>
  <!-- Full Screen Modal -->
  <div
    v-if="showModal"
    class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-xs"
    @click.self="close"
  >
    <div class="relative w-full max-w-2xl p-8 rounded-xl bg-base-content/80 shadow-2xl">
      <button
        @click="close"
        :disabled="isUploading"
        class="absolute top-4 right-4 btn btn-sm btn-circle border-none btn-ghost text-base-100 hover:bg-accent/80 shadow-none disabled:opacity-30"
      >
        <PhX class="w-4 h-4" weight="bold" />
      </button>

      <div class="text-2xl font-bold text-center text-base-100 mb-6">
        {{ t('upload.sampleButton') }}
      </div>

      <!-- Image Grid -->
      <div class="grid grid-cols-4 gap-4 place-items-center mb-6">
        <div
          v-for="(img, index) in sampleImages"
          :key="img.id"
          @click="handleImageSelect(index, img.url)"
          :class="[
            'flex items-center justify-center w-32 h-40 overflow-hidden rounded-lg bg-base-200 cursor-pointer transition-all duration-300',
            selectedImageIndex === index
              ? 'ring-4 ring-primary scale-105'
              : isUploading
                ? 'opacity-50'
                : 'hover:ring-2 hover:ring-primary hover:scale-102',
          ]"
          :style="isUploading ? { pointerEvents: 'none' } : {}"
        >
          <img :src="img.url" :alt="img.alt" class="w-full h-full object-cover rounded-lg" />
        </div>
      </div>

      <!-- Loading Section -->
      <div v-if="isUploading" class="text-center">
        <div class="w-full bg-base-200 rounded-full h-2 mb-3">
          <div class="bg-primary h-2 rounded-full animate-pulse" style="width: 100%"></div>
        </div>
        <p class="text-base-100 text-sm font-medium">
          {{ t('sampleImages.loadingSelected') }}
        </p>
      </div>

      <!-- Error Section -->
      <div v-if="errorMessage" class="text-center mt-4 flex items-center justify-center gap-2">
        <PhWarning class="w-5 h-5 text-red-600 shrink-0" />
        <span class="text-red-600 text-sm font-medium">{{ errorMessage }}</span>
      </div>
    </div>
  </div>
</template>
