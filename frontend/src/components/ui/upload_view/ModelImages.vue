<script setup lang="ts">
import { ref } from 'vue'
import { modelImages } from '../../../data/modelImages'
import { defineExpose } from 'vue'
import { PhX } from '@phosphor-icons/vue'
import { useI18n } from 'vue-i18n'
import { useAppState } from '../../../composables/useAppState'
import { useRouter } from 'vue-router'
import { PhWarning } from '@phosphor-icons/vue'
import hairService from '../../../services/hairService'
import { trackEvent } from '../../../services/analytics'

const { t } = useI18n()
const { isUploading } = useAppState()
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
    if (!navigator.onLine) {
      throw new TypeError('Network offline')
    }
    // Send only the image name to analytics (e.g., guy_1)
    const nameMatch = imageUrl.match(/([a-zA-Z0-9_-]+)\.[a-zA-Z0-9]+$/)
    const imageName = nameMatch ? nameMatch[1] : `model_${index + 1}`
    trackEvent('model_click', { index, image: imageName })
    const response = await fetch(imageUrl)
    const blob = await response.blob()
    const file = new File([blob], `model-image-${index + 1}.jpg`, { type: 'image/jpeg' })

    await hairService.uploadImage(file, 'model_images', imageUrl)

    // Close modal and navigate to processing page
    showModal.value = false
    selectedImageIndex.value = null
    router.push({ name: 'HairColorEditor' })
  } catch (error) {
    console.error('Model image upload failed:', error)
    const message = error instanceof Error ? error.message : String(error)
    if (!navigator.onLine || message === 'Failed to fetch' || /network/i.test(message)) {
      errorMessage.value = t('processing.networkError')
    } else {
      errorMessage.value = t('modelImages.errorMessage')
    }
    selectedImageIndex.value = null
  }
}

defineExpose({ open })
</script>

<template>
  <!-- Full Screen Modal -->
  <div
    v-if="showModal"
    class="no-callout fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-[clamp(12px,5vw,24px)] backdrop-blur-xs select-none"
    @click.self="close"
  >
    <div
      class="bg-base-content/80 relative w-full max-w-2xl rounded-xl p-[clamp(12px,3vw,24px)] shadow-2xl"
    >
      <button
        @click="close"
        :disabled="isUploading"
        class="btn btn-sm btn-circle btn-ghost text-base-100 hover:bg-accent/80 absolute top-4 right-4 border-none shadow-none disabled:opacity-30"
      >
        <PhX class="h-4 w-4" weight="bold" />
      </button>

      <div class="text-base-100 mb-4 text-center text-2xl font-bold">
        {{ t('upload.modelButton') }}
      </div>

      <!-- Image Grid -->
      <div
        class="mb-4 grid grid-cols-2 place-items-center gap-x-2 gap-y-3 sm:grid-cols-3 md:grid-cols-3 md:gap-x-1"
      >
        <div
          v-for="(img, index) in modelImages"
          :key="img.id"
          @click="handleImageSelect(index, img.url)"
          :class="[
            'bg-base-200 flex h-36 w-28 cursor-pointer items-center justify-center overflow-hidden rounded-lg transition-all duration-300 md:h-40 md:w-32',
            selectedImageIndex === index
              ? 'ring-primary scale-105 ring-4'
              : isUploading
                ? 'opacity-50'
                : 'hover:ring-primary hover:scale-102 hover:ring-2',
          ]"
          :style="isUploading ? { pointerEvents: 'none' } : {}"
        >
          <img :src="img.url" :alt="img.alt" class="h-full w-full rounded-lg object-cover" />
        </div>
      </div>

      <!-- Loading Section -->
      <div v-if="isUploading" class="text-center">
        <div class="bg-base-200 mb-3 h-2 w-full rounded-full">
          <div class="bg-primary h-2 animate-pulse rounded-full" style="width: 100%"></div>
        </div>
        <p class="text-base-100 text-sm font-medium">
          {{ t('modelImages.loadingSelected') }}
        </p>
      </div>

      <!-- Error Section -->
      <div v-if="errorMessage" class="mt-4 flex items-center justify-center gap-2 text-center">
        <PhWarning class="h-5 w-5 shrink-0 text-red-600" />
        <span class="text-sm font-medium text-red-600">{{ errorMessage }}</span>
      </div>
    </div>
  </div>
</template>
