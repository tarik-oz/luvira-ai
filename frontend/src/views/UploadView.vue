<script setup lang="ts">
import UploadSection from '../components/ui/UploadSection.vue'
import AppButton from '../components/ui/AppButton.vue'
import SampleImages from '../components/ui/SampleImages.vue'
import CameraCapture from '../components/ui/CameraCapture.vue'
import HairColorShowcase from '../components/ui/HairColorShowcase.vue'
import { PhImage, PhCamera } from '@phosphor-icons/vue'
import { ref } from 'vue'
import { useAppState } from '../composables/useAppState'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()
const { isUploading } = useAppState()

const sampleImagesModal = ref()
const cameraCaptureModal = ref()
</script>

<template>
  <div class="grid grid-cols-1 md:grid-cols-2">
    <!-- Left column -->
    <div class="flex flex-col items-start justify-center px-4 md:px-0">
      <!-- Title -->
      <h1 class="text-base-content mb-6 text-5xl leading-tight font-extrabold md:text-6xl">
        {{ t('upload.title') }}
      </h1>

      <!-- Description -->
      <p class="text-base-content/70 mb-8 max-w-xl text-lg md:text-xl">
        {{ t('upload.description') }}
      </p>

      <!-- UploadSection -->
      <UploadSection class="w-full max-w-xl" />

      <!-- Divider -->
      <div class="divider text-base-content/70 my-6 w-full max-w-xl font-semibold select-none">
        {{ t('upload.divider') }}
      </div>

      <!-- Buttons -->
      <div class="mb-5 flex w-full max-w-xl gap-4">
        <!-- Sample Images Button -->
        <AppButton class="flex-1" :disabled="isUploading" @click="sampleImagesModal.open()">
          <template #icon>
            <PhImage class="h-5 w-5" />
          </template>
          {{ t('upload.sampleButton') }}
        </AppButton>

        <!-- Camera Button -->
        <AppButton class="flex-1" :disabled="isUploading" @click="cameraCaptureModal.open()">
          <template #icon>
            <PhCamera class="h-5 w-5" />
          </template>
          {{ t('upload.cameraButton') }}
        </AppButton>
      </div>

      <!-- SampleImages Modal -->
      <SampleImages ref="sampleImagesModal" />

      <!-- CameraCapture Modal -->
      <CameraCapture ref="cameraCaptureModal" />
    </div>

    <!-- Right column -->
    <div>
      <HairColorShowcase />
    </div>
  </div>
</template>
