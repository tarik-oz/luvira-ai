<script setup lang="ts">
import UploadSection from '../components/ui/UploadSection.vue'
import AppButton from '../components/ui/AppButton.vue'
import SampleImages from '../components/ui/SampleImages.vue'
import CameraCapture from '../components/ui/CameraCapture.vue'
import HairColorShowcase from '../components/ui/HairColorShowcase.vue'
import { PhImage, PhCamera } from '@phosphor-icons/vue'
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

const sampleImagesModal = ref()
const cameraCaptureModal = ref()

const handleFileSelect = (file: File) => {
  // Handle selected file
  console.log('Selected file:', file.name) // TODO: Process file
}
</script>

<template>
  <div class="grid md:grid-cols-2 grid-cols-1">
    <!-- Left column -->
    <div class="flex flex-col items-start justify-center px-4 md:px-0">
      <!-- Title -->
      <h1 class="text-5xl md:text-6xl font-extrabold leading-tight text-base-content mb-6">
        {{ t('upload.title') }}
      </h1>

      <!-- Description -->
      <p class="text-lg md:text-xl max-w-xl mb-8 text-base-content/70">
        {{ t('upload.description') }}
      </p>

      <!-- UploadSection -->
      <UploadSection @file-select="handleFileSelect" class="max-w-xl w-full" />

      <!-- Divider -->
      <div class="divider max-w-xl w-full my-6 font-semibold select-none text-base-content/70">
        {{ t('upload.divider') }}
      </div>

      <!-- Buttons -->
      <div class="flex gap-4 max-w-xl w-full">
        <!-- Sample Images Button -->
        <AppButton class="flex-1" @click="sampleImagesModal.open()">
          <template #icon>
            <PhImage class="w-5 h-5" />
          </template>
          {{ t('upload.sampleButton') }}
        </AppButton>

        <!-- Camera Button -->
        <AppButton class="flex-1" @click="cameraCaptureModal.open()">
          <template #icon>
            <PhCamera class="w-5 h-5" />
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
