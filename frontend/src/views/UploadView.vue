<script setup lang="ts">
import UploadSection from '../components/ui/upload_view/UploadSection.vue'
import AppButton from '../components/ui/base/AppButton.vue'
import ModelImages from '../components/ui/upload_view/ModelImages.vue'
import CameraCapture from '../components/ui/upload_view/CameraCapture.vue'
import HairColorShowcase from '../components/ui/upload_view/HairColorShowcase.vue'
import { PhImage, PhCamera } from '@phosphor-icons/vue'
import { ref } from 'vue'
import { useAppState } from '../composables/useAppState'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()
const { isUploading } = useAppState()

const modelImagesModal = ref()
const cameraCaptureModal = ref()
</script>

<template>
  <div class="grid grid-cols-1 gap-y-6 lg:min-h-[75vh] lg:grid-cols-2 lg:items-center lg:gap-x-8">
    <!-- Left column -->
    <section
      class="order-2 flex flex-col items-center justify-center px-4 text-center lg:order-1 lg:items-start lg:px-0 lg:text-left"
      :aria-busy="isUploading ? 'true' : 'false'"
      aria-describedby="upload-desc"
      aria-labelledby="upload-title"
    >
      <!-- Title -->
      <h1
        id="upload-title"
        class="text-base-content mb-4 text-4xl leading-tight font-extrabold md:text-4xl lg:text-[3.45rem]"
      >
        {{ t('upload.title') }}
      </h1>

      <!-- Description -->
      <p
        id="upload-desc"
        class="text-base-content/70 mb-6 max-w-xl text-base md:text-base lg:text-[1.2rem]"
      >
        {{ t('upload.description') }}
      </p>

      <!-- UploadSection -->
      <UploadSection class="w-full max-w-xl" />

      <!-- Divider -->
      <div
        class="divider text-base-content/70 mx-auto my-6 w-full max-w-xl font-semibold select-none"
      >
        {{ t('upload.divider') }}
      </div>

      <!-- Buttons -->
      <div class="mb-5 flex w-full max-w-xl flex-col gap-3 md:flex-row md:gap-4">
        <!-- Model Images Button -->
        <AppButton
          class="w-full md:flex-1"
          :disabled="isUploading"
          aria-haspopup="dialog"
          @click="modelImagesModal.open()"
        >
          <template #icon>
            <PhImage class="h-5 w-5" />
          </template>
          {{ t('upload.modelButton') }}
        </AppButton>

        <!-- Camera Button -->
        <AppButton
          class="w-full md:flex-1"
          :disabled="isUploading"
          aria-haspopup="dialog"
          @click="cameraCaptureModal.open()"
        >
          <template #icon>
            <PhCamera class="h-5 w-5" />
          </template>
          {{ t('upload.cameraButton') }}
        </AppButton>
      </div>

      <!-- ModelImages Modal -->
      <ModelImages ref="modelImagesModal" />

      <!-- CameraCapture Modal -->
      <CameraCapture ref="cameraCaptureModal" />
    </section>

    <!-- Right column -->
    <div
      class="order-1 mx-auto w-full max-w-xl px-[clamp(16px,5vw,24px)] lg:order-2 lg:mx-0 lg:mt-0 lg:max-w-none lg:px-0 lg:pr-0"
    >
      <div class="mx-auto w-full max-w-[600px]">
        <HairColorShowcase />
      </div>
    </div>
  </div>
</template>
