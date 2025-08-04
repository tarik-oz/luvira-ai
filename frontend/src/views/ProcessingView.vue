<script setup lang="ts">
import { useAppState } from '../composables/useAppState'
import { useRouter } from 'vue-router'
import { onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import ImageDisplay from '../components/ui/ImageDisplay.vue'
import AppButton from '../components/ui/AppButton.vue'
import { PhCaretLeft, PhDownloadSimple } from '@phosphor-icons/vue'

const { uploadedImage, sessionId, resetState } = useAppState()
const router = useRouter()
const { t } = useI18n()

// Compare mode toggle
const isCompareMode = ref(false)

// Check if user has valid session on page load
onMounted(() => {
  if (!sessionId.value || !uploadedImage.value) {
    // No session or image, redirect to upload page
    router.push('/')
  }
})

const goBack = () => {
  resetState()
  router.push('/')
}

const downloadImage = () => {
  // TODO: Download functionality
  console.log('Download clicked')
}
</script>

<template>
  <div class="container mx-auto px-4">
    <!-- Main Content Grid -->
    <div class="grid md:grid-cols-2 grid-cols-1">
      <!-- Left column -->
      <div class="flex flex-col items-center justify-center px-4 md:px-0 space-y-6">
        <!-- Back Button -->
        <div class="w-full flex justify-start">
          <AppButton
            @click="goBack"
            class="bg-base-content hover:bg-base-content/80 active:bg-primary/60 max-w-70 px-4 py-2"
          >
            <template #icon>
              <PhCaretLeft class="w-4 h-4" />
            </template>
            {{ t('processing.backButton') }}
          </AppButton>
        </div>

        <!-- Image Display Container -->
        <div class="w-full max-w-lg mx-auto space-y-4">
          <!-- Image Display -->
          <ImageDisplay />

          <!-- Image Controls -->
          <div class="flex items-center gap-4 justify-between">
            <!-- Download Button -->
            <AppButton @click="downloadImage" class="flex-1 px-4 py-2 max-w-60">
              <template #icon>
                <PhDownloadSimple class="w-4 h-4" />
              </template>
              {{ t('processing.downloadButton') }}
            </AppButton>

            <!-- Compare Mode Toggle -->
            <div class="flex items-center gap-2">
              <span class="text-sm font-medium text-base-content">{{
                t('processing.compareMode')
              }}</span>
              <input type="checkbox" v-model="isCompareMode" class="toggle" />
            </div>
          </div>
        </div>
      </div>

      <!-- Right column -->
      <div></div>
    </div>
  </div>
</template>
