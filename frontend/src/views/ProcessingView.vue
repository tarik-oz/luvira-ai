<script setup lang="ts">
import { useAppState } from '../composables/useAppState'
import { useRouter } from 'vue-router'
import { onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import ImageDisplay from '../components/ui/ImageDisplay.vue'
import AppButton from '../components/ui/AppButton.vue'
import ColorPalette from '../components/ui/ColorPalette.vue'
import DownloadButton from '../components/ui/DownloadButton.vue'
import { PhCaretLeft } from '@phosphor-icons/vue'
import TonePalette from '@/components/ui/TonePalette.vue'

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
</script>

<template>
  <div class="container mx-auto px-4">
    <!-- Back Button -->
    <div class="mb-8">
      <AppButton
        @click="goBack"
        class="bg-base-content hover:bg-base-content/80 active:bg-primary/60 px-4 py-2 max-w-60"
      >
        <template #icon>
          <PhCaretLeft class="w-4 h-4" />
        </template>
        {{ t('processing.backButton') }}
      </AppButton>
    </div>

    <!-- Main Content Grid -->
    <div class="grid md:grid-cols-2 grid-cols-1 gap-8">
      <!-- Left column -->
      <div class="flex flex-col items-center justify-center px-4 md:px-0">
        <!-- Image Display Container -->
        <div class="w-full max-w-lg mx-auto space-y-4">
          <!-- Image Display -->
          <ImageDisplay :compare-mode="isCompareMode" />

          <!-- Image Controls -->
          <div class="flex items-center gap-4 justify-between">
            <!-- Download Button -->
            <DownloadButton />

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
      <div class="flex flex-col px-4 md:px-0 gap-5">
        <ColorPalette />
        <TonePalette />
      </div>
    </div>
  </div>
</template>
