<script setup lang="ts">
import { useAppState } from '../composables/useAppState'
import { useRouter } from 'vue-router'
import { onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import ImageDisplay from '../components/ui/ImageDisplay.vue'
import AppButton from '../components/ui/AppButton.vue'
import ColorPalette from '../components/ui/ColorPalette.vue'
import DownloadButton from '../components/ui/DownloadButton.vue'
import { PhCaretLeft, PhWarning } from '@phosphor-icons/vue'
import TonePalette from '@/components/ui/TonePalette.vue'
import SessionExpiredModal from '@/components/ui/SessionExpiredModal.vue'

const { uploadedImage, sessionId, resetState, sessionError, setSessionError, processingError } =
  useAppState()
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

const confirmSessionExpired = () => {
  setSessionError(null)
  goBack()
}
</script>

<template>
  <div class="container mx-auto px-4">
    <!-- Back Button -->
    <div class="mb-6 lg:mb-8">
      <AppButton
        @click="goBack"
        class="bg-base-content hover:bg-base-content/80 active:bg-primary/60 max-w-60 px-4 py-2"
      >
        <template #icon>
          <PhCaretLeft class="h-4 w-4" />
        </template>
        {{ t('processing.backButton') }}
      </AppButton>
    </div>

    <!-- Main Content Grid -->
    <div class="grid grid-cols-1 gap-8 md:grid-cols-2 md:items-start">
      <!-- Left column -->
      <div
        class="flex flex-col items-center justify-center px-4 md:items-start md:justify-start md:px-0"
      >
        <!-- Image Display Container -->
        <div class="mx-auto w-full max-w-lg space-y-4">
          <!-- Image Display -->
          <ImageDisplay :compare-mode="isCompareMode" />
          <!-- Error Section -->
          <div
            v-if="processingError"
            class="my-4 flex items-center justify-center gap-2 text-center"
          >
            <PhWarning class="h-5 w-5 shrink-0 text-red-600" />
            <span class="text-sm font-medium text-red-600">{{ processingError }}</span>
          </div>
          <SessionExpiredModal
            :visible="sessionError === 'SESSION_EXPIRED'"
            @confirm="confirmSessionExpired"
          />

          <!-- Image Controls -->
          <div class="flex items-center justify-between gap-4">
            <!-- Download Button -->
            <DownloadButton />

            <!-- Compare Mode Toggle -->
            <div class="flex items-center gap-2">
              <span class="text-base-content text-sm font-medium">{{
                t('processing.compareMode')
              }}</span>
              <input type="checkbox" v-model="isCompareMode" class="toggle" />
            </div>
          </div>
        </div>
      </div>

      <!-- Right column -->
      <div class="flex flex-col gap-5 px-4 md:px-0">
        <ColorPalette />
        <TonePalette />
      </div>
    </div>
  </div>
</template>
