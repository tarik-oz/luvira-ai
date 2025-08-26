<script setup lang="ts">
import { useAppState } from '../composables/useAppState'
import { useRouter } from 'vue-router'
import { ref, onBeforeUnmount } from 'vue'
import { useI18n } from 'vue-i18n'
import ImageDisplay from '../components/ui/processing_view/ImageDisplay.vue'
import AppButton from '../components/ui/base/AppButton.vue'
import ColorPalette from '../components/ui/processing_view/ColorPalette.vue'
import DownloadButton from '../components/ui/processing_view/DownloadButton.vue'
import { PhCaretLeft, PhWarning, PhArrowsLeftRight } from '@phosphor-icons/vue'
import TonePalette from '@/components/ui/processing_view/TonePalette.vue'
import MobileColorToneBar from '@/components/ui/processing_view/MobileColorToneBar.vue'
import SessionExpiredModal from '@/components/ui/processing_view/SessionExpiredModal.vue'
import { trackEvent } from '@/services/analytics'

const { resetState, sessionError, setSessionError, processingError, showMobileToneBar } =
  useAppState()
const router = useRouter()
const { t } = useI18n()

// Compare mode toggle
const isCompareMode = ref(false)

const toggleCompareMode = () => {
  const newValue = !isCompareMode.value
  isCompareMode.value = newValue

  // Track when compare mode is enabled
  if (newValue) {
    trackEvent('compare_mode_enabled', {
      action: 'toggle_compare_mode',
      value: true,
    })
  }
}

const goBack = () => {
  router.push('/')
}

const confirmSessionExpired = () => {
  setSessionError(null)
  goBack()
}

onBeforeUnmount(() => {
  resetState()
})
</script>

<template>
  <div class="container mx-auto lg:px-4">
    <!-- Mobile/Tablet Top Actions Bar -->
    <div
      class="-mx-4 mb-4 flex items-center justify-between px-4 py-2 lg:hidden"
      aria-label="Top actions"
    >
      <div class="flex items-center gap-2">
        <!-- Back (AppButton circular, base variant) -->
        <AppButton
          :fullWidth="false"
          variant="base"
          @click="goBack"
          class="h-9 w-9 rounded-full p-0"
          :aria-label="t('processing.backButton') as string"
          :title="t('processing.backButton') as string"
        >
          <template #icon>
            <PhCaretLeft class="text-base-100 h-4 w-4" />
          </template>
        </AppButton>
      </div>
      <div class="flex items-center gap-2">
        <!-- Compare toggle (AppButton rounded, primary when active) -->
        <AppButton
          :fullWidth="false"
          :variant="isCompareMode ? 'primary' : 'accent'"
          @click="toggleCompareMode"
          :aria-pressed="isCompareMode ? 'true' : 'false'"
          class="h-9 w-auto rounded-xl px-3 py-0"
          :aria-label="t('processing.compareMode') as string"
          :title="t('processing.compareMode') as string"
        >
          <template #icon>
            <PhArrowsLeftRight class="text-base-100 h-4 w-4" />
          </template>
        </AppButton>

        <!-- Download (icon-only rounded rect using AppButton inside component) -->
        <DownloadButton :icon-only="true" />
      </div>
    </div>

    <!-- Back Button -->
    <div class="mb-6 hidden lg:mb-8 lg:block">
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
    <div class="grid grid-cols-1 gap-6 lg:grid-cols-2 lg:items-start lg:gap-8">
      <!-- Left column -->
      <section
        class="flex flex-col items-center justify-center px-0 lg:items-start lg:justify-start lg:px-0"
        aria-labelledby="image-display-heading"
      >
        <!-- Image Display Container -->
        <div class="mx-auto w-full max-w-lg space-y-4">
          <!-- Image Display -->
          <h2 id="image-display-heading" class="sr-only">Image display</h2>
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
          <div class="hidden items-center justify-between gap-4 lg:flex">
            <!-- Download Button (desktop) -->
            <DownloadButton />

            <!-- Compare Mode Toggle (desktop) -->
            <div class="flex items-center gap-2">
              <span class="text-base-content text-sm font-medium">{{
                t('processing.compareMode')
              }}</span>
              <input
                type="checkbox"
                :checked="isCompareMode"
                @change="toggleCompareMode"
                class="toggle"
              />
            </div>
          </div>
        </div>
      </section>

      <!-- Right column -->
      <section
        class="flex flex-col gap-4 px-4 lg:gap-5 lg:px-0"
        aria-label="Color and tone selection"
      >
        <!-- ColorPalette: Always show on desktop, conditionally on mobile -->
        <ColorPalette
          :class="{ 'lg:block': true, hidden: showMobileToneBar, block: !showMobileToneBar }"
        />

        <!-- TonePalette: Show on desktop only -->
        <TonePalette class="hidden lg:block" />

        <!-- MobileColorToneBar: Show on mobile only when showMobileToneBar is true -->
        <MobileColorToneBar v-show="showMobileToneBar" />
      </section>
    </div>
  </div>
</template>
