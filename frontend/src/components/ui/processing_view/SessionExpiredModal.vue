<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import AppButton from '../base/AppButton.vue'

interface Props {
  visible: boolean
}

const props = defineProps<Props>()
const emit = defineEmits<{ (e: 'confirm'): void }>()

const { t } = useI18n()

const onConfirm = () => emit('confirm')
</script>

<template>
  <div v-if="props.visible" class="fixed inset-0 z-50 flex items-center justify-center">
    <div class="absolute inset-0 bg-black/60"></div>
    <div
      class="bg-base-300 text-base-content relative z-10 mx-auto mr-2 ml-2 w-full max-w-md rounded-2xl p-6 shadow-2xl"
      role="dialog"
      aria-modal="true"
      :aria-label="t('processing.sessionExpiredTitle') as string"
    >
      <h3 class="mb-2 text-lg font-bold">{{ t('processing.sessionExpiredTitle') }}</h3>
      <p class="mb-6 text-sm opacity-80">{{ t('processing.sessionExpiredMessage') }}</p>
      <div class="flex justify-end">
        <AppButton
          class="bg-primary text-primary-content hover:bg-primary/90 active:bg-primary/80 max-w-40 cursor-pointer px-4 py-2"
          @click="onConfirm"
        >
          {{ t('processing.sessionExpiredConfirm') }}
        </AppButton>
      </div>
    </div>
  </div>
</template>
