<script setup lang="ts">
import { ref } from 'vue'
import { sampleImages } from '../../data/sampleImages'
import { defineExpose } from 'vue'
import { PhX } from '@phosphor-icons/vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

const showModal = ref(false)

const open = () => {
  showModal.value = true
}
const close = () => {
  showModal.value = false
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
        class="absolute top-4 right-4 btn btn-sm btn-circle border-none btn-ghost text-base-100 hover:bg-accent/80 shadow-none"
      >
        <PhX class="w-4 h-4" weight="bold" />
      </button>
      <div class="text-2xl font-bold text-center text-base-100 mb-6">
        {{ t('upload.sampleButton') }}
      </div>
      <div class="grid grid-cols-4 gap-4 place-items-center">
        <div
          v-for="img in sampleImages"
          :key="img.id"
          class="flex items-center justify-center w-32 h-40 overflow-hidden rounded-lg bg-base-200 cursor-pointer hover:ring-2 hover:ring-primary transition"
        >
          <img :src="img.url" :alt="img.alt" class="w-full h-full object-cover rounded-lg" />
        </div>
      </div>
    </div>
  </div>
</template>
