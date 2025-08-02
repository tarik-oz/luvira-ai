<script setup lang="ts">
import { ref } from 'vue'
import { PhUploadSimple, PhInfo } from '@phosphor-icons/vue'
import { useI18n } from 'vue-i18n'

const emit = defineEmits<{ 'file-select': [file: File] }>()

const { t } = useI18n()

// Refs
const fileInput = ref<HTMLInputElement>()
const showTooltip = ref(false)
const isDragOver = ref(false)

// Open file dialog
const openFileDialog = () => fileInput.value?.click()

// Triggered when a file is selected
const handleFileSelect = (event: Event) => {
  const target = event.target as HTMLInputElement
  if (target.files?.[0]) emit('file-select', target.files[0])
}

// Triggered when a file is dropped
const handleDrop = (event: DragEvent) => {
  event.preventDefault()
  if (event.dataTransfer?.files?.[0]) emit('file-select', event.dataTransfer.files[0])
}

// Prevent default drag events and set drag-over state
const handleDragOver = (event: DragEvent) => {
  event.preventDefault()
  isDragOver.value = true
}
const handleDragLeave = (event: DragEvent) => {
  event.preventDefault()
  isDragOver.value = false
}
</script>

<template>
  <div
    :class="[
      'relative flex flex-col items-center justify-center group p-10 rounded-xl border-2 text-center cursor-pointer hover:bg-accent transition-colors duration-300',
      isDragOver ? 'border-accent bg-accent/10' : 'border-base-content/20 bg-base-content',
    ]"
    @click="openFileDialog"
    @drop="handleDrop"
    @dragover="handleDragOver"
    @dragleave="handleDragLeave"
    @dragend="handleDragLeave"
  >
    <!-- Upload icon -->
    <div
      :class="[
        'flex items-center justify-center w-20 h-20 mb-5 rounded-full shadow-sm transition-colors duration-300',
        isDragOver ? 'bg-accent/20' : 'bg-base-100/10',
        !isDragOver && 'group-hover:bg-base-100/20',
      ]"
    >
      <PhUploadSimple
        :class="[
          'w-8 h-8 transition-colors duration-300',
          isDragOver ? 'text-accent/70' : 'text-base-100/80',
          !isDragOver && 'group-hover:text-base-100',
        ]"
      />
    </div>

    <!-- Upload text -->
    <div class="mb-1">
      <h3
        :class="[
          'text-xl font-semibold transition-colors duration-300',
          isDragOver ? 'text-accent' : 'text-base-100/80',
          !isDragOver && 'group-hover:text-base-100',
        ]"
      >
        {{ t('uploadSection.uploadImage') }}
      </h3>
    </div>

    <!-- Hidden file input -->
    <input
      ref="fileInput"
      type="file"
      accept=".jpg,.jpeg,.png"
      class="hidden"
      @change="handleFileSelect"
    />

    <!-- Info icon and tooltip -->
    <div
      class="absolute right-3 bottom-3 z-20 transition-colors duration-300"
      @mouseenter="showTooltip = true"
      @mouseleave="showTooltip = false"
    >
      <PhInfo
        :class="[
          'w-6 h-6 transition-colors duration-300',
          isDragOver ? 'text-accent/70' : 'text-base-100/60 group-hover:text-base-100/70',
          !isDragOver && 'group-hover:text-accent',
        ]"
      />
      <div
        v-if="showTooltip"
        class="absolute right-0 bottom-7 px-3 py-1 rounded bg-base-100/100 text-xs font-bold text-base-content/90 shadow whitespace-nowrap transition-colors duration-300"
      >
        {{ t('uploadSection.tooltip') }}
      </div>
    </div>
  </div>
</template>
