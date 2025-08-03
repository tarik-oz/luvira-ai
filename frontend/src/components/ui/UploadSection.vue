<script setup lang="ts">
import { ref } from 'vue'
import { PhUploadSimple, PhInfo, PhWarning } from '@phosphor-icons/vue'
import { useI18n } from 'vue-i18n'
import { useAppState } from '../../composables/useAppState'
import { useRouter } from 'vue-router'

const emit = defineEmits<{ 'file-select': [file: File] }>()

const { t } = useI18n()
const { uploadFileAndSetSession, isUploading } = useAppState()
const router = useRouter()

// Refs
const fileInput = ref<HTMLInputElement>()
const showTooltip = ref(false)
const isDragOver = ref(false)
const errorMessage = ref<string | null>(null)

// Constants for validation
const MAX_FILE_SIZE = 10 * 1024 * 1024 // 10 MB
const ALLOWED_FILE_TYPES = ['image/jpeg', 'image/png']
const MAX_DIMENSION = 4096 // Maximum width or height

const resizeImage = (file: File): Promise<File> => {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')!
    const img = new Image()

    img.onload = () => {
      const { width, height } = img

      // Check if resizing is needed
      if (width <= MAX_DIMENSION && height <= MAX_DIMENSION) {
        resolve(file) // No resizing needed
        return
      }

      // Calculate new dimensions while maintaining aspect ratio
      let newWidth = width
      let newHeight = height

      if (width > height) {
        if (width > MAX_DIMENSION) {
          newWidth = MAX_DIMENSION
          newHeight = (height * MAX_DIMENSION) / width
        }
      } else {
        if (height > MAX_DIMENSION) {
          newHeight = MAX_DIMENSION
          newWidth = (width * MAX_DIMENSION) / height
        }
      }

      // Set canvas dimensions
      canvas.width = newWidth
      canvas.height = newHeight

      // Draw resized image
      ctx.drawImage(img, 0, 0, newWidth, newHeight)

      // Convert to blob and create new file
      canvas.toBlob(
        (blob) => {
          if (blob) {
            const resizedFile = new File([blob], file.name, {
              type: file.type,
              lastModified: Date.now(),
            })
            resolve(resizedFile)
          } else {
            resolve(file) // Fallback to original file
          }
        },
        file.type,
        0.9,
      ) // 90% quality
    }

    img.src = URL.createObjectURL(file)
  })
}

const processFile = async (file: File) => {
  if (isUploading.value) return
  errorMessage.value = null

  // Log file details for debugging
  console.log('Original file details:', {
    name: file.name,
    size: file.size,
    type: file.type,
    sizeInMB: (file.size / (1024 * 1024)).toFixed(2) + ' MB',
  })

  // Validate file type
  if (!ALLOWED_FILE_TYPES.includes(file.type)) {
    errorMessage.value = t('uploadSection.validationError')
    return
  }

  // Validate file size
  if (file.size > MAX_FILE_SIZE) {
    errorMessage.value = t('uploadSection.validationError')
    return
  }

  try {
    // Resize image if necessary
    const processedFile = await resizeImage(file)

    console.log('Processed file details:', {
      name: processedFile.name,
      size: processedFile.size,
      type: processedFile.type,
      sizeInMB: (processedFile.size / (1024 * 1024)).toFixed(2) + ' MB',
    })

    await uploadFileAndSetSession(processedFile)
    router.push('/color-tone-changer')
  } catch (error) {
    console.error('Upload failed:', error)
    errorMessage.value = t('sampleImages.errorMessage')
  }
}

// Open file dialog
const openFileDialog = () => {
  if (isUploading.value) return
  fileInput.value?.click()
}

// Triggered when a file is selected
const handleFileSelect = (event: Event) => {
  const target = event.target as HTMLInputElement
  if (target.files?.[0]) {
    processFile(target.files[0])
  }
  // Reset file input to allow selecting the same file again
  target.value = ''
}

// Triggered when a file is dropped
const handleDrop = (event: DragEvent) => {
  event.preventDefault()
  isDragOver.value = false
  if (isUploading.value) return
  if (event.dataTransfer?.files?.[0]) {
    processFile(event.dataTransfer.files[0])
  }
}

// Prevent default drag events and set drag-over state
const handleDragOver = (event: DragEvent) => {
  event.preventDefault()
  if (isUploading.value) return
  isDragOver.value = true
}
const handleDragLeave = (event: DragEvent) => {
  event.preventDefault()
  isDragOver.value = false
}
</script>

<template>
  <div>
    <div
      :class="[
        'relative flex flex-col items-center justify-center group p-10 rounded-xl border-2 text-center transition-all duration-300',
        isUploading
          ? 'border-base-content/10 bg-base-content/5 cursor-not-allowed'
          : isDragOver
            ? 'border-accent bg-accent/10'
            : 'border-base-content/20 bg-base-content cursor-pointer hover:bg-accent',
      ]"
      @click="openFileDialog"
      @drop="handleDrop"
      @dragover="handleDragOver"
      @dragleave="handleDragLeave"
      @dragend="handleDragLeave"
    >
      <!-- Loading Spinner -->
      <div
        v-if="isUploading"
        class="absolute inset-0 z-20 flex flex-col items-center justify-center bg-base-content/30 backdrop-blur-sm"
      >
        <!-- Progress bar style loading -->
        <div class="w-32 bg-base-200 rounded-full h-2 mb-4">
          <div class="bg-accent h-2 rounded-full animate-pulse" style="width: 100%"></div>
        </div>
        <p class="font-semibold text-base-100">{{ t('camera.uploading') }}</p>
      </div>

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
        :disabled="isUploading"
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
    <!-- Error Section -->
    <div v-if="errorMessage" class="mt-4 flex items-center justify-center gap-2 text-center">
      <PhWarning class="h-5 w-5 shrink-0 text-red-600" />
      <span class="text-sm font-medium text-red-600">{{ errorMessage }}</span>
    </div>
  </div>
</template>
