<script setup lang="ts">
import { ref, onUnmounted } from 'vue'
import { PhUploadSimple, PhInfo, PhWarning } from '@phosphor-icons/vue'
import { useI18n } from 'vue-i18n'
import { useAppState } from '../../composables/useAppState'
import { useRouter } from 'vue-router'
import hairService from '../../services/hairService'

const { t } = useI18n()
const { isUploading } = useAppState()
const router = useRouter()

// Refs
const fileInput = ref<HTMLInputElement>()
const showTooltip = ref(false)
let tooltipTimeout: number | null = null
const toggleTooltip = () => {
  showTooltip.value = !showTooltip.value
  if (tooltipTimeout) {
    clearTimeout(tooltipTimeout)
    tooltipTimeout = null
  }
  if (showTooltip.value) {
    tooltipTimeout = window.setTimeout(() => {
      showTooltip.value = false
      tooltipTimeout = null
    }, 2500)
  }
}

onUnmounted(() => {
  if (tooltipTimeout) {
    clearTimeout(tooltipTimeout)
    tooltipTimeout = null
  }
})
const isDragOver = ref(false)
const errorMessage = ref<string | null>(null)

// Constants for validation
const MAX_FILE_SIZE = 10 * 1024 * 1024 // 10 MB
const ALLOWED_FILE_TYPES = ['image/jpeg', 'image/png', 'image/jpg']
const MAX_DIMENSION = 1600 // Maximum width or height
const MIN_DIMENSION = 400 // Minimum width and height

const resizeImage = (file: File): Promise<File> => {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')!
    const img = new Image()
    const objectUrl = URL.createObjectURL(file)

    img.onload = () => {
      URL.revokeObjectURL(objectUrl)
      const { width, height } = img

      // Minimum dimension validation
      if (width < MIN_DIMENSION || height < MIN_DIMENSION) {
        reject(new Error('TOO_SMALL'))
        return
      }

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

    img.onerror = () => {
      URL.revokeObjectURL(objectUrl)
      reject(new Error('DECODE_ERROR'))
    }

    img.src = objectUrl
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

  // Validate zero-byte file
  if (file.size === 0) {
    errorMessage.value = t('uploadSection.zeroSize')
    return
  }

  // Validate file type
  if (!ALLOWED_FILE_TYPES.includes(file.type)) {
    errorMessage.value = t('uploadSection.invalidType')
    return
  }

  // Validate file size
  if (file.size > MAX_FILE_SIZE) {
    errorMessage.value = t('uploadSection.maxSize')
    return
  }

  // Resize image if necessary
  let processedFile: File
  try {
    processedFile = await resizeImage(file)
  } catch (error) {
    if (error instanceof Error && error.message === 'TOO_SMALL') {
      errorMessage.value = t('uploadSection.tooSmall', { min: MIN_DIMENSION })
    } else if (error instanceof Error && error.message === 'DECODE_ERROR') {
      errorMessage.value = t('uploadSection.decodeError')
    } else {
      errorMessage.value = t('sampleImages.errorMessage')
    }
    return
  }

  console.log('Processed file details:', {
    name: processedFile.name,
    size: processedFile.size,
    type: processedFile.type,
    sizeInMB: (processedFile.size / (1024 * 1024)).toFixed(2) + ' MB',
  })

  try {
    await hairService.uploadImage(processedFile)
    router.push({ name: 'HairColorEditor' })
  } catch (error) {
    console.error('Upload failed:', error)
    const message = error instanceof Error ? error.message : String(error)
    const errorWithCode = error as Error & { error_code?: string }
    const errorCode =
      error && typeof error === 'object' && 'error_code' in errorWithCode
        ? errorWithCode.error_code
        : undefined
    if (!navigator.onLine || message === 'Failed to fetch' || message === 'NETWORK_ERROR') {
      errorMessage.value = t('uploadSection.networkError')
    } else if (message === 'TIMEOUT') {
      errorMessage.value = t('camera.uploading')
    } else if (errorCode === 'NO_HAIR_DETECTED') {
      errorMessage.value = t('uploadSection.noHairDetected')
    } else {
      errorMessage.value = t('sampleImages.errorMessage')
    }
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
        'group focus-visible:ring-primary/40 no-callout relative flex flex-col items-center justify-center rounded-xl border-2 p-10 text-center transition-all duration-300 select-none focus-visible:ring-2 focus-visible:outline-none active:scale-[0.98]',
        isUploading
          ? 'border-base-content/10 bg-base-content/5 cursor-not-allowed'
          : isDragOver
            ? 'border-accent bg-accent/10'
            : 'border-base-content/20 bg-base-content md:hover:bg-primary cursor-pointer',
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
        class="bg-base-content/30 absolute inset-0 z-20 flex flex-col items-center justify-center backdrop-blur-sm"
      >
        <!-- Progress bar style loading -->
        <div class="bg-base-200 mb-4 h-2 w-32 rounded-full">
          <div class="bg-accent h-2 animate-pulse rounded-full" style="width: 100%"></div>
        </div>
        <p class="text-base-100 font-semibold">{{ t('camera.uploading') }}</p>
      </div>

      <!-- Upload icon -->
      <div
        :class="[
          'mb-5 flex h-20 w-20 items-center justify-center rounded-full shadow-sm transition-colors duration-300',
          isDragOver ? 'bg-accent/20' : 'bg-base-100/10',
          !isDragOver && 'md:group-hover:bg-base-100/20',
        ]"
      >
        <PhUploadSimple
          :class="[
            'h-8 w-8 transition-colors duration-300',
            isDragOver ? 'text-accent/70' : 'text-base-100/80',
            !isDragOver && 'md:group-hover:text-base-100',
          ]"
        />
      </div>

      <!-- Upload text -->
      <div class="mb-1 select-none">
        <h3
          :class="[
            'text-xl font-semibold transition-colors duration-300',
            isDragOver ? 'text-accent' : 'text-base-100/80',
            !isDragOver && 'md:group-hover:text-base-100',
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
        class="absolute right-3 bottom-3 z-20 transition-colors duration-200"
        @mouseenter="showTooltip = true"
        @mouseleave="showTooltip = false"
        @click.stop.prevent="toggleTooltip()"
        @touchstart.stop.prevent="toggleTooltip()"
      >
        <PhInfo
          :class="[
            'h-6 w-6 cursor-help transition-colors duration-200',
            isDragOver ? 'text-accent/70' : 'text-base-100/70',
            !isDragOver && 'md:group-hover:text-base-100/70',
          ]"
        />
        <div
          v-if="showTooltip"
          class="bg-base-100 text-base-content ring-base-content/10 pointer-events-none absolute right-0 bottom-7 z-30 rounded-md px-3 py-1.5 text-xs font-semibold whitespace-nowrap shadow-lg ring-1 transition-all duration-200"
        >
          {{ t('uploadSection.tooltip') }}
        </div>
      </div>
    </div>

    <!-- Quality Note -->
    <div class="mt-4 w-full max-w-xl text-center">
      <p class="text-base-content/60 text-sm">
        {{ t('upload.qualityNote') }}
      </p>
    </div>
    <!-- Error Section -->
    <div v-if="errorMessage" class="mt-4 flex items-center justify-center gap-2 text-center">
      <PhWarning class="h-5 w-5 shrink-0 text-red-600" />
      <span class="text-sm font-medium text-red-600">{{ errorMessage }}</span>
    </div>
  </div>
</template>
