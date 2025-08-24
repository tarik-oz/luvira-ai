<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import { useAppState } from '../../../composables/useAppState'
import AppButton from '../base/AppButton.vue'
import { PhDownloadSimple } from '@phosphor-icons/vue'
import { trackEvent } from '@/services/analytics'

const { iconOnly } = defineProps<{ iconOnly?: boolean }>()

const { t, locale } = useI18n()
const { processedImage, currentColorResult, uploadedImage, selectedTone, isProcessing } =
  useAppState()

const downloadImage = async () => {
  if (!processedImage.value) {
    console.warn('No processed image to download')
    return
  }

  try {
    // Create download filename
    const fileName = generateFileName()

    // Always compose original + current processed image to ensure full image download
    const composedUrl = await composeOverlay(
      uploadedImage.value!,
      processedImage.value!,
      'image/png',
    )
    const resp = await fetch(composedUrl)
    const blob = await resp.blob()
    URL.revokeObjectURL(composedUrl)

    // Download
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = fileName
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)

    console.log('Image downloaded:', fileName)

    // Analytics
    trackEvent('download_image', {
      color: currentColorResult.value?.originalColor || null,
      tone: selectedTone.value || 'base',
      file_name: fileName,
    })
  } catch (error) {
    console.error('Download failed:', error)
  }
}

async function composeOverlay(
  originalUrl: string,
  overlayUrl: string,
  mime: 'image/png' | 'image/webp' = 'image/png',
): Promise<string> {
  const [baseImg, overlayImg] = await Promise.all([loadImage(originalUrl), loadImage(overlayUrl)])
  const width = overlayImg.naturalWidth || baseImg.naturalWidth
  const height = overlayImg.naturalHeight || baseImg.naturalHeight
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(baseImg, 0, 0, width, height)
  ctx.drawImage(overlayImg, 0, 0, width, height)
  const blob: Blob = await new Promise((resolve) => canvas.toBlob((b) => resolve(b!), mime))
  return URL.createObjectURL(blob)
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = url
  })
}

const generateFileName = (): string => {
  if (!currentColorResult.value) {
    return 'LuviraAI_Hair.png'
  }

  const { originalColor } = currentColorResult.value
  const isTrLocale = (locale.value || '').toLowerCase().startsWith('tr')
  const colorName = (t(`colors.${originalColor}`) as string) || originalColor
  const hairWord = isTrLocale ? 'Saç' : 'Hair'

  // Build filename: LuviraAI_Color_Tone_Hair.png
  let fileName = 'LuviraAI'
  fileName += `_${colorName}`

  if (selectedTone.value) {
    // Tones always in English (capitalize first letter)
    const toneCapitalized = selectedTone.value.charAt(0).toUpperCase() + selectedTone.value.slice(1)
    fileName += `_${toneCapitalized}`
  }

  fileName += `_${hairWord}.png`

  return fileName
}
</script>

<template>
  <AppButton
    v-if="!iconOnly"
    @click="downloadImage"
    :disabled="!processedImage || isProcessing"
    :class="
      processedImage && !isProcessing
        ? 'max-w-60 flex-1 px-4 py-2'
        : 'max-w-60 flex-1 cursor-not-allowed px-4 py-2 opacity-50'
    "
    :aria-label="t('processing.downloadButton') as string"
    :title="t('processing.downloadButton') as string"
  >
    <template #icon>
      <PhDownloadSimple class="h-4 w-4" />
    </template>
    {{ t('processing.downloadButton') }}
  </AppButton>
  <AppButton
    v-else
    :fullWidth="false"
    @click="downloadImage"
    :disabled="!processedImage || isProcessing"
    :class="'h-9 w-auto rounded-xl px-3 py-0'"
    :aria-label="t('processing.downloadButton') as string"
    :title="t('processing.downloadButton') as string"
  >
    <template #icon>
      <PhDownloadSimple class="text-base-100 h-4 w-4" />
    </template>
  </AppButton>
</template>
