<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import { useAppState } from '../../composables/useAppState'
import AppButton from './AppButton.vue'
import { PhDownloadSimple } from '@phosphor-icons/vue'

const { t } = useI18n()
const { processedImage, currentColorResult, uploadedImage, selectedTone } = useAppState()

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

    console.log('✅ Image downloaded:', fileName)
  } catch (error) {
    console.error('❌ Download failed:', error)
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
  const currentLocale = t('colors.Black') === 'Siyah' ? 'tr' : 'en' // Detect locale

  // Color names in both languages (only for colors, not tones)
  const colorNames = {
    en: {
      Black: 'Black',
      Blonde: 'Blonde',
      Brown: 'Brown',
      Auburn: 'Auburn',
      Pink: 'Pink',
      Blue: 'Blue',
      Purple: 'Purple',
      Gray: 'Gray',
    },
    tr: {
      Black: 'Siyah',
      Blonde: 'Sarışın',
      Brown: 'Kahverengi',
      Auburn: 'Kestane',
      Pink: 'Pembe',
      Blue: 'Mavi',
      Purple: 'Mor',
      Gray: 'Gri',
    },
  }

  const locale: 'en' | 'tr' = t('colors.Black') === 'Siyah' ? 'tr' : 'en'
  const mapping = colorNames[locale] as Record<string, string>
  const colorName = mapping[originalColor] ?? originalColor
  const hairWord = currentLocale === 'tr' ? 'Saç' : 'Hair'

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
    @click="downloadImage"
    :disabled="!processedImage"
    :class="
      processedImage
        ? 'flex-1 px-4 py-2 max-w-60'
        : 'flex-1 px-4 py-2 max-w-60 opacity-50 cursor-not-allowed'
    "
  >
    <template #icon>
      <PhDownloadSimple class="w-4 h-4" />
    </template>
    {{ t('processing.downloadButton') }}
  </AppButton>
</template>
