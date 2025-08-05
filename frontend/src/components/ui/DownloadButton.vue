<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import { useAppState } from '../../composables/useAppState'
import AppButton from './AppButton.vue'
import { PhDownloadSimple } from '@phosphor-icons/vue'

const { t } = useI18n()
const { processedImage, currentColorResult } = useAppState()

const downloadImage = () => {
  if (!processedImage.value) {
    console.warn('No processed image to download')
    return
  }

  try {
    // Create download filename
    const fileName = generateFileName()

    // Create blob from base64 and download
    const base64Data = processedImage.value.split(',')[1] // Remove data:image/png;base64,
    const byteCharacters = atob(base64Data)
    const byteNumbers = new Array(byteCharacters.length)

    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i)
    }

    const byteArray = new Uint8Array(byteNumbers)
    const blob = new Blob([byteArray], { type: 'image/png' })

    // Create download link
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = fileName
    document.body.appendChild(link)
    link.click()

    // Cleanup
    document.body.removeChild(link)
    URL.revokeObjectURL(url)

    console.log('✅ Image downloaded:', fileName)
  } catch (error) {
    console.error('❌ Download failed:', error)
  }
}

const generateFileName = (): string => {
  if (!currentColorResult.value) {
    return 'LuviraAI_Hair.png'
  }

  const { originalColor, selectedTone } = currentColorResult.value
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

  const colorName = (colorNames[currentLocale] as any)[originalColor] || originalColor
  const hairWord = currentLocale === 'tr' ? 'Sac' : 'Hair'

  // Build filename: LuviraAI_Color_Tone_Hair.png
  let fileName = 'LuviraAI'
  fileName += `_${colorName}`

  if (selectedTone) {
    // Tones always in English (capitalize first letter)
    const toneCapitalized = selectedTone.charAt(0).toUpperCase() + selectedTone.slice(1)
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
