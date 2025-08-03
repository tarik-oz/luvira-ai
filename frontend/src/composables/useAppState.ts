/**
 * Global state management for the hair segmentation app
 */

import { ref, readonly } from 'vue'
import apiService from '../services/apiService'

export type ViewType = 'upload' | 'processing'

export interface ColorChangeResult {
  color: string
  baseResult: string // base64 image
  tones: Record<string, string> // tone name -> base64 image
}

export interface ColorCache {
  [colorName: string]: ColorChangeResult
}

// Global state
const sessionId = ref<string | null>(null)
const uploadedImage = ref<string | null>(null)
const isUploading = ref(false)
const currentColorResult = ref<ColorChangeResult | null>(null)
const processedImage = ref<string | null>(null)
const selectedTone = ref<string | null>(null)
const colorCache = ref<ColorCache>({})

export function useAppState() {
  const uploadFileAndSetSession = async (file: File, originalImageUrl?: string) => {
    setIsUploading(true)
    try {
      // If original URL exists (for sample images), save it
      if (originalImageUrl) {
        setUploadedImage(originalImageUrl)
      } else {
        // Create blob URL for normal file upload
        createImageUrl(file)
      }

      const response = await apiService.uploadAndPrepare(file)
      setSessionId(response.session_id)
      return response.session_id
    } catch (error) {
      throw error
    } finally {
      setIsUploading(false)
    }
  }
  const setSessionId = (id: string | null) => {
    sessionId.value = id
  }

  const setUploadedImage = (imageUrl: string | null) => {
    // Cleanup previous image URL to prevent memory leaks
    if (uploadedImage.value && uploadedImage.value !== imageUrl) {
      URL.revokeObjectURL(uploadedImage.value)
    }
    uploadedImage.value = imageUrl
  }

  const setIsUploading = (loading: boolean) => {
    isUploading.value = loading
  }

  const setCurrentColorResult = (result: ColorChangeResult | null) => {
    if (result) {
      currentColorResult.value = result
      // Cache this color result
      colorCache.value[result.color] = result
      // Set processed image to base result
      processedImage.value = `data:image/png;base64,${result.baseResult}`
      selectedTone.value = null
    } else {
      currentColorResult.value = null
      processedImage.value = null
      selectedTone.value = null
    }
  }

  const setProcessedImageToTone = (toneName: string | null) => {
    if (!currentColorResult.value) return

    selectedTone.value = toneName

    if (toneName === null) {
      // Show base result
      processedImage.value = `data:image/png;base64,${currentColorResult.value.baseResult}`
    } else if (currentColorResult.value.tones[toneName]) {
      // Show specific tone
      processedImage.value = `data:image/png;base64,${currentColorResult.value.tones[toneName]}`
    }
  }

  const getCachedColorResult = (colorName: string): ColorChangeResult | null => {
    return colorCache.value[colorName] || null
  }

  const clearCache = () => {
    colorCache.value = {}
  }

  const resetState = () => {
    // Cleanup image URL
    if (uploadedImage.value) {
      URL.revokeObjectURL(uploadedImage.value)
    }

    // Reset all state
    sessionId.value = null
    uploadedImage.value = null
    isUploading.value = false
    currentColorResult.value = null
    processedImage.value = null
    selectedTone.value = null
    clearCache()
  }

  const createImageUrl = (file: File): string => {
    const imageUrl = URL.createObjectURL(file)
    setUploadedImage(imageUrl)
    return imageUrl
  }

  return {
    // Readonly state
    sessionId: readonly(sessionId),
    uploadedImage: readonly(uploadedImage),
    isUploading: readonly(isUploading),
    currentColorResult: readonly(currentColorResult),
    processedImage: readonly(processedImage),
    selectedTone: readonly(selectedTone),

    // Actions
    setUploadedImage,
    setIsUploading,
    setCurrentColorResult,
    setProcessedImageToTone,
    getCachedColorResult,
    clearCache,
    resetState,
    createImageUrl,
    uploadFileAndSetSession,
  }
}
