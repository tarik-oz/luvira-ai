/**
 * Global state management for the hair segmentation app
 */

import { ref, readonly } from 'vue'
// apiService import removed - now handled by hairService

export type ViewType = 'upload' | 'processing'

export interface ColorChangeResult {
  color: string
  originalColor: string // Ana renk adı (Purple, Black, vs)
  selectedTone: string | null // Seçilen ton (plum, onyx, vs)
  baseResult: string // Composed image URL (PNG/WebP)
  tones: { [tone: string]: string } // Mutable mapping for composed tone URLs
}

export interface ColorCache {
  [colorName: string]: ColorChangeResult
}

// Global state
const sessionId = ref<string | null>(null)
const uploadedImage = ref<string | null>(null)
const isUploading = ref(false)
const isProcessing = ref(false) // Loading state for color change
const currentColorResult = ref<ColorChangeResult | null>(null)
const processedImage = ref<string | null>(null)
const selectedTone = ref<string | null>(null)
const colorCache = ref<ColorCache>({})
const sessionError = ref<string | null>(null)
const processingError = ref<string | null>(null)
const colorToneStates = ref<Record<string, string | null>>({}) // Per-color tone selection state

export function useAppState() {
  const setSessionId = (id: string | null) => {
    sessionId.value = id
    // Clear tone states when new session starts (new image)
    if (id) {
      clearColorToneStates()
    }
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

  const setIsProcessing = (processing: boolean) => {
    isProcessing.value = processing
  }

  const setSessionError = (message: string | null) => {
    sessionError.value = message
  }

  const setProcessingError = (message: string | null) => {
    processingError.value = message
  }

  const setCurrentColorResult = (result: ColorChangeResult | null) => {
    if (result) {
      currentColorResult.value = result

      // Cache this color result with LRU (Last Recently Used) - max 5 colors
      const maxCacheSize = 5
      const cache = colorCache.value

      // Add new result
      cache[result.color] = result

      // If cache exceeds limit, remove oldest entries
      const keys = Object.keys(cache)
      if (keys.length > maxCacheSize) {
        const keysToRemove = keys.slice(0, keys.length - maxCacheSize)
        keysToRemove.forEach((key) => delete cache[key])
      }

      // Set processed image to base result
      processedImage.value = result.baseResult
      selectedTone.value = null
    } else {
      currentColorResult.value = null
      processedImage.value = null
      selectedTone.value = null
    }
  }

  // Incremental cache helpers for streaming
  const setColorBaseResult = (colorName: string, baseUrl: string) => {
    // Ensure cache entry
    const cache = colorCache.value
    const existing = cache[colorName]
    const colorResult: ColorChangeResult = existing
      ? { ...existing, baseResult: baseUrl || existing.baseResult }
      : {
          color: colorName,
          originalColor: colorName,
          selectedTone: null,
          baseResult: baseUrl,
          tones: {},
        }

    cache[colorName] = colorResult

    const priorTone = colorToneStates.value[colorName]
    const switchingToThisColor =
      !currentColorResult.value || currentColorResult.value.color !== colorName

    // Update tone state only if undefined (first time we see this color)
    if (priorTone === undefined) {
      colorToneStates.value[colorName] = null
    }

    // Make this color current
    currentColorResult.value = colorResult

    // Determine selectedTone without clobbering a user-chosen tone
    if (switchingToThisColor) {
      selectedTone.value = priorTone !== undefined ? priorTone : null
    }

    // Only override processed image with base if no tone is selected and we have a base URL
    if ((selectedTone.value === null || selectedTone.value === undefined) && baseUrl) {
      processedImage.value = baseUrl
    }
  }

  const upsertColorTone = (colorName: string, toneName: string, toneUrl: string) => {
    const cache = colorCache.value
    const existing = cache[colorName]
    if (!existing) return
    existing.tones[toneName] = toneUrl
    // If this color is active and selected tone matches, update view
    if (
      currentColorResult.value &&
      currentColorResult.value.color === colorName &&
      selectedTone.value === toneName
    ) {
      processedImage.value = toneUrl
    }
  }

  const setProcessedImageToTone = (toneName: string | null) => {
    if (!currentColorResult.value) return

    selectedTone.value = toneName

    if (toneName === null) {
      // Show base result (URL or data URI)
      processedImage.value = currentColorResult.value.baseResult
    } else if (currentColorResult.value.tones[toneName]) {
      // Show specific tone
      processedImage.value = currentColorResult.value.tones[toneName]
    }
  }

  const getCachedColorResult = (colorName: string): ColorChangeResult | null => {
    return colorCache.value[colorName] || null
  }

  const clearCache = () => {
    colorCache.value = {}
  }

  // Color-specific tone state management
  const getColorToneState = (colorName: string): string | null => {
    return colorToneStates.value[colorName] ?? null // Default to base (null) if not set
  }

  const setColorToneState = (colorName: string, toneName: string | null) => {
    colorToneStates.value[colorName] = toneName
  }

  const clearColorToneStates = () => {
    colorToneStates.value = {}
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
    isProcessing.value = false
    currentColorResult.value = null
    processedImage.value = null
    selectedTone.value = null
    clearCache()
    clearColorToneStates()
    sessionError.value = null
    processingError.value = null
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
    isProcessing: readonly(isProcessing),
    currentColorResult: readonly(currentColorResult),
    processedImage: readonly(processedImage),
    selectedTone: readonly(selectedTone),
    colorCache: readonly(colorCache),
    sessionError: readonly(sessionError),
    processingError: readonly(processingError),

    // Actions
    setSessionId,
    setUploadedImage,
    setIsUploading,
    setIsProcessing,
    setSessionError,
    setProcessingError,
    setCurrentColorResult,
    setProcessedImageToTone,
    setColorBaseResult,
    upsertColorTone,
    getCachedColorResult,
    clearCache,
    resetState,
    createImageUrl,

    // Color tone state management
    getColorToneState,
    setColorToneState,
    clearColorToneStates,
  }
}
