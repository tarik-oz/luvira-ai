/**
 * Global state management for the hair segmentation app
 */

import { ref, readonly } from 'vue'

export type ViewType = 'upload' | 'processing'

export interface ColorChangeResult {
  color: string
  originalColor: string
  selectedTone: string | null
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
const showMobileToneBar = ref(false) // Controls mobile view: ColorPalette vs MobileColorToneBar

export function useAppState() {
  const isBlobUrl = (url: string | null | undefined): boolean => !!url && url.startsWith('blob:')
  const revokeIfBlob = (url: string | null | undefined) => {
    if (isBlobUrl(url)) {
      try {
        URL.revokeObjectURL(url as string)
      } catch {}
    }
  }

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

      // Restore view according to per-color tone state if exists
      const toneState = colorToneStates.value[result.originalColor]
      selectedTone.value = toneState !== undefined ? toneState : null
      if (selectedTone.value && result.tones[selectedTone.value]) {
        processedImage.value = result.tones[selectedTone.value]
      } else {
        processedImage.value = result.baseResult
      }
    } else {
      // Revoke current processed image if it is a blob URL
      revokeIfBlob(processedImage.value)
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
    // Revoke old base blob if replaced
    if (existing && existing.baseResult && existing.baseResult !== baseUrl) {
      revokeIfBlob(existing.baseResult)
    }
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
    // Revoke previous tone URL if replaced
    const prev = existing.tones[toneName]
    if (prev && prev !== toneUrl) revokeIfBlob(prev)
    existing.tones[toneName] = toneUrl
    // If this color is active and selected tone matches, update view
    if (
      currentColorResult.value &&
      currentColorResult.value.color === colorName &&
      selectedTone.value === toneName
    ) {
      // Revoke previously displayed image if it was a blob and will be replaced
      if (processedImage.value && processedImage.value !== toneUrl) {
        revokeIfBlob(processedImage.value)
      }
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
    // Revoke any blob URLs stored in cache
    const cache = colorCache.value
    for (const key of Object.keys(cache)) {
      const entry = cache[key]
      revokeIfBlob(entry.baseResult)
      for (const toneKey of Object.keys(entry.tones)) {
        revokeIfBlob(entry.tones[toneKey])
      }
    }
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

  const setShowMobileToneBar = (show: boolean) => {
    showMobileToneBar.value = show
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
    revokeIfBlob(processedImage.value)
    processedImage.value = null
    selectedTone.value = null
    clearCache()
    clearColorToneStates()
    sessionError.value = null
    processingError.value = null
    showMobileToneBar.value = false
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
    showMobileToneBar: readonly(showMobileToneBar),

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

    // Mobile view state
    setShowMobileToneBar,
  }
}
