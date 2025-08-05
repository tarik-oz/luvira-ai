/**
 * Hair Service - Business logic for hair color processing
 * Combines API calls with state management
 */

import apiService from './apiService'
import { useAppState } from '../composables/useAppState'
import type { ColorChangeResult } from '../composables/useAppState'

class HairService {
  /**
   * Upload image and prepare session
   */
  async uploadImage(file: File, originalImageUrl?: string) {
    const { setIsUploading, setUploadedImage, setSessionId, createImageUrl } = useAppState()

    setIsUploading(true)
    try {
      // Set image display
      if (originalImageUrl) {
        setUploadedImage(originalImageUrl)
      } else {
        createImageUrl(file)
      }

      // Call API
      const response = await apiService.uploadAndPrepare(file)
      setSessionId(response.session_id)

      return response.session_id
    } catch (error) {
      throw error
    } finally {
      setIsUploading(false)
    }
  }

  /**
   * Change hair color with caching
   */
  async changeHairColor(colorName: string, tone?: string) {
    const { sessionId, setIsProcessing, getCachedColorResult, setCurrentColorResult } =
      useAppState()

    if (!sessionId.value) {
      throw new Error('No session ID available')
    }

    // Create cache key for color + tone combination
    const cacheKey = tone ? `${colorName}_${tone}` : colorName

    // Check cache first with tone-specific key
    const cachedResult = getCachedColorResult(cacheKey)
    if (cachedResult) {
      console.log('🗂️ Using cached result for:', cacheKey)
      setCurrentColorResult(cachedResult)
      return
    }

    setIsProcessing(true)
    try {
      console.log('🎨 Requesting color change:', colorName, tone ? `with tone: ${tone}` : '(base)')
      const blob = await apiService.changeHairColor(sessionId.value, colorName, tone)

      // Convert blob to base64
      const base64 = await this.blobToBase64(blob)

      // Create color result with tone-specific cache key
      const colorResult: ColorChangeResult = {
        color: cacheKey, // Use cache key for caching
        originalColor: colorName, // Keep original color name
        selectedTone: tone || null, // Keep selected tone
        baseResult: base64.replace('data:image/png;base64,', ''),
        tones: tone ? { [tone]: base64.replace('data:image/png;base64,', '') } : {},
      }

      setCurrentColorResult(colorResult)
      console.log('✅ Color change completed:', cacheKey)
    } catch (error) {
      console.error('❌ Color change failed:', error)
      throw error
    } finally {
      setIsProcessing(false)
    }
  }

  /**
   * Helper: Convert blob to base64
   */
  private blobToBase64(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onloadend = () => resolve(reader.result as string)
      reader.onerror = reject
      reader.readAsDataURL(blob)
    })
  }
}

// Export singleton instance
export const hairService = new HairService()
export default hairService
