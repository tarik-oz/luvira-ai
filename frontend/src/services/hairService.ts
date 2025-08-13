/**
 * Hair Service - Business logic for hair color processing
 * Combines API calls with state management
 */

import apiService from './apiService'
import { useAppState } from '../composables/useAppState'
import type { ColorChangeResult } from '../composables/useAppState'
import type { JSZipInstance, JSZipFolder, JSZipObject } from 'jszip'

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
      // Map common network/timeout errors for UI to display a better message
      const message = error instanceof Error ? error.message : String(error)
      if (message === 'Failed to fetch' || /network/i.test(message)) {
        throw new Error('NETWORK_ERROR')
      }
      if (message === 'AbortError') {
        throw new Error('TIMEOUT')
      }
      throw error
    } finally {
      setIsUploading(false)
    }
  }

  /**
   * Fetch base + all tones overlays bundle (ZIP) using session and cache the whole set.
   * Caches per base color key (e.g., "Blonde").
   */
  async changeHairColorAllTones(colorName: string) {
    const {
      sessionId,
      setIsProcessing,
      getCachedColorResult,
      setCurrentColorResult,
      setSessionError,
    } = useAppState()

    if (!sessionId.value) {
      throw new Error('No session ID available')
    }

    const cacheKey = colorName // cache per base color
    const cached = getCachedColorResult(cacheKey)
    if (cached) {
      setCurrentColorResult(cached)
      return
    }

    setIsProcessing(true)
    try {
      // Fetch ZIP
      const zipBlob = await apiService.fetchOverlaysBundle(sessionId.value, colorName)
      const arrayBuf = await zipBlob.arrayBuffer()

      // Lazy load JSZip
      const JSZipMod = await import('jszip')
      const zip = await (
        JSZipMod.default as {
          loadAsync: (data: ArrayBuffer | Uint8Array) => Promise<JSZipInstance>
        }
      ).loadAsync(arrayBuf)

      // Helper to convert Uint8Array to data URL (WEBP)
      const u8ToDataUrl = (u8: Uint8Array) => {
        const blob = new Blob([u8], { type: 'image/webp' })
        return URL.createObjectURL(blob)
      }

      // Base overlay
      let baseOverlayUrl = ''
      if (zip.file('base.webp')) {
        const baseU8 = await zip.file('base.webp')!.async('uint8array')
        baseOverlayUrl = u8ToDataUrl(baseU8)
      }

      // Tones
      const tones: Record<string, string> = {}
      const tonesFolder = zip.folder('tones') as JSZipFolder | null
      const toneFiles: Array<Promise<void>> = []
      tonesFolder?.forEach((relPath: string, file: JSZipObject) => {
        if (relPath.endsWith('.webp')) {
          const basename = relPath.split('/').pop() || relPath
          const toneName = basename.replace('.webp', '')
          toneFiles.push(
            file.async('uint8array').then((u8: Uint8Array) => {
              tones[toneName] = u8ToDataUrl(u8)
            }),
          )
        }
      })
      await Promise.all(toneFiles)

      // Compose base overlay over original for immediate display
      const composedBaseUrl = baseOverlayUrl

      // Build result and cache
      const colorResult: ColorChangeResult = {
        color: cacheKey,
        originalColor: colorName,
        selectedTone: null,
        // Store composed base for display
        baseResult: composedBaseUrl,
        tones,
      }

      setCurrentColorResult(colorResult)
    } catch (error: unknown) {
      console.error('❌ All-tones change failed:', error)
      const message = error instanceof Error ? error.message : String(error)
      if (message === 'SESSION_EXPIRED') {
        setSessionError('SESSION_EXPIRED')
        return
      }
      throw error instanceof Error ? error : new Error(message)
    } finally {
      setIsProcessing(false)
    }
  }

  /** Compose original + overlay to a single image and return a Blob URL */
  /** Apply tone locally by switching to the selected overlay (no server request) */
  async applyToneLocally(toneName: string | null) {
    const { currentColorResult, setProcessedImageToTone } = useAppState()
    if (!currentColorResult.value) return
    setProcessedImageToTone(toneName)
  }
}

// Export singleton instance
export const hairService = new HairService()
export default hairService
