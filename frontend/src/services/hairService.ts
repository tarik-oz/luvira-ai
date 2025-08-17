/**
 * Hair Service - Business logic for hair color processing
 * Combines API calls with state management
 */

import apiService from './apiService'
import { useAppState } from '../composables/useAppState'

class HairService {
  private activeController: AbortController | null = null
  private activeRequestId = 0
  private activeColor: string | null = null

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
      getCachedColorResult,
      setCurrentColorResult,
      setSessionError,
      setColorBaseResult,
      upsertColorTone,
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

    try {
      // Seed UI so TonePalette opens immediately; base overlay will replace soon
      setColorBaseResult(cacheKey, '')

      await this.startStream(sessionId.value, cacheKey, true, (name, url) => {
        if (name === 'base') setColorBaseResult(cacheKey, url)
        else upsertColorTone(cacheKey, name, url)
      })
    } catch (error: unknown) {
      console.error('❌ All-tones change failed:', error)
      const message = error instanceof Error ? error.message : String(error)
      if (message === 'SESSION_EXPIRED') {
        setSessionError('SESSION_EXPIRED')
        return
      }
      throw error instanceof Error ? error : new Error(message)
    }
  }

  /** Compose original + overlay to a single image and return a Blob URL */
  /** Apply tone locally by switching to the selected overlay (no server request) */
  async applyToneLocally(toneName: string | null) {
    const { currentColorResult, setProcessedImageToTone } = useAppState()
    if (!currentColorResult.value) return
    setProcessedImageToTone(toneName)
  }

  /** Ensure streaming is running for given color (used when a missing tone is clicked) */
  async ensureColorStream(colorName: string) {
    const { sessionId, setColorBaseResult, upsertColorTone } = useAppState()
    if (!sessionId.value) return
    // If already streaming same color, do nothing
    if (this.activeColor === colorName && this.activeController) return
    await this.startStream(sessionId.value, colorName, false, (name, url) => {
      if (name === 'base') setColorBaseResult(colorName, url)
      else upsertColorTone(colorName, name, url)
    })
  }

  /** Internal: start streaming; markProcessing controls global overlay spinner behavior */
  private async startStream(
    sessionId: string,
    colorName: string,
    markProcessing: boolean,
    onPartUrl: (name: string, url: string) => void,
  ) {
    const { setIsProcessing, selectedTone, getCachedColorResult } = useAppState()
    // Abort previous
    if (this.activeController) {
      try {
        this.activeController.abort()
      } catch {}
    }
    const requestId = ++this.activeRequestId
    this.activeColor = colorName
    const controller = new AbortController()
    this.activeController = controller
    if (markProcessing) setIsProcessing(true)

    try {
      await apiService.streamOverlays(
        sessionId,
        colorName,
        (name, bytes) => {
          if (requestId !== this.activeRequestId) return // stale
          const url = URL.createObjectURL(new Blob([bytes], { type: 'image/webp' }))
          onPartUrl(name, url)
          if (name === 'base' && markProcessing) {
            // If a non-base tone is selected and not yet ready, keep spinner
            const sel = selectedTone.value
            let ready = true
            if (sel) {
              const cached = getCachedColorResult(colorName)
              ready = Boolean(cached && cached.tones && cached.tones[sel])
            }
            if (ready) setIsProcessing(false)
          }
        },
        controller.signal,
      )
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      // Ignore stale/aborted
      if (requestId !== this.activeRequestId) return
      if (controller.signal.aborted) return
      // Surface network issues to caller so UI can show error/revert selection
      if (message === 'Failed to fetch' || /network/i.test(message)) {
        throw new Error('Failed to fetch')
      }
      // Propagate others (e.g., SESSION_EXPIRED)
      throw error instanceof Error ? error : new Error(message)
    } finally {
      if (requestId === this.activeRequestId) {
        this.activeController = null
        this.activeColor = null
        if (markProcessing) setIsProcessing(false)
      }
    }
  }
}

// Export singleton instance
export const hairService = new HairService()
export default hairService
