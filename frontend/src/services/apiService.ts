/**
 * API Service for Hair Segmentation
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export interface UploadResponse {
  session_id: string
  message?: string
}

class ApiService {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  /**
   * Upload image and prepare for processing
   */
  async uploadAndPrepare(file: File): Promise<UploadResponse> {
    const formData = new FormData()
    formData.append('file', file)

    // Create AbortController for timeout
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 30000) // 30 second timeout

    try {
      const response = await fetch(`${this.baseUrl}/upload-and-prepare`, {
        method: 'POST',
        body: formData,
        signal: controller.signal, // Cancel request if it takes too long
      })

      clearTimeout(timeoutId)

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `API Error: ${response.status}`)
      }

      return await response.json()
    } catch (error) {
      clearTimeout(timeoutId)
      console.error('Upload API error:', error)
      // Propagate original error (TypeError => network, DOMException AbortError => timeout)
      throw error
    }
  }

  /**
   * Fetch overlays bundle (ZIP) with base + all tones (WEBP with alpha)
   */
  async fetchOverlaysBundle(sessionId: string, colorName: string): Promise<Blob> {
    const formData = new FormData()
    formData.append('color_name', colorName)
    const response = await fetch(`${this.baseUrl}/overlays-with-session/${sessionId}`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      // Mark SessionExpiredException specially for clean handling
      type Err = { detail?: string; error_code?: string }
      const errorData: Err = await response.json().catch(() => ({}) as Err)
      const msg =
        errorData?.error_code === 'SESSION_EXPIRED' ||
        /session.*expired/i.test(errorData?.detail || '')
          ? 'SESSION_EXPIRED'
          : errorData?.detail || `API error! status: ${response.status}`
      throw new Error(msg)
    }

    return await response.blob()
  }
}

// Export singleton instance
export const apiService = new ApiService()
export default apiService
