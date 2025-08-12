/**
 * API Service for Hair Segmentation
 */

const API_BASE_URL = 'http://localhost:8000'

export interface UploadResponse {
  session_id: string
  message?: string
}

export interface ApiError {
  detail: string
  status?: number
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

      throw new Error('UPLOAD_FAILED')
    }
  }

  /**
   * Fetch overlays bundle (ZIP) with base + all tones (WEBP with alpha)
   */
  async fetchOverlaysBundle(sessionId: string, colorName: string): Promise<Blob> {
    const formData = new FormData()
    formData.append('color_name', colorName)
    try {
      const response = await fetch(`${this.baseUrl}/overlays-with-session/${sessionId}`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `API error! status: ${response.status}`)
      }

      return await response.blob()
    } catch (error) {
      console.error('Overlays bundle API error:', error)
      throw error
    }
  }
}

// Export singleton instance
export const apiService = new ApiService()
export default apiService
