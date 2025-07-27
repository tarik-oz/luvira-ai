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

    try {
      const response = await fetch(`${this.baseUrl}/upload-and-prepare`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      return await response.json()
    } catch (error) {
      console.error('Upload API error:', error)
      throw error
    }
  }

  /**
   * Change hair color with session
   */
  async changeHairColor(sessionId: string, colorName: string, tone?: string): Promise<Blob> {
    const params = new URLSearchParams({
      session_id: sessionId,
      color_name: colorName,
    })

    if (tone) {
      params.append('tone', tone)
    }

    try {
      const response = await fetch(`${this.baseUrl}/change-color-session?${params}`, {
        method: 'POST',
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      return await response.blob()
    } catch (error) {
      console.error('Color change API error:', error)
      throw error
    }
  }

  /**
   * Get available colors
   */
  async getAvailableColors(): Promise<Array<{ name: string; rgb: [number, number, number] }>> {
    try {
      const response = await fetch(`${this.baseUrl}/available-colors`)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      return data.colors_with_rgb || []
    } catch (error) {
      console.error('Get colors API error:', error)
      throw error
    }
  }

  /**
   * Get available tones for a color
   */
  async getAvailableTones(colorName: string): Promise<string[]> {
    try {
      const response = await fetch(`${this.baseUrl}/available-tones/${colorName}`)

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      return data.tones || []
    } catch (error) {
      console.error('Get tones API error:', error)
      throw error
    }
  }

  /**
   * Get hair color change with all tones using session
   */
  async changeHairColorAllTones(sessionId: string, colorName: string): Promise<{
    success: boolean
    color: string
    session_id: string
    base_result: string
    tones: Record<string, string>
  }> {
    const formData = new FormData()
    formData.append('color_name', colorName)

    try {
      const response = await fetch(`${this.baseUrl}/change-hair-color-all-tones-fast/${sessionId}`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      return await response.json()
    } catch (error) {
      console.error('Change hair color all tones API error:', error)
      throw error
    }
  }
}

// Export singleton instance
export const apiService = new ApiService()
export default apiService
