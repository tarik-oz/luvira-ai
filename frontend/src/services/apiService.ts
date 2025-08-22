/**
 * API Service for Hair Segmentation
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export interface UploadResponse {
  session_id: string
  message?: string
}

export interface ApiErrorPayload {
  detail?: string
  error_code?: string
  extra?: Record<string, unknown>
}

class ApiService {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  /**
   * Upload image and prepare for processing
   */
  async uploadAndPrepare(file: File, source: string): Promise<UploadResponse> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('source', source)

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
        const errorData: ApiErrorPayload = await response
          .json()
          .catch(() => ({}) as ApiErrorPayload)
        const err: Error & { error_code?: string } = new Error(
          errorData.detail || `API Error: ${response.status}`,
        ) as Error & { error_code?: string }
        err.error_code = errorData.error_code
        throw err
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
   * Stream overlays (multipart/mixed): calls onPart(name, bytes) as each part arrives.
   */
  async streamOverlays(
    sessionId: string,
    colorName: string,
    onPart: (name: string, bytes: Uint8Array, headers: Record<string, string>) => void,
    signal?: AbortSignal,
  ): Promise<void> {
    const formData = new FormData()
    formData.append('color_name', colorName)

    const response = await fetch(`${this.baseUrl}/overlays-with-session/${sessionId}`, {
      method: 'POST',
      body: formData,
      signal,
    })

    if (!response.ok || !response.body) {
      try {
        const data = await response.json()
        if (data?.error_code === 'SESSION_EXPIRED') throw new Error('SESSION_EXPIRED')
        throw new Error(data?.detail || `API error! status: ${response.status}`)
      } catch {
        throw new Error('Failed to fetch')
      }
    }

    const ct = response.headers.get('Content-Type') || ''
    const boundaryMatch = /boundary=([^;]+)/i.exec(ct)
    if (!boundaryMatch) throw new Error('Invalid stream: boundary not found')
    const boundary = boundaryMatch[1]

    const boundaryBytes = toBytes(`--${boundary}`)
    const boundaryBytesCRLF = toBytes(`\r\n--${boundary}`)
    const endBoundaryBytes = toBytes(`--${boundary}--`)
    const crlfcrlf = toBytes(`\r\n\r\n`)

    const reader = response.body.getReader()
    let buffer = new Uint8Array(0)
    let done = false

    // locate first boundary
    while (!done) {
      const idx = indexOf(buffer, boundaryBytes, 0)
      if (idx >= 0) {
        buffer = buffer.slice(idx + boundaryBytes.length)
        break
      }
      const { value, done: d } = await reader.read()
      if (d) break
      buffer = concat(buffer, value!)
    }

    while (!done) {
      // parse headers
      let headerEndIdx = indexOf(buffer, crlfcrlf, 0)
      while (headerEndIdx < 0) {
        const { value, done: d } = await reader.read()
        if (d) {
          done = true
          break
        }
        buffer = concat(buffer, value!)
        headerEndIdx = indexOf(buffer, crlfcrlf, 0)
      }
      if (done) break

      const headerBytes = buffer.slice(0, headerEndIdx)
      const headersText = new TextDecoder().decode(headerBytes)
      const headers = parseHeaders(headersText)
      buffer = buffer.slice(headerEndIdx + crlfcrlf.length)

      // detect JSON part (session expired)
      const isJson = /application\/json/i.test(headers['content-type'] || '')

      // find next boundary
      let partEndIdx = indexOf(buffer, boundaryBytesCRLF, 0)
      let finalBoundary = false
      while (partEndIdx < 0) {
        const endIdx = indexOf(buffer, endBoundaryBytes, 0)
        if (endIdx >= 0) {
          partEndIdx = endIdx
          finalBoundary = true
          break
        }
        const { value, done: d } = await reader.read()
        if (d) {
          partEndIdx = buffer.length
          done = true
          break
        }
        buffer = concat(buffer, value!)
        partEndIdx = indexOf(buffer, boundaryBytesCRLF, 0)
      }

      const partBytes = buffer.slice(0, partEndIdx)
      buffer = buffer.slice(
        partEndIdx + (finalBoundary ? endBoundaryBytes.length : boundaryBytesCRLF.length),
      )

      if (isJson) {
        const text = new TextDecoder().decode(partBytes)
        try {
          const data = JSON.parse(text)
          if (data?.error_code === 'SESSION_EXPIRED') throw new Error('SESSION_EXPIRED')
          throw new Error(data?.detail || 'STREAM_ERROR')
        } catch (e) {
          throw e instanceof Error ? e : new Error('STREAM_ERROR')
        }
      }

      const disp = headers['content-disposition'] || ''
      const name = parseNameFromDisposition(disp) || 'part'
      onPart(name, partBytes, headers)

      if (finalBoundary) break
    }
  }
}

function toBytes(text: string): Uint8Array {
  return new TextEncoder().encode(text)
}

function concat(a: Uint8Array, b: Uint8Array): Uint8Array {
  const out = new Uint8Array(a.length + b.length)
  out.set(a, 0)
  out.set(b, a.length)
  return out
}

function indexOf(hay: Uint8Array, needle: Uint8Array, from = 0): number {
  if (needle.length === 0) return -1
  outer: for (let i = from; i <= hay.length - needle.length; i++) {
    for (let j = 0; j < needle.length; j++) {
      if (hay[i + j] !== needle[j]) continue outer
    }
    return i
  }
  return -1
}

function parseHeaders(text: string): Record<string, string> {
  const lines = text.split(/\r?\n/)
  const headers: Record<string, string> = {}
  for (const line of lines) {
    const idx = line.indexOf(':')
    if (idx > -1) {
      const k = line.slice(0, idx).trim().toLowerCase()
      const v = line.slice(idx + 1).trim()
      headers[k] = v
    }
  }
  return headers
}

function parseNameFromDisposition(disposition: string): string | null {
  const m = /name=\"([^\"]+)\"/i.exec(disposition)
  return m ? m[1] : null
}

// Export singleton instance
export const apiService = new ApiService()
export default apiService
