/**
 * Shared image validation and resizing utilities
 */

export const MAX_FILE_SIZE = 10 * 1024 * 1024 // 10 MB
export const ALLOWED_FILE_TYPES = ['image/jpeg', 'image/png', 'image/jpg']
export const MAX_DIMENSION = 1600
export const MIN_DIMENSION = 400

export interface ImageValidationOptions {
  maxFileSize: number
  allowedTypes: string[]
  maxDimension: number
  minDimension: number
  outputType?: string
  outputQuality?: number
  /**
   * If true and input is a File, keep its mime type for output when resizing.
   * When false, outputType is used.
   */
  preserveInputTypeForFiles: boolean
}

const defaultOptions: ImageValidationOptions = {
  maxFileSize: MAX_FILE_SIZE,
  allowedTypes: ALLOWED_FILE_TYPES,
  maxDimension: MAX_DIMENSION,
  minDimension: MIN_DIMENSION,
  outputType: 'image/jpeg',
  outputQuality: 0.9,
  preserveInputTypeForFiles: true,
}

function loadImageFromBlob(blob: Blob): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(blob)
    const img = new Image()
    img.onload = () => {
      URL.revokeObjectURL(objectUrl)
      resolve(img)
    }
    img.onerror = () => {
      URL.revokeObjectURL(objectUrl)
      reject(new Error('DECODE_ERROR'))
    }
    img.src = objectUrl
  })
}

/**
 * Validate and resize an image.
 * Throws Error with message codes: ZERO_SIZE | INVALID_TYPE | MAX_SIZE | TOO_SMALL | DECODE_ERROR
 */
export async function validateAndResizeImage(
  input: File | Blob,
  opts?: Partial<ImageValidationOptions>,
): Promise<File> {
  const options = { ...defaultOptions, ...(opts || {}) }

  const size = input.size
  if (size === 0) {
    throw new Error('ZERO_SIZE')
  }
  if (size > options.maxFileSize) {
    throw new Error('MAX_SIZE')
  }

  // Validate type (Files and Blobs both have a type)
  const type = input.type
  if (type && options.allowedTypes.length > 0) {
    if (!options.allowedTypes.includes(type)) {
      throw new Error('INVALID_TYPE')
    }
  }

  // Decode to inspect dimensions
  const img = await loadImageFromBlob(input)
  const width = img.naturalWidth || img.width
  const height = img.naturalHeight || img.height

  if (width < options.minDimension || height < options.minDimension) {
    throw new Error('TOO_SMALL')
  }

  // No resizing needed
  if (width <= options.maxDimension && height <= options.maxDimension) {
    // If it's already a File, return as-is; if Blob, wrap into a File
    if (input instanceof File) return input
    const name = 'image.jpg'
    return new File([input], name, { type: type || options.outputType, lastModified: Date.now() })
  }

  // Resize while preserving aspect ratio
  let newWidth = width
  let newHeight = height
  if (width > height) {
    if (width > options.maxDimension) {
      newWidth = options.maxDimension
      newHeight = Math.round((height * options.maxDimension) / width)
    }
  } else {
    if (height > options.maxDimension) {
      newHeight = options.maxDimension
      newWidth = Math.round((width * options.maxDimension) / height)
    }
  }

  const canvas = document.createElement('canvas')
  canvas.width = newWidth
  canvas.height = newHeight
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    // Fallback: return original if canvas not available
    if (input instanceof File) return input
    return new File([input], 'image.jpg', {
      type: type || options.outputType,
      lastModified: Date.now(),
    })
  }
  ctx.drawImage(img, 0, 0, newWidth, newHeight)

  const outputMime =
    input instanceof File && options.preserveInputTypeForFiles ? input.type : options.outputType

  const blob: Blob = await new Promise((resolve, reject) => {
    canvas.toBlob(
      (b) => {
        if (b) resolve(b)
        else reject(new Error('DECODE_ERROR'))
      },
      outputMime,
      options.outputQuality,
    )
  })

  const filename = input instanceof File ? input.name : 'image.jpg'
  return new File([blob], filename, { type: outputMime || 'image/jpeg', lastModified: Date.now() })
}
