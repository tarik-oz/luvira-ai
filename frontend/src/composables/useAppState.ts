/**
 * App State Management
 */

import { ref, readonly } from 'vue'

export type ViewType = 'upload' | 'processing'

// Global state
const currentView = ref<ViewType>('upload')
const sessionId = ref<string | null>(null)
const uploadedImage = ref<string | null>(null)
const isUploading = ref(false)

export function useAppState() {
  const setCurrentView = (view: ViewType) => {
    currentView.value = view
  }

  const setSessionId = (id: string | null) => {
    sessionId.value = id
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

  const resetState = () => {
    // Cleanup image URL
    if (uploadedImage.value) {
      URL.revokeObjectURL(uploadedImage.value)
    }

    // Reset all state
    currentView.value = 'upload'
    sessionId.value = null
    uploadedImage.value = null
    isUploading.value = false
  }

  const createImageUrl = (file: File): string => {
    const imageUrl = URL.createObjectURL(file)
    setUploadedImage(imageUrl)
    return imageUrl
  }

  return {
    // Readonly state
    currentView: readonly(currentView),
    sessionId: readonly(sessionId),
    uploadedImage: readonly(uploadedImage),
    isUploading: readonly(isUploading),

    // Actions
    setCurrentView,
    setSessionId,
    setUploadedImage,
    setIsUploading,
    resetState,
    createImageUrl,
  }
}
