import { createRouter, createWebHistory } from 'vue-router'
import UploadView from '@/views/UploadView.vue'
import ProcessingView from '@/views/ProcessingView.vue'
import NotFoundView from '@/views/NotFoundView.vue'
import { useAppState } from '@/composables/useAppState'
import type { NavigationGuardNext, RouteLocationNormalized } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'Upload',
    component: UploadView,
  },
  {
    path: '/hair-color-editor',
    name: 'HairColorEditor',
    component: ProcessingView,
    beforeEnter: (
      to: RouteLocationNormalized,
      from: RouteLocationNormalized,
      next: NavigationGuardNext,
    ) => {
      // Check if we're in a browser environment
      if (typeof window === 'undefined') {
        next()
        return
      }

      // Get app state to check session and image
      const { sessionId, uploadedImage } = useAppState()

      // If no session or image, redirect to home
      if (!sessionId.value || !uploadedImage.value) {
        next('/')
        return
      }

      // Allow access
      next()
    },
  },
  // 404 catch-all route - must be last
  {
    path: '/:pathMatch(.*)*',
    name: 'NotFound',
    component: NotFoundView,
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
