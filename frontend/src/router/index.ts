import { createRouter, createWebHistory } from 'vue-router'
import { watch } from 'vue'
import UploadView from '@/views/UploadView.vue'
import ProcessingView from '@/views/ProcessingView.vue'
import NotFoundView from '@/views/NotFoundView.vue'
import { useAppState } from '@/composables/useAppState'
import type { NavigationGuardNext, RouteLocationNormalized } from 'vue-router'
import { applySeo, type RouteSeoMeta } from '@/services/seo'
import i18n from '@/i18n'

type RouteMeta = RouteSeoMeta

const routes = [
  {
    path: '/',
    name: 'Upload',
    component: UploadView,
    meta: {
      titleKey: 'meta.home.title',
      descriptionKey: 'meta.home.description',
    } as RouteMeta,
  },
  {
    path: '/hair-color-editor',
    name: 'HairColorEditor',
    component: ProcessingView,
    meta: {
      titleKey: 'meta.editor.title',
      descriptionKey: 'meta.editor.description',
    } as RouteMeta,
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
    meta: {
      titleKey: 'meta.notFound.title',
      descriptionKey: 'meta.notFound.description',
    } as RouteMeta,
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior(to, from, savedPosition) {
    if (savedPosition) return savedPosition
    if (to.hash) return { el: to.hash, behavior: 'smooth' }
    return { top: 0, behavior: 'smooth' }
  },
})

// Apply meta title/description on navigation
router.afterEach((to) => {
  const meta = (to.meta || {}) as RouteMeta
  applySeo(meta)
})

// Update SEO when language changes
watch(
  () => i18n.global.locale.value,
  (newLocale: string) => {
    document.documentElement.setAttribute('lang', newLocale)
    const route = router.currentRoute.value
    applySeo((route.meta || {}) as RouteMeta)
  },
)

export default router
