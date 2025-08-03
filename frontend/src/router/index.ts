import { createRouter, createWebHistory } from 'vue-router'
import UploadView from '@/views/UploadView.vue'
import ProcessingView from '@/views/ProcessingView.vue'

const routes = [
  {
    path: '/',
    name: 'Upload',
    component: UploadView,
  },
  {
    path: '/color-tone-changer',
    name: 'ColorToneChanger',
    component: ProcessingView,
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
