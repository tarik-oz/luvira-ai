<script setup lang="ts">
import Header from './components/layout/Header.vue'
import Footer from './components/layout/Footer.vue'
import UploadView from './views/UploadView.vue'
import ProcessingView from './views/ProcessingView.vue'
import { useAppState } from './composables/useAppState'

const { currentView } = useAppState()
</script>

<template>
  <div class="h-screen bg-gray-50 flex flex-col overflow-hidden">
    <Header />

    <!-- Main Content -->
    <main class="flex-1 py-6 overflow-y-auto">
      <div class="max-w-6xl mx-auto px-4">
        
        <!-- Upload View -->
        <UploadView v-if="currentView === 'upload'" />

        <!-- Processing View -->
        <ProcessingView v-if="currentView === 'processing'" />
        
      </div>
    </main>

    <Footer />
  </div>
</template>
