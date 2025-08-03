<script setup lang="ts">
import Header from './components/layout/Header.vue'
import Footer from './components/layout/Footer.vue'
</script>

<template>
  <div class="flex flex-col h-screen">
    <Header />

    <!-- Main Content -->
    <main class="flex-1 overflow-y-auto py-6 bg-base-100">
      <div class="max-w-7xl mx-auto px-4 mt-16 pt-10 mb-4">
        <router-view />
      </div>
    </main>

    <Footer />
  </div>
</template>
