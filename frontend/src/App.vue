<script setup lang="ts">
import Header from './components/layout/AppHeader.vue'
import Footer from './components/layout/AppFooter.vue'
</script>

<template>
  <div class="flex h-screen flex-col">
    <Header />

    <!-- Main Content -->
    <main class="bg-base-100 flex-1 overflow-y-auto py-6">
      <div class="mx-auto mt-16 mb-4 max-w-7xl px-4 pt-10">
        <router-view />
      </div>
    </main>

    <Footer />
  </div>
</template>
