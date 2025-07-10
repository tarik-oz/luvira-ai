<script setup lang="ts">
import Header from './components/layout/Header.vue';
import Footer from './components/layout/Footer.vue';
</script>

<template>
  <div class="min-h-screen bg-gray-50">
    <Header />
    <div class="max-w-6xl mx-auto p-8 text-center">
      <h1 class="text-4xl font-bold text-gray-800 mb-4">Hair Segmentation App</h1>
      <p class="text-lg text-gray-600">Welcome to the hair segmentation application!</p>
    </div>
    <Footer />
  </div>
</template>
