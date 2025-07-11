<script setup lang="ts">
import Header from './components/layout/Header.vue'
import Footer from './components/layout/Footer.vue'
import UploadSection from './components/ui/UploadSection.vue'
import SampleImages from './components/ui/SampleImages.vue'
import type { StaticSampleImage } from './data/sampleImages'

const handleFileSelect = (file: File) => {
  console.log('Selected file:', file.name)
}

const handleSampleImageSelect = async (sampleImage: StaticSampleImage) => {
  try {
    console.log('Selected sample image:', sampleImage.name)

    // Fetch the image from public folder and convert to File
    const response = await fetch(sampleImage.url)
    const blob = await response.blob()

    const file = new File([blob], sampleImage.filename, {
      type: blob.type || 'image/jpeg',
    })

    console.log('Created file from sample image:', file.name, file.size)

    // Handle the file the same way as uploaded files
    handleFileSelect(file)
  } catch (error) {
    console.error('Error loading sample image:', error)
  }
}
</script>

<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <Header />

    <!-- Main Content -->
    <main class="flex-1 py-12">
      <div class="max-w-6xl mx-auto px-4">
        <div class="text-center mb-8">
          <h1 class="text-4xl font-bold text-gray-800 mb-4">Try on different hair colors</h1>
          <p class="text-lg text-gray-600">Upload your photo and experiment with new hair colors</p>
        </div>

        <UploadSection @file-select="handleFileSelect" />

        <!-- Simple divider -->
        <div class="text-center my-6">
          <span class="text-gray-400 text-sm">or</span>
        </div>

        <!-- Sample images section -->
        <div class="text-center">
          <SampleImages @select-image="handleSampleImageSelect" />
        </div>
      </div>
    </main>

    <Footer />
  </div>
</template>
