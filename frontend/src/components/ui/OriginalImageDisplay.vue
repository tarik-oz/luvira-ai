<script setup lang="ts">
interface Props {
  uploadedImage: string | null
  sessionId: string | null
}

defineProps<Props>()
</script>

<template>
  <div class="space-y-4">
    <h3 class="text-lg font-semibold text-gray-800">Original Image</h3>
    <div class="bg-white rounded-lg border-2 border-gray-200 p-4">
      <div v-if="uploadedImage" class="aspect-square w-full overflow-hidden rounded-lg bg-gray-100">
        <img :src="uploadedImage" alt="Uploaded image" class="w-full h-full object-cover" />
      </div>
      <div
        v-else
        class="aspect-square w-full flex items-center justify-center bg-gray-100 rounded-lg"
      >
        <p class="text-gray-500">No image uploaded</p>
      </div>
    </div>

    <!-- Session Info -->
    <div v-if="sessionId" class="bg-blue-50 border border-blue-200 rounded-lg p-3">
      <p class="text-sm text-blue-800">
        <span class="font-semibold">Session ID:</span> {{ sessionId }}
      </p>
      <p class="text-xs text-blue-600 mt-1">
        Hair mask has been generated and cached for fast color changes
      </p>
    </div>
  </div>
</template>
