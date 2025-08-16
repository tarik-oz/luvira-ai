<script setup lang="ts">
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import AppButton from '@/components/ui/AppButton.vue'
import { PhHouse, PhWarning, PhArrowLeft } from '@phosphor-icons/vue'

const router = useRouter()
const { t } = useI18n()

const goHome = () => {
  router.push('/')
}
</script>

<template>
  <div class="flex min-h-[60vh] items-center justify-center px-4 py-20">
    <div class="mx-auto max-w-md text-center">
      <!-- 404 Icon -->
      <div class="mb-6 flex justify-center">
        <div class="bg-base-content/10 rounded-full p-4">
          <PhWarning class="text-accent h-16 w-16" weight="duotone" />
        </div>
      </div>

      <!-- 404 Text -->
      <h1 class="text-base-content mb-3 text-5xl font-bold">404</h1>
      <h2 class="text-base-content/80 mb-3 text-xl font-semibold">
        {{ t('notFound.title') }}
      </h2>
      <p class="text-base-content/60 mb-6 text-base">
        {{ t('notFound.description') }}
      </p>

      <!-- Action Buttons - Flexible width for different text lengths -->
      <div class="mx-auto flex w-full max-w-sm flex-col justify-center gap-3 sm:flex-row">
        <AppButton
          @click="goHome"
          class="bg-primary hover:bg-primary/80 text-primary-content min-w-0 flex-1 px-6 py-3"
        >
          <template #icon>
            <PhHouse class="h-4 w-4" />
          </template>
          {{ t('notFound.goHome') }}
        </AppButton>

        <AppButton
          @click="router.go(-1)"
          class="bg-base-content/10 hover:bg-base-content/20 text-base-content min-w-0 flex-1 px-6 py-3"
        >
          <template #icon>
            <PhArrowLeft class="h-4 w-4" />
          </template>
          {{ t('notFound.goBack') }}
        </AppButton>
      </div>

      <!-- Additional Info -->
      <div class="text-base-content/40 mt-8 text-xs">
        <p>{{ t('notFound.help') }}</p>
        <p class="mt-1">
          <a href="mailto:tarikoz201007@gmail.com" class="text-accent hover:underline">
            {{ t('notFound.contact') }}
          </a>
        </p>
      </div>
    </div>
  </div>
</template>
