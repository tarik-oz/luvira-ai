<script setup lang="ts">
import Header from './components/layout/AppHeader.vue'
import Footer from './components/layout/AppFooter.vue'
import { useI18n } from 'vue-i18n'
const { t } = useI18n()
</script>

<template>
  <div class="flex min-h-screen flex-col">
    <a href="#main" class="sr-only focus:not-sr-only focus:underline">{{
      t('a11y.skipToContent')
    }}</a>
    <header>
      <Header />
    </header>

    <main id="main" class="bg-base-100 flex-1 overflow-y-auto py-6" role="main">
      <div class="mx-auto mt-10 mb-4 max-w-7xl px-4 pt-4 md:mt-12 md:pt-10">
        <router-view />
      </div>
    </main>

    <footer>
      <Footer />
    </footer>
  </div>
</template>
