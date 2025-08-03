<script setup lang="ts">
const props = defineProps<{
  fullWidth?: boolean
  class?: string
  disabled?: boolean
}>()
</script>

<template>
  <button
    :disabled="props.disabled"
    :class="[
      'inline-flex items-center justify-center gap-2 px-6 py-3 w-full rounded-lg font-semibold transition-colors duration-200 focus:outline-none',
      props.disabled
        ? 'bg-base-300 text-base-content/50 cursor-not-allowed'
        : 'bg-accent text-base-100 hover:bg-accent/70 active:bg-accent/90 cursor-pointer',
      props.fullWidth ? 'w-full' : '',
      props.class,
    ]"
    type="button"
  >
    <slot name="icon" />
    <span><slot /></span>
  </button>
</template>
