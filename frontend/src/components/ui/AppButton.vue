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
      'focus-visible:ring-primary inline-flex w-full transform items-center justify-center gap-2 rounded-lg px-6 py-3 font-semibold transition-colors duration-200 focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:outline-none active:brightness-95 motion-safe:active:translate-y-[1px] motion-safe:active:scale-[0.99]',
      props.disabled
        ? 'bg-base-300 text-base-content/50 cursor-not-allowed'
        : 'bg-accent text-base-100 hover:bg-accent/70 active:bg-accent/80 cursor-pointer',
      props.fullWidth ? 'w-full' : '',
      props.class,
    ]"
    type="button"
  >
    <slot name="icon" />
    <span><slot /></span>
  </button>
</template>
