<script setup lang="ts">
const props = defineProps<{
  fullWidth?: boolean
  class?: string
}>()
</script>

<template>
  <button
    :class="[
      'inline-flex items-center justify-center gap-2 px-6 py-3 rounded-lg font-semibold transition-colors duration-200 focus:outline-none',
      'bg-accent text-base-100 hover:bg-accent/80 active:bg-accent/90',
      props.fullWidth ? 'w-full' : '',
      props.class,
    ]"
    type="button"
    @click="$emit('click', $event)"
  >
    <slot name="icon" />
    <span><slot /></span>
  </button>
</template>
