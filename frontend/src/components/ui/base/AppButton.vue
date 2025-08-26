<script setup lang="ts">
import { computed, useSlots } from 'vue'
const props = defineProps<{
  fullWidth?: boolean
  class?: string
  disabled?: boolean
  variant?: 'accent' | 'base' | 'primary' | 'ghost'
  type?: 'button' | 'submit' | 'reset'
}>()

const slots = useSlots()
const hasDefaultSlot = computed(() => !!slots.default)

const variantClass = computed(() => {
  if (props.disabled) return 'bg-base-300 text-base-content/50 cursor-not-allowed'
  switch (props.variant) {
    case 'base':
      return 'bg-base-content text-base-100 hover:bg-base-content/80 active:bg-primary/60 cursor-pointer'
    case 'primary':
      return 'bg-primary text-primary-content hover:bg-primary/90 active:bg-primary/80 cursor-pointer'
    case 'ghost':
      return 'bg-transparent text-base-content hover:bg-base-content/10 active:bg-base-content/20 cursor-pointer'
    case 'accent':
    default:
      return 'bg-accent text-base-100 hover:bg-accent/70 active:bg-accent/80 cursor-pointer'
  }
})
</script>

<template>
  <button
    :disabled="props.disabled"
    :class="[
      'focus-visible:ring-primary no-callout inline-flex transform items-center justify-center rounded-lg px-6 py-3 font-semibold transition-all duration-150 select-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:outline-none active:scale-[0.98] active:brightness-95 motion-safe:active:translate-y-[1px]',
      hasDefaultSlot ? 'gap-2' : 'gap-0',
      variantClass,
      props.fullWidth !== false ? 'w-full' : 'w-auto',
      props.class,
    ]"
    :type="props.type ?? 'button'"
  >
    <slot name="icon" />
    <span v-if="hasDefaultSlot"><slot /></span>
  </button>
</template>
