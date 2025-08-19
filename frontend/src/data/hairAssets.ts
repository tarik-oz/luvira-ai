// Utility for accessing hair color preview assets
// Loads all images under assets/hair_pattern_colors and provides helpers

const imageModules = import.meta.glob('../assets/hair_pattern_colors/*/*.webp', {
  import: 'default',
  eager: true,
}) as Record<string, string>

const toSlug = (name: string) => name.toLowerCase()

const resolvePath = (relativePath: string): string | undefined => {
  return imageModules[relativePath]
}

export const getBasePreview = (colorName: string): string => {
  const slug = toSlug(colorName)
  const path = `../assets/hair_pattern_colors/${slug}/${slug}_base.webp`
  return resolvePath(path) || ''
}

export const getTonePreview = (colorName: string, toneName: string): string => {
  const color = toSlug(colorName)
  const tone = toSlug(toneName)
  const path = `../assets/hair_pattern_colors/${color}/${color}_${tone}.webp`
  return resolvePath(path) || ''
}

export default {
  getBasePreview,
  getTonePreview,
}
