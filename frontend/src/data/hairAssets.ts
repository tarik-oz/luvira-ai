// Utility for accessing hair color preview assets
// Loads all images under assets/hair-color-swatches and provides helpers

const imageModules = import.meta.glob('../assets/hair-color-swatches/*/*.webp', {
  import: 'default',
  eager: true,
}) as Record<string, string>

const toSlug = (name: string) => name.toLowerCase()

const resolvePath = (relativePath: string): string | undefined => {
  return imageModules[relativePath]
}

export const getBasePreview = (colorName: string): string => {
  const slug = toSlug(colorName)
  const path = `../assets/hair-color-swatches/${slug}/${slug}-base-hair-swatch.webp`
  return resolvePath(path) || ''
}

export const getTonePreview = (colorName: string, toneName: string): string => {
  const color = toSlug(colorName)
  const tone = toSlug(toneName)
  const path = `../assets/hair-color-swatches/${color}/${color}-${tone}-hair-swatch.webp`
  return resolvePath(path) || ''
}

export default {
  getBasePreview,
  getTonePreview,
}
