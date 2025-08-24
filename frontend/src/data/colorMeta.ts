// Color metadata and helpers for UI ordering

export const PREFERRED_COLOR_ORDER: string[] = [
  'Black',
  'Brown',
  'Blonde',
  'Auburn',
  'Copper',
  'Red',
  'Gray',
  'Purple',
  'Blue',
  'Pink',
  'Green',
  'Teal',
]

export const getPreferredColorOrder = (inputColors: string[]): string[] => {
  const indexOf = (c: string) => {
    const i = PREFERRED_COLOR_ORDER.indexOf(c)
    return i === -1 ? Number.MAX_SAFE_INTEGER : i
  }
  return [...inputColors].sort((a, b) => {
    const ia = indexOf(a)
    const ib = indexOf(b)
    if (ia === ib) return a.localeCompare(b)
    return ia - ib
  })
}

// Tone RGB values mirrored from backend color_config.py CUSTOM_TONES
export const TONE_RGB: Record<string, Record<string, [number, number, number]>> = {
  Black: {
    jet: [2, 2, 2],
    soft: [12, 11, 11],
    onyx: [1, 1, 1],
    charcoal: [8, 8, 9],
  },
  Blonde: {
    platinum: [255, 252, 245],
    golden: [255, 226, 142],
    ash: [223, 218, 201],
    honey: [232, 201, 140],
    beige: [255, 230, 185],
  },
  Copper: {
    bright: [255, 127, 28],
    antique: [134, 91, 58],
    penny: [188, 102, 38],
    rose: [255, 122, 53],
  },
  Brown: {
    chestnut: [138, 87, 46],
    chocolate: [101, 74, 58],
    caramel: [159, 107, 54],
    mahogany: [142, 73, 40],
    espresso: [91, 68, 52],
  },
  Auburn: {
    classic: [174, 78, 30],
    golden: [212, 98, 25],
    dark: [131, 71, 41],
    rust: [179, 76, 22],
  },
  Pink: {
    rose: [255, 137, 182],
    fuchsia: [255, 59, 189],
    blush: [255, 191, 219],
    magenta: [230, 42, 178],
    coral: [255, 110, 153],
  },
  Blue: {
    navy: [0, 0, 217],
    electric: [0, 68, 255],
    ice: [168, 168, 255],
    midnight: [0, 19, 191],
    sky: [89, 89, 255],
  },
  Purple: {
    violet: [142, 0, 142],
    lavender: [161, 106, 179],
    plum: [91, 0, 118],
    amethyst: [161, 0, 161],
    orchid: [151, 61, 153],
  },
  Gray: {
    silver: [194, 190, 189],
    ash: [140, 142, 142],
    charcoal: [108, 109, 109],
    pearl: [176, 171, 170],
    steel: [121, 122, 122],
  },
  Red: {
    cherry: [139, 0, 38],
    rose: [222, 49, 99],
    crimson: [209, 29, 21],
    scarlet: [236, 42, 29],
  },
  Green: {
    emerald: [0, 179, 81],
    forest: [0, 131, 80],
    mint: [78, 223, 140],
    olive: [53, 131, 88],
  },
  Teal: {
    aqua: [0, 201, 170],
    deep: [0, 141, 140],
    pastel: [84, 235, 214],
    seafoam: [0, 210, 189],
  },
}

const luminance = (rgb: [number, number, number]): number => {
  const [r, g, b] = rgb
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

export type ToneItem = { name: string; displayName: string; description: string; preview?: string }

export const getToneSortOrder = (colorName: string, tones: ToneItem[]): ToneItem[] => {
  const rgbMap = TONE_RGB[colorName] || {}
  return [...tones].sort((a, b) => {
    const la = rgbMap[a.name] ? luminance(rgbMap[a.name]) : -Infinity
    const lb = rgbMap[b.name] ? luminance(rgbMap[b.name]) : -Infinity
    // Light to dark: higher luminance first
    if (la === lb) return a.name.localeCompare(b.name)
    return lb - la
  })
}
