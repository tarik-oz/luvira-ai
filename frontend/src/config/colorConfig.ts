export type ToneInfo = { description: string }

export const AVAILABLE_COLORS: string[] = [
  'Black',
  'Blonde',
  'Copper',
  'Brown',
  'Auburn',
  'Pink',
  'Blue',
  'Purple',
  'Gray',
  'Red',
  'Green',
  'Teal',
]

export const TONE_DEFINITIONS: Record<string, Record<string, ToneInfo>> = {
  Black: {
    jet: { description: 'Pure jet black' },
    soft: { description: 'Soft, warm black' },
    onyx: { description: 'Deep, rich onyx black' },
    charcoal: { description: 'Charcoal gray-black' },
  },
  Blonde: {
    platinum: { description: 'Ultra-light platinum blonde' },
    golden: { description: 'Warm golden blonde' },
    ash: { description: 'Cool ash blonde' },
    honey: { description: 'Sweet honey blonde' },
    beige: { description: 'Neutral beige blonde' },
  },
  Copper: {
    bright: { description: 'Bright copper' },
    antique: { description: 'Antique copper' },
    penny: { description: 'Penny copper' },
    rose: { description: 'Rose copper' },
  },
  Brown: {
    chestnut: { description: 'Rich chestnut brown' },
    chocolate: { description: 'Dark chocolate brown' },
    caramel: { description: 'Sweet caramel brown' },
    mahogany: { description: 'Reddish mahogany brown' },
    espresso: { description: 'Deep espresso brown' },
  },
  Auburn: {
    classic: { description: 'Classic auburn' },
    golden: { description: 'Golden auburn' },
    dark: { description: 'Dark auburn' },
    rust: { description: 'Rust auburn' },
  },
  Pink: {
    rose: { description: 'Romantic rose pink' },
    fuchsia: { description: 'Vibrant fuchsia pink' },
    blush: { description: 'Soft blush pink' },
    magenta: { description: 'Bold magenta pink' },
    coral: { description: 'Warm coral pink' },
  },
  Blue: {
    navy: { description: 'Deep navy blue' },
    electric: { description: 'Electric bright blue' },
    ice: { description: 'Ice blue' },
    midnight: { description: 'Midnight blue' },
    sky: { description: 'Sky blue' },
  },
  Purple: {
    violet: { description: 'Rich violet purple' },
    lavender: { description: 'Soft lavender purple' },
    plum: { description: 'Deep plum purple' },
    amethyst: { description: 'Gemstone amethyst purple' },
    orchid: { description: 'Delicate orchid purple' },
  },
  Gray: {
    silver: { description: 'Bright silver gray' },
    ash: { description: 'Cool ash gray' },
    charcoal: { description: 'Dark charcoal gray' },
    pearl: { description: 'Lustrous pearl gray' },
    steel: { description: 'Cool steel gray' },
  },
  Red: {
    burgundy: { description: 'Deep burgundy red' },
    cherry: { description: 'Deep cherry red' },
    rose: { description: 'Soft rose red' },
    crimson: { description: 'Vivid crimson red' },
    scarlet: { description: 'Bright scarlet red' },
  },
  Green: {
    emerald: { description: 'Rich emerald green' },
    forest: { description: 'Deep forest green' },
    mint: { description: 'Light mint green' },
    olive: { description: 'Subdued olive green' },
  },
  Teal: {
    aqua: { description: 'Bright aqua teal' },
    deep: { description: 'Deep ocean teal' },
    pastel: { description: 'Soft pastel teal' },
    seafoam: { description: 'Fresh seafoam teal' },
  },
}
