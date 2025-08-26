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

// Minimal tone lists per color (no descriptions)
export const TONE_MAP: Record<string, string[]> = {
  Black: ['jet', 'soft', 'onyx', 'charcoal'],
  Blonde: ['platinum', 'golden', 'ash', 'honey', 'beige'],
  Copper: ['bright', 'antique', 'penny', 'rose'],
  Brown: ['chestnut', 'chocolate', 'caramel', 'mahogany', 'espresso'],
  Auburn: ['classic', 'golden', 'dark', 'rust'],
  Pink: ['rose', 'fuchsia', 'blush', 'magenta', 'coral'],
  Blue: ['navy', 'electric', 'ice', 'midnight', 'sky'],
  Purple: ['violet', 'lavender', 'plum', 'amethyst', 'orchid'],
  Gray: ['silver', 'ash', 'charcoal', 'pearl', 'steel'],
  Red: ['cherry', 'rose', 'crimson', 'scarlet'],
  Green: ['emerald', 'forest', 'mint', 'olive'],
  Teal: ['aqua', 'deep', 'pastel', 'seafoam'],
}
