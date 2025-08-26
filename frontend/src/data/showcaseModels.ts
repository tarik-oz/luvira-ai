import woman_off_shoulder_blonde_honey_hair from '../assets/hair-color-transformations/woman-off-shoulder-blonde-honey-hair.webp'
import woman_off_shoulder_pink_fuchsia_hair from '../assets/hair-color-transformations/woman-off-shoulder-pink-fuchsia-hair.webp'
import woman_off_shoulder_red_cherry_hair from '../assets/hair-color-transformations/woman-off-shoulder-red-cherry-hair.webp'
import woman_off_shoulder_prob_mask from '../assets/hair-color-transformations/woman-off-shoulder-prob-mask.webp'
import woman_off_shoulder_original_brown_hair_thumb from '../assets/hair-color-transformations/woman-off-shoulder-original-brown-hair-thumb.webp'

import woman_closeup_gray_silver_hair from '../assets/hair-color-transformations/woman-closeup-gray-silver-hair.webp'
import woman_closeup_purple_plum_hair from '../assets/hair-color-transformations/woman-closeup-purple-plum-hair.webp'
import woman_closeup_teal_pastel_hair from '../assets/hair-color-transformations/woman-closeup-teal-pastel-hair.webp'
import woman_closeup_prob_mask from '../assets/hair-color-transformations/woman-closeup-prob-mask.webp'
import woman_closeup_original_ash_hair_thumb from '../assets/hair-color-transformations/woman-closeup-original-ash-hair-thumb.webp'

import woman_hair_bun_bangs_green_mint_color from '../assets/hair-color-transformations/woman-hair-bun-bangs-green-mint-color.webp'
import woman_hair_bun_bangs_blue_ice_color from '../assets/hair-color-transformations/woman-hair-bun-bangs-blue-ice-color.webp'
import woman_hair_bun_bangs_copper_bright_color from '../assets/hair-color-transformations/woman-hair-bun-bangs-copper-bright-color.webp'
import woman_hair_bun_bangs_prob_mask from '../assets/hair-color-transformations/woman-hair-bun-bangs-prob-mask.webp'
import woman_hair_bun_bangs_original_dark_hair_thumb from '../assets/hair-color-transformations/woman-hair-bun-bangs-original-dark-hair-thumb.webp'

export interface ShowcaseColor {
  colorName: string
  toneName: string
  src: string
  alt: string
  altKey?: string
}

export interface ShowcaseModel {
  id: number
  thumbnail: string
  thumbnailAlt: string
  thumbnailAltKey?: string
  mask: string
  colors: ShowcaseColor[]
}

export const showcaseModels: ShowcaseModel[] = [
  {
    id: 1,
    thumbnail: woman_off_shoulder_original_brown_hair_thumb,
    thumbnailAlt:
      'Thumbnail showing the original brown hair color of a woman before digital editing.',
    thumbnailAltKey: 'media.showcase.thumb1',
    mask: woman_off_shoulder_prob_mask,
    colors: [
      {
        colorName: 'Pink',
        toneName: 'Fuchsia',
        src: woman_off_shoulder_pink_fuchsia_hair,
        alt: "A woman's portrait showcasing a vibrant 'Pink Fuchsia' hair color transformation.",
        altKey: 'media.showcase.colors.1.pinkFuchsia',
      },
      {
        colorName: 'Blonde',
        toneName: 'Honey',
        src: woman_off_shoulder_blonde_honey_hair,
        alt: "Portrait of a woman with her hair digitally colored to a 'Blonde Honey' shade.",
        altKey: 'media.showcase.colors.1.blondeHoney',
      },
      {
        colorName: 'Red',
        toneName: 'Cherry',
        src: woman_off_shoulder_red_cherry_hair,
        alt: "Example of a 'Red Cherry' hair color applied to a woman's portrait.",
        altKey: 'media.showcase.colors.1.redCherry',
      },
    ],
  },
  {
    id: 2,
    thumbnail: woman_closeup_original_ash_hair_thumb,
    thumbnailAlt:
      'Thumbnail of a woman with her natural hair color, used for hair color simulation.',
    thumbnailAltKey: 'media.showcase.thumb2',
    mask: woman_closeup_prob_mask,
    colors: [
      {
        colorName: 'Gray',
        toneName: 'Silver',
        src: woman_closeup_gray_silver_hair,
        alt: "Close-up of a woman with a cool-toned 'Gray Silver' hair color edit.",
        altKey: 'media.showcase.colors.2.graySilver',
      },
      {
        colorName: 'Purple',
        toneName: 'Plum',
        src: woman_closeup_purple_plum_hair,
        alt: "A woman's hair digitally changed to a deep 'Purple Plum' color.",
        altKey: 'media.showcase.colors.2.purplePlum',
      },
      {
        colorName: 'Teal',
        toneName: 'Pastel',
        src: woman_closeup_teal_pastel_hair,
        alt: "Example of a bright 'Teal Pastel' hair color applied to a woman's portrait.",
        altKey: 'media.showcase.colors.2.tealPastel',
      },
    ],
  },
  {
    id: 3,
    thumbnail: woman_hair_bun_bangs_original_dark_hair_thumb,
    thumbnailAlt:
      "Thumbnail portrait of a woman with dark hair in a bun, representing the 'before' image.",
    thumbnailAltKey: 'media.showcase.thumb3',
    mask: woman_hair_bun_bangs_prob_mask,
    colors: [
      {
        colorName: 'Green',
        toneName: 'Mint',
        src: woman_hair_bun_bangs_green_mint_color,
        alt: "Portrait of a woman with a vivid 'Green Mint' hair color transformation on her bun and bangs.",
        altKey: 'media.showcase.colors.3.greenMint',
      },
      {
        colorName: 'Blue',
        toneName: 'Ice',
        src: woman_hair_bun_bangs_blue_ice_color,
        alt: "A woman with her hair bun and bangs colored in a striking 'Blue Ice' shade.",
        altKey: 'media.showcase.colors.3.blueIce',
      },
      {
        colorName: 'Copper',
        toneName: 'Bright',
        src: woman_hair_bun_bangs_copper_bright_color,
        alt: "Hair color simulation showing a woman with a warm 'Copper Bright' hair tone.",
        altKey: 'media.showcase.colors.3.copperBright',
      },
    ],
  },
]
