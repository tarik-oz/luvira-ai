import showcase_1_blonde_honey from '../assets/showcase/showcase_1_blonde_honey.webp'
import showcase_1_pink_fuchsia from '../assets/showcase/showcase_1_pink_fuchsia.webp'
import showcase_1_red_cherry from '../assets/showcase/showcase_1_red_cherry.webp'
import showcase_1_prob_mask from '../assets/showcase/showcase_1_prob_mask.png'
import showcase_1_thumb from '../assets/showcase/showcase_1_thumb.jpg'

import showcase_2_gray_silver from '../assets/showcase/showcase_2_gray_silver.webp'
import showcase_2_purple_plum from '../assets/showcase/showcase_2_purple_plum.webp'
import showcase_2_teal_pastel from '../assets/showcase/showcase_2_teal_pastel.webp'
import showcase_2_prob_mask from '../assets/showcase/showcase_2_prob_mask.png'
import showcase_2_thumb from '../assets/showcase/showcase_2_thumb.jpg'

import showcase_3_green_mint from '../assets/showcase/showcase_3_green_mint.webp'
import showcase_3_blue_ice from '../assets/showcase/showcase_3_blue_ice.webp'
import showcase_3_copper_bright from '../assets/showcase/showcase_3_copper_bright.webp'
import showcase_3_prob_mask from '../assets/showcase/showcase_3_prob_mask.png'
import showcase_3_thumb from '../assets/showcase/showcase_3_thumb.jpg'

export interface ShowcaseColor {
  colorName: string
  toneName: string
  src: string
  alt: string
}

export interface ShowcaseModel {
  id: number
  thumbnail: string
  mask: string
  colors: ShowcaseColor[]
}

export const showcaseModels: ShowcaseModel[] = [
  {
    id: 1,
    thumbnail: showcase_1_thumb,
    mask: showcase_1_prob_mask,
    colors: [
      {
        colorName: 'Blonde',
        toneName: 'Honey',
        src: showcase_1_blonde_honey,
        alt: 'Blonde Honey hair color',
      },
      {
        colorName: 'Pink',
        toneName: 'Fuchsia',
        src: showcase_1_pink_fuchsia,
        alt: 'Pink Fuchsia hair color',
      },
      {
        colorName: 'Red',
        toneName: 'Cherry',
        src: showcase_1_red_cherry,
        alt: 'Red Cherry hair color',
      },
    ],
  },
  {
    id: 2,
    thumbnail: showcase_2_thumb,
    mask: showcase_2_prob_mask,
    colors: [
      {
        colorName: 'Gray',
        toneName: 'Silver',
        src: showcase_2_gray_silver,
        alt: 'Gray Silver hair color',
      },
      {
        colorName: 'Purple',
        toneName: 'Plum',
        src: showcase_2_purple_plum,
        alt: 'Purple Plum hair color',
      },
      {
        colorName: 'Teal',
        toneName: 'Pastel',
        src: showcase_2_teal_pastel,
        alt: 'Teal Pastel hair color',
      },
    ],
  },
  {
    id: 3,
    thumbnail: showcase_3_thumb,
    mask: showcase_3_prob_mask,
    colors: [
      {
        colorName: 'Green',
        toneName: 'Mint',
        src: showcase_3_green_mint,
        alt: 'Green Mint hair color',
      },
      {
        colorName: 'Blue',
        toneName: 'Ice',
        src: showcase_3_blue_ice,
        alt: 'Blue Ice hair color',
      },
      {
        colorName: 'Copper',
        toneName: 'Bright',
        src: showcase_3_copper_bright,
        alt: 'Copper Bright hair color',
      },
    ],
  },
]
