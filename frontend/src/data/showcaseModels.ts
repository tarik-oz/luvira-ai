// Import showcase images
import imgBlack1 from '../assets/showcase/1_black.png'
import imgBrown1 from '../assets/showcase/1_brown.png'
import imgPink1 from '../assets/showcase/1_pink.png'
import hairMask1 from '../assets/showcase/1_prob_mask.png'
import thumb1 from '../assets/showcase/1.jpg'

import imgGray3 from '../assets/showcase/3_gray.png'
import imgPink3 from '../assets/showcase/3_pink.png'
import imgPurple3 from '../assets/showcase/3_purple.png'
import hairMask3 from '../assets/showcase/3_prob_mask.png'
import thumb3 from '../assets/showcase/3.jpg'

import imgBlue5 from '../assets/showcase/5_blue.png'
import imgGray5 from '../assets/showcase/5_gray.png'
import imgPink5 from '../assets/showcase/5_pink.png'
import hairMask5 from '../assets/showcase/5_prob_mask.png'
import thumb5 from '../assets/showcase/5.jpg'

export interface ShowcaseColor {
  name: string
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
    thumbnail: thumb1,
    mask: hairMask1,
    colors: [
      { name: 'Black', src: imgBlack1, alt: 'Black hair color' },
      { name: 'Brown', src: imgBrown1, alt: 'Brown hair color' },
      { name: 'Pink', src: imgPink1, alt: 'Pink hair color' },
    ],
  },
  {
    id: 3,
    thumbnail: thumb3,
    mask: hairMask3,
    colors: [
      { name: 'Gray', src: imgGray3, alt: 'Gray hair color' },
      { name: 'Pink', src: imgPink3, alt: 'Pink hair color' },
      { name: 'Purple', src: imgPurple3, alt: 'Purple hair color' },
    ],
  },
  {
    id: 5,
    thumbnail: thumb5,
    mask: hairMask5,
    colors: [
      { name: 'Blue', src: imgBlue5, alt: 'Blue hair color' },
      { name: 'Gray', src: imgGray5, alt: 'Gray hair color' },
      { name: 'Pink', src: imgPink5, alt: 'Pink hair color' },
    ],
  },
]
