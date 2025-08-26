// Import model images
import woman_dark_hair_top_bun_portrait from '../assets/models/woman-dark-hair-top-bun-portrait.webp'
import woman_short_wavy_brown_hair_portrait from '../assets/models/woman-short-wavy-brown-hair-portrait.webp'
import woman_freckles_curly_reddish_hair_portrait from '../assets/models/woman-freckles-curly-reddish-hair-portrait.webp'
import woman_blonde_straight_hair_with_glasses_portrait from '../assets/models/woman-blonde-straight-hair-with-glasses-portrait.webp'
import man_dark_brown_short_hair_portrait from '../assets/models/man-dark-brown-short-hair-portrait.webp'
import man_wavy_light_brown_hair_portrait from '../assets/models/man-wavy-light-brown-hair-portrait.webp'

export interface StaticModelImage {
  id: string
  url: string
  alt: string
  altKey?: string
}

// Static model images data
export const modelImages: StaticModelImage[] = [
  {
    id: '1',
    url: woman_dark_hair_top_bun_portrait,
    alt: 'A smiling young woman with her dark brown hair in a top bun.',
    altKey: 'media.models.1',
  },
  {
    id: '2',
    url: woman_short_wavy_brown_hair_portrait,
    alt: 'A smiling young woman with short, wavy brown hair.',
    altKey: 'media.models.2',
  },
  {
    id: '3',
    url: woman_freckles_curly_reddish_hair_portrait,
    alt: 'A smiling woman with freckles and reddish-brown curly hair.',
    altKey: 'media.models.3',
  },
  {
    id: '4',
    url: woman_blonde_straight_hair_with_glasses_portrait,
    alt: 'A smiling young woman with glasses and long, straight blonde hair.',
    altKey: 'media.models.4',
  },
  {
    id: '5',
    url: man_dark_brown_short_hair_portrait,
    alt: 'A young man with short, dark brown hair looking at the camera.',
    altKey: 'media.models.5',
  },
  {
    id: '6',
    url: man_wavy_light_brown_hair_portrait,
    alt: 'A smiling young man with light brown wavy hair and earrings.',
    altKey: 'media.models.6',
  },
]
