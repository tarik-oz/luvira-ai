// Import sample images
import woman_1 from '../assets/samples/woman_1.webp'
import woman_2 from '../assets/samples/woman_2.webp'
import woman_3 from '../assets/samples/woman_3.webp'
import woman_4 from '../assets/samples/woman_4.webp'
import guy_1 from '../assets/samples/guy_1.webp'
import guy_2 from '../assets/samples/guy_2.webp'

export interface StaticSampleImage {
  id: string
  url: string
  alt: string
}

// Static sample images data
export const sampleImages: StaticSampleImage[] = [
  { id: '1', url: woman_1, alt: 'Woman 1' },
  { id: '2', url: woman_2, alt: 'Woman 2' },
  { id: '3', url: woman_3, alt: 'Woman 3' },
  { id: '4', url: woman_4, alt: 'Woman 4' },
  { id: '5', url: guy_1, alt: 'Guy 1' },
  { id: '6', url: guy_2, alt: 'Guy 2' },
  // trimmed to 6 images
]
