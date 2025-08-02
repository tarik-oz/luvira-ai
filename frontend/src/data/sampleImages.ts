// Import sample images
import sample1 from '../assets/samples/sample1.jpg'
import sample2 from '../assets/samples/sample2.jpg'
import sample3 from '../assets/samples/sample3.jpg'
import sample4 from '../assets/samples/sample4.jpg'
import sample5 from '../assets/samples/sample5.jpg'
import sample6 from '../assets/samples/sample6.jpg'
import sample7 from '../assets/samples/sample7.jpg'
import sample8 from '../assets/samples/sample8.jpg'

export interface StaticSampleImage {
  id: string
  url: string
  alt: string
}

// Static sample images data
export const sampleImages: StaticSampleImage[] = [
  { id: '1', url: sample1, alt: 'Sample 1' },
  { id: '2', url: sample2, alt: 'Sample 2' },
  { id: '3', url: sample3, alt: 'Sample 3' },
  { id: '4', url: sample4, alt: 'Sample 4' },
  { id: '5', url: sample5, alt: 'Sample 5' },
  { id: '6', url: sample6, alt: 'Sample 6' },
  { id: '7', url: sample7, alt: 'Sample 7' },
  { id: '8', url: sample8, alt: 'Sample 8' },
]
