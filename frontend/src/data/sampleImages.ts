// Import sample images
import sample1 from '../assets/samples/sample1.jpg'
import sample2 from '../assets/samples/sample2.jpg'
import sample3 from '../assets/samples/sample3.jpg'
import sample4 from '../assets/samples/sample4.jpg'
import sample5 from '../assets/samples/sample5.jpg'
import sample6 from '../assets/samples/sample6.jpg'

export interface StaticSampleImage {
  id: string
  name: string
  description: string
  url: string
  filename: string
}

// Static sample images data
export const sampleImages: StaticSampleImage[] = [
  {
    id: '1',
    name: 'Sample 1',
    description: 'Female portrait',
    url: sample1,
    filename: 'sample1.jpg',
  },
  {
    id: '2',
    name: 'Sample 2',
    description: 'Female portrait',
    url: sample2,
    filename: 'sample2.jpg',
  },
  {
    id: '3',
    name: 'Sample 3',
    description: 'Female portrait',
    url: sample3,
    filename: 'sample3.jpg',
  },
  {
    id: '4',
    name: 'Sample 4',
    description: 'Female portrait',
    url: sample4,
    filename: 'sample4.jpg',
  },
  {
    id: '5',
    name: 'Sample 5',
    description: 'Female portrait',
    url: sample5,
    filename: 'sample5.jpg',
  },
  {
    id: '6',
    name: 'Sample 6',
    description: 'Female portrait',
    url: sample6,
    filename: 'sample6.jpg',
  },
]
