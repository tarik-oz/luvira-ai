# Vue.js Frontend

This module contains the Vue.js 3 frontend application for the hair segmentation service. It provides a modern, responsive user interface with image upload, camera capture, color selection, and real-time preview capabilities.

## 📋 Table of Contents

- [Setup](#setup)
- [Frontend Architecture](#frontend-architecture)
- [Local Development](#local-development)
- [Components Overview](#components-overview)
- [State Management](#state-management)
- [API Integration](#api-integration)
- [Internationalization](#internationalization)
- [Styling & UI](#styling--ui)
- [Building & Deployment](#building--deployment)
- [Customization](#customization)

## 🚀 Setup

### Prerequisites

- Node.js 18+ and npm/yarn
- Backend API running (see [api/README.md](../api/README.md))

### 1. Navigate to Frontend Directory

```bash
cd frontend
```

### 2. Install Dependencies

```bash
# Using npm
npm install

# Using yarn
yarn install

# Using pnpm
pnpm install
```

### 3. Configure Environment

Create `.env.local` file (optional):

```bash
# API Configuration
VITE_API_BASE_URL=http://localhost:8000

# Development settings
VITE_DEV_MODE=true
VITE_ENABLE_ANALYTICS=false
```

### 4. Start Development Server

```bash
# Using npm
npm run dev

# Using yarn
yarn dev

# Using pnpm
pnpm dev
```

### 5. Access Application

- **Frontend**: http://localhost:5173
- **Hot Reload**: Automatically refreshes on code changes

## 🏗️ Frontend Architecture

### Project Structure

```
frontend/
├── public/                 # Static assets
│   ├── robots.txt         # SEO robots file
│   └── sitemap.xml        # SEO sitemap
├── src/
│   ├── assets/            # Static resources
│   │   └── css/           # Global styles
│   ├── components/        # Vue components
│   │   ├── layout/        # Layout components
│   │   └── ui/            # UI components
│   ├── composables/       # Vue 3 composables
│   ├── data/              # Static data & configuration
│   ├── locales/           # i18n translations
│   ├── router/            # Vue Router configuration
│   ├── services/          # API & external services
│   ├── types/             # TypeScript type definitions
│   ├── utils/             # Utility functions
│   ├── views/             # Page components
│   ├── App.vue           # Root component
│   ├── i18n.ts           # Internationalization setup
│   └── main.ts           # Application entry point
├── index.html            # HTML template
├── package.json          # Dependencies & scripts
├── tsconfig.json         # TypeScript configuration
├── vite.config.ts        # Vite build configuration
└── README.md            # This file
```

### Technology Stack

- **Framework**: Vue.js 3 with Composition API
- **Language**: TypeScript
- **Build Tool**: Vite
- **Router**: Vue Router 4
- **Styling**: CSS3 with CSS Variables
- **HTTP Client**: Fetch API with custom service layer
- **Internationalization**: Vue I18n
- **Icons**: Custom SVG icons
- **State Management**: Vue 3 Composables (no Vuex/Pinia needed)

## 💻 Local Development

### Development Commands

```bash
# Start development server
npm run dev              # Starts Vite dev server on port 5173

# Type checking
npm run type-check       # Run TypeScript compiler check

# Linting
npm run lint             # Run ESLint
npm run lint:fix         # Fix linting issues automatically

# Building
npm run build            # Production build
npm run preview          # Preview production build locally

# Testing (if configured)
npm run test             # Run unit tests
npm run test:watch       # Run tests in watch mode
```

### Backend Integration

The frontend communicates with the FastAPI backend through proxy configuration:

```typescript
// vite.config.ts
export default defineConfig({
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
```

### Environment Variables

```bash
# .env.local
VITE_API_BASE_URL=http://localhost:8000    # Backend API URL
VITE_DEV_MODE=true                         # Development mode
VITE_ENABLE_ANALYTICS=false               # Analytics (production)
VITE_DEFAULT_LANGUAGE=en                   # Default language
```

## 🧩 Components Overview

### Layout Components

#### `AppHeader.vue`

- Application header with branding
- Language switcher
- Navigation elements

#### `AppFooter.vue`

- Footer information
- Links and credits
- Social media links

### Upload View Components

#### `UploadSection.vue`

- Main file upload interface
- Drag & drop functionality
- File validation feedback
- Upload progress indication

```vue
<!-- Usage Example -->
<UploadSection
  @file-selected="handleFileUpload"
  :loading="isUploading"
  :accept="'.jpg,.jpeg,.png'"
  :max-size="10485760"
/>
```

#### `CameraCapture.vue`

- Camera access and photo capture
- Device camera selection
- Real-time preview
- Photo capture with constraints

```vue
<!-- Usage Example -->
<CameraCapture
  @photo-captured="handleCameraPhoto"
  :video-constraints="{ width: 1280, height: 720 }"
/>
```

#### `HairColorShowcase.vue`

- Display of color transformation examples
- Model images with different hair colors
- Interactive color previews

#### `ModelImages.vue`

- Sample images for testing
- Quick start options
- Image gallery with selection

### Processing View Components

#### `ImageDisplay.vue`

- Original and transformed image display
- Side-by-side comparison
- Zoom and pan functionality
- Loading states

```vue
<!-- Usage Example -->
<ImageDisplay
  :original-image="originalImageUrl"
  :transformed-image="transformedImageUrl"
  :loading="isProcessing"
  @zoom-changed="handleZoom"
/>
```

#### `ColorPalette.vue`

- Available hair colors grid
- Color selection interface
- Active color highlighting
- Color name display

```vue
<!-- Usage Example -->
<ColorPalette
  :colors="availableColors"
  :selected-color="selectedColor"
  @color-selected="handleColorSelection"
/>
```

#### `TonePalette.vue`

- Tone variations for selected color
- Base color + tone options
- Real-time preview updates

#### `MobileColorToneBar.vue`

- Mobile-optimized color/tone selector
- Horizontal scroll interface
- Touch-friendly interactions

#### `DownloadButton.vue`

- Download processed image
- Multiple format options
- Download progress feedback

#### `SessionExpiredModal.vue`

- Session expiration notification
- Restart process option
- Error handling UI

### Base UI Components

#### `AppButton.vue`

- Standardized button component
- Multiple variants (primary, secondary, etc.)
- Loading states and icons

```vue
<!-- Usage Examples -->
<AppButton variant="primary" :loading="isLoading">
  Process Image
</AppButton>

<AppButton variant="secondary" icon="download">
  Download Result
</AppButton>
```

## 🔄 State Management

### Global State with Composables

The application uses Vue 3's Composition API for state management:

```typescript
// composables/useAppState.ts
export const useAppState = () => {
  const currentView = ref<'upload' | 'processing'>('upload')
  const selectedImage = ref<File | null>(null)
  const sessionId = ref<string | null>(null)
  const selectedColor = ref<string | null>(null)
  const selectedTone = ref<string | null>(null)
  const isProcessing = ref(false)
  const processedImageUrl = ref<string | null>(null)

  // Actions
  const setCurrentView = (view: 'upload' | 'processing') => {
    currentView.value = view
  }

  const resetState = () => {
    selectedImage.value = null
    sessionId.value = null
    selectedColor.value = null
    selectedTone.value = null
    processedImageUrl.value = null
    isProcessing.value = false
  }

  return {
    // State
    currentView: readonly(currentView),
    selectedImage: readonly(selectedImage),
    sessionId: readonly(sessionId),
    selectedColor: readonly(selectedColor),
    selectedTone: readonly(selectedTone),
    isProcessing: readonly(isProcessing),
    processedImageUrl: readonly(processedImageUrl),

    // Actions
    setCurrentView,
    resetState,
    // ... other actions
  }
}
```

### Usage in Components

```vue
<script setup lang="ts">
import { useAppState } from '@/composables/useAppState'

const { currentView, selectedColor, setCurrentView } = useAppState()

const handleColorSelection = (color: string) => {
  selectedColor.value = color
  // Process color change...
}
</script>
```

## 🌐 API Integration

### Service Layer Architecture

```typescript
// services/apiService.ts
class ApiService {
  private baseURL: string

  constructor() {
    this.baseURL = import.meta.env.VITE_API_BASE_URL || '/api'
  }

  async uploadAndPrepare(file: File, source: string): Promise<{ session_id: string }> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('source', source)

    const response = await fetch(`${this.baseURL}/upload-and-prepare`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      throw await this.handleErrorResponse(response)
    }

    return response.json()
  }

  async changeHairColorWithSession(
    sessionId: string,
    colorName: string,
    tone?: string,
  ): Promise<Blob> {
    const formData = new FormData()
    formData.append('color_name', colorName)
    if (tone) formData.append('tone', tone)

    const response = await fetch(`${this.baseURL}/change-hair-color-with-session/${sessionId}`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      throw await this.handleErrorResponse(response)
    }

    return response.blob()
  }

  private async handleErrorResponse(response: Response) {
    const contentType = response.headers.get('content-type')
    if (contentType?.includes('application/json')) {
      const errorData = await response.json()
      return new Error(errorData.detail || 'API Error')
    }
    return new Error(`HTTP ${response.status}: ${response.statusText}`)
  }
}

export const apiService = new ApiService()
```

### Hair Service Integration

```typescript
// services/hairService.ts
export class HairService {
  private api = apiService

  async processImage(file: File, source: string) {
    try {
      // Upload and prepare
      const { session_id } = await this.api.uploadAndPrepare(file, source)

      // Get available colors
      const colors = await this.api.getAvailableColors()

      return { sessionId: session_id, availableColors: colors }
    } catch (error) {
      this.handleError(error)
      throw error
    }
  }

  async changeColor(sessionId: string, color: string, tone?: string) {
    try {
      const imageBlob = await this.api.changeHairColorWithSession(sessionId, color, tone)

      return URL.createObjectURL(imageBlob)
    } catch (error) {
      this.handleError(error)
      throw error
    }
  }

  private handleError(error: any) {
    // Error logging and analytics
    console.error('Hair Service Error:', error)
  }
}

export const hairService = new HairService()
```

## 🌍 Internationalization

### Language Support

The app supports multiple languages with Vue I18n:

```typescript
// i18n.ts
import { createI18n } from 'vue-i18n'
import en from './locales/en.json'
import tr from './locales/tr.json'

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  fallbackLocale: 'en',
  messages: { en, tr },
})

export default i18n
```

### Translation Files

```json
// locales/en.json
{
  "upload": {
    "title": "Upload Your Photo",
    "dragDrop": "Drag & drop your image here",
    "or": "or",
    "browseFiles": "Browse Files",
    "supportedFormats": "Supported formats: JPG, PNG",
    "maxSize": "Max size: 10MB"
  },
  "colors": {
    "blonde": "Blonde",
    "brown": "Brown",
    "black": "Black",
    "red": "Red"
  },
  "tones": {
    "golden": "Golden",
    "ash": "Ash",
    "platinum": "Platinum"
  }
}
```

### Usage in Components

```vue
<template>
  <div>
    <h1>{{ $t('upload.title') }}</h1>
    <p>{{ $t('upload.supportedFormats') }}</p>
  </div>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n'

const { t, locale } = useI18n()

const changeLanguage = (lang: string) => {
  locale.value = lang
}
</script>
```

## 🎨 Styling & UI

### CSS Architecture

```css
/* assets/css/main.css */

/* CSS Custom Properties (Variables) */
:root {
  /* Colors */
  --color-primary: #6366f1;
  --color-primary-dark: #4f46e5;
  --color-secondary: #8b5cf6;
  --color-success: #10b981;
  --color-warning: #f59e0b;
  --color-error: #ef4444;

  /* Neutrals */
  --color-gray-50: #f9fafb;
  --color-gray-100: #f3f4f6;
  --color-gray-900: #111827;

  /* Typography */
  --font-family-sans: 'Inter', system-ui, sans-serif;
  --font-size-sm: 0.875rem;
  --font-size-base: 1rem;
  --font-size-lg: 1.125rem;
  --font-size-xl: 1.25rem;

  /* Spacing */
  --spacing-1: 0.25rem;
  --spacing-2: 0.5rem;
  --spacing-4: 1rem;
  --spacing-8: 2rem;

  /* Layout */
  --border-radius: 0.5rem;
  --border-radius-lg: 0.75rem;
  --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
  --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
}

/* Base Styles */
body {
  font-family: var(--font-family-sans);
  color: var(--color-gray-900);
  background-color: var(--color-gray-50);
}

/* Utility Classes */
.btn {
  padding: var(--spacing-2) var(--spacing-4);
  border-radius: var(--border-radius);
  font-weight: 500;
  transition: all 0.15s ease;
}

.btn-primary {
  background-color: var(--color-primary);
  color: white;
}

.btn-primary:hover {
  background-color: var(--color-primary-dark);
}

/* Responsive Design */
@media (max-width: 768px) {
  :root {
    --spacing-4: 0.875rem;
    --font-size-base: 0.925rem;
  }
}
```

### Component Styling

```vue
<!-- Example: ColorPalette.vue -->
<template>
  <div class="color-palette">
    <div class="color-grid">
      <button
        v-for="color in colors"
        :key="color.name"
        class="color-button"
        :class="{ active: selectedColor === color.name }"
        @click="selectColor(color.name)"
      >
        <div class="color-swatch" :style="{ backgroundColor: `rgb(${color.rgb.join(',')})` }"></div>
        <span class="color-name">{{ $t(`colors.${color.name.toLowerCase()}`) }}</span>
      </button>
    </div>
  </div>
</template>

<style scoped>
.color-palette {
  padding: var(--spacing-4);
}

.color-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: var(--spacing-4);
}

.color-button {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: var(--spacing-2);
  border: 2px solid transparent;
  border-radius: var(--border-radius-lg);
  background: white;
  cursor: pointer;
  transition: all 0.2s ease;
}

.color-button:hover {
  border-color: var(--color-primary);
  box-shadow: var(--shadow-md);
}

.color-button.active {
  border-color: var(--color-primary);
  background-color: var(--color-primary);
  color: white;
}

.color-swatch {
  width: 60px;
  height: 60px;
  border-radius: 50%;
  border: 3px solid white;
  box-shadow: var(--shadow-sm);
}

.color-name {
  margin-top: var(--spacing-2);
  font-size: var(--font-size-sm);
  font-weight: 500;
}

/* Mobile Responsive */
@media (max-width: 768px) {
  .color-grid {
    grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
    gap: var(--spacing-2);
  }

  .color-swatch {
    width: 50px;
    height: 50px;
  }
}
</style>
```

## 🚀 Building & Deployment

### Development Build

```bash
# Type check
npm run type-check

# Build for production
npm run build

# Preview production build
npm run preview
```

### Production Build Configuration

```typescript
// vite.config.ts
export default defineConfig({
  build: {
    outDir: 'dist',
    sourcemap: false,
    minify: 'terser',
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['vue', 'vue-router', 'vue-i18n'],
          utils: ['lodash', 'date-fns'],
        },
      },
    },
  },
  define: {
    __VUE_OPTIONS_API__: false,
    __VUE_PROD_DEVTOOLS__: false,
  },
})
```

### Docker Deployment

```dockerfile
# Multi-stage build
FROM node:18-alpine as build-stage
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

FROM nginx:alpine as production-stage
COPY --from=build-stage /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Deployment Scripts

```bash
# Build and deploy script
#!/bin/bash
echo "Building frontend..."
npm run build

echo "Deploying to production..."
# Copy dist/ to your web server
rsync -av dist/ user@server:/var/www/html/

echo "Deployment complete!"
```

## 🔧 Customization

### Adding New Components

1. Create component in appropriate directory:

```vue
<!-- components/ui/NewComponent.vue -->
<template>
  <div class="new-component">
    <!-- Component template -->
  </div>
</template>

<script setup lang="ts">
interface Props {
  // Define props
}

interface Emits {
  // Define emits
}

const props = defineProps<Props>()
const emit = defineEmits<Emits>()
</script>

<style scoped>
/* Component styles */
</style>
```

2. Export from index file (if using barrel exports)
3. Import and use in parent components

### Modifying Color Data

```typescript
// data/colorOptions.ts
export interface ColorOption {
  name: string
  rgb: [number, number, number]
  category?: string
}

export const colorOptions: ColorOption[] = [
  { name: 'Blonde', rgb: [220, 208, 186], category: 'Natural' },
  { name: 'Brown', rgb: [101, 67, 33], category: 'Natural' },
  // Add new colors here
]
```

### Custom Themes

```css
/* Define custom theme */
:root[data-theme='dark'] {
  --color-gray-50: #111827;
  --color-gray-900: #f9fafb;
  /* Override other variables */
}

/* Theme switcher implementation */
.theme-switcher {
  /* Theme toggle styles */
}
```

### Adding New Languages

1. Create translation file:

```json
// locales/es.json
{
  "upload": {
    "title": "Sube tu Foto"
    // ... translations
  }
}
```

2. Update i18n configuration:

```typescript
// i18n.ts
import es from './locales/es.json'

const i18n = createI18n({
  // ... existing config
  messages: { en, tr, es },
})
```

## 🔧 Troubleshooting

### Common Issues

#### 1. API Connection Problems

```bash
# Check if backend is running
curl http://localhost:8000/health

# Verify proxy configuration in vite.config.ts
# Check VITE_API_BASE_URL environment variable
```

#### 2. Build Errors

```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Check TypeScript errors
npm run type-check

# Update dependencies
npm update
```

#### 3. CORS Issues

```typescript
// Verify proxy configuration
export default defineConfig({
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
```

#### 4. Image Upload Issues

```javascript
// Check file size and format validation
const validateFile = (file) => {
  const maxSize = 10 * 1024 * 1024 // 10MB
  const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg']

  if (file.size > maxSize) {
    throw new Error('File too large')
  }

  if (!allowedTypes.includes(file.type)) {
    throw new Error('Invalid file type')
  }
}
```

### Development Tips

1. **Hot Reload Issues**: Restart dev server if changes aren't reflecting
2. **TypeScript Errors**: Run `npm run type-check` to see all TS issues
3. **Component Inspector**: Use Vue DevTools browser extension
4. **Performance**: Use Vite's built-in bundle analyzer
5. **Debugging**: Add `debugger` statements and use browser dev tools

### Performance Optimization

```typescript
// Lazy load components
const HeavyComponent = defineAsyncComponent(() => import('./HeavyComponent.vue'))

// Image optimization
const optimizeImage = (file: File, maxWidth = 1024) => {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    const img = new Image()

    img.onload = () => {
      const ratio = Math.min(maxWidth / img.width, maxWidth / img.height)
      canvas.width = img.width * ratio
      canvas.height = img.height * ratio

      ctx?.drawImage(img, 0, 0, canvas.width, canvas.height)
      canvas.toBlob(resolve, 'image/jpeg', 0.8)
    }

    img.src = URL.createObjectURL(file)
  })
}
```

---

For more information about the overall project, see the main [README](../README.md).
