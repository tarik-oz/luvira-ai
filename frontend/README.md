# Vue.js Frontend

## 1. Overview

This module contains the Vue.js 3 frontend application for the LuviraAI project. It provides a modern, responsive user interface with image upload, camera capture, color selection, and real-time preview capabilities for hair color transformation.

## 2. Tech Stack

- **Framework**: Vue.js 3 with Composition API
- **Language**: TypeScript
- **Build Tool**: Vite
- **Routing**: Vue Router 4
- **Internationalization**: Vue I18n
- **Styling**: CSS3 with CSS Variables

## 🚀 Getting Started (Local Development)

This guide will get you a local copy of the frontend up and running for development.

### Prerequisites

- Node.js 18+ and npm/yarn/pnpm
- Backend API running (see [API Module](../api/README.md))

### Installation & Running

1. **Navigate to frontend directory:**

   ```bash
   cd frontend
   ```

2. **Install dependencies:**

   ```bash
   npm install
   ```

3. **Start development server:**
   ```bash
   npm run dev
   ```

**Access the Application:**

- **Frontend**: http://localhost:5173
- **Hot Reload**: Automatically refreshes on code changes

## 🏗️ Frontend Architecture

The application follows Vue.js 3 best practices with a clean, modular structure:

- **Composition API**: Modern Vue.js development with reactive state management
- **Component-Based**: Reusable UI components with clear separation of concerns
- **Service Layer**: Dedicated API services for backend communication
- **Proxy Configuration**: Vite proxy handles API requests during development
- **State Management**: Vue 3 composables for global state (no Pinia needed)

## 🎯 Key Features

- **Responsive Design**: Mobile-first approach with touch-friendly interactions
- **Multi-Input Support**: File upload, camera capture, and sample image gallery
- **Real-time Preview**: Live color transformation preview with session-based processing
- **Internationalization**: Multi-language support (English, Turkish)
- **Theme Support**: Light and dark mode themes for enhanced user experience
- **Modern UI**: Clean, intuitive interface optimized for hair color selection

## 🌐 API Integration

The frontend communicates with the FastAPI backend through:

- **Proxy Configuration**: Vite dev server proxies `/api` requests to backend
- **Service Layer**: Dedicated API service classes handle all backend communication
- **Error Handling**: Comprehensive error handling with user-friendly messages

---

For more information about the overall project, see the main [README](../README.md).
