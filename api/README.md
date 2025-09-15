# API Backend

## 1. Overview

This module contains the FastAPI application that serves as the backend for the LuviraAI project. It handles all image processing, model inference, session management, and provides RESTful endpoints for hair color transformation.

## 2. Tech Stack

- **Framework**: FastAPI
- **Server**: Uvicorn
- **Data Validation**: Pydantic
- **AI/ML**: PyTorch, OpenCV
- **Cloud Integration**: Boto3 (for AWS S3)

## 🚀 Getting Started (Local Development)

This guide will get you a local copy of the API up and running for development and testing.

### Prerequisites

- Python 3.10+
- A virtual environment (e.g., `venv`)
- A trained model file (see [Model Module](../model/README.md))

### Installation & Running

1. **Create and activate a virtual environment from the project root:**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install all dependencies from the root requirements.txt:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install PyTorch (choose CPU or GPU version):**

   ```bash
   # For CPU
   pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

   # For GPU (CUDA 11.8)
   pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Start the development server with hot-reloading:**
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

**Access the API:**

- **Swagger UI**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🏗️ API Architecture & Key Concepts

The API is built on a clean, layered architecture to ensure separation of concerns and maintainability.

- **Layered Architecture (Routes → Services)**: Endpoints defined in the `routes/` directory handle HTTP requests and validation. They then call the appropriate functions in the `services/` directory, which contain the core business logic (model inference, color transformation, etc.). This keeps the API layer thin and the business logic reusable.

- **Session Management**: To provide a fast and interactive user experience, the API uses a session-based workflow. When an image is first uploaded (`/upload-and-prepare`), the hair mask is generated once and cached (either on the local filesystem or AWS S3). Subsequent color change requests use this cached data, bypassing the time-consuming model inference step. Sessions are automatically cleaned up after a configurable timeout.

- **Dependency Injection**: Key components like the `ModelService` and `ColorChangeService` are managed as singletons and injected into the API routes. This ensures that the heavy AI model is loaded into memory only once, improving performance.

## 📚 API Reference

The following is a summary of the available endpoints. All endpoints are also documented and testable via the interactive Swagger UI at `/docs`.

### Public Endpoints (Core Workflow)

#### Upload and Prepare Workflow

![Upload and Prepare](../docs/flows/upload-and-prepare.png)

- **POST /upload-and-prepare**: Uploads an image, generates a hair mask, and creates a user session.

#### Overlays with Session Workflow

![Overlays with Session](../docs/flows/overlays-with-session.png)

- **POST /overlays-with-session/{session_id}**: Streams all color tone variations as transparent overlays for a given session.

### Other Endpoints

<details>
<summary><strong>Click to expand detailed endpoint list...</strong></summary>

**Core Processing**

- POST /predict-mask: Returns only the hair segmentation mask for an image.
- POST /change-hair-color: A slower, single-step endpoint to change hair color without using sessions.

**Information**

- GET /available-colors: Lists all available primary color categories.
- GET /available-tones/{color_name}: Lists all available tones for a specific color.

**Model Management (Development Only)**

- GET /model-info: Get information about the currently loaded model.
- POST /reload-model: Reload the AI model from a specified path.
- POST /clear-model: Clear the model from memory.

**Session Management (Development Only)**

- GET /session-stats: Get statistics about active and expired sessions.
- DELETE /cleanup-session/{session_id}: Manually clean up a specific session.
- POST /cleanup-all-sessions: Manually clean up all sessions.

</details>

## ⚙️ Advanced Configuration

The API can be configured using environment variables. For local development, you can create a `.env` file in the project's root directory.

<details>
<summary><strong>Click to see all environment variables...</strong></summary>

```bash
# Server Configuration
HOST=0.0.0.0          # Server host
PORT=8000             # Server port
APP_ENV=development   # Environment (development/production)

# CORS Configuration
CORS_ORIGINS=*        # Allowed origins

# Model Configuration
MODEL_LOCAL_PATH=model/trained_models/luviraai_weights.pth

# File Validation
MAX_FILE_SIZE=10485760          # Max upload size (10MB)
ALLOWED_IMAGE_TYPES=image/jpeg,image/png,image/jpg

# Session Configuration
SESSION_BACKEND=filesystem      # Backend: filesystem or s3
SESSION_TIMEOUT_MINUTES=30      # Session timeout
SESSION_CACHE_DIR=session_data  # Local cache directory
```

</details>

For more information, see the main project [README](../README.md).
