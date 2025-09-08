# Deep Learning Hair Segmentation

> AI-powered hair color transformation with realistic results in seconds

A comprehensive hair segmentation and realistic color transformation system powered by deep learning. This project provides both a REST API and a Vue.js frontend for accurate hair mask prediction and natural-looking hair recoloring.

🌐 **Live Demo**: [luviraai.app](https://luviraai.app)

## 🎥 System in Action

<video width="100%" autoplay muted loop>
  <source src="./docs/luvira-ai-demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

_Real-time hair color transformation with AI-powered segmentation_

## 🌟 Features

- **AI-Powered Hair Segmentation**: Attention U-Net model for precise hair boundary detection
- **Realistic Color Transformation**: HSV-based color change with natural blending and tone variations
- **Session-Based Processing**: Fast color trials without regenerating masks
- **Mobile-Optimized UI**: Responsive Vue.js frontend with camera capture support
- **REST API**: FastAPI backend with comprehensive endpoints
- **Multiple Deployment Options**: Docker Compose for easy local development

## 🏗️ System Architecture

![System Architecture](./docs/system-architecture-diagram.png)

The system follows a microservices architecture with clean separation between frontend, backend API, and core processing modules:

- **Vue.js Frontend**: Responsive UI with real-time preview and mobile optimization
- **FastAPI Backend**: RESTful API with session management and async processing
- **Deep Learning Module**: Attention U-Net for precise hair segmentation
- **Color Transformation Engine**: HSV-based realistic color blending
- **Session Storage**: Filesystem or S3-based caching for fast color trials

## 🚀 Quick Start with Docker

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd deep-learning-hair-segmentation
```

### 2. Start the Application

```bash
# Start all services (API + Frontend)
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

### 3. Access the Application

- **Frontend**: http://localhost:5173
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/

### 4. Stop the Application

```bash
docker-compose down
```

## 📁 Project Structure

```
deep-learning-hair-segmentation/
├── 📁 api/                    # FastAPI Backend
│   ├── routes/                # API endpoints
│   ├── services/              # Business logic
│   ├── core/                  # Core components
│   └── README.md             # Backend documentation
├── 📁 frontend/               # Vue.js Frontend
│   ├── src/                   # Source code
│   ├── public/                # Static assets
│   └── README.md             # Frontend documentation
├── 📁 model/                  # Deep Learning Models
│   ├── models/                # Model architectures
│   ├── training/              # Training scripts
│   ├── inference/             # Prediction logic
│   └── README.md             # Model documentation
├── 📁 color_changer/          # Color Transformation
│   ├── core/                  # Main transformer
│   ├── transformers/          # HSV processing
│   ├── utils/                 # Helper functions
│   └── README.md             # Color changer documentation
├── docker-compose.yml        # Multi-service orchestration
├── Dockerfile                # Backend container
└── requirements.txt          # Python dependencies
```

## 🔧 Module Documentation

Each module has detailed documentation for local development and advanced usage:

- **[🤖 Model Training & Inference](./model/README.md)** - Train new models, run predictions, CLI tools
- **[🚀 API Backend](./api/README.md)** - FastAPI development, endpoints, local setup
- **[🎨 Color Transformation](./color_changer/README.md)** - Color algorithms, custom colors, testing tools
- **[💻 Frontend](./frontend/README.md)** - Vue.js development, components, UI customization

## 🛠️ Development Setup (Without Docker)

For module-specific development, see individual README files. Each module can be run independently:

1. **Model Training**: See [model/README.md](./model/README.md)
2. **API Development**: See [api/README.md](./api/README.md)
3. **Color Testing**: See [color_changer/README.md](./color_changer/README.md)
4. **Frontend Development**: See [frontend/README.md](./frontend/README.md)

## 📝 API Usage Examples

### Upload and Process Image

```bash
# Upload image and get session ID
curl -X POST "http://localhost:8000/upload-and-prepare" \
  -F "file=@image.jpg" \
  -F "source=upload_section"

# Change hair color using session
curl -X POST "http://localhost:8000/change-hair-color-with-session/session_abc123" \
  -F "color_name=Blonde" \
  -F "tone=golden"
```

### Get Available Colors

```bash
curl "http://localhost:8000/available-colors"
curl "http://localhost:8000/available-tones/Blonde"
```

### Logs

```bash
# View API logs
docker-compose logs api

# View frontend logs
docker-compose logs frontend

# View all logs
docker-compose logs -f
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Tarık Öz**

---

_For detailed module documentation, please refer to the individual README files in each directory._
