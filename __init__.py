"""
Flexible and modular hair segmentation and color change system.

Includes:
- PyTorch-based U-Net and Attention U-Net models for segmentation
- Advanced color changer utilities for realistic hair recoloring
- FastAPI-based REST API for mask prediction and color change
- Tools for training, evaluation, and dataset management

v3.0.0 summary:
- Model: training checkpoints/resume, Kaggle training setup, richer augmentation
- Color changer: per-color special handlers and extended tone configs
- API: frontend endpoints and session-based workflow
- Frontend: responsive design UI
"""

__version__ = "3.0.0"
__author__ = "Tarik"
__description__ = (
    "Hair segmentation model training, hair color changer, API endpoints, and responsive UI design."
)