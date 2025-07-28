# Kaggle Folder Usage Guide

## 📁 Files

- `config.py` - Configuration settings for Kaggle environment
- `train.py` - Main training script (uses updated model structure)
- `setup.py` - Kaggle environment preparation and verification script
- `create_zip.py` - Kaggle zip file creator

## 🚀 Usage

### 1. Creating Kaggle ZIP

Create zip file for uploading to Kaggle:

```bash
cd model/kaggle
python create_zip.py
```

This command:

- Collects all necessary model files
- Creates `hair_segmentation_kaggle_YYYYMMDD_HHMMSS.zip` file
- Saves it in the main directory (project root)

### 2. Using on Kaggle

#### 2.1. Upload ZIP to Kaggle as Dataset

1. Upload the created zip file to Kaggle as a dataset
2. Note the dataset name (e.g., `my-hair-segmentation`)

#### 2.2. Extract in Kaggle Notebook

```python
import zipfile
import os

# Extract the zip file
with zipfile.ZipFile('/kaggle/input/my-hair-segmentation/hair_segmentation_kaggle_*.zip', 'r') as zipf:
    zipf.extractall('/kaggle/working/')

# Change to working directory
os.chdir('/kaggle/working')
```

#### 2.3. Prepare Kaggle Environment (Optional)

```python
# Run setup script
from model.kaggle.setup import main as setup_main
setup_main()
```

#### 2.4. Start Training

```python
# Run main training script
%run main_kaggle_train.py

# OR directly with import:
from model.kaggle.train import main
main()
```

### 3. Configuration

Modify settings in `model/kaggle/config.py`:

```python
# Change dataset name
DATASET_NAME = "your-dataset-name-in-kaggle"

# Training settings
TRAINING_CONFIG = {
    "batch_size": 24,        # Adjust according to GPU memory
    "epochs": 50,
    "learning_rate": 2e-4,
    "model_type": "attention_unet",  # or "unet"
    # ...
}
```

## 📋 Requirements

### Kaggle Dataset Structure

```
your-dataset/
├── images/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
└── masks/
    ├── 0.webp
    ├── 1.webp
    └── ...
```

### Python Packages

- Usually pre-installed in Kaggle environment
- Missing packages will be auto-installed by `setup.py`

## 🔧 Troubleshooting

### Dataset Not Found

```python
# Check dataset name in config.py
DATASET_NAME = "correct-dataset-name"
```

### Memory Error

```python
# Reduce batch size in config.py
TRAINING_CONFIG["batch_size"] = 16  # or 8
```

### Import Errors

```python
# Make sure you extracted the zip file correctly
import sys
sys.path.insert(0, '/kaggle/working')
```

## 💡 Tips

1. **GPU Usage**: batch_size=24 is optimized for Kaggle P100 GPU
2. **Dataset Size**: Lazy loading is used for 30K+ images
3. **Checkpoint**: Model is automatically saved to `/kaggle/working/trained_models/`
4. **Log File**: Training logs are stored in `/kaggle/working/training.log`

## 📞 Quick Start

To do everything with one command:

```bash
# 1. Create zip
python model/kaggle/create_zip.py

# 2. Upload to Kaggle and in notebook:
%run main_kaggle_train.py
```
