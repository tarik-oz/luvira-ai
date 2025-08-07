from pathlib import Path

KAGGLE_WORKING_DIR = Path("/kaggle/working/")

KAGGLE_INPUT_DIR = Path("/kaggle/input/hair-dataset-99")

DATA_DIR = KAGGLE_INPUT_DIR

IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR = DATA_DIR / "masks"

PROCESSED_DATA_DIR = KAGGLE_WORKING_DIR / "processed_data"

TRAINED_MODELS_DIR = KAGGLE_WORKING_DIR / "trained_models"

TEST_RESULTS_DIR = KAGGLE_WORKING_DIR / "test_results"

for directory in [PROCESSED_DATA_DIR, TRAINED_MODELS_DIR, TEST_RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

MODEL_CONFIG = {
    "input_shape": (3, 256, 256),
    "num_filters": [64, 128, 256, 512],
    "bridge_filters": 512,
    "output_channels": 1,
    "activation": "sigmoid",
    "model_type": "attention_unet"
}

TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 150,
    "learning_rate": 1e-4,
    "validation_split": 0.15,
    "random_seed": 42,
    "loss_function": "total",
    "bce_weight": 0.3,
    "dice_weight": 0.4,
    "boundary_weight": 0.3,
    "optimizer": "adamw",
    "device": "auto",

    "resume_training": False,
    "checkpoint_path": None,
}

DATA_CONFIG = {
    "image_size": (256, 256),
    "normalization_factor": 255.0,
    "mask_threshold": 0.5,
    "lazy_loading": True,
    "use_augmentation": True,
    "num_workers": 4,
}

CALLBACKS_CONFIG = {
    "checkpoint_monitor": "val_dice",
    "reduce_lr_monitor": "val_loss",
    "reduce_lr_patience": 5,
    "reduce_lr_factor": 0.5,
    "reduce_lr_min_lr": 1e-7,
    "early_stopping_monitor": "val_dice",
    "early_stopping_patience": 15
}

FILE_PATTERNS = {
    "images": ["*.jpg", "*.jpeg", "*.png"],
    "masks": ["*.webp", "*.png", "*.jpg"],
    "processed_images": "train_images.npy",
    "processed_masks": "train_masks.npy",
    "validation_images": "val_images.npy",
    "validation_masks": "val_masks.npy"
}

print("Kaggle config.py loaded successfully - Optimized for 30K Hair Dataset")
print(f"Data will be read from: {DATA_DIR}")
print(f"Models will be saved to: {TRAINED_MODELS_DIR}")
print(f"Training config: {TRAINING_CONFIG['epochs']} epochs, batch_size={TRAINING_CONFIG['batch_size']}")
print(f"Model: {MODEL_CONFIG['model_type']} with {MODEL_CONFIG['bridge_filters']} bridge filters")
print(f"Optimizer: {TRAINING_CONFIG['optimizer']} with LR={TRAINING_CONFIG['learning_rate']}")