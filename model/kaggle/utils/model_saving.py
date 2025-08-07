import json
import shutil
from datetime import datetime
from pathlib import Path

def create_timestamped_folder(trained_models_dir: Path, best_val_accuracy: float) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    accuracy_str = f"acc{best_val_accuracy:.4f}"
    folder_name = f"{timestamp}_{accuracy_str}"
    folder_path = trained_models_dir / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"Created model folder: {folder_path}")
    return folder_path

def save_config_json(folder_path: Path, model_config: dict, training_config: dict, callbacks_config: dict, dataset_info: dict, training_results: dict) -> None:
    config_data = {
        "model_config": model_config,
        "training_config": training_config,
        "callbacks_config": callbacks_config,
        "dataset_info": dataset_info or {},
        "training_results": training_results,
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    }
    config_path = folder_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    print(f"Saved config to: {config_path}")

def save_training_log(folder_path: Path, device, history: dict, summary: dict) -> None:
    log_path = folder_path / "training_log.txt"
    with open(log_path, 'w') as f:
        f.write("=== Training Started ===\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n\n")
        f.write("=== Epoch Details ===\n")
        for epoch, (train_loss, train_acc, train_dice, train_mse, val_loss, val_acc, val_dice, val_mse) in enumerate(
            zip(history['train_loss'], history['train_accuracy'], history['train_dice'], history['train_mse'],
                history['val_loss'], history['val_accuracy'], history['val_dice'], history['val_mse']), 1):
            f.write(f"Epoch {epoch}:\n")
            f.write(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Dice: {train_dice:.4f}, Train MSE: {train_mse:.6f}\n")
            f.write(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Dice: {val_dice:.4f}, Val MSE: {val_mse:.6f}\n\n")
        f.write("=== Best Model Info ===\n")
        f.write(f"Best Val Accuracy: {summary.get('best_val_accuracy', 0):.4f}\n")
        f.write(f"Best Val Dice: {summary.get('best_val_dice', 0):.4f}\n")
        f.write(f"Best Val MSE: {summary.get('best_val_mse', 0):.6f}\n")
        f.write(f"Best Val Loss: {summary.get('best_val_loss', 0):.4f}\n\n")
        f.write("=== Training Summary ===\n")
        f.write(f"Total Epochs: {summary.get('total_epochs', 0)}\n")
        f.write(f"Final Train Accuracy: {summary.get('final_train_accuracy', 0):.4f}\n")
        f.write(f"Final Train Dice: {summary.get('final_train_dice', 0):.4f}\n")
        f.write(f"Final Train MSE: {summary.get('final_train_mse', 0):.6f}\n")
        f.write(f"Final Val Accuracy: {summary.get('final_val_accuracy', 0):.4f}\n")
        f.write(f"Final Val Dice: {summary.get('final_val_dice', 0):.4f}\n")
        f.write(f"Final Val MSE: {summary.get('final_val_mse', 0):.6f}\n")
    print(f"Saved training log to: {log_path}")

def save_models_to_folder(folder_path: Path, best_model_path: Path, latest_model_path: Path) -> None:
    best_model_dest = folder_path / "best_model.pth"
    if best_model_path.exists():
        shutil.copy2(best_model_path, best_model_dest)
        print(f"Saved best model to: {best_model_dest}")
    latest_model_dest = folder_path / "latest_model.pth"
    if latest_model_path.exists():
        shutil.copy2(latest_model_path, latest_model_dest)
        print(f"Saved latest model to: {latest_model_dest}") 