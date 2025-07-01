import os
import json
import logging
from pathlib import Path
from config import FILE_PATTERNS

def get_latest_timestamp(directory, patterns=None):
    if patterns is None:
        patterns = FILE_PATTERNS["masks"] if "mask" in str(directory).lower() else FILE_PATTERNS["images"]
    latest = 0
    for pattern in patterns:
        for file in Path(directory).glob(pattern):
            ts = os.path.getmtime(file)
            if ts > latest:
                latest = ts
    return latest

def save_timestamps(processed_dir, image_ts, mask_ts):
    ts_path = Path(processed_dir) / "timestamps.json"
    with open(ts_path, "w") as f:
        json.dump({"images": image_ts, "masks": mask_ts}, f)
    logging.info(f"Timestamps file saved: {ts_path}")

def load_timestamps(processed_dir):
    ts_path = Path(processed_dir) / "timestamps.json"
    if not ts_path.exists():
        logging.info("Timestamps file not found.")
        return None
    with open(ts_path, "r") as f:
        logging.info(f"Timestamps file loaded: {ts_path}")
        return json.load(f)

def needs_processing(images_dir, masks_dir, processed_dir):
    image_ts = get_latest_timestamp(images_dir)
    mask_ts = get_latest_timestamp(masks_dir)
    saved_ts = load_timestamps(processed_dir)
    if saved_ts is None:
        logging.info("No timestamps found. Data will be processed for the first time.")
        return True
    if image_ts != saved_ts["images"] or mask_ts != saved_ts["masks"]:
        logging.info("Change detected in images or masks. Data will be reprocessed.")
        return True
    logging.info("Data is up-to-date. Existing processed npy files will be used.")
    return False 