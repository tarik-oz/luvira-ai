# Hair Segmentation & Color Change

Flexible and modular system for hair segmentation and color change.

- **Model:** PyTorch-based U-Net and Attention U-Net for hair segmentation
- **Color Changer:** Realistic hair recoloring using HSV and natural blending
- **API:** FastAPI REST endpoints for mask prediction and hair color change

## Features

- Train and evaluate segmentation models on custom datasets
- Predict hair masks and recolor hair in images via API
- Robust validation, error handling, and modular code structure

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd deep-learning-hair-segmentation
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API:**
   ```bash
   python -m api.run_api
   ```

4. **Try the endpoints:**
   - `POST /predict-mask` — Upload an image, get hair mask
   - `POST /change-hair-color` — Upload an image and RGB values, get recolored image

## Project Structure

- `model/` — Segmentation models, training, and inference
- `color_changer/` — Hair color change utilities
- `api/` — FastAPI application and endpoints

## License

MIT

---

*For more details, see the code and comments in each module!*
