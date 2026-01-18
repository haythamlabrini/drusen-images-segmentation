# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Medical image segmentation project for detecting **drusen** (deposits under the retina) in OCT (Optical Coherence Tomography) eye scans. Uses deep learning with PyTorch and segmentation-models-pytorch for binary semantic segmentation.

## Architecture

### Model
- **Framework**: PyTorch Lightning + segmentation-models-pytorch (smp)
- **Architecture**: Configurable via `smp.create_model()` (supports UNet, FPN, DeepLabV3+, etc.)
- **Loss Function**: Dice Loss (binary mode)
- **Metrics**: IoU, Accuracy, Precision, Recall, F1

### Key Class: `DrusenModel`
Located in both notebooks. A PyTorch Lightning module that:
- Normalizes images using encoder-specific preprocessing params
- Outputs binary segmentation masks (drusen vs background)
- Tracks per-image and dataset-wide IoU metrics
- Uses Adam optimizer with lr=0.0001

### Data Pipeline
- **Input**: 1024x885 JPG images (resized to dimensions divisible by 32)
- **Augmentation**: Random rotation (±10°), horizontal/vertical flips
- **Normalization**: Encoder-specific mean/std from smp
- **DataLoader**: Batch size 10, multiprocessing enabled

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `drsuen_tests.ipynb` | Model training, validation, testing with data splits (70/15/15) |
| `Run_model_on_all_flattened_images.ipynb` | Batch inference on full dataset, generates predictions and CSV metrics |

## Running in Google Colab

Both notebooks are designed for Colab with GPU runtime:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install segmentation-models-pytorch pytorch-lightning torchvision tensorflow pandas
```

### Expected Drive Paths
- Input images: `/content/drive/MyDrive/Drusen Project/complete-batch-flat-finale/`
- Model checkpoints: `/content/drive/MyDrive/Drusen Project/syncedLatest/syncedLatest/logs/tb_logs/`
- Predictions output: `/content/drive/MyDrive/Drusen Project/feb7_results/`

## Output Structure

For each processed image, creates a folder containing:
```
predictions_path/
  image_name/
    original.png      # Original image
    prediction.png    # Predicted mask (white = drusen)
    figure.png        # Side-by-side comparison
    metrics.csv       # Object count and area
```

Final combined metrics: `combined_metrics.csv` with columns: filename, prediction_count, prediction_area

## Key Functions

| Function | Purpose |
|----------|---------|
| `resize_image_for_model()` | Resize to dimensions divisible by 32 + optional augmentation |
| `set_data()` | Prepare image/mask pairs for DataLoader |
| `process_dataloader()` | Run inference, save predictions with memory optimization |
| `count_white_objects_and_area()` | OpenCV contour analysis for drusen metrics |
