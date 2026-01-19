# Drusen Image Segmentation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haythamlabrini/drusen-images-segmentation/blob/main/Run_model_on_all_flattened_images.ipynb)

Medical image segmentation project for automated detection and quantification of **drusen** in OCT (Optical Coherence Tomography) retinal scans using deep learning.

## What are Drusen?

Drusen are yellow deposits of fatty proteins (lipids) that accumulate under the retina. They are a key biomarker for Age-related Macular Degeneration (AMD), one of the leading causes of vision loss in people over 50. Early detection and quantification of drusen is critical for:

- Early AMD diagnosis
- Disease progression monitoring
- Treatment planning and response assessment
- Clinical research studies

## Features

- **Automated Segmentation**: Binary semantic segmentation to identify drusen regions
- **Batch Processing**: Process thousands of OCT images automatically
- **Quantitative Analysis**: Count drusen objects and measure total affected area
- **Visualization**: Side-by-side comparison of original images and predicted masks
- **Metrics Export**: CSV output with per-image and combined statistics

## Model Architecture

### Framework Stack

| Component | Technology |
|-----------|------------|
| Deep Learning | PyTorch + PyTorch Lightning |
| Segmentation | [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models_pytorch) (smp) |
| Image Processing | OpenCV, PIL, torchvision |
| GPU Acceleration | CUDA via Google Colab |

### Neural Network

The `DrusenModel` class is a PyTorch Lightning module supporting multiple encoder-decoder architectures:

- **Configurable Architectures**: UNet, FPN, DeepLabV3+, PSPNet, and more via `smp.create_model()`
- **Pre-trained Encoders**: ImageNet weights with encoder-specific preprocessing
- **Binary Segmentation**: Single-class output (drusen vs. background)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss Function | Dice Loss (binary mode, from_logits=True) |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Batch Size | 10 |
| Gradient Accumulation | 10 batches |
| Input Normalization | Encoder-specific mean/std from smp |

### Metrics Tracked

- **IoU (Intersection over Union)**: Per-image and dataset-wide
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: Harmonic mean of precision and recall

## Dataset Specifications

| Property | Value |
|----------|-------|
| Image Format | JPG |
| Original Resolution | 1024 × 885 pixels |
| Model Input Size | Resized to dimensions divisible by 32 |
| Total Images | ~4,170 (validated for consistency) |
| File Structure | Flat directory with unique filenames |

### Data Validation

The notebooks include automatic validation for:
- Duplicate detection (0 duplicates in final dataset)
- File extension verification (all .jpg)
- Resolution consistency check (all 1024×885)

## Data Augmentation

During training, images undergo the following transformations:

```python
transforms.Compose([
    transforms.RandomRotation(degrees=(-10, 10)),  # ±10° rotation
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])
```

## Project Structure

```
drusen-images-segmentation/
├── drsuen_tests.ipynb                      # Model training & evaluation
├── Run_model_on_all_flattened_images.ipynb # Batch inference on full dataset
├── CLAUDE.md                               # Claude Code AI assistant context
├── PRD-CLAUDE-MD-IMPROVEMENTS.md           # Development planning document
└── README.md                               # This file
```

## Notebooks

### 1. Training Notebook (`drsuen_tests.ipynb`)

Full training pipeline including:

1. **Environment Setup**: GPU runtime configuration, dependency installation
2. **Data Loading**: CSV-based dataset management with train/val/test splits (70/15/15)
3. **Augmentation**: torchvision transforms for training robustness
4. **Model Definition**: PyTorch Lightning module with configurable architecture
5. **Training Loop**: Multi-epoch training with TensorBoard logging
6. **Validation**: Per-epoch metrics tracking
7. **Testing**: Final model evaluation
8. **Inference**: Single-image prediction examples

### 2. Batch Inference Notebook (`Run_model_on_all_flattened_images.ipynb`)

Production inference pipeline:

1. **Data Validation**: Verify dataset integrity (count, duplicates, dimensions)
2. **Model Loading**: Load trained checkpoint from Google Drive
3. **Batch Processing**: Memory-optimized inference with GPU acceleration
4. **Output Generation**:
   - Original image preservation
   - Binary prediction masks
   - Side-by-side visualization figures
5. **Metrics Extraction**: OpenCV contour analysis for drusen quantification
6. **CSV Export**: Per-image and combined dataset metrics

## Output Structure

For each processed image, a results folder is created:

```
predictions_path/
├── image_001/
│   ├── original.png      # Original OCT image
│   ├── prediction.png    # Binary segmentation mask (white = drusen)
│   ├── figure.png        # Side-by-side comparison visualization
│   └── metrics.csv       # Object count and area for this image
├── image_002/
│   └── ...
└── combined_metrics.csv  # Aggregated metrics for all images
```

### Metrics CSV Format

| Column | Description |
|--------|-------------|
| filename | Image identifier (without extension) |
| prediction_count | Number of distinct drusen objects detected |
| prediction_area | Total white pixel count (drusen area in pixels) |

## Usage

### Prerequisites

- Google account for Colab access
- Google Drive storage for data and checkpoints
- GPU runtime (CUDA-enabled)

### Quick Start

1. **Open in Colab**: Click the badge at the top of this README
2. **Mount Drive**: Allow access to your Google Drive
3. **Install Dependencies**:
   ```python
   !pip install segmentation-models-pytorch pytorch-lightning torchvision tensorflow pandas
   ```
4. **Update Paths**: Modify these variables to match your Drive structure:
   ```python
   complete_batch_path = '/content/drive/MyDrive/Drusen Project/complete-batch-flat-finale'
   predictions_path = '/content/drive/MyDrive/Drusen Project/feb7_results'
   latest_version_checkpoint = '/content/drive/MyDrive/.../epoch=9-step=180.ckpt'
   ```
5. **Run All Cells**: Execute the notebook sequentially

### Training a New Model

1. Open `drsuen_tests.ipynb`
2. Prepare your labeled dataset (images + binary masks)
3. Update configuration:
   ```python
   ENCODER = "timm-efficientnet-b0"  # Or other smp-supported encoders
   WEIGHT = "imagenet"
   EPOCH = 25
   LR = 0.003
   BATCH_SIZE = 16
   ```
4. Run training cells
5. Model checkpoints saved to TensorBoard log directory

### Running Inference

1. Open `Run_model_on_all_flattened_images.ipynb`
2. Point to your trained checkpoint
3. Set input/output directories
4. Run all cells
5. Results saved to specified predictions folder

## Technical Details

### Memory Optimization

The batch inference notebook includes memory management for processing large datasets:

```python
# Non-blocking GPU transfer
image = batch["image"].cuda(non_blocking=True)

# Immediate memory cleanup
del image_to_plot, predicted_mask_to_plot
torch.cuda.empty_cache()
```

### Drusen Quantification

OpenCV-based contour analysis extracts drusen metrics:

```python
def count_white_objects_and_area(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_white_objects = len(contours)
    num_white_pixels = np.sum(image == 255)
    return num_white_objects, num_white_pixels
```

### Image Preprocessing

Images are resized to dimensions divisible by 32 (required by encoder-decoder skip connections):

```python
def resize_image_for_model(image, should_augment):
    original_width, original_height = image.size
    desired_width = original_width - (original_width % 32)
    desired_height = original_height - (original_height % 32)
    resized_image = image.resize((desired_width, desired_height))
    # Optional augmentation...
    return resized_image
```

## Expected Google Drive Structure

```
Google Drive/
└── Drusen Project/
    ├── complete-batch-flat-finale/    # Input OCT images
    │   ├── image_001.jpg
    │   ├── image_002.jpg
    │   └── ...
    ├── syncedLatest/                  # Model checkpoints
    │   └── logs/tb_logs/
    │       └── drusen_model_v1/
    │           └── version_17/
    │               └── checkpoints/
    │                   └── epoch=9-step=180.ckpt
    └── feb7_results/                  # Prediction outputs
        ├── image_001/
        └── combined_metrics.csv
```

## Dependencies

```
segmentation-models-pytorch>=0.3.0
pytorch-lightning>=2.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.0.0
pillow>=9.0.0
numpy>=1.20.0
pandas>=1.5.0
matplotlib>=3.5.0
tensorflow>=2.10.0  # For TensorBoard logging
```

## Model Checkpoints

Pre-trained model checkpoints are stored in Google Drive:
- Path: `/Drusen Project/syncedLatest/syncedLatest/logs/tb_logs/drusen_model_v1/`
- Format: PyTorch Lightning `.ckpt` files
- Loading: `DrusenModel.load_from_checkpoint(checkpoint_path)`

## Performance Notes

- **GPU Required**: CUDA-enabled GPU recommended (Colab provides free T4/P100)
- **Processing Time**: ~4,170 images processed in batch mode with checkpointing
- **Memory**: Batch processing with explicit memory cleanup prevents OOM errors
- **Resume Capability**: Skips already-processed images if output exists

## Filename Convention

OCT image filenames follow this pattern:
```
{patient_id}_{date}_{eye}_{location}_{scan_number}.jpg
```

Example: `019_2023-03-06_OD_T_10.jpg`
- `019`: Patient ID
- `2023-03-06`: Scan date
- `OD`: Right eye (Oculus Dexter) / `OS`: Left eye (Oculus Sinister)
- `T`: Top / `B`: Bottom (scan location)
- `10`: Scan number in series

## License

This project is for research and educational purposes. Please ensure compliance with any data usage agreements when working with medical imaging data.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{drusen_segmentation,
  author = {Haytham Labrini},
  title = {Drusen Image Segmentation},
  url = {https://github.com/haythamlabrini/drusen-images-segmentation},
  year = {2024}
}
```

## Acknowledgments

- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models_pytorch) by Pavel Yakubovskiy
- [PyTorch Lightning](https://lightning.ai/) for training framework
- Google Colab for GPU compute resources
