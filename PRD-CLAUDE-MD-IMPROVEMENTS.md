# PRD: CLAUDE.md Specification Improvements

## Document Information
- **Project**: drusen-images-segmentation
- **Target File**: CLAUDE.md
- **Created**: 2026-01-18
- **Status**: Draft
- **Priority**: Medium

---

## Problem Statement

The current CLAUDE.md provides basic project documentation but lacks critical information for developers to effectively work with the codebase. Key gaps identified by expert panel review:

1. No local development path (Colab-only assumption)
2. Missing environment/dependency requirements
3. No error handling or troubleshooting guidance
4. No verification steps to confirm setup works
5. Missing model checkpoint details and compatibility notes

**Current Quality Score**: 6.8/10

---

## Goals

| Goal | Success Metric |
|------|----------------|
| Enable local development | Developer can run inference without Colab |
| Reduce setup failures | Include all prerequisite requirements |
| Improve troubleshooting | Document common errors and solutions |
| Enable verification | Provide steps to confirm working setup |
| Improve actionability | Add concrete workflow examples |

**Target Quality Score**: 8.5/10

---

## Proposed Changes

### 1. Add Environment Requirements Section

**Location**: After "Project Overview", before "Architecture"

```markdown
## Requirements

### Python Environment
- Python 3.8 - 3.10 (PyTorch Lightning compatibility)
- CUDA 11.x or 12.x (for GPU acceleration)

### Hardware
- GPU: NVIDIA with 8GB+ VRAM (for batch_size=10)
- RAM: 16GB+ recommended for large datasets
- Storage: ~500MB for model checkpoint

### Dependencies
```bash
pip install segmentation-models-pytorch==0.3.3
pip install pytorch-lightning==2.0.0
pip install torchvision>=0.15.0
pip install opencv-python>=4.5.0
pip install pandas>=1.5.0
pip install matplotlib>=3.5.0
pip install Pillow>=9.0.0
```

### Optional: requirements.txt
Consider creating a `requirements.txt` for reproducibility.
```

---

### 2. Add Local Development Section

**Location**: After "Running in Google Colab"

```markdown
## Running Locally

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Configuration
Update paths in notebooks before running:

| Variable | Colab Default | Local Example |
|----------|---------------|---------------|
| `complete_batch_path` | `/content/drive/MyDrive/...` | `./data/images/` |
| `predictions_path` | `/content/drive/MyDrive/...` | `./output/predictions/` |
| `latest_version_checkpoint` | `/content/drive/MyDrive/...` | `./checkpoints/epoch=9-step=180.ckpt` |

### Running Inference
```python
# In Python or Jupyter locally
from drusen_model import DrusenModel  # if extracted to module

model = DrusenModel.load_from_checkpoint("./checkpoints/model.ckpt")
model.eval()
# ... proceed with inference
```
```

---

### 3. Add Model Checkpoint Details

**Location**: New subsection under "Architecture"

```markdown
### Model Checkpoint

- **Current Checkpoint**: `epoch=9-step=180.ckpt`
- **Encoder**: (specify actual encoder used, e.g., ResNet34, EfficientNet-B0)
- **Input Size**: Images resized to dimensions divisible by 32 (max 1024x864 from 1024x885)
- **Output**: Single-channel binary mask (0=background, 1=drusen)

#### Loading Custom Checkpoints
```python
model = DrusenModel.load_from_checkpoint(
    "path/to/checkpoint.ckpt",
    arch="UNet",           # or FPN, DeepLabV3+
    encoder_name="resnet34",  # must match training
    in_channels=3,
    out_classes=1
)
```

#### Checkpoint Compatibility
Checkpoints are tied to:
- Model architecture (arch parameter)
- Encoder backbone (encoder_name)
- segmentation-models-pytorch version
```

---

### 4. Add Troubleshooting Section

**Location**: New section before "Key Functions"

```markdown
## Troubleshooting

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `CUDA out of memory` | Batch size too large for GPU | Reduce `batch_size` in DataLoader (try 4 or 2) |
| `Image dimensions not divisible by 32` | Input image wrong size | Images are auto-resized; ensure input is valid JPG |
| `RuntimeError: size mismatch` | Checkpoint/model mismatch | Verify encoder_name matches checkpoint |
| `FileNotFoundError` on checkpoint | Wrong path | Update `latest_version_checkpoint` path |

### Image Size Handling
- **Expected**: 1024x885 pixels
- **Other sizes**: Will be resized to nearest dimensions divisible by 32
- **Validation**: Code filters out non-1024x885 images (see cell [15] output)

### Memory Optimization
For large datasets or limited GPU memory:
```python
# Reduce batch size
dataloader = DataLoader(dataset, batch_size=4, ...)  # instead of 10

# Enable memory-efficient processing (already in process_dataloader)
torch.cuda.empty_cache()  # Called after each batch
```
```

---

### 5. Add Verification Section

**Location**: After "Running Locally" or "Running in Google Colab"

```markdown
## Verification

### Quick Verification Steps

1. **Check GPU availability**:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should print: True
   ```

2. **Load model successfully**:
   ```python
   model = DrusenModel.load_from_checkpoint("path/to/checkpoint.ckpt")
   print(model)  # Should print model architecture
   ```

3. **Run single image inference**:
   ```python
   # Process one image to verify pipeline
   test_image = prepare_data(["test_image.jpg"])
   with torch.no_grad():
       output = model(test_image)
   print(output.shape)  # Expected: torch.Size([1, 1, H, W])
   ```

### Expected Performance Baselines
| Metric | Expected Range | Notes |
|--------|----------------|-------|
| Test IoU | > 0.70 | Per-image IoU on held-out test set |
| Inference Time | ~0.5s/image | On NVIDIA T4 GPU |
| GPU Memory | ~6GB | With batch_size=10 |
```

---

### 6. Add Workflow Examples

**Location**: New section after "Notebooks"

```markdown
## Common Workflows

### Workflow 1: Run Inference on New Images

```
1. Place JPG images (1024x885) in input directory
2. Open `Run_model_on_all_flattened_images.ipynb`
3. Update paths:
   - `complete_batch_path` → your input directory
   - `predictions_path` → your output directory
4. Run cells 1-30 (setup through model loading)
5. Run cell 44 (batch inference)
6. Results in: predictions_path/{image_name}/prediction.png
```

### Workflow 2: Retrain Model

```
1. Prepare dataset with image/mask pairs
2. Open `drsuen_tests.ipynb`
3. Update data paths and split ratios if needed
4. Run training cells
5. Monitor TensorBoard: tensorboard --logdir=./logs
6. Checkpoint saved to: ./logs/tb_logs/drusen_model_v1/
```

### Workflow 3: Generate Metrics Report

```
1. Run inference workflow first
2. Cell 49: Generates per-image metrics.csv
3. Cell 50: Combines all into combined_metrics.csv
4. Output columns: filename, prediction_count, prediction_area
```
```

---

## Implementation Notes

### Files to Create/Modify

| File | Action | Priority |
|------|--------|----------|
| `CLAUDE.md` | Update with all sections above | High |
| `requirements.txt` | Create with pinned versions | Medium |
| `models/drusen_model.py` | Extract DrusenModel class (optional) | Low |

### Section Order in Updated CLAUDE.md

1. Project Overview (existing)
2. **Requirements** (new)
3. Architecture (existing, enhanced)
4. Notebooks (existing)
5. **Common Workflows** (new)
6. Running in Google Colab (existing)
7. **Running Locally** (new)
8. **Verification** (new)
9. Output Structure (existing)
10. **Troubleshooting** (new)
11. Key Functions (existing)

---

## Out of Scope

- Extracting `DrusenModel` to separate module (code refactoring)
- Creating automated tests
- CI/CD pipeline setup
- Docker containerization

---

## Acceptance Criteria

- [ ] Developer can set up local environment following documentation
- [ ] All dependencies have version specifications
- [ ] Common errors have documented solutions
- [ ] Verification steps confirm working setup
- [ ] At least 3 workflow examples provided
- [ ] Model checkpoint requirements documented
