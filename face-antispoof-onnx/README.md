<div align="center">

# Lightweight Face Antispoof (MiniFAS, ONNX)

[![License](https://img.shields.io/badge/License-Apache%202.0-black.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-black)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-black)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-black)](https://onnx.ai/)

![Demo](assets/demo.gif)

</div>

A lightweight face anti-spoofing model that distinguishes real faces from spoofing attempts (printed photos, screen displays, etc.).

### Performance Demo

> **Note:** The demo above was recorded on legacy hardware to showcase the model's efficiency on low-spec devices.

| Metric | Details |
| :--- | :--- |
| **CPU** | Intel® Core™ i7-2630QM @ 2.00GHz (4C/8T) |
| **GPU** | None (Inference run on CPU) |


---

## Model

The trained model is a tiny classifier that predicts two classes: **Real** or **Spoof**.

| ONNX | Quantized ONNX | PyTorch | Input | Arch |
|:---------:|:---------------:|:------------:|:-----:|:------------:|
| 1.82 MB | **600 KB** | 1.95 MB | 128×128 RGB | MiniFAS |


### Model Performance

| Metric | Model | Quantized |
|:-------|:-----:|:---------:|
| **Model Size** | 1.82 MB | **600 KB** |
| **Overall Accuracy** | **98.20%** | **98.20%** |
| Real Accuracy | 97.58% | 97.55% |
| Spoof Accuracy | 98.73% | 98.73% |
| **ROC-AUC** | **0.9984** | **0.9984** |
| **Average Precision** | **0.9987** | **0.9987** |

> Tested on CelebA Spoof (70k+ samples). Quantization has no accuracy drop.

**[Detailed metrics →](docs/METRICS.md)** | **[Previous results →](docs/PREVIOUS_RESULTS.md)**

---

## Pre-trained

Pre-trained models are available in the `models/` directory:

| Model | Size | Format | Use Case |
|:------|:----:|:------:|:---------|
| `best_model.pth` | 1.95 MB | PyTorch | Training, fine-tuning, PyTorch inference |
| `best_model.onnx` | 1.82 MB | ONNX | General deployment, cross-platform inference |
| `best_model_quantized.onnx` | **600 KB** | ONNX (INT8) | **Production deployment** |

---

## Why MiniFAS?

The first version used MobileNetV4 (still in `src/mobilenetv4` for reference). It worked, but the model was larger than necessary and the training was more complex.

MiniFAS turned out to be a better fit:
- Smaller model, faster inference
- Built specifically for anti-spoofing, not a general-purpose backbone
- Uses Fourier Transform auxiliary loss during training—this helps the model learn frequency-domain patterns that distinguish real skin texture from printed photos and screen displays
- SE (Squeeze-and-Excitation) blocks for adaptive channel attention

> The MobileNetV4 code remains in `src/mobilenetv4/` for future experiments and reference. All current training uses MiniFASNet V2 SE.

**[Architecture details →](docs/ARCHITECTURE.md)**

---

## Quick Start

### 1. Create and activate a virtual environment (Recommended)

**Using Conda:**
```bash
conda create -n face-antispoof python
conda activate face-antispoof
```

**OR using venv:**
```bash
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> [!IMPORTANT]
> **Python Version:** This project requires **Python 3.8.0 or higher**.

### Compatibility Note: Python 3.7.x
Tested on **Python 3.7.16** and was confirmed that they are **not compatible**. Attempting to install dependencies on Python 3.7.x will result in a `subprocess-exited-with-error` during the `pip` installation of backend dependencies.

**Error Example:**
```text
ERROR: Ignored the following versions that require a different python version: 0.1.0 Requires-Python >=3.9; ...
ERROR: Could not find a version that satisfies the requirement puccinialin
ERROR: No matching distribution found for puccinialin
```

> **Note:** To run on GPU, install `onnxruntime-gpu` instead of `onnxruntime`.

### Run the Demo

**Webcam:**
```bash
python demo.py
```
or

```bash
python demo.py --camera <index>
```

**Single image:**
```bash
python demo.py --image <path>
```

Green bbox = real. 
Red bbox = spoof.

---

## Training

### 1. Prepare the Dataset

The dataset needs:
- Face images (`.jpg` or `.png`)
- Bounding box files: for `image.jpg`, a corresponding `image_BB.txt` with `x y w h`
- Label files: `metas/labels/train_label.json` and `metas/labels/test_label.json`

![Data preparation overview](assets/data_prep.png)

Run the prep script to crop faces:

```bash
python scripts/prepare_data.py \
  --orig_dir <path> \
  --crop_dir <path> \
  --size <number> \
  --bbox_expansion_factor <float> \
  --spoof_types <number> [<number> ...]
```

This reads images, crops faces using the bounding boxes (with some padding), resizes to the specified size, and organizes everything into `train/` and `test/` folders.

→ [**Why these preprocessing choices?**](docs/DATA_PREPARATION.md) (interpolation methods, padding strategy, etc.)

![Data prep result](assets/data_prep2.png)

### 2. Train

```bash
python scripts/train.py \
  --crop_dir <path> \
  --input_size <number> \
  --batch_size <number> \
  --output_dir <path>
```

Checkpoints and TensorBoard logs go to `<output_dir>/MINIFAS/`.

**Resume training:**
```bash
python scripts/train.py \
  --crop_dir <path> \
  --resume <checkpoint_path>
```

### 3. Prepare Model

Extract clean model weights from checkpoint (removes optimizer state, FTGenerator, DataParallel prefixes):

```bash
python scripts/prepare_best_model.py <epoch_checkpoint> \
  --output <path> \
  --input_size <number>
```

This creates a clean, inference-ready PyTorch model.

### 4. Export to ONNX

**Regular ONNX export:**
```bash
python scripts/export_onnx.py <checkpoint_path> \
  --input_size <number> \
  --output <path>
```

**Quantized ONNX:**
```bash
python scripts/quantize_onnx.py <checkpoint_path> \
  --input_size <number> \
  --output <path>
```

---

## Repo Structure

```
├── demo.py              # Inference demo
├── src/
│   ├── detection/       # Face detection
│   ├── inference/       # Model inference
│   ├── minifasv2/       # Training code
│   └── mobilenetv4/     # Legacy
├── scripts/             # Data prep, training, export
├── models/              # Pre-trained models
├── docs/                # Documentation
└── assets/              # Demo assets & results
```

---

## Limitations

Works best with well-lit, frontal faces. See [**Limitations & Notes**](docs/LIMITATIONS.md) for edge cases and tips.

---

## Acknowledgment

This project is based on the MiniFAS architecture from the
Silent Face Anti-Spoofing project by Minivision AI, licensed under Apache-2.0.

> This repository provides an independent training pipeline, ONNX export,
quantization, and deployment tooling.

---

## License

Apache-2.0. See `LICENSE`.
