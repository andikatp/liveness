# Model Integration Guide — `passive_liveness`

> **Purpose:** This document is the single source of truth for integrating any
> anti-spoofing TFLite model into the `passive_liveness` package. Read this
> before swapping models.

---

## 1. Model Input / Output Contract

### Input Tensor

| Property | Value |
|---|---|
| **Data type** | `float32` |
| **Layout** | Auto-detected: `NHWC [1, H, W, 3]` or `NCHW [1, 3, H, W]` |
| **Spatial size** | Auto-detected from tensor shape (e.g. `128`, `80`) |
| **Channel order** | RGB by default. Set `isBgr: true` if model expects BGR. |
| **Value range** | Depends on `NormalizationScheme` (see §2) |

The Dart engine reads the TFLite input tensor shape on `initModel` and automatically
configures `isNativeNchw`, `modelTargetSize`, and `useNchw`.

### Output Tensor

| Property | Value |
|---|---|
| **Shape** | `[1, N]` where N = number of classes (2 or 3) |
| **2-class** | `[class0_logit, class1_logit]` — mapping set by `ModelClassOrder` |
| **3-class** | `[2D_spoof, 3D_spoof, real]` (MiniFASNet convention) |

---

## 2. Normalization Schemes

| Scheme | Formula | When to use |
|---|---|---|
| `zeroToOne` | `val / 255.0` | Training used only `transforms.ToTensor()` |
| `minusOneToOne` | `(val - 127.5) / 127.5` | Training used `transforms.Normalize([0.5]*3, [0.5]*3)` |
| `imageNet` | `(val/255 - mean) / std` | Training used `transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])` |

**How to check:** Search your training code for `transforms.Normalize(...)`. If absent,
use `zeroToOne`. If present, match the mean/std values to the scheme above.

---

## 3. Class Ordering (Critical!)

PyTorch `datasets.ImageFolder` assigns class indices by **alphabetical sort** of
the subdirectory names.

| Directory names | Sorted | Index 0 | Index 1 | `ModelClassOrder` |
|---|---|---|---|---|
| `real/`, `spoof/` | `real` < `spoof` | **Real** | **Spoof** | `realFirst` (default) |
| `fake/`, `real/` | `fake` < `real` | **Spoof** | **Real** | `spoofFirst` |
| `live/`, `spoof/` | `live` < `spoof` | **Real** | **Spoof** | `realFirst` |
| `0_spoof/`, `1_real/` | `0` < `1` | **Spoof** | **Real** | `spoofFirst` |

**How to verify:** Check your training output for `Class Mapping: {...}`.
Use `spoofFirst` only if index 0 maps to the spoof/fake class.

---

## 4. Conversion Pipeline

### Recommended: `ai-edge-torch` (direct PyTorch → TFLite)

```python
import ai_edge_torch, torch

model.eval().cpu()
edge_model = ai_edge_torch.convert(model, (torch.randn(1, 3, 128, 128),))
edge_model.export("best_model.tflite")
```

- Preserves original NCHW layout → TFLite input will be `[1, 3, 128, 128]`
- Dart engine auto-detects NCHW and sets `isNativeNchw = true`
- No ONNX intermediate step → fewer silent conversion bugs

### Fallback: ONNX → TFLite via `onnx2tf`

```python
# ONNX export
torch.onnx.export(model, dummy, "model.onnx", opset_version=16, ...)
# onnx2tf conversion
!onnx2tf -i model.onnx -o tflite_out --non_verbose
```

- Transposes NCHW→NHWC → TFLite input will be `[1, 128, 128, 3]`
- **MUST verify:** Compare PyTorch vs TFLite outputs (see §5)
- Complex architectures (squeeze-excite, attention) may convert incorrectly

### Always Verify After Conversion

```python
import numpy as np, tensorflow as tf
from torchvision import transforms
from PIL import Image

img = Image.open("test_real.jpg").convert('RGB')
pt_in = transforms.ToTensor()(img).unsqueeze(0)

# PyTorch
with torch.no_grad():
    pt_out = model(pt_in).numpy()

# TFLite
interp = tf.lite.Interpreter(model_path="best_model.tflite")
interp.allocate_tensors()
inp = interp.get_input_details()[0]

if inp['shape'].tolist() == [1, 3, 128, 128]:  # NCHW (ai-edge-torch)
    tf_in = pt_in.numpy()
else:  # NHWC (onnx2tf)
    tf_in = pt_in.numpy().transpose(0, 2, 3, 1)

interp.set_tensor(inp['index'], tf_in.astype(np.float32))
interp.invoke()
tf_out = interp.get_tensor(interp.get_output_details()[0]['index'])

print(f"PyTorch: {pt_out}")
print(f"TFLite:  {tf_out}")
print(f"Max diff: {np.max(np.abs(pt_out - tf_out)):.6f}")
# diff < 0.01 = correct conversion
# diff > 0.5  = broken conversion
```

---

## 5. Dart Integration

### Initialization

```dart
await detector.initialize(
  assetPath: 'assets/best_model.tflite',   // or filePath / modelBytes
  classOrder: ModelClassOrder.realFirst,    // Match your training class_to_idx
);
```

### Detection

```dart
final result = await detector.detectLivenessFromCameraImage(
  cameraImage,
  boundingBox: faceBbox,
  rotation: sensorRotation,
  normalizationScheme: NormalizationScheme.zeroToOne, // Match your training transforms
  // isBgr: false,              // true only if model was trained on BGR images
  // enableContrastStretch: false, // true only for legacy MiniFASNet 80×80
);
```

### What the engine auto-detects (you do NOT set these):

| Property | How detected | Example |
|---|---|---|
| `isNativeNchw` | `inputShape[1] == 3` → NCHW, else NHWC | `[1,3,128,128]` → NCHW |
| `modelTargetSize` | NCHW: `shape[2]`, NHWC: `shape[1]` | `128` |
| `useNchw` | Follows `isNativeNchw` | automatic |

---

## 6. Model History

| Date | Model | Architecture | Input | Classes | Normalization | ClassOrder | Conversion | Notes |
|---|---|---|---|---|---|---|---|---|
| Original | MiniFASNet v2 SE | Custom CNN | 80×80, BGR, NCHW | 3-class `[2D, 3D, Real]` | `zeroToOne` | N/A (3-class) | Pre-converted | Legacy default model |
| 2026-08 (Failed) | MobileNetV4-small | `timm` pretrained | 128×128, RGB, varies | 2-class `{real:0, spoof:1}` | `zeroToOne` | `realFirst` | `onnx2tf` | Reverted due to accuracy regression |
| 2026-08 (Current) | MiniFASNet v2 SE | Custom CNN (Fourier Loss + SE) | 128×128, RGB, NHWC | 2-class `[real, spoof]` | `zeroToOne` | `realFirst` | `onnx2tf` | Validated `facenox/face-antispoof-onnx` 1.84MB model |

---

## 7. Troubleshooting

### Model returns inverted scores (real face → spoof)

1. **Check class order**: Print `class_to_idx` from training. Use `spoofFirst` if
   index 0 is spoof.
2. **Check conversion**: Run verification script (§4). If outputs diverge, reconvert
   with `ai-edge-torch`.
3. **Check normalization**: Compare on-device tensor stats (`min/max/mean`) with
   what the training pipeline produces for the same image.

### `ChromVar: 0.0` or `SatVar: 0.0000`

Fixed in commit adding single-plane NV21 support to `ColorSpaceAnalyzer`.
Ensure you're on latest version.

### Model runs but accuracy is poor despite high Kaggle val accuracy

This is **domain shift**. Camera YUV buffers differ from JPEG-compressed web images.
Solutions:
- Train with camera-captured datasets (CelebA-Spoof, CASIA-FASD)
- Add `GaussianBlur`, `RandomAffine`, `RandomPerspective` augmentation
- Capture your own real/spoof training data from the target device
