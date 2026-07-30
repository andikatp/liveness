# passive_liveness

[![pub package](https://img.shields.io/pub/v/passive_liveness.svg)](https://pub.dev/packages/passive_liveness)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-flutter%20%7C%20android%20%7C%20ios-blue.svg)](https://pub.dev/packages/passive_liveness)

An ultra-lightweight, high-performance passive face anti-spoofing (liveness) detection package for Flutter powered by **LiteRT (TensorFlow Lite)** edge inference on Android and iOS.

---

## Features

- ⚡ **LiteRT Next Hardware Acceleration**: Powered by `flutter_litert` `CompiledModel` with zero-copy hardware acceleration (GPU / NPU / CPU fallback).
- 🔓 **100% Standalone (No ML Kit Required)**: Works directly out-of-the-box on raw images or camera streams without needing Google ML Kit or third-party face detectors.
- 📷 **Zero-Copy Camera Stream Processing**: Preprocesses raw Flutter `CameraImage` byte buffers (`NV21` / `YUV420` on Android, `BGRA8888` on iOS) directly to TFLite tensors without main-thread image decoding.
- 👓 **Glasses Glare Resistance (Asymmetric EMA)**: Uses Asymmetric Exponential Moving Average filtering ($\alpha=0.1$ for score drops, $\alpha=0.4$ for recovery) to resist momentary specular reflections on glasses.
- 🏃 **Motion Gate & Stability Heuristic**: Tracks bounding box delta shifts to bypass inference on motion-blurred or out-of-focus frames during user movement.
- 🖼️ **Static Photo & File Detection**: Uses Flutter's built-in C++ Skia engine codecs (`dart:ui`) to evaluate liveness from static images (`File` or `Uint8List`) with **zero external image package dependencies**.
- 📱 **Android Rotated Bounding Box Mapping**: Built-in `isRotatedBoundingBox` auto-detection and `FaceBoundingBox.toRawBufferSpace()` transformation for portrait ML Kit face detection bounding boxes on Android (`0°`, `90°`, `180°`, `270°`).
- 💡 **Adaptive Contrast Stretching & Edge Clamping**: Automatically normalizes dark backlit faces and uses Edge Pixel Replication (`BORDER_REPLICATE`) to eliminate black border artifacts.

---

## Do I Need Google ML Kit or a Face Detector?

### Short Answer: **NO!** 

The `boundingBox` parameter is **100% OPTIONAL**. 

If you don't pass a `boundingBox`, `passive_liveness` automatically uses the **entire image frame** as the face region. You can run liveness detection in just **one line of code**!

### How Bounding Box Works:

- **Without Face Detector (Default - Super Easy):**
  Simply pass your camera buffer, photo file, or bytes. `passive_liveness` evaluates the entire image automatically:
  ```dart
  // Works out of the box without ML Kit!
  final result = await detector.detectLivenessFromImageBytes(imageBytes);
  ```

- **With Face Detector (Optional - Advanced Precision):**
  If your app already uses a face detector (like `google_mlkit_face_detection`), passing `FaceBoundingBox.fromRect(face.boundingBox)` crops directly onto the detected face for pinpoint accuracy:
  ```dart
  // Optional: pass boundingBox if you have ML Kit
  final result = await detector.detectLivenessFromImageBytes(
    imageBytes,
    boundingBox: FaceBoundingBox.fromRect(faceRect),
  );
  ```

---

## Installation

Add `passive_liveness` to your project:

```bash
flutter pub add passive_liveness
```

---

## Usage

### 1. Initialize Detector Engine

Initialize `PassiveLivenessDetector`. The bundled MiniFAS model is loaded automatically:

```dart
import 'package:passive_liveness/passive_liveness.dart';

final detector = PassiveLivenessDetector();
await detector.initialize();
```

---

### 2. Real-Time Camera Stream Detection (`CameraImage`)

#### Option A: Simple (Without ML Kit - Full Frame)
```dart
import 'package:camera/camera.dart';
import 'package:passive_liveness/passive_liveness.dart';

void processCameraFrame(CameraImage cameraImage, int sensorRotation) async {
  final buffer = LivenessImageBuffer(
    width: cameraImage.width,
    height: cameraImage.height,
    format: cameraImage.format.group == ImageFormatGroup.bgra8888
        ? LivenessImageFormat.bgra8888
        : (cameraImage.planes.length == 1
            ? LivenessImageFormat.nv21
            : LivenessImageFormat.yuv420),
    planes: cameraImage.planes
        .map((p) => LivenessImagePlane(
              bytes: p.bytes,
              bytesPerRow: p.bytesPerRow,
              bytesPerPixel: p.bytesPerPixel,
            ))
        .toList(),
  );

  // No boundingBox passed - evaluates full frame!
  final LivenessResult result = await detector.detectLivenessFromBuffer(
    buffer,
    rotation: sensorRotation,
  );

  if (result.isReal) {
    print('Real human face!');
  } else {
    print('Spoof face detected!');
  }
}
```

#### Option B: Advanced (With ML Kit Face Detector)
```dart
  // Pass faceRect from ML Kit
  final boundingBox = faceRect != null ? FaceBoundingBox.fromRect(faceRect) : null;

  final LivenessResult result = await detector.detectLivenessFromBuffer(
    buffer,
    boundingBox: boundingBox,
    rotation: sensorRotation,
    isRotatedBoundingBox: Platform.isAndroid,
  );
```

---

### 3. Using `LivenessFrameProcessor` for Smooth Streaming

For stream processing with automatic motion gating and frame throttling:

```dart
final processor = LivenessFrameProcessor(
  detector: detector,
  throttleInterval: const Duration(milliseconds: 150),
);

// In your camera stream listener:
final LivenessResult? result = await processor.processBufferFrame(
  buffer,
  rotation: sensorRotation,
);
```

---

### 4. Detect Liveness from Static Photo File (`File`)

Evaluate a photo file picked via `image_picker` or taken with `takePicture()`:

```dart
import 'dart:io';
import 'package:passive_liveness/passive_liveness.dart';

Future<void> checkPhotoLiveness(File imageFile) async {
  // Works directly on the file - boundingBox is optional!
  final LivenessResult result = await detector.detectLivenessFromImageFile(
    imageFile,
  );

  print('Is Real: ${result.isReal}');
  print('Real Score: ${result.realScore}');
  print('Inference Time: ${result.inferenceTime.inMilliseconds}ms');
}
```

---

### 5. Detect Liveness from Image Bytes (`Uint8List`)

Evaluate liveness directly from in-memory image bytes:

```dart
import 'dart:typed_data';
import 'package:passive_liveness/passive_liveness.dart';

Future<void> checkBytesLiveness(Uint8List imageBytes) async {
  // Works directly on raw bytes - boundingBox is optional!
  final LivenessResult result = await detector.detectLivenessFromImageBytes(
    imageBytes,
  );

  print('Is Real: ${result.isReal}');
  print('Real Score: ${result.realScore}');
}
```

---

## Clean Up

When you are done using the detector (e.g. in a State's `dispose` method):

```dart
@override
void dispose() {
  detector.dispose();
  super.dispose();
}
```

---

## Core Classes Reference

| Class | Description |
|---|---|
| `PassiveLivenessDetector` | Main engine class for initializing the LiteRT model and running inferences. |
| `LivenessFrameProcessor` | Stream processor with motion-gating heuristic and frame throttling. |
| `LivenessImageBuffer` | Lightweight container for camera raw byte planes (`NV21`, `YUV420`, `BGRA8888`). |
| `FaceBoundingBox` | Coordinates (`x`, `y`, `width`, `height`) defining the face area. **Optional**. Supports `.fromRect(Rect)`. |
| `LivenessResult` | Detection result containing `isReal`, `realScore`, `spoofScore`, `logitDiff`, and `inferenceTime`. |

---

## Recent Improvements & Changelog

### 🚀 Performance & Accuracy Upgrades
- **LiteRT Next `CompiledModel` Integration:** Upgraded engine to `flutter_litert` `CompiledModel` with zero-copy hardware acceleration (GPU / NPU / CPU fallback).
- **Standalone Support (Optional BoundingBox):** Bounding box parameter is optional; default fallback automatically processes the full image frame.
- **Asymmetric EMA Glare Resistance:** Implemented Asymmetric Exponential Moving Average ($\alpha=0.1$ for score drops, $\alpha=0.4$ for recovery) to prevent specular lens reflections on **glasses** from causing false spoof drops.
- **Bounding Box Motion Stability Gate:** Added motion stability heuristic ($5\%$ position/size shift threshold) in `LivenessFrameProcessor` to bypass TFLite inference during user movement.
- **Android Sensor Coordinate Rotation:** Auto-transforms ML Kit portrait face bounding boxes to match raw landscape sensor buffers (`0°`, `90°`, `180°`, `270°`).
- **Edge Pixel Replication (`BORDER_REPLICATE`):** Switched from zero-padding to coordinate clamping (`rawX.round().clamp(0, rawW - 1)`), eliminating pitch-black border artifacts on edge face crops.
- **Adaptive Contrast Stretching:** Added `enableContrastStretch` option to automatically brighten dark, backlit face crops without distorting skin moiré signals.
- **Visual Tensor Dumper:** Added `ImagePreprocessor.saveTensorToDisk(...)` to export 128x128 Float32List tensors to PNG/PPM images for debugging.

---

## Model Attribution & Credits

This package utilizes the **MiniFASNet v2 SE** passive face anti-spoofing model architecture, derived and converted from [facenox/face-antispoof-onnx](https://github.com/facenox/face-antispoof-onnx) (MiniFASNetV2 SE trained with Fourier Transform frequency spectrum loss).

---

## Author

**Andika Tri Prasetya**
- GitHub: [github.com/andikatp](https://github.com/andikatp)

---

## License

This project is licensed under the OSI-approved **MIT License** - see the [LICENSE](LICENSE) file for details.
