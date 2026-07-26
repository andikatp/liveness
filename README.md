# passive_liveness

[![pub package](https://img.shields.io/pub/v/passive_liveness.svg)](https://pub.dev/packages/passive_liveness)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-flutter%20%7C%20android%20%7C%20ios-blue.svg)](https://pub.dev/packages/passive_liveness)

An ultra-lightweight, high-performance passive face anti-spoofing (liveness) detection package for Flutter powered by **LiteRT (TensorFlow Lite)** edge inference on Android and iOS.

---

## Features

- ⚡ **High-Performance Edge Inference**: Uses MiniFASNet v2 SE model via `flutter_litert`, achieving **~20–40ms** inference time on mobile devices.
- 📷 **Zero-Copy Camera Stream Processing**: Preprocesses raw Flutter `CameraImage` byte buffers (`NV21` / `YUV420` on Android, `BGRA8888` on iOS) directly to TFLite tensors without main-thread image decoding.
- 🖼️ **Static Photo & File Detection**: Uses Flutter's built-in C++ Skia engine codecs (`dart:ui`) to evaluate liveness from static images (`File` or `Uint8List`) with **zero external image package dependencies**.
- 📱 **Rotation & Sensor Alignment**: Closed-form pixel sampling matching OpenCV `BORDER_REFLECT_101` supporting `0°`, `90°`, `180°`, and `270°` camera sensor rotations.
- 💡 **Low-Light Shadow-Lift Compensation**: Automatically un-clips shadow gradients in dark or underexposed room environments.

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

Pass raw `CameraImage` frames from `camera` package along with an optional face bounding box (e.g. from Google ML Kit Face Detection):

```dart
import 'package:camera/camera.dart';
import 'package:passive_liveness/passive_liveness.dart';

void processCameraFrame(CameraImage cameraImage, Rect? faceRect, int sensorRotation) async {
  // Convert CameraImage into LivenessImageBuffer
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

  // Convert Rect to FaceBoundingBox
  final boundingBox = faceRect != null ? FaceBoundingBox.fromRect(faceRect) : null;

  final LivenessResult result = await detector.detectLivenessFromBuffer(
    buffer,
    boundingBox: boundingBox,
    rotation: sensorRotation, // e.g., 90 for iOS/Android front camera
  );

  if (result.isReal) {
    print('Real human face! Real score: ${result.realScore.toStringAsFixed(3)}');
  } else {
    print('Spoof face detected! Spoof score: ${result.spoofScore.toStringAsFixed(3)}');
  }
}
```

---

### 3. Detect Liveness from Static Photo File (`File`)

Evaluate a photo file picked via `image_picker` or taken with `takePicture()`:

```dart
import 'dart:io';
import 'package:passive_liveness/passive_liveness.dart';

Future<void> checkPhotoLiveness(File imageFile, Rect? faceRect) async {
  final boundingBox = faceRect != null ? FaceBoundingBox.fromRect(faceRect) : null;

  final LivenessResult result = await detector.detectLivenessFromImageFile(
    imageFile,
    boundingBox: boundingBox,
  );

  print('Is Real: ${result.isReal}');
  print('Real Score: ${result.realScore}');
  print('Inference Time: ${result.inferenceTime.inMilliseconds}ms');
}
```

---

### 4. Detect Liveness from Image Bytes (`Uint8List`)

Evaluate liveness directly from in-memory image bytes:

```dart
import 'dart:typed_data';
import 'package:passive_liveness/passive_liveness.dart';

Future<void> checkBytesLiveness(Uint8List imageBytes, {Rect? faceRect}) async {
  final boundingBox = faceRect != null ? FaceBoundingBox.fromRect(faceRect) : null;

  final LivenessResult result = await detector.detectLivenessFromImageBytes(
    imageBytes,
    boundingBox: boundingBox,
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
| `PassiveLivenessDetector` | Main engine class for initializing the model and running inferences. |
| `LivenessImageBuffer` | Lightweight container for camera raw byte planes (`NV21`, `YUV420`, `BGRA8888`). |
| `FaceBoundingBox` | Coordinates (`x`, `y`, `width`, `height`) defining the face area. Supports `.fromRect(Rect)`. |
| `LivenessResult` | Detection result containing `isReal`, `realScore`, `spoofScore`, `logitDiff`, and `inferenceTime`. |

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
