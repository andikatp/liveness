# passive_liveness

[![pub package](https://img.shields.io/pub/v/passive_liveness.svg)](https://pub.dev/packages/passive_liveness)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An ultra-lightweight, high-performance passive face anti-spoofing detection package for Flutter using **LiteRT (TensorFlow Lite)** edge inference on Android and iOS.

---

## Features

- **⚡ High Performance Edge Inference**: Powered by LiteRT (`flutter_litert`), achieving **~20–40ms** inference time on mobile devices.
- **📷 Zero-Copy Camera Stream Processing**: Preprocesses raw `CameraImage` byte buffers (`NV21` / `YUV420` on Android, `BGRA8888` on iOS) directly to TFLite tensors without main-thread image decoding.
- **🖼️ Zero-Dependency Static Photo File Detection**: Uses Flutter's built-in C++ Skia engine codecs (`dart:ui`) to decode static images (`File` or `Uint8List`) with **0 external image package dependencies**.
- **📱 Rotation & Sensor Alignment**: Closed-form pixel sampling matching OpenCV `BORDER_REFLECT_101` supporting `0°`, `90°`, `180°`, and `270°` camera sensor rotation.
- **💡 Low-Light Shadow-Lift Compensation**: Automatically un-clips shadow gradients in dark or underexposed room environments.

---

## Installation

Add `passive_liveness` to your `pubspec.yaml`:

```yaml
dependencies:
  passive_liveness: ^0.0.1
```

Or run:

```bash
flutter pub add passive_liveness
```

Ensure your `pubspec.yaml` bundles the bundled MiniFAS TFLite model asset:

```yaml
flutter:
  assets:
    - packages/passive_liveness/assets/best_model.tflite
```

---

## Usage

### 1. Initialize Detector Engine

```dart
import 'package:passive_liveness/passive_liveness.dart';

final detector = PassiveLivenessDetector();
await detector.initialize();
```

---

### 2. Detect Liveness from Camera Frame Stream (`CameraImage`)

```dart
import 'package:camera/camera.dart';
import 'package:passive_liveness/passive_liveness.dart';

void onFrameReceived(CameraImage cameraImage, FaceBoundingBox? boundingBox, int sensorRotation) async {
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

  final LivenessResult result = await detector.detectLivenessFromBuffer(
    buffer,
    boundingBox: boundingBox, // From ML Kit face detection (in raw buffer coords)
    rotation: sensorRotation,  // e.g., 90 for iOS front camera
  );

  if (result.isReal) {
    print('Real human face detected! Score: ${result.realScore.toStringAsFixed(3)}');
  } else {
    print('Spoof face detected! Score: ${result.spoofScore.toStringAsFixed(3)}');
  }
}
```

---

### 3. Detect Liveness from Static Photo File (`File` / `ImagePicker`)

```dart
import 'dart:io';
import 'package:passive_liveness/passive_liveness.dart';

Future<void> checkPhotoLiveness(File imageFile, FaceBoundingBox? boundingBox) async {
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

### 4. Detect Liveness from Encoded Image Bytes (`Uint8List`)

```dart
import 'dart:typed_data';
import 'package:passive_liveness/passive_liveness.dart';

Future<void> checkBytesLiveness(Uint8List imageBytes) async {
  final LivenessResult result = await detector.detectLivenessFromImageBytes(
    imageBytes,
  );

  print('Is Real: ${result.isReal}');
}
```

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
