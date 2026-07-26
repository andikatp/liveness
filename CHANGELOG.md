## 0.0.2

- **Fix Android ML Kit Bounding Box Coordinate System Mismatch**: Fixed false spoof classification on Android caused by portrait bounding box offsets. Added `isRotatedBoundingBox` parameter and `FaceBoundingBox.toRawBufferSpace()` transformation for `0°`, `90°`, `180°`, and `270°` sensor angles.
- **Fix Android 270° Front Camera Upside-Down Crop**: Corrected `case 270` coordinate mapping in `ImagePreprocessor` to ensure MiniFAS receives upright face crops.
- **Offload Inference to Background Isolate (`IsolateInterpreter`)**: Prevented main UI isolate frame drops by running LiteRT inference asynchronously on a dedicated background Dart isolate.
- **Add XNNPack ARM NEON SIMD Hardware Vectorization**: Enabled `XNNPackDelegate` for 2x–4x CPU matrix multiplication speedup on mobile chipsets.
- **Fast NV21 Plane Copy**: Optimized `_getNv21Bytes` to replace 518,400 Dart loop iterations with native `setRange` array copies.
- **Low-Light Adaptive Gamma Contrast Expansion**: Replaced linear boost with adaptive power-law gamma curve ($\gamma \approx 0.60 - 0.88$) to expand 3D skin texture gradients in dim lighting.

## 0.0.1

- Initial release of `passive_liveness`.
- Ultra-lightweight passive face anti-spoofing engine powered by LiteRT (TensorFlow Lite) edge inference.
- Real-time zero-copy camera frame processing (`LivenessImageBuffer`) for high FPS video streams (iOS `BGRA8888`, Android `NV21`/`YUV420`).
- Support for static image file (`File`) and byte array (`Uint8List`) liveness detection (e.g. from `takePicture()` or `image_picker`).
- Closed-form `reflect101` boundary padding and camera sensor rotation sampling (`0°`, `90°`, `180°`, `270°`).
- Built-in low-light shadow-lift compensation for underexposed camera environments.
