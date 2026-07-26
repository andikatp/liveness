## 0.0.1

- Initial release of `passive_liveness`.
- Ultra-lightweight passive face anti-spoofing engine powered by LiteRT (TensorFlow Lite) edge inference.
- Real-time zero-copy camera frame processing (`LivenessImageBuffer`) for high FPS video streams (iOS `BGRA8888`, Android `NV21`/`YUV420`).
- Support for static image file (`File`) and byte array (`Uint8List`) liveness detection (e.g. from `takePicture()` or `image_picker`).
- Closed-form `reflect101` boundary padding and camera sensor rotation sampling (`0°`, `90°`, `180°`, `270°`).
- Built-in low-light shadow-lift compensation for underexposed camera environments.
