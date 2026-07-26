import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter_litert/flutter_litert.dart';

import 'models/face_bounding_box.dart';
import 'models/liveness_image_buffer.dart';
import 'models/liveness_result.dart';
import 'utils/image_preprocessor.dart';

/// Core passive liveness detector engine using MiniFAS LiteRT (TFLite) model.
///
/// Uses the **MiniFASNet v2 SE** architecture derived from
/// [facenox/face-antispoof-onnx](https://github.com/facenox/face-antispoof-onnx).
class PassiveLivenessDetector {
  static const String defaultAssetPath =
      'packages/passive_liveness/assets/best_model.tflite';
  static const String fallbackAssetPath = 'assets/best_model.tflite';

  Interpreter? _interpreter;
  IsolateInterpreter? _isolateInterpreter;
  bool _isInitialized = false;

  /// Whether the detector interpreter is initialized and ready for inference.
  bool get isInitialized => _isInitialized && _interpreter != null;

  /// Initialize the LiteRT interpreter for passive liveness detection.
  Future<void> initialize({
    String? assetPath,
    String? filePath,
    Uint8List? modelBytes,
    InterpreterOptions? options,
  }) async {
    if (_isInitialized) return;

    final opts =
        options ??
        (InterpreterOptions()
          ..threads = 4
          ..addDelegate(XNNPackDelegate()));

    if (modelBytes != null) {
      _interpreter = Interpreter.fromBuffer(modelBytes, options: opts);
    } else if (filePath != null) {
      _interpreter = Interpreter.fromFile(File(filePath), options: opts);
    } else {
      final path = assetPath ?? defaultAssetPath;
      try {
        _interpreter = await Interpreter.fromAsset(path, options: opts);
      } catch (_) {
        _interpreter = await Interpreter.fromAsset(
          fallbackAssetPath,
          options: opts,
        );
      }
    }

    if (_interpreter != null) {
      try {
        _isolateInterpreter = await IsolateInterpreter.create(
          address: _interpreter!.address,
        );
      } catch (_) {
        _isolateInterpreter = null;
      }
    }

    _isInitialized = true;
  }

  /// Run passive liveness detection directly on a raw camera buffer ([LivenessImageBuffer]).
  ///
  /// The [boundingBox] should be in raw buffer coordinate space (as returned
  /// by ML Kit face detection).
  /// The [rotation] is the camera rotation in degrees CW needed to make the buffer upright.
  Future<LivenessResult> detectLivenessFromBuffer(
    LivenessImageBuffer buffer, {
    FaceBoundingBox? boundingBox,
    int rotation = 0,
    bool? isRotatedBoundingBox,
    double threshold = 0.0,
    double expansionFactor = ImagePreprocessor.defaultExpansionFactor,
  }) async {
    if (!isInitialized || _interpreter == null) {
      throw StateError(
        'PassiveLivenessDetector is not initialized. Call initialize() first.',
      );
    }

    final stopwatch = Stopwatch()..start();

    // 1. Preprocess crop to 128x128 Float32 NHWC tensor [1, 128, 128, 3]
    final tensorData = ImagePreprocessor.preprocessBufferToTensor(
      buffer,
      boundingBox: boundingBox,
      rotation: rotation,
      isRotatedBoundingBox: isRotatedBoundingBox,
      expansionFactor: expansionFactor,
    );

    final input = tensorData.reshape([
      1,
      ImagePreprocessor.defaultModelSize,
      ImagePreprocessor.defaultModelSize,
      3,
    ]);

    final output = List.generate(1, (_) => List<double>.filled(2, 0.0));

    if (_isolateInterpreter != null) {
      await _isolateInterpreter!.run(input, output);
    } else {
      _interpreter!.run(input, output);
    }
    stopwatch.stop();

    final logits = output[0];

    return LivenessResult.fromLogits(
      realLogit: logits[0],
      spoofLogit: logits[1],
      threshold: threshold,
      inferenceTime: stopwatch.elapsed,
    );
  }

  /// Run passive liveness detection on an in-memory encoded image byte array
  /// (e.g. JPEG / PNG bytes from camera `takePicture()` or `ImagePicker`).
  ///
  /// Uses Flutter's built-in C++ Skia engine codecs (`dart:ui`) with zero external dependencies.
  /// The [boundingBox] should be in pixel coordinates of the static image `[0..width, 0..height]`.
  Future<LivenessResult> detectLivenessFromImageBytes(
    Uint8List imageBytes, {
    FaceBoundingBox? boundingBox,
    double threshold = 0.0,
    double expansionFactor = ImagePreprocessor.defaultExpansionFactor,
  }) async {
    if (!isInitialized || _interpreter == null) {
      throw StateError(
        'PassiveLivenessDetector is not initialized. Call initialize() first.',
      );
    }

    final stopwatch = Stopwatch()..start();

    final ui.Codec codec = await ui.instantiateImageCodec(imageBytes);
    final ui.FrameInfo frameInfo = await codec.getNextFrame();
    final ui.Image image = frameInfo.image;
    final ByteData? byteData = await image.toByteData(
      format: ui.ImageByteFormat.rawRgba,
    );

    if (byteData == null) {
      image.dispose();
      codec.dispose();
      throw ArgumentError('Failed to extract raw RGBA pixel bytes from image.');
    }

    final rgbaBytes = byteData.buffer.asUint8List();
    final width = image.width;
    final height = image.height;

    image.dispose();
    codec.dispose();

    final tensorData = ImagePreprocessor.preprocessRgbaBytesToTensor(
      rgbaBytes,
      width,
      height,
      boundingBox: boundingBox,
      expansionFactor: expansionFactor,
    );

    final input = tensorData.reshape([
      1,
      ImagePreprocessor.defaultModelSize,
      ImagePreprocessor.defaultModelSize,
      3,
    ]);

    final output = List.generate(1, (_) => List<double>.filled(2, 0.0));

    if (_isolateInterpreter != null) {
      await _isolateInterpreter!.run(input, output);
    } else {
      _interpreter!.run(input, output);
    }
    stopwatch.stop();

    final logits = output[0];

    return LivenessResult.fromLogits(
      realLogit: logits[0],
      spoofLogit: logits[1],
      threshold: threshold,
      inferenceTime: stopwatch.elapsed,
    );
  }

  /// Run passive liveness detection on a static image file
  /// (e.g. `XFile.path` from camera package or `File` from `image_picker`).
  ///
  /// The [boundingBox] should be in pixel coordinates of the static image `[0..width, 0..height]`.
  Future<LivenessResult> detectLivenessFromImageFile(
    File file, {
    FaceBoundingBox? boundingBox,
    double threshold = 0.0,
    double expansionFactor = ImagePreprocessor.defaultExpansionFactor,
  }) async {
    final bytes = await file.readAsBytes();
    return detectLivenessFromImageBytes(
      bytes,
      boundingBox: boundingBox,
      threshold: threshold,
      expansionFactor: expansionFactor,
    );
  }

  /// Close LiteRT interpreter.
  void dispose() {
    _isolateInterpreter?.close();
    _isolateInterpreter = null;
    _interpreter?.close();
    _interpreter = null;
    _isInitialized = false;
  }
}
