import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/services.dart';
import 'package:flutter_litert/flutter_litert.dart';

import 'models/face_bounding_box.dart';
import 'models/liveness_image_buffer.dart';
import 'models/liveness_result.dart';
import 'utils/color_space_analyzer.dart';
import 'utils/face_proximity_gate.dart';
import 'utils/high_res_screen_analyzer.dart';
import 'utils/image_preprocessor.dart';
import 'utils/lbp_hog_analyzer.dart';
import 'utils/liveness_logger.dart';

/// Core passive liveness detector engine using MiniFAS LiteRT (TFLite) model.
///
/// Supports LiteRT Next (`CompiledModel`) for zero-copy hardware acceleration (GPU / CPU)
/// with classic `Interpreter` fallback.
class PassiveLivenessDetector {
  /// Default asset path for the TFLite model package asset.
  static const String defaultAssetPath =
      'packages/passive_liveness/assets/best_model.tflite';

  /// Fallback asset path when loaded directly within host application.
  static const String fallbackAssetPath = 'assets/best_model.tflite';

  /// Default Exponential Moving Average (EMA) alpha for score smoothing.
  static const double defaultEmaAlpha = 0.3;

  /// Creates a new [PassiveLivenessDetector] instance.
  PassiveLivenessDetector();

  CompiledModel? _compiledModel;
  Interpreter? _interpreter;
  IsolateInterpreter? _isolateInterpreter;
  bool _isInitialized = false;

  /// Input tensor shape of the loaded model (e.g. [1, 128, 128, 3] or [1, 3, 128, 128]).
  List<int>? _modelInputShape;

  /// Whether the model natively expects NCHW format (`[1, 3, H, W]`).
  bool _isNativeNchw = false;

  /// Target spatial resolution expected by model (e.g. 128 or 80).
  int _modelTargetSize = ImagePreprocessor.defaultModelSize;

  /// The current Exponential Moving Average (EMA) of the real score.
  double? _emaRealScore;

  /// Returns the current EMA real score.
  double? get emaRealScore => _emaRealScore;

  /// Resets the EMA real score tracker.
  void resetEma() {
    _emaRealScore = null;
  }

  /// Whether the detector is initialized and ready for inference.
  bool get isInitialized =>
      _isInitialized && (_compiledModel != null || _interpreter != null);

  /// Initialize the LiteRT model for passive liveness detection.
  ///
  /// Set [useCompiledModel] to `true` (default) to use LiteRT Next `CompiledModel` for zero-copy
  /// GPU/hardware acceleration.
  Future<void> initialize({
    String? assetPath,
    String? filePath,
    Uint8List? modelBytes,
    Set<Accelerator>? accelerators,
    InterpreterOptions? options,
    bool enableIsolate = false,
    bool useCompiledModel = true,
  }) async {
    if (_isInitialized) return;

    final accels = accelerators ?? {Accelerator.gpu, Accelerator.cpu};

    // Always create classic interpreter for shape inspection and fallback
    final opts =
        options ??
        (InterpreterOptions()
          ..threads = 2
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
        final inputTensor = _interpreter!.getInputTensor(0);
        _modelInputShape = List<int>.from(inputTensor.shape);

        if (_modelInputShape != null && _modelInputShape!.length == 4) {
          if (_modelInputShape![1] == 3) {
            _isNativeNchw = true;
            _modelTargetSize = _modelInputShape![2];
          } else {
            _isNativeNchw = false;
            _modelTargetSize = _modelInputShape![1];
          }
        }

        LivenessLogger.logModelInit(
          inputShape: _modelInputShape,
          tensorType: inputTensor.type.toString(),
          isNativeNchw: _isNativeNchw,
          targetSize: _modelTargetSize,
          isCompiledModel: _compiledModel != null,
          isIsolateInterpreter: _isolateInterpreter != null,
        );
      } catch (e) {
        LivenessLogger.log('Failed to inspect model input shape: $e');
        _modelInputShape = null;
      }
    }

    // Try compiled model for LiteRT Next zero-copy path if requested and no custom options/isolates
    if (useCompiledModel && options == null && !enableIsolate) {
      try {
        if (modelBytes != null) {
          _compiledModel = CompiledModel.fromBuffer(
            modelBytes,
            accelerators: accels,
          );
        } else if (filePath != null) {
          _compiledModel = CompiledModel.fromFile(
            filePath,
            accelerators: accels,
          );
        } else {
          final path = assetPath ?? defaultAssetPath;
          try {
            final bd = await rootBundle.load(path);
            final bytes = bd.buffer.asUint8List(
              bd.offsetInBytes,
              bd.lengthInBytes,
            );
            _compiledModel = CompiledModel.fromBuffer(
              bytes,
              accelerators: accels,
            );
          } catch (_) {
            final bd = await rootBundle.load(fallbackAssetPath);
            final bytes = bd.buffer.asUint8List(
              bd.offsetInBytes,
              bd.lengthInBytes,
            );
            _compiledModel = CompiledModel.fromBuffer(
              bytes,
              accelerators: accels,
            );
          }
        }
      } catch (e) {
        LivenessLogger.log('CompiledModel fallback to Interpreter: $e');
        _compiledModel = null;
      }
    }

    if (_interpreter != null && enableIsolate && _compiledModel == null) {
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

  void _ensureInputShapeAllocated({
    required int targetSize,
    required bool useNchw,
  }) {
    if (_interpreter == null) return;
    final targetShape = useNchw
        ? [1, 3, targetSize, targetSize]
        : [1, targetSize, targetSize, 3];

    final currentShape = _interpreter!.getInputTensor(0).shape;
    bool shapesMatch = currentShape.length == targetShape.length;
    if (shapesMatch) {
      for (int i = 0; i < currentShape.length; i++) {
        if (currentShape[i] != targetShape[i]) {
          shapesMatch = false;
          break;
        }
      }
    }

    if (!shapesMatch) {
      _interpreter!.resizeInputTensor(0, targetShape);
      _interpreter!.allocateTensors();
    }
  }

  void _logTensorStats(Float32List tensorData) {
    LivenessLogger.logTensorStats(
      tensorData,
      inputShape: _interpreter?.getInputTensor(0).shape,
    );
  }

  Future<List<double>> _runInference({
    required Float32List tensorData,
    required bool effectiveUseNchw,
    required int effectiveTargetSize,
    required bool runOnIsolate,
  }) async {
    if (_compiledModel != null) {
      final outputs = _compiledModel!.run([tensorData]);
      return outputs[0];
    } else {
      final input = effectiveUseNchw
          ? tensorData.reshape([1, 3, effectiveTargetSize, effectiveTargetSize])
          : tensorData.reshape([
              1,
              effectiveTargetSize,
              effectiveTargetSize,
              3,
            ]);

      final output = List.generate(1, (_) => List<double>.filled(2, 0.0));

      if (!_isInitialized || _interpreter == null) {
        return output[0];
      }

      if (runOnIsolate && _isolateInterpreter != null) {
        await _isolateInterpreter!.run(input, output);
      } else {
        _interpreter!.run(input, output);
      }
      return output[0];
    }
  }

  /// Runs passive face liveness and anti-spoofing detection directly on a raw camera image buffer ([LivenessImageBuffer]).
  ///
  /// Returns a [LivenessResult] containing `isReal`, classification `status`, real/spoof confidence scores,
  /// and detailed physical heuristic metrics (LBP ratio, HOG dominance, chrominance variance, patch Laplacian focus dispersal).
  ///
  /// ### Simple Usage (Recommended)
  /// All multi-layer anti-spoofing heuristic engines (Proximity Gate, Micro-Texture Analysis,
  /// YCbCr Color Space Analysis, and 2D Laplacian High-Res Screen Analysis) are **enabled by default** with
  /// optimal production settings. You only need to pass the camera [buffer] and optional [boundingBox] & [rotation]:
  ///
  /// ```dart
  /// final result = await detector.detectLivenessFromBuffer(
  ///   buffer,
  ///   boundingBox: faceBbox, // Optional (crops face if provided, uses full frame if null)
  ///   rotation: sensorRotation, // Camera sensor rotation in degrees (0, 90, 180, 270)
  /// );
  ///
  /// if (result.isReal) {
  ///   print('Real face verified!');
  /// } else {
  ///   print('Spoof detected: ${result.status.name}');
  /// }
  /// ```
  ///
  /// ### Parameters:
  /// - [buffer]: The raw camera frame image plane buffer (`NV21`, `YUV420`, or `BGRA8888`).
  /// - [boundingBox]: Optional facial bounding box from a face detector (e.g. ML Kit). If `null`, full frame is evaluated.
  /// - [rotation]: Sensor rotation angle in degrees (`0`, `90`, `180`, `270`). Default is `0`.
  /// - [isRotatedBoundingBox]: Whether bounding box coordinates are in rotated frame space. Auto-detected if `null`.
  /// - [threshold]: Logit decision threshold (default `0.0`). Positive values demand higher neural model confidence.
  /// - [expansionFactor]: Square crop margin around the bounding box (default `1.5x`).
  /// - [emaAlpha]: Exponential Moving Average smoothing factor across consecutive stream frames (default `0.3`).
  /// - [enableProximityGate]: Enable face area coverage ($5\% \le \text{ratio} \le 85\%$) & aspect ratio validation. Default `true`.
  /// - [enableTextureAnalysis]: Enable LBP halftone print noise & glasses-debiased HOG grid analysis. Default `true`.
  /// - [enableColorSpaceAnalysis]: Enable YCbCr sub-pixel chrominance variance ($\sigma^2_{CbCr}$) screen replay analysis. Default `true`.
  /// - [enableHighResScreenAnalysis]: Enable 2D Laplacian focus depth dispersal & specular glass highlight analysis for OLED/4K screens. Default `true`.
  Future<LivenessResult> detectLivenessFromBuffer(
    LivenessImageBuffer buffer, {
    FaceBoundingBox? boundingBox,
    int rotation = 0,
    bool? isRotatedBoundingBox,
    double threshold = 0.0,
    double expansionFactor = ImagePreprocessor.defaultExpansionFactor,
    double emaAlpha = defaultEmaAlpha,
    bool enableProximityGate = true,
    bool enableTextureAnalysis = true,
    bool enableColorSpaceAnalysis = true,
    bool enableHighResScreenAnalysis = true,
  }) async {
    final proximityGate = const FaceProximityGate();
    final textureAnalyzer = const LbpHogAnalyzer();
    final colorSpaceAnalyzer = const ColorSpaceAnalyzer();
    final highResScreenAnalyzer = const HighResScreenAnalyzer();
    if (!isInitialized) {
      throw StateError(
        'PassiveLivenessDetector is not initialized. Call initialize() first.',
      );
    }

    final stopwatch = Stopwatch()..start();

    // 1. Proximity & Aspect Ratio Gate Check
    double? faceAreaRatio;
    if (enableProximityGate && boundingBox != null) {
      final gateResult = proximityGate.evaluate(
        boundingBox: boundingBox,
        frameWidth: buffer.width,
        frameHeight: buffer.height,
      );
      faceAreaRatio = gateResult.faceAreaRatio;

      if (!gateResult.isValid) {
        stopwatch.stop();
        return LivenessResult.pending(
          threshold: threshold,
          status: gateResult.status,
        );
      }
    }

    // 2. Micro-Texture LBP / HOG Post-Processing
    double? lbpRatio;
    double? hogDominance;
    bool isPrintSpoof = false;
    bool isScreenGridSpoof = false;

    if (enableTextureAnalysis) {
      final highResCrop = ImagePreprocessor.extractHighResCrop(
        buffer,
        boundingBox: boundingBox,
        rotation: rotation,
      );
      final textureResult = textureAnalyzer.analyzeGrayscaleCrop(
        highResCrop,
        256,
        256,
      );
      lbpRatio = textureResult.lbpNonUniformRatio;
      hogDominance = textureResult.hogPeakDominance;
      isPrintSpoof = textureResult.isPrintSpoof;
      isScreenGridSpoof = textureResult.isScreenGridSpoof;
    }

    // 3. YCbCr Chrominance Variance Analysis
    double? chrominanceVar;
    bool isScreenReplaySpoof = false;

    if (enableColorSpaceAnalysis) {
      final colorResult = colorSpaceAnalyzer.analyzeBuffer(buffer);
      chrominanceVar = colorResult.chrominanceVariance;
      isScreenReplaySpoof = colorResult.isScreenReplaySpoof;
    }

    // 4. 2D Laplacian Frequency & Focus Depth Analysis for High-Res Screens
    bool isHighResScreenSpoof = false;

    if (enableHighResScreenAnalysis) {
      final highResCrop = ImagePreprocessor.extractHighResCrop(
        buffer,
        boundingBox: boundingBox,
        rotation: rotation,
      );
      final highResResult = highResScreenAnalyzer.analyzeGrayscaleCrop(
        highResCrop,
        256,
        256,
      );
      isHighResScreenSpoof = highResResult.isHighResScreenSpoof;
    }

    final effectiveUseNchw = _isNativeNchw;
    final effectiveTargetSize = _modelTargetSize;

    _ensureInputShapeAllocated(
      targetSize: effectiveTargetSize,
      useNchw: effectiveUseNchw,
    );

    final tensorData = ImagePreprocessor.preprocessBufferToTensor(
      buffer,
      boundingBox: boundingBox,
      rotation: rotation,
      isRotatedBoundingBox: isRotatedBoundingBox,
      expansionFactor: expansionFactor,
      targetSize: effectiveTargetSize,
      useNchw: effectiveUseNchw,
      isBgr: false,
      enableContrastStretch: true,
    );

    _logTensorStats(tensorData);

    final runOnIsolate = _isolateInterpreter != null;
    final logits = await _runInference(
      tensorData: tensorData,
      effectiveUseNchw: effectiveUseNchw,
      effectiveTargetSize: effectiveTargetSize,
      runOnIsolate: runOnIsolate,
    );
    stopwatch.stop();

    const realIdx = 0;
    const spoofIdx = 1;

    final realLogit = logits[realIdx];
    final spoofLogit = logits[spoofIdx];

    final rawResult = LivenessResult.fromLogits(
      realLogit: realLogit,
      spoofLogit: spoofLogit,
      threshold: threshold,
      inferenceTime: stopwatch.elapsed,
      lbpUniformityScore: lbpRatio,
      hogGridDominance: hogDominance,
      faceAreaRatio: faceAreaRatio,
      chrominanceVariance: chrominanceVar,
    );

    // Balanced EMA Calculation:
    final currentRealProb = 1.0 / (1.0 + math.exp(spoofLogit - realLogit));

    final double effectiveAlpha;
    if (_emaRealScore == null) {
      _emaRealScore = currentRealProb;
      effectiveAlpha = 1.0;
    } else {
      effectiveAlpha = emaAlpha;
      _emaRealScore =
          (currentRealProb * effectiveAlpha) +
          (_emaRealScore! * (1.0 - effectiveAlpha));
    }

    final safeEma = _emaRealScore!.clamp(1e-7, 1.0 - 1e-7);
    final spoofScoreEma = 1.0 - safeEma;
    final smoothedDiff = math.log(safeEma / (1.0 - safeEma));

    LivenessStatus calculatedStatus = (smoothedDiff >= threshold)
        ? LivenessStatus.real
        : LivenessStatus.spoof;

    // Multi-Factor Decision Fusion Engine:
    // Protect genuine users with glasses from single-metric false positives (e.g. HOG frame edges),
    // while catching definitive screen replay attacks (high chrominance variance).
    // Note: isHighResScreenSpoof is a soft signal only (not a hard override) because smartphone
    // front cameras have wide depth-of-field, making patch focus CV unreliable as a standalone gate.
    if (calculatedStatus == LivenessStatus.real) {
      int heuristicSpoofCount = 0;
      if (isPrintSpoof) heuristicSpoofCount++;
      if (isScreenGridSpoof) heuristicSpoofCount++;
      if (isScreenReplaySpoof) heuristicSpoofCount++;
      if (isHighResScreenSpoof) heuristicSpoofCount++;

      final isStrongNeuralReal = currentRealProb >= 0.70;

      // Only chrominance variance (isScreenReplaySpoof) is used as a hard override.
      // All other heuristics require multi-factor agreement.
      if (isScreenReplaySpoof ||
          (isStrongNeuralReal && heuristicSpoofCount >= 2) ||
          (!isStrongNeuralReal && heuristicSpoofCount >= 1)) {
        if (isPrintSpoof &&
            !isHighResScreenSpoof &&
            !isScreenGridSpoof &&
            !isScreenReplaySpoof) {
          calculatedStatus = LivenessStatus.printSpoof;
        } else {
          calculatedStatus = LivenessStatus.screenReplaySpoof;
        }
      }
    }

    final result = LivenessResult(
      isReal: calculatedStatus == LivenessStatus.real,
      status: calculatedStatus,
      realScore: safeEma,
      spoofScore: spoofScoreEma,
      realLogit: rawResult.realLogit,
      spoofLogit: rawResult.spoofLogit,
      logitDiff: smoothedDiff,
      confidence: smoothedDiff.abs(),
      threshold: threshold,
      inferenceTime: stopwatch.elapsed,
      rawRealScore: rawResult.realScore,
      rawSpoofScore: rawResult.spoofScore,
      rawLogitDiff: rawResult.logitDiff,
      rawIsReal: rawResult.isReal,
      lbpUniformityScore: lbpRatio,
      hogGridDominance: hogDominance,
      faceAreaRatio: faceAreaRatio,
      chrominanceVariance: chrominanceVar,
    );

    LivenessLogger.logInferenceResult(
      realLogit: realLogit,
      spoofLogit: spoofLogit,
      logitDiff: smoothedDiff,
      currentRealProb: currentRealProb,
      emaRealScore: safeEma,
      isReal: result.isReal,
      status: result.status,
      threshold: threshold,
      inferenceTime: stopwatch.elapsed,
    );

    return result;
  }

  /// Runs passive face liveness and anti-spoofing detection directly on raw image bytes ([Uint8List]).
  ///
  /// Evaluates static photo bytes (e.g. JPEG, PNG) using the full multi-layer anti-spoofing heuristic suite
  /// (micro-texture analysis, YCbCr chrominance variance, and 2D Laplacian high-res screen analysis).
  ///
  /// ### Parameters:
  /// - [imageBytes]: The raw image file bytes (`Uint8List`).
  /// - [boundingBox]: Optional facial bounding box from a face detector. If `null`, the full image is evaluated.
  /// - [threshold]: Logit decision threshold (default `0.0`). Positive values demand higher neural model confidence.
  /// - [expansionFactor]: Square crop margin around face bounding box (default `1.5x`).
  Future<LivenessResult> detectLivenessFromImageBytes(
    Uint8List imageBytes, {
    FaceBoundingBox? boundingBox,
    double threshold = 0.0,
    double expansionFactor = ImagePreprocessor.defaultExpansionFactor,
  }) async {
    if (!isInitialized) {
      throw StateError(
        'PassiveLivenessDetector is not initialized. Call initialize() first.',
      );
    }

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

    final buffer = LivenessImageBuffer(
      width: width,
      height: height,
      format: LivenessImageFormat.bgra8888,
      planes: [
        LivenessImagePlane(
          bytes: rgbaBytes,
          bytesPerRow: width * 4,
          bytesPerPixel: 4,
        ),
      ],
    );

    return detectLivenessFromBuffer(
      buffer,
      boundingBox: boundingBox,
      rotation: 0,
      threshold: threshold,
      expansionFactor: expansionFactor,
      enableProximityGate:
          false, // Proximity gate is disabled for static image crops
    );
  }

  /// Runs passive face liveness and anti-spoofing detection directly on a static image [File].
  ///
  /// Evaluates static photo files (e.g. from `image_picker` or camera capture) using the full anti-spoofing heuristic suite.
  ///
  /// ### Parameters:
  /// - [file]: The static image file to analyze.
  /// - [boundingBox]: Optional facial bounding box from a face detector. If `null`, the full image is evaluated.
  /// - [threshold]: Logit decision threshold (default `0.0`). Positive values demand higher neural model confidence.
  /// - [expansionFactor]: Square crop margin around face bounding box (default `1.5x`).
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

  /// Close LiteRT models and clear EMA tracker.
  void dispose() {
    resetEma();
    _isolateInterpreter?.close();
    _isolateInterpreter = null;
    _interpreter?.close();
    _interpreter = null;
    _compiledModel?.close();
    _compiledModel = null;
    _isInitialized = false;
  }
}
