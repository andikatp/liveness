import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/services.dart';

import 'models/face_bounding_box.dart';
import 'models/liveness_image_buffer.dart';
import 'models/liveness_result.dart';
import 'native_liveness_engine.dart';
import 'utils/color_space_analyzer.dart';
import 'utils/face_proximity_gate.dart';
import 'utils/high_res_screen_analyzer.dart';
import 'utils/image_preprocessor.dart';
import 'utils/lbp_hog_analyzer.dart';
import 'utils/liveness_logger.dart';

/// Core passive liveness detector engine using native TFLite inference.
///
/// Uses native platform channels for model inference:
/// - **Android**: Google Play Services TFLite (0 MB size impact)
/// - **iOS**: TensorFlowLiteSwift (~3-5 MB size impact)
///
/// All image preprocessing and heuristic analysis runs in Dart.
/// Only the neural network inference is delegated to native code.
class PassiveLivenessDetector {
  /// Default asset path for the TFLite model package asset.
  static const String defaultAssetPath =
      'packages/passive_liveness/assets/best_model.tflite';

  /// Fallback asset path when loaded directly within host application.
  static const String fallbackAssetPath = 'assets/best_model.tflite';

  /// Default Exponential Moving Average (EMA) alpha for score smoothing.
  static const double defaultEmaAlpha = 0.3;

  final NativeLivenessEngine _engine = NativeLivenessEngine();

  /// Creates a new [PassiveLivenessDetector] instance.
  PassiveLivenessDetector();

  bool _isInitialized = false;

  /// Input tensor shape of the loaded model (e.g. [1, 128, 128, 3] or [1, 3, 128, 128]).
  List<int>? get modelInputShape => _engine.modelInputShape;

  /// Whether the model natively expects NCHW format (`[1, 3, H, W]`).
  bool get isNativeNchw => _engine.isNativeNchw;

  /// Target spatial resolution expected by model (e.g. 128 or 80).
  int get modelTargetSize => _engine.modelTargetSize;

  /// The current Exponential Moving Average (EMA) of the real score.
  double? _emaRealScore;

  /// Returns the current EMA real score.
  double? get emaRealScore => _emaRealScore;

  /// Resets the EMA real score tracker.
  void resetEma() {
    _emaRealScore = null;
  }

  /// Whether the detector is initialized and ready for inference.
  bool get isInitialized => _isInitialized && _engine.isModelLoaded;

  /// Initialize the native TFLite model for passive liveness detection.
  ///
  /// Loads the model bytes and sends them to the native platform (Android/iOS)
  /// for interpreter initialization. On Android, this uses Google Play Services
  /// TFLite runtime (0 MB APK impact). On iOS, it uses TensorFlowLiteSwift.
  Future<void> initialize({
    String? assetPath,
    String? filePath,
    Uint8List? modelBytes,
  }) async {
    if (_isInitialized) return;

    Uint8List bytes;

    if (modelBytes != null) {
      bytes = modelBytes;
    } else if (filePath != null) {
      bytes = await File(filePath).readAsBytes();
    } else {
      final path = assetPath ?? defaultAssetPath;
      try {
        final bd = await rootBundle.load(path);
        bytes = bd.buffer.asUint8List(bd.offsetInBytes, bd.lengthInBytes);
      } catch (_) {
        final bd = await rootBundle.load(fallbackAssetPath);
        bytes = bd.buffer.asUint8List(bd.offsetInBytes, bd.lengthInBytes);
      }
    }

    await _engine.loadModel(bytes);

    LivenessLogger.logModelInit(
      inputShape: _engine.modelInputShape,
      tensorType: 'float32',
      isNativeNchw: _engine.isNativeNchw,
      targetSize: _engine.modelTargetSize,
    );

    _isInitialized = true;
  }

  Future<List<double>> _runInference({required Float32List tensorData}) async {
    return await _engine.runInference(tensorData);
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
      final colorResult = colorSpaceAnalyzer.analyzeBuffer(
        buffer,
        boundingBox: boundingBox,
      );
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

    final effectiveUseNchw = isNativeNchw;
    final effectiveTargetSize = modelTargetSize;

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

    LivenessLogger.logTensorStats(tensorData);

    final logits = await _runInference(tensorData: tensorData);
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

      // Chrominance variance (isScreenReplaySpoof) and 2D flat screen focus (isHighResScreenSpoof)
      // act as primary screen replay indicators to catch high-resolution OLED screen attacks.
      if (isScreenReplaySpoof ||
          isHighResScreenSpoof ||
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

  /// Close native TFLite interpreter and clear EMA tracker.
  Future<void> dispose() async {
    resetEma();
    await _engine.close();
    _isInitialized = false;
  }
}
