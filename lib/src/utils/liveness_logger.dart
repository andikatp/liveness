import 'package:flutter/foundation.dart';

import '../models/face_bounding_box.dart';
import '../models/liveness_result.dart';

/// Centralized diagnostic logging utility for passive face liveness detection.
class LivenessLogger {
  const LivenessLogger._();

  /// Global flag to enable or disable diagnostic print/developer logging.
  static bool enableLogging = true;

  /// Log a message with standard `[PASSIVE_LIVENESS]` prefix.
  static void log(String message) {
    if (!enableLogging) return;
    debugPrint('[PASSIVE_LIVENESS] $message');
  }

  /// Log model initialization specs.
  static void logModelInit({
    required List<int>? inputShape,
    required String tensorType,
    required bool isNativeNchw,
    required int targetSize,
    String engineName = 'Native Platform Channel (Play Services / TFLiteSwift)',
  }) {
    if (!enableLogging) return;
    log(
      'Model Init -> Shape: $inputShape | Type: $tensorType | Format: ${isNativeNchw ? "NCHW [1, 3, $targetSize, $targetSize]" : "NHWC [1, $targetSize, $targetSize, 3]"} | TargetSize: $targetSize | Engine: $engineName',
    );
  }

  /// Log image buffer crop and bounding box metrics.
  static void logCropStats({
    required int rawWidth,
    required int rawHeight,
    required int rotation,
    required FaceBoundingBox? boundingBox,
    required double expansionFactor,
    required double cropLeft,
    required double cropTop,
    required double cropWidth,
    required double cropHeight,
  }) {
    if (!enableLogging) return;
    final bboxStr = boundingBox != null
        ? 'x: ${boundingBox.x.toStringAsFixed(1)}, y: ${boundingBox.y.toStringAsFixed(1)}, w: ${boundingBox.width.toStringAsFixed(1)}, h: ${boundingBox.height.toStringAsFixed(1)}'
        : 'Full Frame';
    log(
      'Crop Stats -> Raw: ${rawWidth}x$rawHeight (rot: $rotation°) | FaceBox: [$bboxStr] | Expansion: ${expansionFactor.toStringAsFixed(2)}x | CropRegion: (L: ${cropLeft.toStringAsFixed(1)}, T: ${cropTop.toStringAsFixed(1)}, W: ${cropWidth.toStringAsFixed(1)}, H: ${cropHeight.toStringAsFixed(1)})',
    );
  }

  /// Log input tensor stats (min, max, mean, length, input shape).
  static void logTensorStats(Float32List tensorData, {List<int>? inputShape}) {
    if (!enableLogging || tensorData.isEmpty) return;
    double minVal = tensorData[0];
    double maxVal = tensorData[0];
    double sumVal = 0.0;
    for (int i = 0; i < tensorData.length; i++) {
      final v = tensorData[i];
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
      sumVal += v;
    }
    final double meanVal = sumVal / tensorData.length;
    log(
      'Input Tensor -> Shape: $inputShape | Stats -> min: ${minVal.toStringAsFixed(4)}, max: ${maxVal.toStringAsFixed(4)}, mean: ${meanVal.toStringAsFixed(4)}, len: ${tensorData.length}',
    );
  }

  /// Log inference logits, softmax scores, EMA smoothing, threshold, and time.
  static void logInferenceResult({
    required double realLogit,
    required double spoofLogit,
    required double logitDiff,
    required double currentRealProb,
    required double? emaRealScore,
    required bool isReal,
    required LivenessStatus status,
    required double threshold,
    required Duration inferenceTime,
  }) {
    if (!enableLogging) return;
    final emaStr = emaRealScore != null
        ? emaRealScore.toStringAsFixed(4)
        : 'none';
    log(
      'Inference Result -> Logits: [real: ${realLogit.toStringAsFixed(4)}, spoof: ${spoofLogit.toStringAsFixed(4)}] | LogitDiff: ${logitDiff.toStringAsFixed(4)} | CurrentProb: ${currentRealProb.toStringAsFixed(4)} | EMA: $emaStr | Status: ${status.name.toUpperCase()} (isReal: $isReal, threshold: ${threshold.toStringAsFixed(2)}) | Time: ${inferenceTime.inMilliseconds}ms',
    );
  }

  /// Log motion stability metrics for streaming camera frames.
  static void logMotionStability({
    required bool isStable,
    required double dx,
    required double dy,
    required double dw,
    required double dh,
  }) {
    if (!enableLogging) return;
    log(
      'Motion Check -> ${isStable ? "STABLE" : "UNSTABLE (Motion Detected)"} | Deltas -> dx: ${dx.toStringAsFixed(3)}, dy: ${dy.toStringAsFixed(3)}, dw: ${dw.toStringAsFixed(3)}, dh: ${dh.toStringAsFixed(3)}',
    );
  }
}
