import 'dart:async';

import 'models/face_bounding_box.dart';
import 'models/liveness_image_buffer.dart';
import 'models/liveness_result.dart';
import 'passive_liveness_detector.dart';

/// Helper to handle live camera frame streaming without UI lag or frame dropping.
class LivenessFrameProcessor {
  /// The passive liveness detector engine instance.
  final PassiveLivenessDetector detector;

  /// Minimum time interval between consecutive frame processing calls.
  final Duration throttleInterval;

  bool _isProcessing = false;
  DateTime? _lastProcessedTime;

  /// Creates a [LivenessFrameProcessor] with the given [detector] and [throttleInterval].
  LivenessFrameProcessor({
    required this.detector,
    this.throttleInterval = const Duration(milliseconds: 150),
  });

  /// Whether a frame is currently being processed.
  bool get isProcessing => _isProcessing;

  /// Process a single frame from raw [LivenessImageBuffer].
  /// Returns `null` if the processor is busy or within throttle interval.
  Future<LivenessResult?> processBufferFrame(
    LivenessImageBuffer buffer, {
    FaceBoundingBox? boundingBox,
    int rotation = 0,
    bool? isRotatedBoundingBox,
    double threshold = 0.0,
    double expansionFactor = 1.5,
  }) async {
    final now = DateTime.now();

    if (_isProcessing) {
      return null;
    }

    if (_lastProcessedTime != null &&
        now.difference(_lastProcessedTime!) < throttleInterval) {
      return null;
    }

    _isProcessing = true;
    _lastProcessedTime = now;

    try {
      final result = await detector.detectLivenessFromBuffer(
        buffer,
        boundingBox: boundingBox,
        rotation: rotation,
        isRotatedBoundingBox: isRotatedBoundingBox,
        threshold: threshold,
        expansionFactor: expansionFactor,
      );
      return result;
    } finally {
      _isProcessing = false;
    }
  }

  /// Reset internal frame processor state.
  void reset() {
    _isProcessing = false;
    _lastProcessedTime = null;
  }
}
