import 'dart:async';
import 'dart:ui' as ui;

import 'models/face_bounding_box.dart';
import 'models/liveness_image_buffer.dart';
import 'models/liveness_result.dart';
import 'passive_liveness_detector.dart';
import 'utils/liveness_logger.dart';

/// Helper to handle live camera frame streaming without UI lag or frame dropping.
class LivenessFrameProcessor {
  /// The passive liveness detector engine instance.
  final PassiveLivenessDetector detector;

  /// Minimum time interval between consecutive frame processing calls.
  final Duration throttleInterval;

  bool _isProcessing = false;
  DateTime? _lastProcessedTime;

  /// Previous frame bounding box for motion stability check.
  ui.Rect? _lastFaceBox;

  /// Cached previous inference result returned when frame is unstable due to motion.
  LivenessResult? _lastResult;

  /// Creates a [LivenessFrameProcessor] with the given [detector] and [throttleInterval].
  LivenessFrameProcessor({
    required this.detector,
    this.throttleInterval = const Duration(milliseconds: 150),
  });

  /// Whether a frame is currently being processed.
  bool get isProcessing => _isProcessing;

  /// Checks whether the face bounding box is stable relative to the previous frame.
  ///
  /// Returns `false` if motion deltas exceed 5% (0.05), indicating motion blur
  /// or camera AE/AF adjustment.
  bool _isFaceStable(ui.Rect currentBox, int frameWidth, int frameHeight) {
    if (_lastFaceBox == null) {
      _lastFaceBox = currentBox;
      return false; // Need a baseline frame
    }

    final currCx = currentBox.center.dx;
    final currCy = currentBox.center.dy;
    final lastCx = _lastFaceBox!.center.dx;
    final lastCy = _lastFaceBox!.center.dy;

    final dx = (currCx - lastCx).abs() / frameWidth;
    final dy = (currCy - lastCy).abs() / frameHeight;

    final lastWidth = _lastFaceBox!.width > 0 ? _lastFaceBox!.width : 1.0;
    final lastHeight = _lastFaceBox!.height > 0 ? _lastFaceBox!.height : 1.0;

    final dw = (currentBox.width - _lastFaceBox!.width).abs() / lastWidth;
    final dh = (currentBox.height - _lastFaceBox!.height).abs() / lastHeight;

    _lastFaceBox = currentBox;

    final isStable = !(dx > 0.05 || dy > 0.05 || dw > 0.05 || dh > 0.05);

    LivenessLogger.logMotionStability(
      isStable: isStable,
      dx: dx,
      dy: dy,
      dw: dw,
      dh: dh,
    );

    return isStable;
  }

  /// Processes a single frame from a raw camera [LivenessImageBuffer].
  ///
  /// Throttles frame evaluation frequency according to [throttleInterval] to prevent UI lag,
  /// and automatically bypasses inference during user motion blur.
  ///
  /// Returns `null` if the processor is busy or within the throttle interval.
  Future<LivenessResult?> processBufferFrame(
    LivenessImageBuffer buffer, {
    FaceBoundingBox? boundingBox,
    int rotation = 0,
    bool? isRotatedBoundingBox,
    double threshold = 0.0,
    double expansionFactor = 1.5,
    bool enableProximityGate = true,
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
      if (boundingBox != null) {
        final rect = ui.Rect.fromLTWH(
          boundingBox.x,
          boundingBox.y,
          boundingBox.width,
          boundingBox.height,
        );
        final isStable = _isFaceStable(rect, buffer.width, buffer.height);
        if (!isStable) {
          return _lastResult ?? LivenessResult.pending();
        }
      }

      final result = await detector.detectLivenessFromBuffer(
        buffer,
        boundingBox: boundingBox,
        rotation: rotation,
        isRotatedBoundingBox: isRotatedBoundingBox,
        threshold: threshold,
        expansionFactor: expansionFactor,
        enableProximityGate: enableProximityGate,
      );
      _lastResult = result;
      return result;
    } finally {
      _isProcessing = false;
    }
  }

  /// Reset internal frame processor state and clear detector EMA tracker.
  void reset() {
    _isProcessing = false;
    _lastProcessedTime = null;
    _lastFaceBox = null;
    _lastResult = null;
    detector.resetEma();
  }
}
