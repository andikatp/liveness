import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'face_anti_spoofing_detector_platform_interface.dart';

/// An implementation of [LivenessDetectorPlatform] that uses method channels.
class MethodChannelFaceAntiSpoofingDetector
    extends FaceAntiSpoofingDetectorPlatform {
  /// The method channel used to interact with the native platform.
  @visibleForTesting
  final methodChannel = const MethodChannel('com.leng.dev/liveness_detector');

  /// Last debug log from native detector (iOS only).
  String? lastNativeDebugLog;

  @override
  Future<String?> getPlatformVersion() async {
    final version = await methodChannel.invokeMethod<String>(
      'getPlatformVersion',
    );
    return version;
  }

  @override
  Future<bool> initialize() async {
    final result = await methodChannel.invokeMethod<bool>('initialize');
    return result ?? false;
  }

  @override
  Future<bool> destroy() async {
    final result = await methodChannel.invokeMethod<bool>('destroy');
    return result ?? false;
  }

  @override
  Future<double?> detectLiveness({
    required Uint8List yuvBytes,
    required int previewWidth,
    required int previewHeight,
    required int orientation,
    required Rect faceContour,
  }) async {
    final result = await methodChannel.invokeMethod('detect_liveness', {
      'yuvBytes': yuvBytes,
      'previewWidth': previewWidth,
      'previewHeight': previewHeight,
      'orientation': orientation,
      'faceBox': {
        'left': faceContour.left.toInt(),
        'top': faceContour.top.toInt(),
        'right': faceContour.right.toInt(),
        'bottom': faceContour.bottom.toInt(),
      },
    });

    // iOS returns a Map with 'score' and 'debug' keys
    if (result is Map) {
      final debug = result['debug'];
      if (debug is String && debug.isNotEmpty) {
        lastNativeDebugLog = debug;
      }
      final score = result['score'];
      if (score == null || score is! num) return null;
      return score.toDouble();
    }

    // Android returns a plain double (backward compatible)
    if (result is num) {
      lastNativeDebugLog = null;
      return result.toDouble();
    }

    lastNativeDebugLog = null;
    return null;
  }
}
