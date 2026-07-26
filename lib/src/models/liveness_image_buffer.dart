import 'dart:typed_data';

/// Image format group supported for passive liveness detection.
enum LivenessImageFormat {
  /// YUV 4:2:0 format (typical for Android camera streams).
  yuv420,

  /// NV21 interleaved format.
  nv21,

  /// BGRA 8888 32-bit format (typical for iOS camera streams).
  bgra8888,
}

/// Represents a single plane in a raw camera frame buffer.
class LivenessImagePlane {
  /// Byte array for this plane.
  final Uint8List bytes;

  /// Number of bytes per row (row stride).
  final int bytesPerRow;

  /// Number of bytes per pixel (pixel stride), if applicable.
  final int? bytesPerPixel;

  /// Creates a [LivenessImagePlane] with [bytes], [bytesPerRow], and optional [bytesPerPixel].
  const LivenessImagePlane({
    required this.bytes,
    required this.bytesPerRow,
    this.bytesPerPixel,
  });
}

/// Pure Dart lightweight model representing a raw camera image buffer.
///
/// Designed to decouple `passive_liveness` from any specific Flutter camera package.
class LivenessImageBuffer {
  /// Buffer width in pixels.
  final int width;

  /// Buffer height in pixels.
  final int height;

  /// Buffer color format (`yuv420`, `nv21`, `bgra8888`).
  final LivenessImageFormat format;

  /// List of image planes.
  final List<LivenessImagePlane> planes;

  /// Creates a [LivenessImageBuffer] with [width], [height], [format], and [planes].
  const LivenessImageBuffer({
    required this.width,
    required this.height,
    required this.format,
    required this.planes,
  });
}
