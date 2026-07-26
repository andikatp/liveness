import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import '../models/face_bounding_box.dart';
import '../models/liveness_image_buffer.dart';

/// Preprocessing utilities for passive face anti-spoofing input.
class ImagePreprocessor {
  /// Default input size expected by MiniFAS model (128x128).
  static const int defaultModelSize = 128;

  /// Default bounding box expansion factor recommended by MiniFAS (2.0x).
  static const double defaultExpansionFactor = 2.0;

  /// Closed-form reflect-101 index wrapping matching OpenCV `BORDER_REFLECT_101`.
  /// O(1) — no while loops. Maps [p] into `[0, max-1]` using mirrored reflection.
  static int reflect101(int p, int max) {
    if (max <= 1) return 0;
    final period = 2 * (max - 1);
    p = p % period;
    if (p < 0) p += period;
    return p < max ? p : period - p;
  }

  /// Closed-form double-precision reflect-101 index wrapping.
  /// O(1) — maps [p] into `[0.0, max-1.0]` using mirrored reflection.
  static double reflect101Double(double p, int max) {
    if (max <= 1) return 0.0;
    final period = 2.0 * (max - 1);
    p = p % period;
    if (p < 0.0) p += period;
    return p < max ? p : period - p;
  }

  /// Preprocesses a raw camera [LivenessImageBuffer] into a Float32 NHWC tensor
  /// `[1, targetSize, targetSize, 3]`.
  ///
  /// The [boundingBox] can be in raw buffer space or rotated frame space (e.g. ML Kit on Android).
  /// The [rotation] parameter (0, 90, 180, 270 degrees CW) specifies how many degrees
  /// the raw buffer must be rotated to become upright. The pixel sampling rotates
  /// the cropped face patch so that the output tensor is always **upright**
  /// (eyes at top, chin at bottom) for MiniFAS inference.
  static Float32List preprocessBufferToTensor(
    LivenessImageBuffer buffer, {
    FaceBoundingBox? boundingBox,
    int rotation = 0,
    bool? isRotatedBoundingBox,
    bool enableShadowLift = true,
    double expansionFactor = defaultExpansionFactor,
    int targetSize = defaultModelSize,
  }) {
    final rawW = buffer.width;
    final rawH = buffer.height;
    final normRotation = ((rotation % 360) + 360) % 360;

    final faceBbox = boundingBox ??
        FaceBoundingBox(
          x: 0,
          y: 0,
          width: rawW.toDouble(),
          height: rawH.toDouble(),
        );

    // Auto-detect if bounding box is in rotated frame space [0..rotW, 0..rotH]
    // (ML Kit on Android returns face bounding boxes in rotated frame space).
    final bool isRotated = isRotatedBoundingBox ??
        (Platform.isAndroid && (normRotation == 90 || normRotation == 270)) ||
        ((normRotation == 90 || normRotation == 270) &&
            (faceBbox.centerY > rawH || faceBbox.centerX > rawW));

    final effectiveBbox = isRotated
        ? faceBbox.toRawBufferSpace(rawW, rawH, normRotation)
        : faceBbox;

    // Expand the bounding box and compute square crop in raw buffer space.
    final maxDim = math.max(effectiveBbox.width, effectiveBbox.height);
    final cropSize = math.max(1.0, maxDim * expansionFactor);
    final cropLeft = effectiveBbox.centerX - cropSize / 2.0;
    final cropTop = effectiveBbox.centerY - cropSize / 2.0;

    final tensor = Float32List(1 * targetSize * targetSize * 3);

    final isBgra = buffer.format == LivenessImageFormat.bgra8888;

    final Uint8List plane0 = buffer.planes[0].bytes;
    final int p0Stride = buffer.planes[0].bytesPerRow;

    final Uint8List? plane1 =
        buffer.planes.length > 1 ? buffer.planes[1].bytes : null;
    final Uint8List? plane2 =
        buffer.planes.length > 2 ? buffer.planes[2].bytes : null;

    final int uvStride =
        plane1 != null ? buffer.planes[1].bytesPerRow : 0;
    final int uvPixelStride =
        plane1 != null ? (buffer.planes[1].bytesPerPixel ?? 1) : 1;

    final int rawWm1 = rawW - 1;
    final int rawHm1 = rawH - 1;
    const double inv255 = 1.0 / 255.0;

    for (int y = 0; y < targetSize; y++) {
      final double normUY = (y + 0.5) / targetSize;

      for (int x = 0; x < targetSize; x++) {
        final double normUX = (x + 0.5) / targetSize;

        // Map normalized upright coordinates (normUX, normUY) back to
        // raw buffer relative crop coordinates (relX, relY) based on rotation.
        double relX, relY;
        switch (normRotation) {
          case 90:
            relX = (1.0 - normUY) * cropSize;
            relY = normUX * cropSize;
            break;
          case 180:
            relX = (1.0 - normUX) * cropSize;
            relY = (1.0 - normUY) * cropSize;
            break;
          case 270:
            relX = (1.0 - normUY) * cropSize;
            relY = (1.0 - normUX) * cropSize;
            break;
          case 0:
          default:
            relX = normUX * cropSize;
            relY = normUY * cropSize;
            break;
        }

        final double srcXf = reflect101Double(cropLeft + relX, rawW);
        final double srcYf = reflect101Double(cropTop + relY, rawH);

        final int srcX = srcXf.toInt().clamp(0, rawWm1);
        final int srcY = srcYf.toInt().clamp(0, rawHm1);

        int r, g, b;

        if (isBgra) {
          final offset = srcY * p0Stride + (srcX << 2);
          if (offset + 2 < plane0.length) {
            b = plane0[offset];
            g = plane0[offset + 1];
            r = plane0[offset + 2];
          } else {
            r = g = b = 0;
          }
        } else {
          // YUV420 / NV21
          final yIdx = srcY * p0Stride + srcX;
          final yVal = yIdx < plane0.length ? plane0[yIdx] : 0;

          int uVal = 128;
          int vVal = 128;

          if (plane1 != null && plane2 != null) {
            final uvRow = srcY >> 1;
            final uvCol = srcX >> 1;
            final uvOff = uvRow * uvStride + uvCol * uvPixelStride;

            if (uvOff < plane1.length && uvOff < plane2.length) {
              if (buffer.format == LivenessImageFormat.nv21) {
                vVal = plane1[uvOff];
                uVal = plane2[uvOff];
              } else {
                uVal = plane1[uvOff];
                vVal = plane2[uvOff];
              }
            }
          } else if (plane1 != null) {
            // Dual plane (Y in plane0, interleaved UV/VU in plane1)
            final uvRow = srcY >> 1;
            final uvCol = srcX >> 1;
            final p1PixelStride = buffer.planes[1].bytesPerPixel ?? 2;
            final uvOff = uvRow * uvStride + uvCol * p1PixelStride;

            if (uvOff + 1 < plane1.length) {
              if (buffer.format == LivenessImageFormat.nv21) {
                vVal = plane1[uvOff];
                uVal = plane1[uvOff + 1];
              } else {
                uVal = plane1[uvOff];
                vVal = plane1[uvOff + 1];
              }
            }
          } else if (plane0.length >= rawW * rawH * 3 ~/ 2) {
            // Single plane packed NV21 / NV12 in plane0
            final uvOff =
                p0Stride * rawH + (srcY >> 1) * p0Stride + ((srcX >> 1) << 1);
            if (uvOff + 1 < plane0.length) {
              if (buffer.format == LivenessImageFormat.nv21) {
                vVal = plane0[uvOff];
                uVal = plane0[uvOff + 1];
              } else {
                uVal = plane0[uvOff];
                vVal = plane0[uvOff + 1];
              }
            }
          }

          // BT.601 YUV→RGB conversion
          final u = uVal - 128;
          final v = vVal - 128;
          r = (yVal + 1.402 * v).round().clamp(0, 255);
          g = (yVal - 0.344136 * u - 0.714136 * v).round().clamp(0, 255);
          b = (yVal + 1.772 * u).round().clamp(0, 255);
        }

        final idx = (y * targetSize + x) * 3;
        tensor[idx] = r * inv255;
        tensor[idx + 1] = g * inv255;
        tensor[idx + 2] = b * inv255;
      }
    }

    // Low-light & pitch-shadow adaptive compensation:
    // If the image crop is underexposed or shadowed (mean RGB < 0.35),
    // apply an adaptive linear offset (+0.04 to +0.085) to un-clip shadow
    // gradients so low-light pitch/angle 3D skin features remain clear for MiniFAS.
    double meanBrightness = 0.0;
    for (int i = 0; i < tensor.length; i += 3) {
      meanBrightness +=
          0.299 * tensor[i] + 0.587 * tensor[i + 1] + 0.114 * tensor[i + 2];
    }
    meanBrightness /= (targetSize * targetSize);

    if (enableShadowLift && meanBrightness > 0.01 && meanBrightness < 0.38) {
      // Adaptive gamma curve (0.60 to 0.88) lifts dark shadows (eye sockets, cheeks)
      // while expanding skin texture contrast for MiniFAS in dim light.
      final double gamma = 0.60 + (meanBrightness / 0.38) * 0.28;
      final double boostOffset = (0.38 - meanBrightness) * 0.10;
      for (int i = 0; i < tensor.length; i++) {
        final val = tensor[i];
        if (val > 0) {
          tensor[i] = (math.pow(val, gamma).toDouble() + boostOffset).clamp(0.0, 1.0);
        }
      }
    }

    return tensor;
  }

  /// Preprocesses a raw RGBA 32-bit pixel byte buffer into a Float32 NHWC tensor
  /// `[1, targetSize, targetSize, 3]`.
  ///
  /// The [boundingBox] is expected in static image pixel coordinates `[0..rawW, 0..rawH]`.
  static Float32List preprocessRgbaBytesToTensor(
    Uint8List rgbaBytes,
    int rawW,
    int rawH, {
    FaceBoundingBox? boundingBox,
    double expansionFactor = defaultExpansionFactor,
    int targetSize = defaultModelSize,
  }) {
    final faceBbox = boundingBox ??
        FaceBoundingBox(
          x: 0,
          y: 0,
          width: rawW.toDouble(),
          height: rawH.toDouble(),
        );

    final maxDim = math.max(faceBbox.width, faceBbox.height);
    final cropSize = math.max(1.0, maxDim * expansionFactor);
    final cropLeft = faceBbox.centerX - cropSize / 2.0;
    final cropTop = faceBbox.centerY - cropSize / 2.0;

    final tensor = Float32List(1 * targetSize * targetSize * 3);
    final double scale = cropSize / targetSize;
    final int rawWm1 = rawW - 1;
    final int rawHm1 = rawH - 1;
    const double inv255 = 1.0 / 255.0;

    for (int y = 0; y < targetSize; y++) {
      final double srcYf = reflect101Double(cropTop + (y + 0.5) * scale, rawH);
      final int srcY = srcYf.toInt().clamp(0, rawHm1);

      for (int x = 0; x < targetSize; x++) {
        final double srcXf =
            reflect101Double(cropLeft + (x + 0.5) * scale, rawW);
        final int srcX = srcXf.toInt().clamp(0, rawWm1);

        final offset = (srcY * rawW + srcX) * 4;
        final r = offset < rgbaBytes.length ? rgbaBytes[offset].toDouble() : 0.0;
        final g =
            offset + 1 < rgbaBytes.length ? rgbaBytes[offset + 1].toDouble() : 0.0;
        final b =
            offset + 2 < rgbaBytes.length ? rgbaBytes[offset + 2].toDouble() : 0.0;

        final idx = (y * targetSize + x) * 3;
        tensor[idx] = r * inv255;
        tensor[idx + 1] = g * inv255;
        tensor[idx + 2] = b * inv255;
      }
    }

    double meanBrightness = 0.0;
    for (int i = 0; i < tensor.length; i += 3) {
      meanBrightness +=
          0.299 * tensor[i] + 0.587 * tensor[i + 1] + 0.114 * tensor[i + 2];
    }
    meanBrightness /= (targetSize * targetSize);

    if (meanBrightness > 0.01 && meanBrightness < 0.38) {
      final double gamma = 0.60 + (meanBrightness / 0.38) * 0.28;
      final double boostOffset = (0.38 - meanBrightness) * 0.10;
      for (int i = 0; i < tensor.length; i++) {
        final val = tensor[i];
        if (val > 0) {
          tensor[i] = (math.pow(val, gamma).toDouble() + boostOffset).clamp(0.0, 1.0);
        }
      }
    }

    return tensor;
  }
}

