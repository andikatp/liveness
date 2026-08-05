import 'dart:math' as math;
import 'dart:typed_data';

import '../models/face_bounding_box.dart';
import '../models/liveness_image_buffer.dart';

/// Result of evaluating YCbCr color space chrominance distributions.
class ColorSpaceAnalysisResult {
  /// Calculated chrominance variance ($\sigma^2_{CbCr} = \sigma^2_{Cb} + \sigma^2_{Cr}$).
  final double chrominanceVariance;

  /// Mean Cb channel chrominance value ($0 \dots 255$).
  final double meanCb;

  /// Mean Cr channel chrominance value ($0 \dots 255$).
  final double meanCr;

  /// Whether the chrominance distribution indicates a digital screen replay attack.
  final bool isScreenReplaySpoof;

  const ColorSpaceAnalysisResult({
    required this.chrominanceVariance,
    required this.meanCb,
    required this.meanCr,
    required this.isScreenReplaySpoof,
  });
}

/// Evaluates YCbCr / YUV chrominance sub-sampling metrics ($\sigma^2_{CbCr}$)
/// to detect emissive RGB digital display screen replay attacks.
class ColorSpaceAnalyzer {
  /// Upper bound threshold for chrominance variance (default: 160.0). Higher values indicate screen sub-pixel dispersion.
  final double maxVarianceThreshold;

  /// Lower bound threshold for chrominance variance (default: 0.5). Extremely low values indicate flat monochrome / synthetic photos.
  final double minVarianceThreshold;

  const ColorSpaceAnalyzer({
    this.maxVarianceThreshold = 160.0,
    this.minVarianceThreshold = 0.5,
  });

  /// Evaluates a raw [LivenessImageBuffer] in YUV420 or BGRA format.
  ColorSpaceAnalysisResult analyzeBuffer(
    LivenessImageBuffer buffer, {
    FaceBoundingBox? boundingBox,
  }) {
    final width = buffer.width;
    final height = buffer.height;

    if (width <= 0 || height <= 0 || buffer.planes.isEmpty) {
      return const ColorSpaceAnalysisResult(
        chrominanceVariance: 0.0,
        meanCb: 128.0,
        meanCr: 128.0,
        isScreenReplaySpoof: false,
      );
    }

    int startX = 0;
    int startY = 0;
    int endX = width;
    int endY = height;

    if (boundingBox != null) {
      // Analyze only within the face region to avoid false positives from colorful backgrounds
      startX = math.max(0, boundingBox.x.toInt());
      startY = math.max(0, boundingBox.y.toInt());
      endX = math.min(width, (boundingBox.x + boundingBox.width).ceil());
      endY = math.min(height, (boundingBox.y + boundingBox.height).ceil());
    }

    final isBgra = buffer.format == LivenessImageFormat.bgra8888;
    final isRgba = buffer.format == LivenessImageFormat.rgba8888;

    double sumCb = 0.0;
    double sumCr = 0.0;
    int sampleCount = 0;

    // Subsample every 4th pixel for high performance without loss of statistical accuracy
    const step = 4;

    final maxSamples =
        ((width + step - 1) ~/ step) * ((height + step - 1) ~/ step);
    final cbList = Float64List(maxSamples);
    final crList = Float64List(maxSamples);

    if (isBgra || isRgba) {
      final plane0 = buffer.planes[0].bytes;
      final stride = buffer.planes[0].bytesPerRow;

      for (int y = startY; y < endY; y += step) {
        final rowOffset = y * stride;
        for (int x = startX; x < endX; x += step) {
          final offset = rowOffset + (x << 2);
          if (offset + 2 < plane0.length) {
            final b = (isBgra ? plane0[offset] : plane0[offset + 2]).toDouble();
            final g = plane0[offset + 1].toDouble();
            final r = (isBgra ? plane0[offset + 2] : plane0[offset]).toDouble();

            // Skip specular glare pixels (high brightness >= 245)
            final luma = 0.299 * r + 0.587 * g + 0.114 * b;
            if (luma >= 245) continue;

            // BT.601 RGB -> YCbCr formulas
            final cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b;
            final cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b;

            cbList[sampleCount] = cb;
            crList[sampleCount] = cr;
            sumCb += cb;
            sumCr += cr;
            sampleCount++;
          }
        }
      }
    } else {
      // YUV420 / NV21 buffer planes
      final plane1 = buffer.planes.length > 1 ? buffer.planes[1].bytes : null;
      final plane2 = buffer.planes.length > 2 ? buffer.planes[2].bytes : null;

      if (plane1 != null) {
        final uvStride = buffer.planes[1].bytesPerRow;
        final uvPixelStride = buffer.planes[1].bytesPerPixel ?? 1;

        for (int y = startY; y < endY; y += step) {
          final uvRow = (y >> 1) * uvStride;
          for (int x = startX; x < endX; x += step) {
            final uvCol = x >> 1;
            final uvOff = uvRow + uvCol * uvPixelStride;

            int uVal = 128;
            int vVal = 128;

            if (uvPixelStride == 2 && uvOff + 1 < plane1.length) {
              if (buffer.format == LivenessImageFormat.nv21) {
                vVal = plane1[uvOff];
                uVal = plane1[uvOff + 1];
              } else {
                uVal = plane1[uvOff];
                vVal = plane1[uvOff + 1];
              }
            } else if (plane2 != null &&
                uvOff < plane1.length &&
                uvOff < plane2.length) {
              if (buffer.format == LivenessImageFormat.nv21) {
                vVal = plane1[uvOff];
                uVal = plane2[uvOff];
              } else {
                uVal = plane1[uvOff];
                vVal = plane2[uvOff];
              }
            }

            final cb = uVal.toDouble();
            final cr = vVal.toDouble();

            cbList[sampleCount] = cb;
            crList[sampleCount] = cr;
            sumCb += cb;
            sumCr += cr;
            sampleCount++;
          }
        }
      }
    }

    if (sampleCount == 0) {
      return const ColorSpaceAnalysisResult(
        chrominanceVariance: 0.0,
        meanCb: 128.0,
        meanCr: 128.0,
        isScreenReplaySpoof: false,
      );
    }

    final meanCb = sumCb / sampleCount;
    final meanCr = sumCr / sampleCount;

    double varCb = 0.0;
    double varCr = 0.0;

    for (int i = 0; i < sampleCount; i++) {
      final diffCb = cbList[i] - meanCb;
      final diffCr = crList[i] - meanCr;
      varCb += diffCb * diffCb;
      varCr += diffCr * diffCr;
    }

    varCb /= sampleCount;
    varCr /= sampleCount;

    final totalChrominanceVar = varCb + varCr;
    final isScreenReplaySpoof =
        totalChrominanceVar > maxVarianceThreshold ||
        totalChrominanceVar < minVarianceThreshold;

    return ColorSpaceAnalysisResult(
      chrominanceVariance: totalChrominanceVar,
      meanCb: meanCb,
      meanCr: meanCr,
      isScreenReplaySpoof: isScreenReplaySpoof,
    );
  }
}
