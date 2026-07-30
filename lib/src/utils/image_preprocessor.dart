import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import '../models/face_bounding_box.dart';
import '../models/liveness_image_buffer.dart';

/// Preprocessing utilities for passive face anti-spoofing input.
class ImagePreprocessor {
  const ImagePreprocessor._();

  /// Default input size expected by MiniFAS model (128x128).
  static const int defaultModelSize = 128;

  /// Default bounding box expansion factor recommended by MiniFAS (2.7x).
  static const double defaultExpansionFactor = 2.7;

  /// Preprocesses a raw camera [LivenessImageBuffer] into a Float32 tensor.
  ///
  /// Stores normalized RGB pixel values in `[0.0, 1.0]` range.
  /// Out-of-boundary pixels use **Edge Pixel Replication** (`BORDER_REPLICATE` coordinate clamping)
  /// to avoid artificial black border artifacts.
  ///
  /// If [useNchw] is `true` (default), the tensor is structured in NCHW format
  /// `[1, 3, targetSize, targetSize]`. Otherwise, NHWC format `[1, targetSize, targetSize, 3]` is used.
  static Float32List preprocessBufferToTensor(
    LivenessImageBuffer buffer, {
    FaceBoundingBox? boundingBox,
    int rotation = 0,
    bool? isRotatedBoundingBox,
    double expansionFactor = defaultExpansionFactor,
    int targetSize = defaultModelSize,
    bool useNchw = true,
    bool isBgr = false,
    bool enableContrastStretch = false,
  }) {
    final rawW = buffer.width;
    final rawH = buffer.height;
    final normRotation = ((rotation % 360) + 360) % 360;

    final faceBbox =
        boundingBox ??
        FaceBoundingBox(
          x: 0,
          y: 0,
          width: rawW.toDouble(),
          height: rawH.toDouble(),
        );

    // Auto-detect if bounding box is in rotated frame space [0..rotW, 0..rotH]
    final bool isRotated =
        isRotatedBoundingBox ??
        (Platform.isAndroid && (normRotation == 90 || normRotation == 270)) ||
            ((normRotation == 90 || normRotation == 270) &&
                (faceBbox.centerY > rawH || faceBbox.centerX > rawW));

    final effectiveBbox = isRotated
        ? faceBbox.toRawBufferSpace(rawW, rawH, normRotation)
        : faceBbox;

    // Python crop logic replication:
    // scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, expansionFactor)
    final boxW = math.max(1.0, effectiveBbox.width);
    final boxH = math.max(1.0, effectiveBbox.height);
    final scale = math.min(
      math.min((rawH - 1) / boxH, (rawW - 1) / boxW),
      expansionFactor,
    );
    final newW = boxW * scale;
    final newH = boxH * scale;

    final cropLeft = effectiveBbox.centerX - newW / 2.0;
    final cropTop = effectiveBbox.centerY - newH / 2.0;

    final tensor = Float32List(1 * targetSize * targetSize * 3);
    final hw = targetSize * targetSize;

    final isBgra = buffer.format == LivenessImageFormat.bgra8888;

    final Uint8List plane0 = buffer.planes[0].bytes;
    final int p0Stride = buffer.planes[0].bytesPerRow;

    final Uint8List? plane1 = buffer.planes.length > 1
        ? buffer.planes[1].bytes
        : null;
    final Uint8List? plane2 = buffer.planes.length > 2
        ? buffer.planes[2].bytes
        : null;

    final int uvStride = plane1 != null ? buffer.planes[1].bytesPerRow : 0;
    final int uvPixelStride = plane1 != null
        ? (buffer.planes[1].bytesPerPixel ?? 1)
        : 1;

    for (int y = 0; y < targetSize; y++) {
      final double normUY = (y + 0.5) / targetSize;

      for (int x = 0; x < targetSize; x++) {
        final double normUX = (x + 0.5) / targetSize;

        // Map normalized upright coordinates (normUX, normUY) back to
        // raw buffer relative crop coordinates (relX, relY) based on rotation.
        double relX, relY;
        switch (normRotation) {
          case 90:
            relX = normUY * newW;
            relY = (1.0 - normUX) * newH;
            break;
          case 180:
            relX = (1.0 - normUX) * newW;
            relY = (1.0 - normUY) * newH;
            break;
          case 270:
            relX = (1.0 - normUY) * newW;
            relY = normUX * newH;
            break;
          case 0:
          default:
            relX = normUX * newW;
            relY = normUY * newH;
            break;
        }

        final double rawX = cropLeft + relX;
        final double rawY = cropTop + relY;

        // Edge Pixel Replication: Clamp coordinates to raw image boundaries
        final int srcX = rawX.round().clamp(0, rawW - 1);
        final int srcY = rawY.round().clamp(0, rawH - 1);

        int r = 0, g = 0, b = 0;

        if (isBgra) {
          final offset = srcY * p0Stride + (srcX << 2);
          if (offset + 2 < plane0.length) {
            b = plane0[offset];
            g = plane0[offset + 1];
            r = plane0[offset + 2];
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

            if (uvPixelStride == 2) {
              if (uvOff + 1 < plane1.length) {
                if (buffer.format == LivenessImageFormat.nv21) {
                  vVal = plane1[uvOff];
                  uVal = plane1[uvOff + 1];
                } else {
                  uVal = plane1[uvOff];
                  vVal = plane1[uvOff + 1];
                }
              } else if (uvOff < plane1.length) {
                uVal = plane1[uvOff];
                vVal = uvOff < plane2.length ? plane2[uvOff] : 128;
              }
            } else {
              if (uvOff < plane1.length && uvOff < plane2.length) {
                if (buffer.format == LivenessImageFormat.nv21) {
                  vVal = plane1[uvOff];
                  uVal = plane2[uvOff];
                } else {
                  uVal = plane1[uvOff];
                  vVal = plane2[uvOff];
                }
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

        // Store normalized 0.0 - 1.0 Float32 values
        final spatialIdx = y * targetSize + x;
        final c0Norm = (isBgr ? b : r) / 255.0;
        final c1Norm = g / 255.0;
        final c2Norm = (isBgr ? r : b) / 255.0;

        if (useNchw) {
          tensor[spatialIdx] = c0Norm;
          tensor[hw + spatialIdx] = c1Norm;
          tensor[2 * hw + spatialIdx] = c2Norm;
        } else {
          final idx = spatialIdx * 3;
          tensor[idx] = c0Norm;
          tensor[idx + 1] = c1Norm;
          tensor[idx + 2] = c2Norm;
        }
      }
    }

    _applyAdaptiveContrastStretch(
      tensor,
      targetSize: targetSize,
      useNchw: useNchw,
      enableContrastStretch: enableContrastStretch,
    );

    return tensor;
  }

  /// Preprocesses a raw RGBA 32-bit pixel byte buffer into a Float32 tensor.
  ///
  /// Stores normalized RGB pixel values in `[0.0, 1.0]` range.
  /// Out-of-boundary pixels use **Edge Pixel Replication** (`BORDER_REPLICATE` coordinate clamping).
  static Float32List preprocessRgbaBytesToTensor(
    Uint8List rgbaBytes,
    int rawW,
    int rawH, {
    FaceBoundingBox? boundingBox,
    double expansionFactor = defaultExpansionFactor,
    int targetSize = defaultModelSize,
    bool useNchw = true,
    bool isBgr = false,
    bool enableContrastStretch = false,
  }) {
    final faceBbox =
        boundingBox ??
        FaceBoundingBox(
          x: 0,
          y: 0,
          width: rawW.toDouble(),
          height: rawH.toDouble(),
        );

    final boxW = math.max(1.0, faceBbox.width);
    final boxH = math.max(1.0, faceBbox.height);
    final scale = math.min(
      math.min((rawH - 1) / boxH, (rawW - 1) / boxW),
      expansionFactor,
    );
    final newW = boxW * scale;
    final newH = boxH * scale;

    final cropLeft = faceBbox.centerX - newW / 2.0;
    final cropTop = faceBbox.centerY - newH / 2.0;

    final tensor = Float32List(1 * targetSize * targetSize * 3);
    final hw = targetSize * targetSize;

    for (int y = 0; y < targetSize; y++) {
      final double normUY = (y + 0.5) / targetSize;
      final double rawY = cropTop + normUY * newH;

      for (int x = 0; x < targetSize; x++) {
        final double normUX = (x + 0.5) / targetSize;
        final double rawX = cropLeft + normUX * newW;

        // Edge Pixel Replication: Clamp coordinates to raw image boundaries
        final int srcX = rawX.round().clamp(0, rawW - 1);
        final int srcY = rawY.round().clamp(0, rawH - 1);

        double r = 0.0, g = 0.0, b = 0.0;

        final offset = (srcY * rawW + srcX) * 4;
        if (offset + 2 < rgbaBytes.length) {
          r = rgbaBytes[offset].toDouble();
          g = rgbaBytes[offset + 1].toDouble();
          b = rgbaBytes[offset + 2].toDouble();
        }

        final spatialIdx = y * targetSize + x;
        final c0Norm = (isBgr ? b : r) / 255.0;
        final c1Norm = g / 255.0;
        final c2Norm = (isBgr ? r : b) / 255.0;

        if (useNchw) {
          tensor[spatialIdx] = c0Norm;
          tensor[hw + spatialIdx] = c1Norm;
          tensor[2 * hw + spatialIdx] = c2Norm;
        } else {
          final idx = spatialIdx * 3;
          tensor[idx] = c0Norm;
          tensor[idx + 1] = c1Norm;
          tensor[idx + 2] = c2Norm;
        }
      }
    }

    _applyAdaptiveContrastStretch(
      tensor,
      targetSize: targetSize,
      useNchw: useNchw,
      enableContrastStretch: enableContrastStretch,
    );

    return tensor;
  }

  /// Luma-Preserving Contrast Stretch to prevent RGB sensor noise amplification.
  static void _applyAdaptiveContrastStretch(
    Float32List tensor, {
    required int targetSize,
    bool useNchw = true,
    bool enableContrastStretch = false,
  }) {
    if (!enableContrastStretch || tensor.isEmpty) return;

    final hw = targetSize * targetSize;
    if (hw == 0) return;

    double minY = 1.0;
    double maxY = 0.0;
    double sumY = 0.0;

    // 1. Calculate relative luma Y = 0.299*R + 0.587*G + 0.114*B and find min/max Y
    for (int i = 0; i < hw; i++) {
      double r, g, b;
      if (useNchw) {
        r = tensor[i];
        g = tensor[hw + i];
        b = tensor[2 * hw + i];
      } else {
        final idx = i * 3;
        r = tensor[idx];
        g = tensor[idx + 1];
        b = tensor[idx + 2];
      }

      final yVal = 0.299 * r + 0.587 * g + 0.114 * b;
      if (yVal < minY) minY = yVal;
      if (yVal > maxY) maxY = yVal;
      sumY += yVal;
    }

    final meanY = sumY / hw;

    // 2. If crop is too dark (< 0.35 mean luma), stretch luma while preserving color ratios
    if (meanY < 0.35 && (maxY - minY) > 0.05) {
      final lumaRange = maxY - minY;

      for (int i = 0; i < hw; i++) {
        double r, g, b;
        if (useNchw) {
          r = tensor[i];
          g = tensor[hw + i];
          b = tensor[2 * hw + i];
        } else {
          final idx = i * 3;
          r = tensor[idx];
          g = tensor[idx + 1];
          b = tensor[idx + 2];
        }

        final yVal = 0.299 * r + 0.587 * g + 0.114 * b;
        final newY = (yVal - minY) / lumaRange;
        final ratio = newY / (yVal + 1e-7);

        final newR = (r * ratio).clamp(0.0, 1.0);
        final newG = (g * ratio).clamp(0.0, 1.0);
        final newB = (b * ratio).clamp(0.0, 1.0);

        if (useNchw) {
          tensor[i] = newR;
          tensor[hw + i] = newG;
          tensor[2 * hw + i] = newB;
        } else {
          final idx = i * 3;
          tensor[idx] = newR;
          tensor[idx + 1] = newG;
          tensor[idx + 2] = newB;
        }
      }
    }
  }

  /// Saves a Float32List tensor (128x128) as a PNG/PPM image file on disk for visual debugging.
  ///
  /// Supports NCHW (`isNchw: true`) and NHWC (`isNchw: false`) layouts.
  /// Default save path: `/tmp/liveness_tensor_debug.png`.
  static Future<File> saveTensorToDisk(
    Float32List tensor, {
    bool isNchw = true,
    String? filePath,
    int targetSize = defaultModelSize,
  }) async {
    final path = filePath ?? '/tmp/liveness_tensor_debug.png';
    final file = File(path);
    await file.parent.create(recursive: true);

    final hw = targetSize * targetSize;
    final rgbaBytes = Uint8List(hw * 4);

    for (int y = 0; y < targetSize; y++) {
      for (int x = 0; x < targetSize; x++) {
        final spatialIdx = y * targetSize + x;
        final outIdx = spatialIdx * 4;

        double rVal, gVal, bVal;

        if (isNchw) {
          rVal = tensor[spatialIdx];
          gVal = tensor[hw + spatialIdx];
          bVal = tensor[2 * hw + spatialIdx];
        } else {
          final inIdx = spatialIdx * 3;
          rVal = tensor[inIdx];
          gVal = tensor[inIdx + 1];
          bVal = tensor[inIdx + 2];
        }

        // Scale [0.0, 1.0] to [0, 255] if normalized
        final rPixel = rVal <= 1.0 ? (rVal * 255.0) : rVal;
        final gPixel = gVal <= 1.0 ? (gVal * 255.0) : gVal;
        final bPixel = bVal <= 1.0 ? (bVal * 255.0) : bVal;

        rgbaBytes[outIdx] = rPixel.round().clamp(0, 255);
        rgbaBytes[outIdx + 1] = gPixel.round().clamp(0, 255);
        rgbaBytes[outIdx + 2] = bPixel.round().clamp(0, 255);
        rgbaBytes[outIdx + 3] = 255;
      }
    }

    if (path.endsWith('.ppm')) {
      final header = 'P6\n$targetSize $targetSize\n255\n';
      final rgbBytes = Uint8List(hw * 3);
      for (int i = 0; i < hw; i++) {
        rgbBytes[i * 3] = rgbaBytes[i * 4];
        rgbBytes[i * 3 + 1] = rgbaBytes[i * 4 + 1];
        rgbBytes[i * 3 + 2] = rgbaBytes[i * 4 + 2];
      }
      final builder = BytesBuilder();
      builder.add(header.codeUnits);
      builder.add(rgbBytes);
      await file.writeAsBytes(builder.toBytes());
    } else {
      final descriptor = ui.ImageDescriptor.raw(
        await ui.ImmutableBuffer.fromUint8List(rgbaBytes),
        width: targetSize,
        height: targetSize,
        pixelFormat: ui.PixelFormat.rgba8888,
      );
      final codec = await descriptor.instantiateCodec();
      final frameInfo = await codec.getNextFrame();
      final pngByteData = await frameInfo.image.toByteData(
        format: ui.ImageByteFormat.png,
      );
      if (pngByteData != null) {
        await file.writeAsBytes(pngByteData.buffer.asUint8List());
      }
      frameInfo.image.dispose();
      codec.dispose();
      descriptor.dispose();
    }

    return file;
  }
}
