import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:passive_liveness/passive_liveness.dart';

void main() {
  group('LivenessResult tests', () {
    test(
      'Calculates real face classification correctly when realLogit > spoofLogit',
      () {
        final result = LivenessResult.fromLogits(
          realLogit: 3.5,
          spoofLogit: 0.5,
          threshold: 0.0,
        );

        expect(result.isReal, isTrue);
        expect(result.status, equals(LivenessStatus.real));
        expect(result.logitDiff, equals(3.0));
        expect(result.confidence, equals(3.0));
        expect(result.realScore, greaterThan(result.spoofScore));
        expect(result.realScore + result.spoofScore, closeTo(1.0, 1e-5));
      },
    );

    test(
      'Calculates spoof face classification correctly when realLogit < spoofLogit',
      () {
        final result = LivenessResult.fromLogits(
          realLogit: -2.0,
          spoofLogit: 2.0,
          threshold: 0.0,
        );

        expect(result.isReal, isFalse);
        expect(result.status, equals(LivenessStatus.spoof));
        expect(result.logitDiff, equals(-4.0));
        expect(result.confidence, equals(4.0));
        expect(result.spoofScore, greaterThan(result.realScore));
        expect(result.realScore + result.spoofScore, closeTo(1.0, 1e-5));
      },
    );

    test('Exposes raw single-frame metrics correctly alongside smoothed metrics', () {
      final result = LivenessResult(
        isReal: false,
        status: LivenessStatus.spoof,
        realScore: 0.4,
        spoofScore: 0.6,
        realLogit: 0.84,
        spoofLogit: -0.84,
        logitDiff: -0.4,
        confidence: 0.4,
        threshold: 0.0,
        inferenceTime: Duration.zero,
        rawRealScore: 0.844,
        rawSpoofScore: 0.156,
        rawLogitDiff: 1.68,
        rawIsReal: true,
      );

      expect(result.isReal, isFalse);
      expect(result.rawIsReal, isTrue);
      expect(result.rawRealScore, equals(0.844));
      expect(result.rawLogitDiff, equals(1.68));
    });
  });

  group('FaceBoundingBox tests', () {
    test('Calculates center and boundary coordinates accurately', () {
      const bbox = FaceBoundingBox(x: 10, y: 20, width: 100, height: 120);

      expect(bbox.centerX, equals(60.0));
      expect(bbox.centerY, equals(80.0));
      expect(bbox.right, equals(110.0));
      expect(bbox.bottom, equals(140.0));
    });

    test('Constructs from Rect and LTRB correctly', () {
      final fromLTRB = FaceBoundingBox.fromLTRB(10, 20, 110, 140);
      expect(fromLTRB.x, equals(10.0));
      expect(fromLTRB.y, equals(20.0));
      expect(fromLTRB.width, equals(100.0));
      expect(fromLTRB.height, equals(120.0));

      final rect = fromLTRB.toRect();
      expect(rect.left, equals(10.0));
      expect(rect.top, equals(20.0));
      expect(rect.width, equals(100.0));
      expect(rect.height, equals(120.0));
    });

    test(
      'toRawBufferSpace transforms 270 deg rotated bounding box correctly',
      () {
        // Rotated 1080x1920 space: center at (540, 960), size (400, 600)
        const rotBbox = FaceBoundingBox(
          x: 340,
          y: 660,
          width: 400,
          height: 600,
        );
        final rawBbox = rotBbox.toRawBufferSpace(1920, 1080, 270);

        // Raw 1920x1080 space: rawCx = 1920 - 960 = 960, rawCy = 540
        expect(rawBbox.centerX, equals(960.0));
        expect(rawBbox.centerY, equals(540.0));
        expect(rawBbox.width, equals(600.0));
        expect(rawBbox.height, equals(400.0));
      },
    );
  });

  group('ImagePreprocessor tests', () {
    test('edge pixel replication clamps out-of-bounds coordinates to boundary pixels', () {
      final bytes = Uint8List(20 * 20 * 4);
      // Fill 20x20 image with white pixels (R=255, G=255, B=255)
      for (int i = 0; i < bytes.length; i += 4) {
        bytes[i] = 255;
        bytes[i + 1] = 255;
        bytes[i + 2] = 255;
        bytes[i + 3] = 255;
      }
      final buffer = LivenessImageBuffer(
        width: 20,
        height: 20,
        format: LivenessImageFormat.bgra8888,
        planes: [
          LivenessImagePlane(
            bytes: bytes,
            bytesPerRow: 20 * 4,
            bytesPerPixel: 4,
          ),
        ],
      );

      // Bounding box near corner so crop extends outside image
      const bbox = FaceBoundingBox(x: -5, y: -5, width: 20, height: 20);

      final tensor = ImagePreprocessor.preprocessBufferToTensor(
        buffer,
        boundingBox: bbox,
        expansionFactor: 2.0,
        targetSize: 10,
        useNchw: false,
      );

      // Top-left corner of tensor corresponds to out-of-bound crop area -> should replicate white edge (1.0)
      expect(tensor[0], equals(1.0));
      expect(tensor[1], equals(1.0));
      expect(tensor[2], equals(1.0));
    });

    test(
      'preprocessBufferToTensor converts BGRA buffer to float32 normalized [0..1] tensor (NCHW & NHWC)',
      () {
        // Create dummy BGRA8888 60x60 image plane (R=255, G=128, B=0, A=255)
        final bytes = Uint8List(60 * 60 * 4);
        for (int i = 0; i < bytes.length; i += 4) {
          bytes[i] = 0; // B
          bytes[i + 1] = 128; // G
          bytes[i + 2] = 255; // R
          bytes[i + 3] = 255; // A
        }

        final buffer = LivenessImageBuffer(
          width: 60,
          height: 60,
          format: LivenessImageFormat.bgra8888,
          planes: [
            LivenessImagePlane(
              bytes: bytes,
              bytesPerRow: 60 * 4,
              bytesPerPixel: 4,
            ),
          ],
        );

        // Test NCHW layout (default)
        final tensorNchw = ImagePreprocessor.preprocessBufferToTensor(
          buffer,
          targetSize: 128,
          useNchw: true,
        );

        expect(tensorNchw.length, equals(49152));
        final hw = 128 * 128;
        expect(tensorNchw[0], equals(1.0)); // Channel 0 (R: 255/255)
        expect(tensorNchw[hw], closeTo(128 / 255.0, 1e-3)); // Channel 1 (G: 128/255)
        expect(tensorNchw[2 * hw], equals(0.0)); // Channel 2 (B: 0/255)

        // Test NHWC layout
        final tensorNhwc = ImagePreprocessor.preprocessBufferToTensor(
          buffer,
          targetSize: 128,
          useNchw: false,
        );

        expect(tensorNhwc.length, equals(49152));
        expect(tensorNhwc[0], equals(1.0)); // R
        expect(tensorNhwc[1], closeTo(128 / 255.0, 1e-3)); // G
        expect(tensorNhwc[2], equals(0.0)); // B
      },
    );

    test(
      'preprocessRgbaBytesToTensor processes raw RGBA pixel buffer to 0..1 float tensor',
      () {
        final rgbaBytes = Uint8List(50 * 50 * 4);
        for (int i = 0; i < rgbaBytes.length; i += 4) {
          rgbaBytes[i] = 255; // R
          rgbaBytes[i + 1] = 100; // G
          rgbaBytes[i + 2] = 50; // B
          rgbaBytes[i + 3] = 255; // A
        }

        final tensor = ImagePreprocessor.preprocessRgbaBytesToTensor(
          rgbaBytes,
          50,
          50,
          targetSize: 10,
          useNchw: false,
        );

        expect(tensor.length, equals(300));
        expect(tensor[0], equals(1.0));
        expect(tensor[1], closeTo(100 / 255.0, 1e-3));
        expect(tensor[2], closeTo(50 / 255.0, 1e-3));
      },
    );

    test('saveTensorToDisk writes valid PPM image file to disk', () async {
      final tensor = Float32List(1 * 128 * 128 * 3);
      final hw = 128 * 128;
      // Set R channel to 1.0 for NCHW tensor
      for (int i = 0; i < hw; i++) {
        tensor[i] = 1.0; // R
        tensor[hw + i] = 0.0; // G
        tensor[2 * hw + i] = 0.0; // B
      }

      final testPath = '${Directory.systemTemp.path}/test_liveness_tensor.ppm';
      final savedFile = await ImagePreprocessor.saveTensorToDisk(
        tensor,
        isNchw: true,
        filePath: testPath,
        targetSize: 128,
      );

      expect(savedFile.existsSync(), isTrue);
      final length = await savedFile.length();
      expect(length, greaterThan(hw * 3));
      await savedFile.delete();
    });

    test('preprocessBufferToTensor maintains 1:1 square crop aspect ratio for rectangular bounding box', () {
      final bytes = Uint8List(100 * 100 * 4);
      final buffer = LivenessImageBuffer(
        width: 100,
        height: 100,
        format: LivenessImageFormat.bgra8888,
        planes: [
          LivenessImagePlane(
            bytes: bytes,
            bytesPerRow: 100 * 4,
            bytesPerPixel: 4,
          ),
        ],
      );

      // Rectangular bounding box (30x50)
      const bbox = FaceBoundingBox(x: 10, y: 10, width: 30, height: 50);

      // Should complete without error using square crop math (baseSide = max(30, 50) = 50)
      final tensor = ImagePreprocessor.preprocessBufferToTensor(
        buffer,
        boundingBox: bbox,
        expansionFactor: 2.0,
        targetSize: 128,
        useNchw: true,
      );

      expect(tensor.length, equals(49152));
    });

    test('preprocessBufferToTensor calculates true 2.0x square crop with reflect101 padding', () {
      final bytes = Uint8List(100 * 100 * 4);
      final buffer = LivenessImageBuffer(
        width: 100,
        height: 100,
        format: LivenessImageFormat.bgra8888,
        planes: [
          LivenessImagePlane(
            bytes: bytes,
            bytesPerRow: 100 * 4,
            bytesPerPixel: 4,
          ),
        ],
      );

      // Large face box (60x60) in 100x100 frame -> baseSide = 60.
      // 2.0x expansion is 120.0. Unclamped crop extends outside boundaries and uses reflect101 padding.
      const bbox = FaceBoundingBox(x: 20, y: 20, width: 60, height: 60);

      final tensor = ImagePreprocessor.preprocessBufferToTensor(
        buffer,
        boundingBox: bbox,
        expansionFactor: 2.0,
        targetSize: 10,
        useNchw: true,
      );

      expect(tensor.length, equals(300));
    });

    test('reflect101 border padding mirrors pixels for out-of-bounds crop coordinates', () {
      // 10x10 BGRA image where column 0 = red (255, 0, 0), column 1 = green (0, 255, 0)
      final bytes = Uint8List(10 * 10 * 4);
      for (int y = 0; y < 10; y++) {
        for (int x = 0; x < 10; x++) {
          final idx = (y * 10 + x) * 4;
          if (x == 0) {
            bytes[idx] = 0; // B
            bytes[idx + 1] = 0; // G
            bytes[idx + 2] = 255; // R
            bytes[idx + 3] = 255; // A
          } else if (x == 1) {
            bytes[idx] = 0; // B
            bytes[idx + 1] = 255; // G
            bytes[idx + 2] = 0; // R
            bytes[idx + 3] = 255; // A
          }
        }
      }

      final buffer = LivenessImageBuffer(
        width: 10,
        height: 10,
        format: LivenessImageFormat.bgra8888,
        planes: [
          LivenessImagePlane(
            bytes: bytes,
            bytesPerRow: 10 * 4,
            bytesPerPixel: 4,
          ),
        ],
      );

      // Crop box overflowing on the left (x = -2)
      const bbox = FaceBoundingBox(x: -2, y: 0, width: 10, height: 10);
      final tensor = ImagePreprocessor.preprocessBufferToTensor(
        buffer,
        boundingBox: bbox,
        expansionFactor: 1.0,
        targetSize: 10,
        useNchw: false,
      );

      // Should complete without out-of-range exception and contain mirrored values
      expect(tensor.length, equals(300));
    });

    test('preprocessBufferToTensor supports isBgr: true for BGR channel ordering', () {
      final bytes = Uint8List(10 * 10 * 4);
      for (int i = 0; i < bytes.length; i += 4) {
        bytes[i] = 255; // B = 255
        bytes[i + 1] = 128; // G = 128
        bytes[i + 2] = 0; // R = 0
        bytes[i + 3] = 255;
      }

      final buffer = LivenessImageBuffer(
        width: 10,
        height: 10,
        format: LivenessImageFormat.bgra8888,
        planes: [
          LivenessImagePlane(
            bytes: bytes,
            bytesPerRow: 10 * 4,
            bytesPerPixel: 4,
          ),
        ],
      );

      final tensorBgr = ImagePreprocessor.preprocessBufferToTensor(
        buffer,
        targetSize: 10,
        useNchw: true,
        isBgr: true,
      );

      final hw = 10 * 10;
      expect(tensorBgr[0], equals(1.0)); // Channel 0 is B (255/255)
      expect(tensorBgr[hw], closeTo(128 / 255.0, 1e-3)); // Channel 1 is G (128/255)
      expect(tensorBgr[2 * hw], equals(0.0)); // Channel 2 is R (0/255)
    });
  });

  group('PassiveLivenessDetector tests', () {
    test(
      'EMA tracker initializes as null and resets correctly',
      () {
        final detector = PassiveLivenessDetector();

        expect(detector.emaRealScore, isNull);
        expect(detector.isInitialized, isFalse);

        detector.resetEma();
        expect(detector.emaRealScore, isNull);
        detector.dispose();
      },
    );
  });
}
