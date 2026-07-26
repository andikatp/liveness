import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:passive_liveness/passive_liveness.dart';

void main() {
  group('LivenessResult tests', () {
    test('Calculates real face classification correctly when realLogit > spoofLogit', () {
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
    });

    test('Calculates spoof face classification correctly when realLogit < spoofLogit', () {
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

    test('toRawBufferSpace transforms 270 deg rotated bounding box correctly', () {
      // Rotated 1080x1920 space: center at (540, 960), size (400, 600)
      const rotBbox = FaceBoundingBox(x: 340, y: 660, width: 400, height: 600);
      final rawBbox = rotBbox.toRawBufferSpace(1920, 1080, 270);

      // Raw 1920x1080 space: rawCx = 1920 - 960 = 960, rawCy = 540
      expect(rawBbox.centerX, equals(960.0));
      expect(rawBbox.centerY, equals(540.0));
      expect(rawBbox.width, equals(600.0));
      expect(rawBbox.height, equals(400.0));
    });
  });

  group('ImagePreprocessor tests', () {
    test('reflect101 algorithm mirrors indices correctly', () {
      expect(ImagePreprocessor.reflect101(0, 100), equals(0));
      expect(ImagePreprocessor.reflect101(99, 100), equals(99));
      expect(ImagePreprocessor.reflect101(-1, 100), equals(1));
      expect(ImagePreprocessor.reflect101(-2, 100), equals(2));
      expect(ImagePreprocessor.reflect101(100, 100), equals(98));
      expect(ImagePreprocessor.reflect101(101, 100), equals(97));

      expect(ImagePreprocessor.reflect101Double(0.0, 100), equals(0.0));
      expect(ImagePreprocessor.reflect101Double(-1.5, 100), equals(1.5));
      expect(ImagePreprocessor.reflect101Double(101.5, 100), equals(96.5));
    });

    test('preprocessBufferToTensor converts BGRA buffer to float32 NHWC tensor', () {
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
          LivenessImagePlane(bytes: bytes, bytesPerRow: 60 * 4, bytesPerPixel: 4),
        ],
      );

      final tensor = ImagePreprocessor.preprocessBufferToTensor(
        buffer,
        targetSize: 128,
      );

      // Shape: 1 x 128 x 128 x 3 = 49152 values
      expect(tensor.length, equals(49152));

      // Pixel 0 (R=255/255=1.0, G=128/255≈0.5019, B=0/255=0.0)
      expect(tensor[0], closeTo(1.0, 1e-3));
      expect(tensor[1], closeTo(128 / 255.0, 1e-3));
      expect(tensor[2], equals(0.0));
    });

    test('preprocessBufferToTensor crops bounding box directly from raw buffer coordinates', () {
      // 100x100 BGRA buffer initialized to zeros
      final bytes = Uint8List(100 * 100 * 4);

      // Fill a 20x20 box at (x: 40..60, y: 40..60) with red (BGRA = 0, 0, 255, 255)
      for (int y = 40; y < 60; y++) {
        for (int x = 40; x < 60; x++) {
          final offset = (y * 100 + x) * 4;
          bytes[offset] = 0;       // B
          bytes[offset + 1] = 0;   // G
          bytes[offset + 2] = 255; // R
          bytes[offset + 3] = 255; // A
        }
      }

      final buffer = LivenessImageBuffer(
        width: 100,
        height: 100,
        format: LivenessImageFormat.bgra8888,
        planes: [
          LivenessImagePlane(bytes: bytes, bytesPerRow: 100 * 4, bytesPerPixel: 4),
        ],
      );

      // Pass bounding box covering the red square (x:40, y:40, w:20, h:20) with expansionFactor=1.0
      const bbox = FaceBoundingBox(x: 40, y: 40, width: 20, height: 20);

      final tensor = ImagePreprocessor.preprocessBufferToTensor(
        buffer,
        boundingBox: bbox,
        expansionFactor: 1.0,
        enableShadowLift: false,
        targetSize: 10,
      );

      // Center pixel of target (5, 5) should correspond to (x:45, y:45) in raw buffer, which is RED (R=1.0)
      final centerIdx = (5 * 10 + 5) * 3;
      expect(tensor[centerIdx], closeTo(1.0, 1e-3)); // R
      expect(tensor[centerIdx + 1], equals(0.0));     // G
      expect(tensor[centerIdx + 2], equals(0.0));     // B
    });

    test('preprocessRgbaBytesToTensor processes raw RGBA pixel buffer correctly', () {
      final rgbaBytes = Uint8List(50 * 50 * 4);
      for (int i = 0; i < rgbaBytes.length; i += 4) {
        rgbaBytes[i] = 255;   // R
        rgbaBytes[i + 1] = 100; // G
        rgbaBytes[i + 2] = 50;  // B
        rgbaBytes[i + 3] = 255; // A
      }

      final tensor = ImagePreprocessor.preprocessRgbaBytesToTensor(
        rgbaBytes,
        50,
        50,
        targetSize: 10,
      );

      expect(tensor.length, equals(300));
      expect(tensor[0], closeTo(1.0, 1e-3));
      expect(tensor[1], closeTo(100 / 255.0, 1e-3));
      expect(tensor[2], closeTo(50 / 255.0, 1e-3));
    });
  });
}
