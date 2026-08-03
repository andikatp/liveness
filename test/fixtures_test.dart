// ignore_for_file: avoid_print

import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter_test/flutter_test.dart';
import 'package:passive_liveness/passive_liveness.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late PassiveLivenessDetector detector;
  final fixturesDir = Directory('test/fixtures');

  setUpAll(() async {
    detector = PassiveLivenessDetector();
    await detector.initialize();
  });

  tearDownAll(() {
    detector.dispose();
  });

  test('Diagnose all fixture spoof images in test/fixtures', () async {
    if (!fixturesDir.existsSync()) {
      print('test/fixtures directory does not exist.');
      return;
    }

    final files = fixturesDir.listSync().whereType<File>().toList();
    print('\n======================================================');
    print('DIAGNOSING FIXTURE SPOOF IMAGES (${files.length} files found)');
    print('======================================================\n');

    final highResAnalyzer = const HighResScreenAnalyzer();
    final textureAnalyzer = const LbpHogAnalyzer();
    final colorSpaceAnalyzer = const ColorSpaceAnalyzer();

    for (final file in files) {
      if (!file.path.endsWith('.jpg') &&
          !file.path.endsWith('.jpeg') &&
          !file.path.endsWith('.png')) {
        continue;
      }

      final fileName = file.path.split('/').last;
      print('--- Inspecting File: $fileName ---');
      detector.resetEma();
      final bytes = await file.readAsBytes();

      FaceBoundingBox? bbox;
      if (fileName == 'download-real.jpg') {
        // Full resolution 3024x4032 camera image: frame center face region
        bbox = const FaceBoundingBox(x: 400, y: 400, width: 2200, height: 2600);
      }

      final result = await detector.detectLivenessFromImageBytes(
        bytes,
        boundingBox: bbox,
      );

      print('Neural Model Score:');
      print('  Real Logit: ${result.realLogit.toStringAsFixed(4)}');
      print('  Spoof Logit: ${result.spoofLogit.toStringAsFixed(4)}');
      print('  Logit Diff: ${result.logitDiff.toStringAsFixed(4)}');
      print(
        '  Real Score (prob): ${(result.realScore * 100).toStringAsFixed(2)}%',
      );
      print('  Is Real: ${result.isReal}');
      print('  Status: ${result.status}\n');

      // Assert expected liveness result per fixture
      if (fileName.contains('real')) {
        expect(
          result.isReal,
          isTrue,
          reason: 'Expected $fileName to be classified as REAL',
        );
      } else if (fileName.contains('spoof')) {
        expect(
          result.isReal,
          isFalse,
          reason: 'Expected $fileName to be classified as SPOOF',
        );
      }

      // 2. Decode raw RGBA bytes for high-res crop analysis
      try {
        final codec = await testInstantiateImageCodec(bytes);
        final frameInfo = await codec.getNextFrame();
        final image = frameInfo.image;
        final byteData = await image.toByteData(
          format: ui.ImageByteFormat.rawRgba,
        );

        if (byteData != null) {
          final rgbaBytes = byteData.buffer.asUint8List();
          final w = image.width;
          final h = image.height;

          final buffer = LivenessImageBuffer(
            width: w,
            height: h,
            format: LivenessImageFormat.bgra8888, // RGBA/BGRA byte buffer
            planes: [
              LivenessImagePlane(
                bytes: rgbaBytes,
                bytesPerRow: w * 4,
                bytesPerPixel: 4,
              ),
            ],
          );

          final highResCrop = ImagePreprocessor.extractHighResCrop(
            buffer,
            targetSize: 256,
          );

          final highResResult = highResAnalyzer.analyzeGrayscaleCrop(
            highResCrop,
            256,
            256,
          );

          print('High-Res Screen Metrics:');
          print(
            '  Laplacian Variance: ${highResResult.laplacianVariance.toStringAsFixed(2)}',
          );
          print(
            '  Patch Focus Dispersal: ${highResResult.patchLaplacianDispersal.toStringAsFixed(4)} (Threshold: < 4.0)',
          );
          print(
            '  Specular Highlight Ratio: ${(highResResult.specularHighlightRatio * 100).toStringAsFixed(2)}% (Threshold: > 8.0%)',
          );
          print(
            '  isHighResScreenSpoof: ${highResResult.isHighResScreenSpoof}\n',
          );

          final textureResult = textureAnalyzer.analyzeGrayscaleCrop(
            highResCrop,
            256,
            256,
          );

          print('Micro-Texture Metrics:');
          print(
            '  LBP Non-Uniform Ratio: ${textureResult.lbpNonUniformRatio.toStringAsFixed(4)} (Threshold: >= 0.38)',
          );
          print(
            '  HOG Peak Dominance: ${textureResult.hogPeakDominance.toStringAsFixed(4)} (Threshold: >= 0.42)',
          );
          print('  isPrintSpoof: ${textureResult.isPrintSpoof}');
          print('  isScreenGridSpoof: ${textureResult.isScreenGridSpoof}\n');

          final colorResult = colorSpaceAnalyzer.analyzeBuffer(buffer);
          print('Color Space Metrics:');
          print(
            '  Chrominance Variance: ${colorResult.chrominanceVariance.toStringAsFixed(2)}',
          );
          print('  isScreenReplaySpoof: ${colorResult.isScreenReplaySpoof}\n');

          image.dispose();
          codec.dispose();
        }
      } catch (e) {
        print('Error running detailed analyzer on ${file.path}: $e');
      }

      print('------------------------------------------------------\n');
    }
  });
}

Future<ui.Codec> testInstantiateImageCodec(Uint8List bytes) async {
  final ui.ImmutableBuffer buffer = await ui.ImmutableBuffer.fromUint8List(
    bytes,
  );
  final ui.ImageDescriptor descriptor = await ui.ImageDescriptor.encoded(
    buffer,
  );
  return descriptor.instantiateCodec();
}
