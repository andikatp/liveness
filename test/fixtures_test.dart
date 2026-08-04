// ignore_for_file: avoid_print

import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:passive_liveness/passive_liveness.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late PassiveLivenessDetector detector;
  final fixturesDir = Directory('test/fixtures');

  List<double> mockLogits = [3.5, 0.5];

  setUpAll(() async {
    const channel = MethodChannel('com.andikatp.passiveLiveness');
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
      if (methodCall.method == 'initModel') {
        return {
          'inputShape': [1, 3, 128, 128],
          'isNchw': true,
          'targetSize': 128,
        };
      }
      if (methodCall.method == 'runInference') {
        return mockLogits;
      }
      if (methodCall.method == 'closeModel') {
        return null;
      }
      return null;
    });

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

    final files = fixturesDir.listSync().whereType<File>().toList()
      ..sort((a, b) => a.path.compareTo(b.path));
    print('\n======================================================');
    print('DIAGNOSING FIXTURE SPOOF IMAGES (${files.length} files found)');
    print('======================================================\n');

    final summaryLines = <String>[];
    summaryLines.add(
      'FILENAME            | EXPECTED | IS_REAL | STATUS            | CHROM_VAR | DISPERSAL | SPECULAR% | LBP_RATIO | HOG_DOM',
    );
    summaryLines.add(
      '--------------------+----------+---------+-------------------+-----------+-----------+-----------+-----------+---------',
    );

    for (final file in files) {
      final lowerPath = file.path.toLowerCase();
      if (!lowerPath.endsWith('.jpg') &&
          !lowerPath.endsWith('.jpeg') &&
          !lowerPath.endsWith('.png')) {
        continue;
      }

      final fileName = file.path.split('/').last;
      detector.resetEma();
      mockLogits = [3.5, 0.5]; // Test heuristic override when neural model predicts REAL
      final bytes = await file.readAsBytes();

      FaceBoundingBox? bbox;
      if (fileName == 'download-real.jpg') {
        bbox = const FaceBoundingBox(x: 400, y: 400, width: 2200, height: 2600);
      } else if (fileName.toLowerCase().contains('real6')) {
        bbox = const FaceBoundingBox(x: 468, y: 1144, width: 800, height: 800);
      } else if (fileName.toLowerCase() == 'spoof4.jpeg') {
        bbox = const FaceBoundingBox(x: 176, y: 680, width: 800, height: 1000);
      }

      final result = await detector.detectLivenessFromImageBytes(
        bytes,
        boundingBox: bbox,
      );

      final expected = fileName.contains('real') ? 'REAL' : 'SPOOF';
      final isRealStr = result.isReal ? 'TRUE' : 'FALSE';

      final chromVar = result.chrominanceVariance?.toStringAsFixed(1) ?? 'N/A';
      final lbp = result.lbpUniformityScore?.toStringAsFixed(3) ?? 'N/A';
      final hog = result.hogGridDominance?.toStringAsFixed(3) ?? 'N/A';

      summaryLines.add(
        '${fileName.padRight(19)} | ${expected.padRight(8)} | BBox: ${isRealStr.padRight(5)} | Status: ${result.status.name.padRight(17)} | Chrom: ${chromVar.padRight(6)} | LBP: ${lbp.padRight(5)} | HOG: ${hog.padRight(5)}',
      );

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
          reason: 'Expected $fileName to be classified as SPOOF (heuristics must override neural real logit)',
        );
      }
    }

    print('\n${summaryLines.join('\n')}\n');
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
