import 'dart:math' as math;
import 'dart:typed_data';

/// Result of evaluating Local Binary Pattern (LBP) and Histogram of Oriented Gradients (HOG) micro-textures.
class TextureAnalysisResult {
  /// Ratio of non-uniform LBP patterns ($0.0 \dots 1.0$). High values ($>0.35$) indicate paper/inkjet print noise.
  final double lbpNonUniformRatio;

  /// Peak gradient orientation dominance ratio ($0.0 \dots 1.0$). High values ($>0.40$) indicate digital screen grid alignment.
  final double hogPeakDominance;

  /// Whether the micro-texture indicates an inkjet paper print spoof.
  final bool isPrintSpoof;

  /// Whether the micro-texture indicates a digital screen grid replay spoof.
  final bool isScreenGridSpoof;

  const TextureAnalysisResult({
    required this.lbpNonUniformRatio,
    required this.hogPeakDominance,
    required this.isPrintSpoof,
    required this.isScreenGridSpoof,
  });

  /// True if any micro-texture spoof pattern was detected.
  bool get isTextureSpoof => isPrintSpoof || isScreenGridSpoof;
}

/// Lightweight micro-texture analysis engine using Local Binary Pattern (LBP) and Histogram of Oriented Gradients (HOG).
class LbpHogAnalyzer {
  /// Threshold for non-uniform LBP pattern ratio indicating print spoofing (default: 0.38).
  final double lbpPrintThreshold;

  /// Threshold for HOG orientation peak dominance indicating screen grid replay (default: 0.42).
  final double hogScreenThreshold;

  const LbpHogAnalyzer({
    this.lbpPrintThreshold = 0.38,
    this.hogScreenThreshold = 0.42,
  });

  /// Pre-computed lookup table for 8-bit uniform LBP pattern check.
  static final Uint8List _uniformLbpLookup = _buildUniformLbpLookup();

  static Uint8List _buildUniformLbpLookup() {
    final table = Uint8List(256);
    for (int code = 0; code < 256; code++) {
      int transitions = 0;
      for (int i = 0; i < 8; i++) {
        final currentBit = (code >> i) & 1;
        final nextBit = (code >> ((i + 1) % 8)) & 1;
        if (currentBit != nextBit) {
          transitions++;
        }
      }
      // Uniform patterns have at most 2 bit transitions
      table[code] = transitions <= 2 ? 1 : 0;
    }
    return table;
  }

  /// Analyzes a grayscale pixel crop ([width] x [height] Uint8List).
  TextureAnalysisResult analyzeGrayscaleCrop(
    Uint8List grayBytes,
    int width,
    int height,
  ) {
    if (width < 8 || height < 8 || grayBytes.length < width * height) {
      return const TextureAnalysisResult(
        lbpNonUniformRatio: 0.0,
        hogPeakDominance: 0.0,
        isPrintSpoof: false,
        isScreenGridSpoof: false,
      );
    }

    int nonUniformLbpCount = 0;
    int totalLbpEvaluated = 0;

    // 8-neighbor offsets: (dx, dy)
    final dx = const [-1, 0, 1, 1, 1, 0, -1, -1];
    final dy = const [-1, -1, -1, 0, 1, 1, 1, 0];

    // 1. Calculate LBP histogram
    for (int y = 1; y < height - 1; y++) {
      final yOffset = y * width;
      for (int x = 1; x < width - 1; x++) {
        final centerPixel = grayBytes[yOffset + x];
        int lbpCode = 0;

        for (int i = 0; i < 8; i++) {
          final neighborPixel = grayBytes[(y + dy[i]) * width + (x + dx[i])];
          if (neighborPixel >= centerPixel) {
            lbpCode |= (1 << i);
          }
        }

        if (_uniformLbpLookup[lbpCode] == 0) {
          nonUniformLbpCount++;
        }
        totalLbpEvaluated++;
      }
    }

    final lbpRatio = totalLbpEvaluated > 0
        ? nonUniformLbpCount / totalLbpEvaluated
        : 0.0;

    // 2. Calculate HOG orientation distribution (9 bins from 0 to 180 degrees)
    final hogBins = Float64List(9);
    double totalGradientMagnitude = 0.0;

    for (int y = 1; y < height - 1; y++) {
      final yOffset = y * width;
      for (int x = 1; x < width - 1; x++) {
        final gx = (grayBytes[yOffset + x + 1] - grayBytes[yOffset + x - 1])
            .toDouble();
        final gy =
            (grayBytes[(y + 1) * width + x] - grayBytes[(y - 1) * width + x])
                .toDouble();

        final mag = math.sqrt(gx * gx + gy * gy);
        if (mag < 10.0) continue; // Ignore weak background gradients

        // Calculate angle in degrees [0, 180)
        double angleRad = math.atan2(gy, gx);
        if (angleRad < 0) angleRad += math.pi;
        double angleDeg = angleRad * (180.0 / math.pi);
        if (angleDeg >= 180.0) angleDeg = 179.9;

        final binIdx = (angleDeg / 20.0).floor().clamp(0, 8);
        hogBins[binIdx] += mag;
        totalGradientMagnitude += mag;
      }
    }

    double maxBinEnergy = 0.0;
    if (totalGradientMagnitude > 0) {
      for (int b = 0; b < 9; b++) {
        if (hogBins[b] > maxBinEnergy) {
          maxBinEnergy = hogBins[b];
        }
      }
    }

    final hogDominance = totalGradientMagnitude > 0
        ? maxBinEnergy / totalGradientMagnitude
        : 0.0;

    final isPrintSpoof = lbpRatio >= lbpPrintThreshold;
    final isScreenGridSpoof = hogDominance >= hogScreenThreshold;

    return TextureAnalysisResult(
      lbpNonUniformRatio: lbpRatio,
      hogPeakDominance: hogDominance,
      isPrintSpoof: isPrintSpoof,
      isScreenGridSpoof: isScreenGridSpoof,
    );
  }
}
