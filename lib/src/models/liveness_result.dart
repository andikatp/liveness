import 'dart:math' as math;

/// Status of passive liveness detection result.
enum LivenessStatus {
  /// Classified as a real face.
  real,

  /// Classified as a spoof attempt.
  spoof,
}

/// Result of passive liveness detection for a face.
class LivenessResult {
  /// Whether the face is classified as real (live person).
  final bool isReal;

  /// Human-readable status classification (`real` or `spoof`).
  final LivenessStatus status;

  /// Softmax probability of being a real face (0.0 to 1.0).
  final double realScore;

  /// Softmax probability of being a spoof face (0.0 to 1.0).
  final double spoofScore;

  /// Raw logit score for real class from ONNX model.
  final double realLogit;

  /// Raw logit score for spoof class from ONNX model.
  final double spoofLogit;

  /// Logit difference (`realLogit - spoofLogit`).
  final double logitDiff;

  /// Absolute confidence metric based on logit difference.
  final double confidence;

  /// Logit threshold used for classification.
  final double threshold;

  /// Time taken to execute preprocessing and model inference.
  final Duration inferenceTime;

  /// Creates a [LivenessResult] containing complete classification metrics.
  const LivenessResult({
    required this.isReal,
    required this.status,
    required this.realScore,
    required this.spoofScore,
    required this.realLogit,
    required this.spoofLogit,
    required this.logitDiff,
    required this.confidence,
    required this.threshold,
    required this.inferenceTime,
  });

  /// Factory constructor for pending/unstable frames.
  factory LivenessResult.pending({
    double threshold = 0.0,
  }) {
    return LivenessResult(
      isReal: false,
      status: LivenessStatus.spoof,
      realScore: 0.5,
      spoofScore: 0.5,
      realLogit: 0.0,
      spoofLogit: 0.0,
      logitDiff: 0.0,
      confidence: 0.0,
      threshold: threshold,
      inferenceTime: Duration.zero,
    );
  }

  /// Factory constructor to calculate probabilities and classification from raw logits.
  factory LivenessResult.fromLogits({
    required double realLogit,
    required double spoofLogit,
    double threshold = 0.0,
    Duration inferenceTime = Duration.zero,
  }) {
    final logitDiff = realLogit - spoofLogit;
    final isReal = logitDiff >= threshold;

    // Numerically stable softmax
    final maxLogit = math.max(realLogit, spoofLogit);
    final expReal = math.exp(realLogit - maxLogit);
    final expSpoof = math.exp(spoofLogit - maxLogit);
    final sumExp = expReal + expSpoof;

    final realScore = expReal / sumExp;
    final spoofScore = expSpoof / sumExp;
    final confidence = logitDiff.abs();

    return LivenessResult(
      isReal: isReal,
      status: isReal ? LivenessStatus.real : LivenessStatus.spoof,
      realScore: realScore,
      spoofScore: spoofScore,
      realLogit: realLogit,
      spoofLogit: spoofLogit,
      logitDiff: logitDiff,
      confidence: confidence,
      threshold: threshold,
      inferenceTime: inferenceTime,
    );
  }

  /// Convert result to a JSON map representation.
  Map<String, dynamic> toJson() => {
        'isReal': isReal,
        'status': status.name,
        'realScore': realScore,
        'spoofScore': spoofScore,
        'realLogit': realLogit,
        'spoofLogit': spoofLogit,
        'logitDiff': logitDiff,
        'confidence': confidence,
        'threshold': threshold,
        'inferenceTimeMs': inferenceTime.inMilliseconds,
      };

  @override
  String toString() {
    return 'LivenessResult(isReal: $isReal, status: ${status.name}, realScore: ${realScore.toStringAsFixed(4)}, logitDiff: ${logitDiff.toStringAsFixed(4)}, inferenceTimeMs: ${inferenceTime.inMilliseconds})';
  }
}

/// Alias for [LivenessResult] to prevent class name collisions in host applications.
typedef PassiveLivenessResult = LivenessResult;

