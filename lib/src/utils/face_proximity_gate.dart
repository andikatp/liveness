import '../models/face_bounding_box.dart';
import '../models/liveness_result.dart';

/// Result of evaluating face proximity and aspect ratio constraints.
class FaceGateResult {
  /// Whether the face bounding box passes proximity and aspect ratio checks.
  final bool isValid;

  /// Specific failure status if invalid (`tooFar`, `tooClose`, or `invalidAspectRatio`).
  final LivenessStatus status;

  /// Calculated ratio of face area relative to overall camera frame area ($0.0 \dots 1.0$).
  final double faceAreaRatio;

  /// Calculated aspect ratio ($\text{width} / \text{height}$) of the face bounding box.
  final double aspectRatio;

  const FaceGateResult({
    required this.isValid,
    required this.status,
    required this.faceAreaRatio,
    required this.aspectRatio,
  });

  /// Factory constructor for a valid face gate evaluation.
  factory FaceGateResult.valid({
    required double faceAreaRatio,
    required double aspectRatio,
  }) {
    return FaceGateResult(
      isValid: true,
      status: LivenessStatus.real,
      faceAreaRatio: faceAreaRatio,
      aspectRatio: aspectRatio,
    );
  }
}

/// Evaluates face area coverage and bounding box aspect ratios to reject
/// presentation attacks using small printed photos or extreme lens close-ups.
class FaceProximityGate {
  /// Minimum allowed face area ratio relative to frame size (default: 5% / 0.05).
  final double minFaceAreaRatio;

  /// Maximum allowed face area ratio relative to frame size (default: 85% / 0.85).
  final double maxFaceAreaRatio;

  /// Minimum valid face bounding box aspect ratio (width / height, default: 0.50).
  final double minAspectRatio;

  /// Maximum valid face bounding box aspect ratio (width / height, default: 1.25).
  final double maxAspectRatio;

  /// Creates a [FaceProximityGate] with customizable thresholds.
  const FaceProximityGate({
    this.minFaceAreaRatio = 0.05,
    this.maxFaceAreaRatio = 0.85,
    this.minAspectRatio = 0.50,
    this.maxAspectRatio = 1.25,
  });

  /// Evaluates a [boundingBox] against camera frame dimensions ([frameWidth], [frameHeight]).
  FaceGateResult evaluate({
    required FaceBoundingBox boundingBox,
    required int frameWidth,
    required int frameHeight,
    int rotation = 0,
  }) {
    if (frameWidth <= 0 || frameHeight <= 0) {
      return const FaceGateResult(
        isValid: false,
        status: LivenessStatus.tooFar,
        faceAreaRatio: 0.0,
        aspectRatio: 1.0,
      );
    }

    final frameArea = (frameWidth * frameHeight).toDouble();
    final faceArea = (boundingBox.width * boundingBox.height).abs();
    final faceAreaRatio = (faceArea / frameArea).clamp(0.0, 1.0);

    final normRotation = ((rotation % 360) + 360) % 360;
    final isTransposed = normRotation == 90 || normRotation == 270;

    final uprightWidth = isTransposed ? boundingBox.height : boundingBox.width;
    final uprightHeight = isTransposed ? boundingBox.width : boundingBox.height;

    final safeHeight = uprightHeight > 0 ? uprightHeight : 1.0;
    final aspectRatio = uprightWidth / safeHeight;

    if (faceAreaRatio < minFaceAreaRatio) {
      return FaceGateResult(
        isValid: false,
        status: LivenessStatus.tooFar,
        faceAreaRatio: faceAreaRatio,
        aspectRatio: aspectRatio,
      );
    }

    if (faceAreaRatio > maxFaceAreaRatio) {
      return FaceGateResult(
        isValid: false,
        status: LivenessStatus.tooClose,
        faceAreaRatio: faceAreaRatio,
        aspectRatio: aspectRatio,
      );
    }

    if (aspectRatio < minAspectRatio || aspectRatio > maxAspectRatio) {
      return FaceGateResult(
        isValid: false,
        status: LivenessStatus.invalidAspectRatio,
        faceAreaRatio: faceAreaRatio,
        aspectRatio: aspectRatio,
      );
    }

    return FaceGateResult.valid(
      faceAreaRatio: faceAreaRatio,
      aspectRatio: aspectRatio,
    );
  }
}
