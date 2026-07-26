import 'dart:ui';

/// Represents a bounding box for a detected face in an image.
class FaceBoundingBox {
  /// X-coordinate of top-left corner of the face bounding box.
  final double x;

  /// Y-coordinate of top-left corner of the face bounding box.
  final double y;

  /// Width of the face bounding box.
  final double width;

  /// Height of the face bounding box.
  final double height;

  /// Creates a [FaceBoundingBox] with top-left coordinates ([x], [y]) and dimensions ([width], [height]).
  const FaceBoundingBox({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
  });

  /// Factory constructor from Flutter [Rect].
  factory FaceBoundingBox.fromRect(Rect rect) {
    return FaceBoundingBox(
      x: rect.left,
      y: rect.top,
      width: rect.width,
      height: rect.height,
    );
  }

  /// Factory constructor from Left, Top, Right, Bottom coordinates.
  factory FaceBoundingBox.fromLTRB(
    double left,
    double top,
    double right,
    double bottom,
  ) {
    return FaceBoundingBox(
      x: left,
      y: top,
      width: right - left,
      height: bottom - top,
    );
  }

  /// Convert to Flutter [Rect].
  Rect toRect() => Rect.fromLTWH(x, y, width, height);

  /// Center X coordinate.
  double get centerX => x + (width / 2);

  /// Center Y coordinate.
  double get centerY => y + (height / 2);

  /// Right boundary X coordinate.
  double get right => x + width;

  /// Bottom boundary Y coordinate.
  double get bottom => y + height;

  /// Convert bounding box to a JSON map representation.
  Map<String, dynamic> toJson() => {
        'x': x,
        'y': y,
        'width': width,
        'height': height,
      };

  @override
  String toString() =>
      'FaceBoundingBox(x: $x, y: $y, width: $width, height: $height)';
}
