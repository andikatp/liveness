# Limitations & Technical Notes

Anti-spoofing depends on texture analysis, so input quality matters. Notes from testing:

## 1. Environmental Constraints

* **Lighting matters:** Fourier Transform patterns need decent lighting. Low-light or harsh backlight = noise, which causes misclassification.
* **What works:** Even lighting on the face. Bright windows behind the subject = bad.

## 2. Input & Preprocessing Requirements

Trained on a specific pipeline, so inputs need to match.

* **1.5x padding:** Tight crops (just eyes/forehead) lose context. The padding gives enough "head space" to see 3D structure vs flat screens/paper.
* **Resolution:** Resizes to 128x128, but source face should be 64x64 minimum. Upscaling tiny blurry faces loses the spoofing artifacts (screen pixels, print dots).

## 3. Pose & Occlusion

* **Angles:** Best with frontal views (+/-30 deg yaw/pitch). Profile views drop accuracy.
* **Obstructions:** Masks, hands over face, thick glasses with reflections mess with texture extraction.

## 4. Known Edge Cases

* **Attack types:** Good at printed photos and screens. Not trained on 3D silicone masks or prosthetics.
* **Motion blur:** Fast movement smears textures. For video, use temporal filtering (require 3-5 consecutive "Real" frames).

## 5. Security Tuning

The default threshold is balanced for general use (FPR < 2%). Security is a trade-off:

* **High-security:** If you can't afford any spoofs getting through, you can bump the threshold (e.g., 0.5 → 0.8). This will also increase false rejects for real users.
* **High-convenience:** If you want a smoother experience and can tolerate minor risks, lower thresholds should work fine.

## Implementation Example

The 1.5x padding is handled as follows when cropping faces:

```python
import cv2
import numpy as np

def crop_face_with_padding(image, bbox, padding_factor=1.5):
    """
    Crop face with proper padding for anti-spoofing model.
    
    Args:
        image: Input image (numpy array)
        bbox: Bounding box as (x, y, w, h)
        padding_factor: Padding multiplier (default: 1.5)
    
    Returns:
        Cropped face image
    """
    x, y, w, h = bbox
    
    # Calculate center and expanded dimensions
    center_x = x + w / 2
    center_y = y + h / 2
    max_dim = max(w, h)
    new_size = int(max_dim * padding_factor)
    
    # Calculate new bounding box
    x1 = int(center_x - new_size / 2)
    y1 = int(center_y - new_size / 2)
    x2 = x1 + new_size
    y2 = y1 + new_size
    
    # Clamp to image boundaries
    h_img, w_img = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_img, x2)
    y2 = min(h_img, y2)
    
    # Crop and resize to 128x128
    face_crop = image[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, (128, 128), interpolation=cv2.INTER_LANCZOS4)
    
    return face_resized
```

For temporal filtering in video streams:

```python
from collections import deque

class TemporalFilter:
    """Require N consecutive 'real' predictions before accepting."""
    
    def __init__(self, required_frames=3):
        self.required_frames = required_frames
        self.history = deque(maxlen=required_frames)
    
    def update(self, is_real: bool) -> bool:
        """Update filter and return final decision."""
        self.history.append(is_real)
        
        if len(self.history) < self.required_frames:
            return False  # Not enough frames yet
        
        return all(self.history)  # All must be 'real'
```
