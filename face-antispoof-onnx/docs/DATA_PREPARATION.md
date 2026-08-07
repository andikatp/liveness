# Data Preparation

Preprocessing details for `scripts/prepare_data.py`.

---

## The Pipeline

For each image:

1. Read the bounding box from `image_BB.txt`
2. Expand the box by 1.5× (configurable)
3. Make it square (use the longer side)
4. Pad if the crop goes outside image bounds
5. Resize to 128×128
6. Save


---

## Why Expand the Bounding Box?

Face detectors give tight boxes. Just the face, sometimes cutting off forehead or chin. Anti-spoofing needs context:

- **Skin texture around face** - real skin vs printed paper
- **Hair and ears** - often show printing artifacts
- **Face-background boundary** - edges of printed photos or phone bezels

Default expansion is **1.5x** (50% larger on each side).

![Bbox](../assets/docs/bbox.png)

---

## Why Square Crops?

128x128 square input. Stretching rectangles distorts proportions, and distortion patterns differ between real/spoof. Don't want the model learning "stretched = spoof."

Approach:
1. Take longer side of bbox
2. Make square crop centered on face
3. Resize to 128x128

---

## The Padding Problem

Sometimes expanded crop goes outside image bounds (face near edge, large expansion, tight original image).

Options:
1. **Skip image** - loses data
2. **Black padding** - harsh artificial edges
3. **Reflected pixels** - natural extension

Using option 3: `cv2.BORDER_REFLECT_101`.

### Why BORDER_REFLECT_101?

| Mode | Result |
|:-----|:-------|
| `BORDER_REFLECT` | `abcdef` -> `fedcba|abcdef|fedcba` |
| `BORDER_REFLECT_101` | `abcdef` -> `gfedcb|abcdef|edcbag` |
| `BORDER_REPLICATE` | `abcdef` -> `aaaaaa|abcdef|ffffff` |

`BORDER_REFLECT_101` = smoothest. No edge pixel duplication, no visible seam.

- Black padding = edges model might wrongly learn as "spoof"
- Replicate = visible color bands
- Reflect 101 = natural extension

![Padding comparison](../assets/docs/padding_comparison.png)

---

## Resizing: LANCZOS vs AREA

```python
interp = cv2.INTER_LANCZOS4 if crop_size < target_size else cv2.INTER_AREA
```

### Upscaling (small crop -> 128x128)

Small face, scaling up: **LANCZOS4**.

High-quality interpolation using windowed sinc function. Sharp results, no blocky artifacts. Small faces = far from camera, so preserve texture.

![Upscale comparison](../assets/docs/upscale_comparison.png)

### Downscaling (large crop -> 128x128)

Large face, scaling down: **AREA**.

Averages pixels that map to each output pixel. Proper downsampling that considers all source pixels. Bilinear/bicubic can miss details or create aliasing. AREA = true average, keeps texture.

![Downscale comparison](../assets/docs/downscale_comparison.png)

### The Quick Version

| Situation | Method | Why |
|:----------|:-------|:----|
| Upscaling | LANCZOS4 | Sharp edges, preserves detail |
| Downscaling | AREA | Proper averaging, no aliasing |

---

## What Gets Saved

After processing:

```
{crop_dir}/
+-- train/
|   +-- {same structure as original}
+-- test/
|   +-- {same structure as original}
+-- metas/
    +-- labels/
        +-- train_label.json
        +-- test_label.json
```

Original folder structure preserved. Label JSONs copied to `metas/labels/`.

---

## TL;DR

| Decision | Choice | Reason |
|:---------|:-------|:-------|
| Bbox expansion | 1.5× | Include context (hair, ears, skin texture) |
| Crop shape | Square | No distortion, consistent input |
| Padding | REFLECT_101 | Natural extension, no artificial edges |
| Upscale | LANCZOS4 | Sharp, preserves detail |
| Downscale | AREA | Proper averaging, no aliasing |


