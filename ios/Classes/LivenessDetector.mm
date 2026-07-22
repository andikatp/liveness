#import "LivenessDetector.h"
#include <TargetConditionals.h>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#if TARGET_OS_SIMULATOR
#undef HAS_NCNN
#define HAS_NCNN 0
#else
#ifndef HAS_NCNN
#define HAS_NCNN 1
#endif
#endif

#if HAS_NCNN
#include <ncnn/net.h>
#endif

struct ModelConfig {
  float scale;
  int height;
  int width;
  std::string name;
};

class LivenessDetectorImpl {
public:
#if HAS_NCNN
  std::vector<ncnn::Net *> nets;
#else
  std::vector<void *> nets;
#endif
  std::vector<ModelConfig> configs;
  bool is_loaded = false;

  ~LivenessDetectorImpl() {
#if HAS_NCNN
    for (auto net : nets) {
      delete net;
    }
#endif
  }
};

@implementation LivenessDetector {
  LivenessDetectorImpl *impl;
  NSMutableString *_debugLog;
}

@synthesize lastDebugLog = _lastDebugLog;

- (instancetype)init {
  self = [super init];
  if (self) {
    impl = new LivenessDetectorImpl();
    _lastDebugLog = @"";
  }
  return self;
}

- (int)loadModel:(NSString *)modelPath configPath:(NSString *)configPath {
  if (impl->is_loaded)
    return 0;

  if (impl->configs.empty()) {
    impl->configs.push_back({2.7f, 80, 80, "model_1"});
    impl->configs.push_back({4.0f, 80, 80, "model_2"});
  }

#if HAS_NCNN
  for (const auto &cfg : impl->configs) {
    ncnn::Net *net = new ncnn::Net();
    net->opt.use_vulkan_compute = false;
    net->opt.use_fp16_storage = true;

    NSString *paramPath =
        [NSString stringWithFormat:@"%@/%s.param", modelPath, cfg.name.c_str()];
    NSString *binPath =
        [NSString stringWithFormat:@"%@/%s.bin", modelPath, cfg.name.c_str()];

    if (net->load_param([paramPath UTF8String]) != 0 ||
        net->load_model([binPath UTF8String]) != 0) {
      delete net;
      for (auto n : impl->nets)
        delete n;
      impl->nets.clear();
      return -1;
    }
    impl->nets.push_back(net);
  }
#endif

  impl->is_loaded = true;
  return 0;
}

- (float)detectLiveness:(NSData *)bgraData
                  width:(int)width
                 height:(int)height
            orientation:(int)orientation
                   left:(int)left
                    top:(int)top
                  right:(int)right
                 bottom:(int)bottom {
  // Reset debug log for this call
  _debugLog = [NSMutableString string];
  _lastDebugLog = @"";

  if (!impl->is_loaded) {
    _lastDebugLog = @"[LivenessDetector] model not loaded";
    return -1.0f;
  }

#if HAS_NCNN
  if (!bgraData) {
    _lastDebugLog = @"[LivenessDetector] bgraData is nil";
    return -1.0f;
  }

  // Debug: Print raw box from dart
  [_debugLog appendFormat:
      @"RAW_BOX: L=%d T=%d R=%d B=%d imgW=%d imgH=%d orient=%d dataLen=%d",
      left, top, right, bottom, width, height, orientation,
      (int)[bgraData length]];

  // If the box is already in the ROTATED space (e.g. its Y coordinates exceed
  // the unrotated height), we must un-rotate it back to the UNROTATED space
  // before clamping and passing through the rest of the pipeline.
  if (orientation > 4 && orientation <= 8 && width > height) {
    if (bottom > height || top >= height) {
      int origLeft = left, origTop = top, origRight = right,
          origBottom = bottom;
      if (orientation == 6) { // Rotate 90 CW (un-rotate is 90 CCW)
        left = origTop;
        top = height - origRight;
        right = origBottom;
        bottom = height - origLeft;
      } else if (orientation == 8) { // Rotate 270 CW (un-rotate is 90 CW)
        left = width - origBottom;
        top = origLeft;
        right = width - origTop;
        bottom = origRight;
      }
      [_debugLog appendFormat:@" | UN_ROTATED: L=%d T=%d R=%d B=%d", left,
                              top, right, bottom];
    }
  }

  // --- Clamp coordinates to image boundaries ---
  int preClampL = left, preClampT = top, preClampR = right, preClampB = bottom;
  left = std::max(0, left);
  top = std::max(0, top);
  right = std::min(width, right);
  bottom = std::min(height, bottom);

  if (preClampL != left || preClampT != top || preClampR != right ||
      preClampB != bottom) {
    [_debugLog appendFormat:@" | CLAMPED: L=%d T=%d R=%d B=%d", left, top,
                            right, bottom];
  }

  if (right <= left || bottom <= top) {
    [_debugLog appendString:@" | INVALID_BOX_AFTER_CLAMP"];
    _lastDebugLog = [_debugLog copy];
    return -1.0f;
  }

  int expectedSize = width * height * 4;
  if ((int)[bgraData length] < expectedSize &&
      (int)[bgraData length] < width * 4) {
    [_debugLog appendFormat:@" | DATA_TOO_SMALL: expected=%d got=%d",
                            expectedSize, (int)[bgraData length]];
    _lastDebugLog = [_debugLog copy];
    return -1.0f;
  }

  // --- Strip row-stride padding from BGRA data ---
  const unsigned char *srcPixels = (const unsigned char *)[bgraData bytes];
  const unsigned char *pixels = srcPixels;
  std::vector<unsigned char> strippedBuf;

  if ((int)[bgraData length] > expectedSize) {
    int stride = (int)[bgraData length] / height;
    [_debugLog appendFormat:@" | STRIDE_STRIP: stride=%d expected=%d", stride,
                            width * 4];
    strippedBuf.resize(expectedSize);
    for (int row = 0; row < height; row++) {
      memcpy(strippedBuf.data() + row * width * 4, srcPixels + row * stride,
             width * 4);
    }
    pixels = strippedBuf.data();
  }

  ncnn::Mat img;
  if (orientation > 1 && orientation <= 8) {
    int outw = (orientation >= 5) ? height : width;
    int outh = (orientation >= 5) ? width : height;

    std::vector<unsigned char> rotated_bgra(outw * outh * 4);
    int ncnn_orientation = orientation;

    ncnn::kanna_rotate_c4(pixels, width, height, rotated_bgra.data(), outw,
                          outh, ncnn_orientation);

    img = ncnn::Mat::from_pixels(rotated_bgra.data(), ncnn::Mat::PIXEL_BGRA2BGR,
                                 outw, outh);

    // Rotate bounding box to match the new rotated image dimensions
    int new_left = left, new_top = top, new_right = right, new_bottom = bottom;

    if (ncnn_orientation == 2) { // Mirror horizontal
      new_left = width - right;
      new_top = top;
      new_right = width - left;
      new_bottom = bottom;
    } else if (ncnn_orientation == 3) { // Rotate 180
      new_left = width - right;
      new_top = height - bottom;
      new_right = width - left;
      new_bottom = height - top;
    } else if (ncnn_orientation == 4) { // Mirror vertical
      new_left = left;
      new_top = height - bottom;
      new_right = right;
      new_bottom = height - top;
    } else if (ncnn_orientation == 5) { // Transpose
      new_left = top;
      new_top = left;
      new_right = bottom;
      new_bottom = right;
    } else if (ncnn_orientation == 6) { // Rotate 90 CW
      new_left = height - bottom;
      new_top = left;
      new_right = height - top;
      new_bottom = right;
    } else if (ncnn_orientation == 7) { // Transverse
      new_left = height - bottom;
      new_top = width - right;
      new_right = height - top;
      new_bottom = width - left;
    } else if (ncnn_orientation == 8) { // Rotate 270 CW (90 CCW)
      new_left = top;
      new_top = width - right;
      new_right = bottom;
      new_bottom = width - left;
    }

    left = std::max(0, new_left);
    top = std::max(0, new_top);
    right = std::min(outw, new_right);
    bottom = std::min(outh, new_bottom);

    [_debugLog appendFormat:@" | ROTATED_IMG: %dx%d BOX_AFTER_ROT: L=%d T=%d "
                            @"R=%d B=%d",
                            outw, outh, left, top, right, bottom];

    if (right <= left || bottom <= top) {
      [_debugLog appendString:@" | INVALID_BOX_AFTER_ROT"];
      _lastDebugLog = [_debugLog copy];
      return -1.0f;
    }

    width = outw;
    height = outh;
  } else {
    img = ncnn::Mat::from_pixels(pixels, ncnn::Mat::PIXEL_BGRA2BGR, width,
                                 height);
  }

  float total_score = 0.0f;
  int valid_models = 0;

  for (size_t i = 0; i < impl->nets.size(); i++) {
    ncnn::Net *net = impl->nets[i];
    const auto &cfg = impl->configs[i];

    int box_width = right - left;
    int box_height = bottom - top;

    int side = std::max(box_width, box_height);

    float scale = std::min(height / (float)side,
                           std::min(width / (float)side, cfg.scale));

    float w = side * scale;
    float h = side * scale;

    float cx = left + box_width / 2.0f;
    float cy = top + box_height / 2.0f;

    float x = cx - w / 2.0f;
    float y = cy - h / 2.0f;

    x = std::max(0.0f, x);
    y = std::max(0.0f, y);
    w = std::min((float)width - x, w);
    h = std::min((float)height - y, h);

    [_debugLog
        appendFormat:
            @" | M%zu: crop(x=%.0f y=%.0f w=%.0f h=%.0f) scale=%.2f",
            i, x, y, w, h, scale];

    ncnn::Mat face;
    ncnn::copy_cut_border(img, face, (int)y, (int)(height - y - h), (int)x,
                          (int)(width - x - w));

    ncnn::Mat in;
    ncnn::resize_bilinear(face, in, cfg.width, cfg.height);

    ncnn::Extractor ex = net->create_extractor();
    if (ex.input("data", in) != 0) {
      [_debugLog appendFormat:@" M%zu:INPUT_FAIL", i];
      continue;
    }

    ncnn::Mat out;
    if (ex.extract("softmax", out) != 0 || out.empty()) {
      [_debugLog appendFormat:@" M%zu:EXTRACT_FAIL", i];
      continue;
    }

    // Log softmax outputs
    [_debugLog
        appendFormat:@" softmax[%.4f,%.4f](w=%d)", out.w > 0 ? out[0] : -1.0f,
                     out.w > 1 ? out[1] : -1.0f, out.w];

    if (out.w >= 2) {
      total_score += out[1];
    } else {
      total_score += out[0];
    }
    valid_models++;
  }

  if (valid_models == 0) {
    [_debugLog appendString:@" | NO_VALID_MODELS"];
    _lastDebugLog = [_debugLog copy];
    return -1.0f;
  }

  float finalScore = total_score / valid_models;
  [_debugLog appendFormat:@" | FINAL=%.4f (%d models)", finalScore,
                          valid_models];
  _lastDebugLog = [_debugLog copy];

  return finalScore;
#else
  _lastDebugLog = @"[LivenessDetector] SIMULATOR — ncnn disabled";
  return 0.0f;
#endif
}

- (void)destroy {
  if (impl) {
    delete impl;
    impl = new LivenessDetectorImpl();
  }
}

- (void)dealloc {
  if (impl) {
    delete impl;
    impl = nullptr;
  }
#if !__has_feature(objc_arc)
  [super dealloc];
#endif
}

@end