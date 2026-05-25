#import "LivenessDetector.h"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// ncnn headers
#if __has_include(<ncnn/net.h>)
    #include <ncnn/net.h>
    #define HAS_NCNN 1
#elif __has_include(<ncnn/ncnn/net.h>)
    #include <ncnn/ncnn/net.h>
    #define HAS_NCNN 1
#elif __has_include("ncnn/net.h")
    #include "ncnn/net.h"
    #define HAS_NCNN 1
#endif

struct ModelConfig {
  float scale;
  float shift_x;
  float shift_y;
  int height;
  int width;
  std::string name;
  bool org_resize;
};

class LivenessDetectorImpl {
public:
#ifdef HAS_NCNN
  std::vector<ncnn::Net *> nets;
#else
  std::vector<void *> nets;
#endif
  std::vector<ModelConfig> configs;
  bool is_loaded = false;

  ~LivenessDetectorImpl() {
#ifdef HAS_NCNN
    for (auto net : nets) {
      delete net;
    }
#endif
  }
};

@implementation LivenessDetector {
  LivenessDetectorImpl *impl;
}

- (instancetype)init {
  self = [super init];
  if (self) {
    impl = new LivenessDetectorImpl();
  }
  return self;
}

- (int)loadModel:(NSString *)modelPath configPath:(NSString *)configPath {
  if (impl->is_loaded)
    return 0;

  // Scale 2.7 and 4.0 are standard for MiniFASNet models
  ModelConfig cfg1 = {2.7f, 0.0f, 0.0f, 80, 80, "model_1", false};
  ModelConfig cfg2 = {4.0f, 0.0f, 0.0f, 80, 80, "model_2", false};

  impl->configs.push_back(cfg1);
  impl->configs.push_back(cfg2);

#ifdef HAS_NCNN
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
      return -1;
    }
    impl->nets.push_back(net);
  }
#endif

  impl->is_loaded = true;
  return 0;
}

- (float)detectLiveness:(NSData *)yuvData
                  width:(int)width
                 height:(int)height
            orientation:(int)orientation
                   left:(int)left
                    top:(int)top
                  right:(int)right
                 bottom:(int)bottom {
  if (!impl->is_loaded)
    return 0.0f;

#ifdef HAS_NCNN
  // 1. Create Mat from pixels (iOS usually provides BGRA)
  ncnn::Mat img =
      ncnn::Mat::from_pixels((const unsigned char *)[yuvData bytes],
                             ncnn::Mat::PIXEL_BGRA2RGB, width, height);

  float total_score = 0.0f;

  for (size_t i = 0; i < impl->nets.size(); i++) {
    ncnn::Net *net = impl->nets[i];
    const auto &cfg = impl->configs[i];

    // 2. Crop Face area with scale
    int box_width = right - left;
    int box_height = bottom - top;

    float scale = std::min(height / (float)box_height,
                           std::min(width / (float)box_width, cfg.scale));

    float w = box_width * scale;
    float h = box_height * scale;
    float x = left - (w - box_width) / 2.0f;
    float y = top - (h - box_height) / 2.0f;

    // Clamp bounds
    x = std::max(0.0f, x);
    y = std::max(0.0f, y);
    w = std::min((float)width - x, w);
    h = std::min((float)height - y, h);

    ncnn::Mat face;
    ncnn::copy_cut_border(img, face, (int)y, (int)(height - y - h),
                          (int)x, (int)(width - x - w));

    // 3. Resize to model input (80x80)
    ncnn::Mat in;
    ncnn::resize_bilinear(face, in, cfg.width, cfg.height);

    // 4. Run Inference
    ncnn::Extractor ex = net->create_extractor();
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("softmax", out);

    // out[1] is the "Real" score for these models
    if (out.w >= 2) {
      total_score += out[1];
    } else {
      total_score += out[0];
    }
  }

  return total_score / impl->nets.size();
#else
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
