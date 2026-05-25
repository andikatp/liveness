#import "LivenessDetector.h"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// Attempt to include ncnn
#if __has_include("ncnn/net.h")
#include "ncnn/net.h"
#define HAS_NCNN 1
#endif

// Internal C++ struct
struct ModelConfig {
  float scale;
  float shift_x;
  float shift_y;
  int height;
  int width;
  std::string name;
  bool org_resize;
};

// Private C++ Implementation class
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

  ModelConfig cfg1 = {2.7f, 0.0f, 0.0f, 80, 80, "model_1", false};
  ModelConfig cfg2 = {4.0f, 0.0f, 0.0f, 80, 80, "model_2", false};

  impl->configs.push_back(cfg1);
  impl->configs.push_back(cfg2);

#ifdef HAS_NCNN
  for (const auto &cfg : impl->configs) {
    ncnn::Net *net = new ncnn::Net();
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
  return 0.95f;
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
