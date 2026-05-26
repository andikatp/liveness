Pod::Spec.new do |s|
  s.name             = 'face_anti_spoofing_detector'
  s.version          = '0.0.4'
  s.summary          = 'A Flutter plugin for passive face liveness detection.'
  s.description      = <<-DESC
Flutter plugin that provides passive liveness detection for facial recognition systems.
  DESC
  s.homepage         = 'https://github.com/horlengg/face_anti_spoofing_detector'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Your Company' => 'email@example.com' }
  s.source           = { :path => '.' }

  s.source_files     = 'Classes/**/*'
  s.dependency 'Flutter'
  s.platform = :ios, '12.0'

  # ❌ REMOVE vendored_frameworks completely
  # s.vendored_frameworks = ...

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',

    # prevent simulator from touching ncnn
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'arm64',

    # header search (needed for compile)
    'HEADER_SEARCH_PATHS' => '$(inherited) "${PODS_TARGET_SRCROOT}/ncnn.xcframework/ios-arm64/ncnn.framework/Headers"',

    # 🔥 link ONLY on real device
    'LIBRARY_SEARCH_PATHS[sdk=iphoneos*]' => '$(inherited) "${PODS_TARGET_SRCROOT}/ncnn.xcframework/ios-arm64"',
    'OTHER_LDFLAGS[sdk=iphoneos*]' => '$(inherited) -framework ncnn'
  }

  s.swift_version = '5.0'
end