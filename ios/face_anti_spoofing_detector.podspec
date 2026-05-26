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
  s.static_framework = true
  s.dependency 'Flutter'
  s.platform = :ios, '12.0'

  current_dir = File.dirname(__FILE__)

  # Conditionally vendor frameworks only for device
  # For simulator: no vendored frameworks, ncnn is stubbed via __has_include
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',

    # Device: link ncnn + openmp via vendored path
    'OTHER_LDFLAGS[sdk=iphoneos*]' =>
      "$(inherited)" \
      " -force_load \"#{current_dir}/ncnn.xcframework/ios-arm64/ncnn.framework/Versions/A/ncnn\"" \
      " -force_load \"#{current_dir}/openmp.xcframework/ios-arm64/openmp.framework/Versions/A/openmp\"",

    'FRAMEWORK_SEARCH_PATHS[sdk=iphoneos*]' =>
      "$(inherited)" \
      " \"#{current_dir}/ncnn.xcframework/ios-arm64\"" \
      " \"#{current_dir}/openmp.xcframework/ios-arm64\"",

    'HEADER_SEARCH_PATHS[sdk=iphoneos*]' =>
      "$(inherited)" \
      " \"#{current_dir}/ncnn.xcframework/ios-arm64/ncnn.framework/Headers\"" \
      " \"#{current_dir}/openmp.xcframework/ios-arm64/openmp.framework/Headers\"",

    # Simulator: nothing — __has_include returns false, HAS_NCNN undefined,
    # LivenessDetector.mm compiles the stub returning 0.0f
    'OTHER_LDFLAGS[sdk=iphonesimulator*]'          => '$(inherited)',
    'FRAMEWORK_SEARCH_PATHS[sdk=iphonesimulator*]' => '$(inherited)',
    'HEADER_SEARCH_PATHS[sdk=iphonesimulator*]'    => '$(inherited)',
  }

  s.swift_version = '5.0'
  s.resource_bundles = {
    'face_anti_spoofing_detector_privacy' => ['Resources/PrivacyInfo.xcprivacy'],
    'face_anti_spoofing_detector_assets'  => ['Assets/**/*']
  }
end