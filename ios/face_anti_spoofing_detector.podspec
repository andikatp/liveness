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

  # Only vendor frameworks for real device builds
  # Simulator has no ncnn slice — LivenessDetector.mm stubs it via __has_include
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',

    # Simulator: don't link ncnn, don't search for it
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]'           => 'arm64 i386',
    'OTHER_LDFLAGS[sdk=iphonesimulator*]'            => '$(inherited)',
    'FRAMEWORK_SEARCH_PATHS[sdk=iphonesimulator*]'   => '$(inherited)',
    'HEADER_SEARCH_PATHS[sdk=iphonesimulator*]'      => '$(inherited)',

    # Device: force-link ncnn + openmp
    'OTHER_LDFLAGS[sdk=iphoneos*]' =>
      "$(inherited)" \
      " -force_load \"#{current_dir}/ncnn.xcframework/ios-arm64/ncnn.framework/ncnn\"" \
      " \"#{current_dir}/openmp.xcframework/ios-arm64/openmp.framework/openmp\"",

    'FRAMEWORK_SEARCH_PATHS[sdk=iphoneos*]' =>
      "$(inherited)" \
      " \"#{current_dir}/ncnn.xcframework/ios-arm64\"" \
      " \"#{current_dir}/openmp.xcframework/ios-arm64\"",

    'HEADER_SEARCH_PATHS[sdk=iphoneos*]' =>
      "$(inherited)" \
      " \"#{current_dir}/ncnn.xcframework/ios-arm64/ncnn.framework/Headers\"" \
      " \"#{current_dir}/openmp.xcframework/ios-arm64/openmp.framework/Headers\"",
  }

  s.swift_version = '5.0'
  s.resource_bundles = {
    'face_anti_spoofing_detector_privacy' => ['Resources/PrivacyInfo.xcprivacy'],
    'face_anti_spoofing_detector_assets'  => ['Assets/**/*']
  }
end