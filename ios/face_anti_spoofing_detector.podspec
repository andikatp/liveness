Pod::Spec.new do |s|
  s.name             = 'face_anti_spoofing_detector'
  s.version          = '0.0.4'
  s.summary          = 'A Flutter plugin for passive face liveness detection.'
  s.description      = <<-DESC
Flutter plugin that provides passive liveness detection for facial recognition systems — ensuring that the detected face belongs to a live person rather than a photo, video.
                       DESC
  s.homepage         = 'https://github.com/horlengg/face_anti_spoofing_detector'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Your Company' => 'email@example.com' }
  s.source           = { :path => '.' }
  s.source_files = 'Classes/**/*'
  s.static_framework = true
  s.preserve_paths = 'ncnn.xcframework', 'openmp.xcframework'
  s.dependency 'Flutter'
  s.platform = :ios, '12.0'
  s.pod_target_xcconfig = { 
    'DEFINES_MODULE' => 'YES', 
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
    'OTHER_LDFLAGS[sdk=iphoneos*]' => '$(inherited) -framework "ncnn" -framework "openmp"',
    'FRAMEWORK_SEARCH_PATHS[sdk=iphoneos*]' => '$(inherited) "${PODS_TARGET_SRCROOT}" "${PODS_TARGET_SRCROOT}/ncnn.xcframework/ios-arm64" "${PODS_TARGET_SRCROOT}/openmp.xcframework/ios-arm64"',
    'HEADER_SEARCH_PATHS[sdk=iphoneos*]' => '$(inherited) "${PODS_TARGET_SRCROOT}/ncnn.xcframework/ios-arm64/ncnn.framework/Headers" "${PODS_TARGET_SRCROOT}/openmp.xcframework/ios-arm64/openmp.framework/Headers"',
    'CLANG_WARN_DOCUMENTATION_COMMENTS' => 'NO'
  }
  s.swift_version = '5.0'

  s.resource_bundles = {
    'face_anti_spoofing_detector_privacy' => ['Resources/PrivacyInfo.xcprivacy'],
    'face_anti_spoofing_detector_assets' => ['Assets/**/*']
  }
end
