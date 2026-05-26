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
  s.source_files     = 'Classes/**/*'
  s.static_framework = true

  # This is what actually copies and links the frameworks
  s.vendored_frameworks = 'ncnn.xcframework', 'openmp.xcframework'

  s.dependency 'Flutter'
  s.platform = :ios, '12.0'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    # Exclude arm64 simulator (no simulator slice in xcframework)
    # x86_64 simulator still works, arm64 sim is excluded
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'arm64 i386',
    # Do NOT define HAS_NCNN here — LivenessDetector.mm handles it
    # via __has_include so simulator gets the stub automatically
  }

  s.swift_version = '5.0'
  s.resource_bundles = {
    'face_anti_spoofing_detector_privacy' => ['Resources/PrivacyInfo.xcprivacy'],
    'face_anti_spoofing_detector_assets'  => ['Assets/**/*']
  }
end