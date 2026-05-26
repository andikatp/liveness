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

  s.vendored_frameworks = 'ncnn.xcframework', 'openmp.xcframework'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',

    # 🔥 KEY: disable arm64 simulator so it never tries to use ncnn
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'arm64',

    # 🔥 KEY: ONLY link framework on real device
    'OTHER_LDFLAGS[sdk=iphoneos*]' => '$(inherited) -framework ncnn -framework openmp',

    # simulator → no linking at all
    'OTHER_LDFLAGS[sdk=iphonesimulator*]' => '$(inherited)'
  }

  s.swift_version = '5.0'

  s.resource_bundles = {
    'face_anti_spoofing_detector_privacy' => ['Resources/PrivacyInfo.xcprivacy'],
    'face_anti_spoofing_detector_assets'  => ['Assets/**/*']
  }
end