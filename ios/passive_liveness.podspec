Pod::Spec.new do |s|
  s.name             = 'passive_liveness'
  s.version          = '0.0.5'
  s.summary          = 'Passive face liveness detection using native TFLite inference.'
  s.description      = <<-DESC
Ultra-lightweight passive face liveness and anti-spoofing detection plugin using native TensorFlow Lite engine.
                       DESC
  s.homepage         = 'https://github.com/andikatp/passive_liveness'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'andikatp' => 'andikatp@example.com' }
  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*'
  s.dependency 'Flutter'
  s.dependency 'TensorFlowLiteSwift', '~> 2.14'
  s.platform         = :ios, '12.0'

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.swift_version    = '5.0'
end
