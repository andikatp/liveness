import Flutter
import UIKit

public class LivenessDetectorPlugin: NSObject, FlutterPlugin {
  private var detector: LivenessDetector?

  private static var registrar: FlutterPluginRegistrar?

  public static func register(with registrar: FlutterPluginRegistrar) {
    let channel = FlutterMethodChannel(name: "com.leng.dev/liveness_detector", binaryMessenger: registrar.messenger())
    let instance = LivenessDetectorPlugin()
    self.registrar = registrar
    registrar.addMethodCallDelegate(instance, channel: channel)
  }

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    switch call.method {
    case "getPlatformVersion":
      result("iOS " + UIDevice.current.systemVersion)
    case "initialize":
      if detector == nil {
          detector = LivenessDetector()
      }
      
      guard let registrar = LivenessDetectorPlugin.registrar else {
          result(false)
          return
      }

      // Look up asset key from flutter package
      var key = registrar.lookupKey(forAsset: "android/src/main/assets/live/config.json", fromPackage: "face_anti_spoofing_detector")
      var configPath = Bundle.main.path(forResource: key, ofType: nil) ?? ""

      // Fallback if not found
      if configPath.isEmpty {
          key = registrar.lookupKey(forAsset: "android/src/main/assets/live/config.json")
          configPath = Bundle.main.path(forResource: key, ofType: nil) ?? ""
      }

      let assetPath = configPath.isEmpty ? "" : (configPath as NSString).deletingLastPathComponent
      let status = detector?.loadModel(assetPath, configPath: configPath) ?? -1
      result(status == 0)
    case "detect_liveness":
      guard let args = call.arguments as? [String: Any],
            let yuvData = args["yuvBytes"] as? FlutterStandardTypedData,
            let width = args["previewWidth"] as? Int32,
            let height = args["previewHeight"] as? Int32,
            let orientation = args["orientation"] as? Int32,
            let faceBox = args["faceBox"] as? [String: Int32] else {
          result(FlutterError(code: "INVALID_ARGS", message: "Arguments mismatch", details: nil))
          return
      }
      
      let score = detector?.detectLiveness(yuvData.data, 
                                          width: width, 
                                          height: height, 
                                          orientation: orientation, 
                                          left: faceBox["left"] ?? 0, 
                                          top: faceBox["top"] ?? 0, 
                                          right: faceBox["right"] ?? 0, 
                                          bottom: faceBox["bottom"] ?? 0) ?? 0.0
      result(Double(score))
    case "destroy":
      detector?.destroy()
      detector = nil
      result(true)
    default:
      result(FlutterMethodNotImplemented)
    }
  }
}
