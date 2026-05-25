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
      
      // Get the correct path for the plugin asset
      let assetKey = LivenessDetectorPlugin.registrar?.lookupKey(forAsset: "android/src/main/assets/live/") ?? "android/src/main/assets/live/"
      let assetPath = Bundle.main.path(forResource: assetKey, ofType: nil) ?? ""
      
      let status = detector?.loadModel(assetPath, configPath: assetPath + "/config.json") ?? -1
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
