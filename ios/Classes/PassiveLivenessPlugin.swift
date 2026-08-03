import Flutter
import UIKit
import TensorFlowLite

public class PassiveLivenessPlugin: NSObject, FlutterPlugin {
    private var interpreter: Interpreter?
    private var isNativeNchw: Bool = false
    private var targetSize: Int = 128
    private var inputShape: [Int] = []

    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(name: "com.andikatp.passiveLiveness", binaryMessenger: registrar.messenger())
        let instance = PassiveLivenessPlugin()
        registrar.addMethodCallDelegate(instance, channel: channel)
    }

    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "initModel":
            guard let args = call.arguments as? [String: Any],
                  let typedData = args["modelBytes"] as? FlutterStandardTypedData else {
                result(FlutterError(code: "INVALID_ARGUMENT", message: "modelBytes is missing or invalid", details: nil))
                return
            }
            initializeModel(modelData: typedData.data, result: result)

        case "runInference":
            guard let args = call.arguments as? [String: Any] else {
                result(FlutterError(code: "INVALID_ARGUMENT", message: "Arguments must be a Map", details: nil))
                return
            }

            var floatArray: [Float] = []

            if let typedFloatData = args["inputData"] as? FlutterStandardFloat32Data {
                let data = typedFloatData.data
                floatArray = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            } else if let doubleArray = args["inputData"] as? [Double] {
                floatArray = doubleArray.map { Float($0) }
            } else if let array = args["inputData"] as? [Float] {
                floatArray = array
            } else {
                result(FlutterError(code: "INVALID_ARGUMENT", message: "inputData is missing or invalid format", details: nil))
                return
            }

            runInference(inputFloats: floatArray, result: result)

        case "closeModel":
            closeModel()
            result(nil)

        default:
            result(FlutterMethodNotImplemented)
        }
    }

    private func initializeModel(modelData: Data, result: FlutterResult) {
        do {
            var options = InterpreterOptions()
            options.threadCount = 2

            let newInterpreter = try Interpreter(modelData: modelData, options: options)
            try newInterpreter.allocateTensors()

            let inputTensor = try newInterpreter.input(at: 0)
            let shape = inputTensor.shape.dimensions
            inputShape = shape

            if shape.count == 4 {
                if shape[1] == 3 {
                    isNativeNchw = true
                    targetSize = shape[2]
                } else {
                    isNativeNchw = false
                    targetSize = shape[1]
                }
            }

            interpreter = newInterpreter

            let response: [String: Any] = [
                "inputShape": shape,
                "isNchw": isNativeNchw,
                "targetSize": targetSize
            ]
            result(response)
        } catch {
            result(FlutterError(code: "INIT_FAILED", message: "Failed to initialize TFLite Interpreter: \(error.localizedDescription)", details: nil))
        }
    }

    private func runInference(inputFloats: [Float], result: FlutterResult) {
        guard let interpreter = interpreter else {
            result(FlutterError(code: "NOT_INITIALIZED", message: "Interpreter is not initialized", details: nil))
            return
        }

        do {
            let inputData = inputFloats.withUnsafeBufferPointer { Data(buffer: $0) }
            try interpreter.copy(inputData, toInputAt: 0)
            try interpreter.invoke()

            let outputTensor = try interpreter.output(at: 0)
            let outputFloats = outputTensor.data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }

            if outputFloats.count >= 2 {
                let realLogit = Double(outputFloats[0])
                let spoofLogit = Double(outputFloats[1])
                result([realLogit, spoofLogit])
            } else {
                result([0.0, 0.0])
            }
        } catch {
            result(FlutterError(code: "INFERENCE_FAILED", message: "Native inference execution error: \(error.localizedDescription)", details: nil))
        }
    }

    private func closeModel() {
        interpreter = nil
    }
}
