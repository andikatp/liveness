package com.andikatp.passiveLiveness

import android.content.Context
import androidx.annotation.NonNull
import com.google.android.gms.tflite.client.TfLiteInitializationOptions
import com.google.android.gms.tflite.gpu.support.TfLiteGpu
import com.google.android.gms.tflite.java.TfLite
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.gpu.GpuDelegateFactory
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** PassiveLivenessPlugin using Google Play Services TFLite */
class PassiveLivenessPlugin : FlutterPlugin, MethodCallHandler {
    private lateinit var channel: MethodChannel
    private lateinit var context: Context
    private var interpreter: InterpreterApi? = null
    private var inputShape: IntArray? = null
    private var isNativeNchw: Boolean = false
    private var targetSize: Int = 128

    override fun onAttachedToEngine(@NonNull flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
        context = flutterPluginBinding.applicationContext
        channel = MethodChannel(flutterPluginBinding.binaryMessenger, "com.andikatp.passiveLiveness")
        channel.setMethodCallHandler(this)
    }

    override fun onMethodCall(@NonNull call: MethodCall, @NonNull result: Result) {
        when (call.method) {
            "initModel" -> {
                val modelBytes = call.argument<ByteArray>("modelBytes")
                if (modelBytes == null) {
                    result.error("INVALID_ARGUMENT", "modelBytes cannot be null", null)
                    return
                }
                initializeModel(modelBytes, result)
            }
            "runInference" -> {
                val rawInput = call.argument<Any>("inputData")
                if (rawInput == null) {
                    result.error("INVALID_ARGUMENT", "inputData cannot be null", null)
                    return
                }

                val floatArray: FloatArray? = when (rawInput) {
                    is ByteArray -> {
                        val floatBuffer = ByteBuffer.wrap(rawInput).order(ByteOrder.nativeOrder()).asFloatBuffer()
                        FloatArray(floatBuffer.remaining()).also { floatBuffer.get(it) }
                    }
                    is FloatArray -> rawInput
                    is DoubleArray -> FloatArray(rawInput.size) { i -> rawInput[i].toFloat() }
                    is List<*> -> FloatArray(rawInput.size) { i -> (rawInput[i] as? Number)?.toFloat() ?: 0f }
                    else -> null
                }
 
                if (floatArray == null) {
                    result.error("INVALID_ARGUMENT", "Unsupported inputData type: ${rawInput.javaClass.name}", null)
                    return
                }

                runInference(floatArray, result)
            }
            "closeModel" -> {
                closeModel()
                result.success(null)
            }
            else -> {
                result.notImplemented()
            }
        }
    }

    private fun initializeModel(modelBytes: ByteArray, result: Result) {
        val initOptions = TfLiteInitializationOptions.builder()
            .setEnableGpuDelegateSupport(true)
            .build()

        TfLite.initialize(context, initOptions).addOnSuccessListener {
            TfLiteGpu.isGpuDelegateAvailable(context).addOnSuccessListener { isGpuAvailable ->
                createInterpreter(modelBytes, isGpuAvailable, result)
            }.addOnFailureListener {
                createInterpreter(modelBytes, false, result)
            }
        }.addOnFailureListener { e ->
            result.error("PLAY_SERVICES_FAILED", "Failed to initialize Google Play Services TFLite: ${e.message}", null)
        }
    }

    private fun createInterpreter(modelBytes: ByteArray, useGpu: Boolean, result: Result) {
        try {
            val buffer = ByteBuffer.allocateDirect(modelBytes.size).apply {
                order(ByteOrder.nativeOrder())
                put(modelBytes)
                rewind()
            }

            val options = InterpreterApi.Options().apply {
                setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY)
                if (useGpu) {
                    addDelegateFactory(GpuDelegateFactory())
                } else {
                    setNumThreads(2)
                }
            }

            val newInterpreter = InterpreterApi.create(buffer, options)
            interpreter = newInterpreter
            val tensor = newInterpreter.getInputTensor(0)
            val shape = tensor.shape()
            inputShape = shape

            if (shape.size == 4) {
                if (shape[1] == 3) {
                    isNativeNchw = true
                    targetSize = shape[2]
                } else {
                    isNativeNchw = false
                    targetSize = shape[1]
                }
            }

            val response = mapOf(
                "inputShape" to shape.toList(),
                "isNchw" to isNativeNchw,
                "targetSize" to targetSize
            )
            result.success(response)
        } catch (e: Exception) {
            if (useGpu) {
                android.util.Log.w("PassiveLivenessPlugin", "GPU Interpreter creation failed: ${e.message}, falling back to CPU", e)
                createInterpreter(modelBytes, false, result)
            } else {
                result.error("INIT_FAILED", "Failed to create TFLite Interpreter: ${e.message}", null)
            }
        }
    }

    private fun runInference(inputData: FloatArray, result: Result) {
        val currentInterpreter = interpreter
        if (currentInterpreter == null) {
            result.error("NOT_INITIALIZED", "Interpreter is not initialized", null)
            return
        }

        try {
            val inputBuffer = ByteBuffer.allocateDirect(inputData.size * 4).apply {
                order(ByteOrder.nativeOrder())
                asFloatBuffer().put(inputData)
                rewind()
            }

            val outputMap = HashMap<Int, Any>()
            val outputFloats = Array(1) { FloatArray(2) }
            outputMap[0] = outputFloats

            currentInterpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)

            val realLogit = outputFloats[0][0].toDouble()
            val spoofLogit = outputFloats[0][1].toDouble()

            result.success(listOf(realLogit, spoofLogit))
        } catch (e: Exception) {
            result.error("INFERENCE_FAILED", "Native inference execution error: ${e.message}", null)
        }
    }

    private fun closeModel() {
        interpreter?.close()
        interpreter = null
    }

    override fun onDetachedFromEngine(@NonNull binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
        closeModel()
    }
}

