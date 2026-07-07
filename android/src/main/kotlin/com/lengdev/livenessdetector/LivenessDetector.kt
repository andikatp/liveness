package com.lengdev.livenessdetector

import android.content.res.AssetManager
import android.util.Log
import androidx.annotation.Keep
import org.json.JSONArray
import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * Data class representing a detected face bounding box
 */
@Keep
data class FaceBox(
    val left: Int,
    val top: Int,
    val right: Int,
    val bottom: Int,
    var confidence: Float
)

/**
 * Model configuration matching the C++ ModelConfig structure
 */
@Keep
data class ModelConfig(
    var scale: Float = 0F,
    var shift_x: Float = 0F,
    var shift_y: Float = 0F,
    var height: Int = 0,
    var width: Int = 0,
    var name: String = "",
    var org_resize: Boolean = false
)

/**
 * Kotlin wrapper for the native LivenessDetector
 * Created by Ly Houleng
 */
class LivenessDetector {

    companion object {
        const val TAG = "LivenessDetector"

        init {
            try {
                System.loadLibrary("liveness_detector")
                Log.d(TAG, "Native library 'liveness_detector' loaded successfully")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load native library 'liveness_detector'", e)
                e.printStackTrace()
                throw RuntimeException(
                    "Failed to load liveness_detector native library. " + "Make sure the .so files are included in your APK.", e
                )
            }
        }
    }

    @Keep
    private var nativeHandler: Long = 0

    init {
        nativeHandler = allocate()
        if (nativeHandler == 0L) {
            throw RuntimeException("Failed to create native LivenessDetector instance")
        }
        Log.d(TAG, "Native instance created: $nativeHandler")
    }

    /**
     * Load models from Android assets
     * Reads configuration from assets/live/config.json
     * @param assetManager Android AssetManager
     * @return 0 on success, negative value on error
     */
    fun loadModel(assetManager: AssetManager): Int {
        if (nativeHandler == 0L) {
            Log.e(TAG, "LivenessDetector is destroyed or not initialized")
            return -1
        }

        val configs = parseConfig(assetManager)

        if (configs.isEmpty()) {
            Log.e(TAG, "Failed to parse model config")
            return -1
        }

        Log.d(TAG, "Loaded ${configs.size} model configurations")
        return nativeLoadModel(assetManager, configs)
    }

    /**
     * Detect liveness from YUV camera data
     * @param yuv YUV420 image data from camera
     * @param previewWidth Camera preview width
     * @param previewHeight Camera preview height
     * @param orientation Camera orientation (0, 90, 180, 270)
     * @param faceBox Detected face bounding box
     * @return Liveness score (0.0 = fake, 1.0 = real), or -1.0f on invalid input/error
     */
    fun detect(
        yuv: ByteArray,
        previewWidth: Int,
        previewHeight: Int,
        orientation: Int,
        faceBox: FaceBox
    ): Float {
        if (nativeHandler == 0L) {
            Log.e(TAG, "LivenessDetector is destroyed or not initialized")
            return -1.0f
        }

        if (previewWidth <= 0 || previewHeight <= 0) {
            Log.e(TAG, "Invalid preview dimensions: ${previewWidth}x$previewHeight")
            return -1.0f
        }

        val expectedSize = previewWidth * previewHeight * 3 / 2
        if (yuv.size < expectedSize) {
            Log.e(TAG, "Invalid YUV data size. Expected at least: $expectedSize, got: ${yuv.size}")
            return -1.0f
        }

        var left = faceBox.left
        var top = faceBox.top
        var right = faceBox.right
        var bottom = faceBox.bottom

        // If orientation is 5..8 (e.g. 270 deg rotation where width and height swap in C++),
        // C++ expects the input box to be in the UNROTATED preview space [0..previewWidth, 0..previewHeight].
        // If the box passed in is already in the ROTATED space (e.g. bottom > previewHeight),
        // we un-rotate it before passing to nativeDetectYuv so C++ doesn't rotate it a second time out-of-bounds.
        if (orientation in 5..8 && previewWidth > previewHeight) {
            if (bottom > previewHeight || top >= previewHeight) {
                if (orientation == 8) { // Rotate 270 CW (90 CCW)
                    val origLeft = left
                    val origTop = top
                    val origRight = right
                    val origBottom = bottom
                    left = previewWidth - origBottom
                    top = origLeft
                    right = previewWidth - origTop
                    bottom = origBottom
                } else if (orientation == 6) { // Rotate 90 CW
                    val origLeft = left
                    val origTop = top
                    val origRight = right
                    val origBottom = bottom
                    left = origTop
                    top = previewHeight - origRight
                    right = origBottom
                    bottom = previewHeight - origLeft
                }
            }
        }

        // Safety check: ensure box coordinates are ordered correctly
        if (right <= left || bottom <= top) {
            Log.w(TAG, "Invalid face box dimensions: left=$left, top=$top, right=$right, bottom=$bottom")
            return -1.0f
        }

        // Clamp coordinates to image boundaries [0, previewWidth] and [0, previewHeight]
        // to prevent native C++/OpenCV copy_cut_border out-of-bounds segfaults.
        left = left.coerceIn(0, previewWidth - 1)
        top = top.coerceIn(0, previewHeight - 1)
        right = right.coerceIn(left + 1, previewWidth)
        bottom = bottom.coerceIn(top + 1, previewHeight)

        return nativeDetectYuv(
            yuv,
            previewWidth,
            previewHeight,
            orientation,
            left,
            top,
            right,
            bottom
        )
    }


    /**
     * Release native resources
     * Should be called when the detector is no longer needed
     */
    fun destroy() {
        if (nativeHandler != 0L) {
            deallocate()
            Log.d(TAG, "Native instance destroyed")
            nativeHandler = 0
        }
    }

    /**
     * Parse model configuration from assets/live/config.json
     * Expected JSON format:
     * [
     *   {
     *     "name": "model1",
     *     "width": 256,
     *     "height": 256,
     *     "scale": 1.0,
     *     "shift_x": 0.0,
     *     "shift_y": 0.0,
     *     "org_resize": false
     *   }
     * ]
     */
    private fun parseConfig(assetManager: AssetManager): List<ModelConfig> {
        return try {
            val inputStream = assetManager.open("live/config.json")
            val br = BufferedReader(InputStreamReader(inputStream))
            val jsonContent = br.use { it.readText() }

            val jsonArray = JSONArray(jsonContent)
            val configList = mutableListOf<ModelConfig>()

            for (i in 0 until jsonArray.length()) {
                val jsonObject = jsonArray.getJSONObject(i)
                val config = ModelConfig().apply {
                    name = jsonObject.optString("name", "")
                    width = jsonObject.optInt("width", 256)
                    height = jsonObject.optInt("height", 256)
                    scale = jsonObject.optDouble("scale", 1.0).toFloat()
                    shift_x = jsonObject.optDouble("shift_x", 0.0).toFloat()
                    shift_y = jsonObject.optDouble("shift_y", 0.0).toFloat()
                    org_resize = jsonObject.optBoolean("org_resize", false)
                }
                configList.add(config)
                Log.d(TAG, "Loaded config: ${config.name} (${config.width}x${config.height})")
            }

            configList
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing config.json", e)
            emptyList()
        }
    }

    protected fun finalize() {
        destroy()
    }

    ///////////////////////////////////// Native Methods ////////////////////////////////////

    @Keep
    private external fun allocate(): Long

    @Keep
    private external fun deallocate()

    @Keep
    private external fun nativeLoadModel(
        assetManager: AssetManager,
        configs: List<ModelConfig>
    ): Int

    @Keep
    private external fun nativeDetectYuv(
        yuv: ByteArray,
        previewWidth: Int,
        previewHeight: Int,
        orientation: Int,
        left: Int,
        top: Int,
        right: Int,
        bottom: Int
    ): Float
}
