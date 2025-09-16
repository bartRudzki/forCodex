package com.example.grapevineguardian.ml

import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.Closeable
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Executes inference against a TensorFlow Lite object detection model whose path is supplied
 * at runtime. The engine keeps the interpreter cached and reloads it whenever a different
 * model path is provided so the host application can swap models without shipping them as
 * APK assets.
 */
class DetectionEngine(
    private val interpreterOptions: Interpreter.Options = Interpreter.Options()
) : Closeable {

    data class InputNormalization(val mean: Float, val std: Float) {
        init {
            require(std > 0f) { "Standard deviation must be greater than zero" }
        }

        companion object {
            val NONE = InputNormalization(mean = 0f, std = 1f)
            val DEFAULT = InputNormalization(mean = 127.5f, std = 127.5f)
        }
    }

    data class ModelConfig(
        val modelPath: String,
        val labelPath: String? = null,
        val confidenceThreshold: Float = DEFAULT_CONFIDENCE_THRESHOLD,
        val maxResults: Int = DEFAULT_MAX_RESULTS,
        val inputNormalization: InputNormalization = InputNormalization.DEFAULT
    ) {
        init {
            require(confidenceThreshold in 0f..1f) {
                "confidenceThreshold must be within [0, 1]"
            }
            require(maxResults > 0) { "maxResults must be greater than zero" }
        }
    }

    data class Detection(
        val boundingBox: RectF,
        val label: String,
        val confidence: Float
    )

    private data class ModelMetadata(
        val inputWidth: Int,
        val inputHeight: Int,
        val inputDataType: DataType,
        val maxDetections: Int,
        val outputTensorCount: Int
    )

    private data class DetectionOutputs(
        val boxes: Array<Array<FloatArray>>,
        val classes: Array<FloatArray>,
        val scores: Array<FloatArray>,
        val count: FloatArray,
        val hasCountTensor: Boolean,
        val map: MutableMap<Int, Any>
    )

    private val lock = ReentrantLock()

    @Volatile
    private var interpreter: Interpreter? = null

    @Volatile
    private var loadedModelPath: String? = null

    @Volatile
    private var loadedModelLastModified: Long = -1L

    @Volatile
    private var modelMetadata: ModelMetadata? = null

    @Volatile
    private var labels: List<String> = emptyList()

    @Volatile
    private var loadedLabelPath: String? = null

    @Volatile
    private var loadedLabelLastModified: Long = -1L

    @Volatile
    private var activeConfig: ModelConfig? = null

    /**
     * Loads or reloads the TensorFlow Lite model described by [config]. The interpreter is reused
     * when the same model (path, timestamps, and config) is supplied repeatedly.
     */
    @Throws(IOException::class)
    fun loadModel(config: ModelConfig) {
        lock.withLock {
            val modelFile = File(config.modelPath)
            require(modelFile.exists()) { "Model file not found at ${config.modelPath}" }
            require(modelFile.canRead()) { "Model file at ${config.modelPath} is not readable" }

            val lastModified = modelFile.lastModified()
            val cachedInterpreter = interpreter
            val shouldReloadInterpreter = cachedInterpreter == null ||
                loadedModelPath != config.modelPath ||
                loadedModelLastModified != lastModified

            val labelFile = config.labelPath?.let { File(it) }
            val labelLastModified = labelFile
                ?.takeIf { it.exists() && it.canRead() }
                ?.lastModified() ?: -1L
            val shouldReloadLabels = loadedLabelPath != config.labelPath ||
                loadedLabelLastModified != labelLastModified

            if (!shouldReloadInterpreter && !shouldReloadLabels) {
                activeConfig = config
                return
            }

            if (!shouldReloadInterpreter && shouldReloadLabels) {
                labels = loadLabels(config.labelPath)
                loadedLabelPath = config.labelPath
                loadedLabelLastModified = labelLastModified
                activeConfig = config
                return
            }

            closeInterpreterLocked()

            val buffer = loadModelFile(modelFile)
            val newInterpreter = Interpreter(buffer, interpreterOptions)
            interpreter = newInterpreter
            loadedModelPath = config.modelPath
            loadedModelLastModified = lastModified
            activeConfig = config

            modelMetadata = extractMetadata(newInterpreter)
            labels = loadLabels(config.labelPath)
            loadedLabelPath = config.labelPath
            loadedLabelLastModified = labelLastModified
        }
    }

    /** Returns `true` when the interpreter has been initialized. */
    fun isModelLoaded(): Boolean = interpreter != null

    /**
     * Runs inference for models that accept a single input/output pair.
     */
    fun runInference(input: Any, output: Any) {
        lock.withLock {
            val interpreter = requireInterpreterLocked()
            interpreter.run(input, output)
        }
    }

    /**
     * Runs inference for models that expose multiple inputs/outputs.
     */
    fun runInference(inputs: Array<Any>, outputs: MutableMap<Int, Any>) {
        lock.withLock {
            val interpreter = requireInterpreterLocked()
            interpreter.runForMultipleInputsOutputs(inputs, outputs)
        }
    }

    /**
     * Executes object detection for the supplied [bitmap] using the currently loaded model and
     * returns the top results subject to the active configuration.
     */
    fun detect(bitmap: Bitmap): List<Detection> {
        lock.withLock {
            val interpreter = requireInterpreterLocked()
            val metadata = requireNotNull(modelMetadata) {
                "Model metadata unavailable. Ensure loadModel() succeeds before detect()."
            }
            val config = requireNotNull(activeConfig) {
                "Model configuration unavailable. Ensure loadModel() succeeds before detect()."
            }

            val inputBuffer = preprocessBitmap(bitmap, metadata, config.inputNormalization)
            val outputs = createOutputBuffer(metadata)

            interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs.map)

            return parseDetections(
                outputs = outputs,
                originalWidth = bitmap.width,
                originalHeight = bitmap.height,
                config = config
            )
        }
    }

    override fun close() {
        lock.withLock {
            closeInterpreterLocked()
            loadedModelPath = null
            loadedModelLastModified = -1L
            modelMetadata = null
            labels = emptyList()
            loadedLabelPath = null
            loadedLabelLastModified = -1L
            activeConfig = null
        }
    }

    private fun requireInterpreterLocked(): Interpreter {
        return interpreter
            ?: error("TensorFlow Lite model has not been loaded. Call loadModel() with a valid configuration before running inferences.")
    }

    private fun closeInterpreterLocked() {
        interpreter?.close()
        interpreter = null
    }

    private fun extractMetadata(interpreter: Interpreter): ModelMetadata {
        if (interpreter.inputTensorCount == 0) {
            error("Model has no input tensors")
        }

        val inputTensor = interpreter.getInputTensor(0)
        val inputShape = inputTensor.shape()
        if (inputShape.size < 4) {
            error("Expected 4D input tensor but found shape=${inputShape.contentToString()}")
        }

        val height = inputShape[1]
        val width = inputShape[2]
        val dataType = inputTensor.dataType()
        val maxDetections = calculateMaxDetections(interpreter)

        return ModelMetadata(
            inputWidth = width,
            inputHeight = height,
            inputDataType = dataType,
            maxDetections = maxDetections,
            outputTensorCount = interpreter.outputTensorCount
        )
    }

    private fun calculateMaxDetections(interpreter: Interpreter): Int {
        if (interpreter.outputTensorCount == 0) {
            return DEFAULT_MAX_RESULTS
        }

        val outputTensor = interpreter.getOutputTensor(0)
        val outputShape = outputTensor.shape()
        return when {
            outputShape.size >= 2 -> outputShape[1]
            outputTensor.numElements() > 0 -> outputTensor.numElements()
            else -> DEFAULT_MAX_RESULTS
        }
    }

    private fun preprocessBitmap(
        bitmap: Bitmap,
        metadata: ModelMetadata,
        normalization: InputNormalization
    ): ByteBuffer {
        val width = metadata.inputWidth
        val height = metadata.inputHeight

        val scaledBitmap = if (bitmap.width != width || bitmap.height != height) {
            Bitmap.createScaledBitmap(bitmap, width, height, true)
        } else {
            bitmap
        }

        val bytesPerChannel = when (metadata.inputDataType) {
            DataType.UINT8 -> 1
            DataType.FLOAT32 -> 4
            else -> error("Unsupported input data type: ${metadata.inputDataType}")
        }
        val inputBuffer = ByteBuffer.allocateDirect(width * height * CHANNELS * bytesPerChannel)
        inputBuffer.order(ByteOrder.nativeOrder())

        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = scaledBitmap.getPixel(x, y)
                val r = (pixel shr 16 and 0xFF)
                val g = (pixel shr 8 and 0xFF)
                val b = (pixel and 0xFF)

                when (metadata.inputDataType) {
                    DataType.UINT8 -> {
                        inputBuffer.put(r.toByte())
                        inputBuffer.put(g.toByte())
                        inputBuffer.put(b.toByte())
                    }

                    DataType.FLOAT32 -> {
                        inputBuffer.putFloat((r - normalization.mean) / normalization.std)
                        inputBuffer.putFloat((g - normalization.mean) / normalization.std)
                        inputBuffer.putFloat((b - normalization.mean) / normalization.std)
                    }

                    else -> error("Unsupported input data type: ${metadata.inputDataType}")
                }
            }
        }

        inputBuffer.rewind()
        return inputBuffer
    }

    private fun createOutputBuffer(metadata: ModelMetadata): DetectionOutputs {
        val maxDetections = metadata.maxDetections.coerceAtLeast(DEFAULT_MAX_RESULTS)
        val boxes = Array(1) { Array(maxDetections) { FloatArray(4) } }
        val classes = Array(1) { FloatArray(maxDetections) }
        val scores = Array(1) { FloatArray(maxDetections) }
        val count = FloatArray(1)
        val hasCountTensor = metadata.outputTensorCount > 3
        if (!hasCountTensor) {
            count[0] = maxDetections.toFloat()
        }
        val map = mutableMapOf<Int, Any>()
        map[0] = boxes
        if (metadata.outputTensorCount > 1) {
            map[1] = classes
        }
        if (metadata.outputTensorCount > 2) {
            map[2] = scores
        }
        if (hasCountTensor) {
            map[3] = count
        }
        return DetectionOutputs(boxes, classes, scores, count, hasCountTensor, map)
    }

    private fun parseDetections(
        outputs: DetectionOutputs,
        originalWidth: Int,
        originalHeight: Int,
        config: ModelConfig
    ): List<Detection> {
        val availableDetections = min(
            outputs.boxes[0].size,
            min(outputs.classes[0].size, outputs.scores[0].size)
        )

        val detectionCount = if (outputs.hasCountTensor) {
            outputs.count.firstOrNull()
                ?.roundToInt()
                ?.coerceIn(0, availableDetections)
                ?: availableDetections
        } else {
            availableDetections
        }

        val maxResults = min(config.maxResults, detectionCount)
        if (maxResults == 0) {
            return emptyList()
        }

        val results = mutableListOf<Detection>()
        for (index in 0 until detectionCount) {
            val confidence = outputs.scores[0][index]
            if (confidence < config.confidenceThreshold) {
                continue
            }

            val labelIndex = outputs.classes[0][index].roundToInt()
            val label = labels.getOrElse(labelIndex) { "#$labelIndex" }

            val box = outputs.boxes[0][index]
            if (box.size < 4) {
                continue
            }

            val top = clamp(box[0]) * originalHeight
            val left = clamp(box[1]) * originalWidth
            val bottom = clamp(box[2]) * originalHeight
            val right = clamp(box[3]) * originalWidth

            results += Detection(
                boundingBox = RectF(left, top, right, bottom),
                label = label,
                confidence = confidence
            )
        }

        return results
            .sortedByDescending { it.confidence }
            .take(maxResults)
    }

    private fun clamp(value: Float, minValue: Float = 0f, maxValue: Float = 1f): Float {
        return when {
            value < minValue -> minValue
            value > maxValue -> maxValue
            else -> value
        }
    }

    @Throws(IOException::class)
    private fun loadModelFile(modelFile: File): MappedByteBuffer {
        FileInputStream(modelFile).use { input ->
            val channel: FileChannel = input.channel
            return channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
        }
    }

    private fun loadLabels(labelPath: String?): List<String> {
        if (labelPath.isNullOrBlank()) {
            return emptyList()
        }

        val file = File(labelPath)
        if (!file.exists() || !file.canRead()) {
            return emptyList()
        }

        BufferedReader(InputStreamReader(file.inputStream())).use { reader ->
            return reader.lineSequence()
                .map { it.trim() }
                .filter { it.isNotEmpty() }
                .toList()
        }
    }

    companion object {
        private const val CHANNELS = 3
        private const val DEFAULT_MAX_RESULTS = 10
        private const val DEFAULT_CONFIDENCE_THRESHOLD = 0.5f
    }
}
