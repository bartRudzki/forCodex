package com.example.grapevineguardian.ml

import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Thin wrapper around a [Interpreter] that loads a TensorFlow Lite model from a
 * path supplied at runtime rather than relying on a bundled asset.
 */
class DetectionEngine(
    private val interpreterOptions: Interpreter.Options = Interpreter.Options()
) : Closeable {

    private val lock = ReentrantLock()

    @Volatile
    private var interpreter: Interpreter? = null

    @Volatile
    private var loadedModelPath: String? = null

    /**
     * Loads (or reloads) the TensorFlow Lite model from the supplied [modelPath].
     * If the same path is provided consecutively, the existing interpreter is reused.
     */
    @Throws(IOException::class)
    fun loadModel(modelPath: String) {
        lock.withLock {
            if (loadedModelPath == modelPath && interpreter != null) {
                return
            }

            val modelFile = File(modelPath)
            require(modelFile.exists()) { "Model file not found at $modelPath" }
            require(modelFile.canRead()) { "Model file at $modelPath is not readable" }

            closeInterpreterLocked()

            interpreter = Interpreter(loadModelFile(modelFile), interpreterOptions)
            loadedModelPath = modelPath
        }
    }

    /** Returns `true` when the interpreter is ready to execute inferences. */
    fun isModelLoaded(): Boolean = interpreter != null

    /** Runs inference for models that accept a single input / output pair. */
    fun runInference(input: Any, output: Any) {
        lock.withLock {
            val interpreter = requireInterpreterLocked()
            interpreter.run(input, output)
        }
    }

    /** Runs inference for models that expose multiple inputs / outputs. */
    fun runInference(inputs: Array<Any>, outputs: MutableMap<Int, Any>) {
        lock.withLock {
            val interpreter = requireInterpreterLocked()
            interpreter.runForMultipleInputsOutputs(inputs, outputs)
        }
    }

    override fun close() {
        lock.withLock {
            closeInterpreterLocked()
            loadedModelPath = null
        }
    }

    private fun requireInterpreterLocked(): Interpreter {
        return interpreter ?: error("TensorFlow Lite model has not been loaded. Call loadModel() with a valid path before running inference.")
    }

    private fun closeInterpreterLocked() {
        interpreter?.close()
        interpreter = null
    }

    @Throws(IOException::class)
    private fun loadModelFile(modelFile: File): MappedByteBuffer {
        FileInputStream(modelFile).use { input ->
            val channel = input.channel
            return channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
        }
    }
}
