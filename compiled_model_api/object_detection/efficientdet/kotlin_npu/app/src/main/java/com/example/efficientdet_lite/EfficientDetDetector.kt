package com.example.efficientdet_lite

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer

class EfficientDetDetector(
    context: Context,
    private val modelAssetName: String = MODEL_ASSET_NAME,
    confidenceThreshold: Float = 0.35f,
) : AutoCloseable {
    private val appContext = context.applicationContext
    private val labels = loadLabels(appContext)
    private val postprocessor = EfficientDetPostprocessor(labels, confidenceThreshold)
    private var activeModel: ActiveModel? = null
    private var inferenceCount = 0

    val selectedBackend: String
        get() = activeModel?.backendName ?: "uninitialized"

    @Synchronized
    fun detect(bitmap: Bitmap): List<Detection> {
        val startNs = SystemClock.elapsedRealtimeNanos()
        val preprocessed = EfficientDetPreprocessor.preprocess(bitmap)
        val preprocessNs = SystemClock.elapsedRealtimeNanos()
        var model = activeModel ?: createFirstAvailableModel()

        val detections = try {
            runModel(model, preprocessed)
        } catch (throwable: Throwable) {
            Log.w(TAG, "${model.backendName} inference failed; trying fallback", throwable)
            model.close()
            activeModel = null
            model = createFirstAvailableModel(skipBackendsThrough = model.backend)
            runModel(model, preprocessed)
        }
        val endNs = SystemClock.elapsedRealtimeNanos()
        inferenceCount += 1
        if (inferenceCount <= INITIAL_PERF_LOGS || inferenceCount % PERF_LOG_INTERVAL == 0) {
            Log.i(
                "EfficientDetPerf",
                "backend=${model.backendName} total=${nsToMs(endNs - startNs)}ms " +
                    "preprocess=${nsToMs(preprocessNs - startNs)}ms " +
                    "run+post=${nsToMs(endNs - preprocessNs)}ms detections=${detections.size}",
            )
        }
        return detections
    }

    private fun runModel(model: ActiveModel, preprocessed: PreprocessedFrame): List<Detection> {
        require(preprocessed.input.size == EXPECTED_INPUT_BYTES) {
            "EfficientDet input mismatch: got ${preprocessed.input.size}, expected $EXPECTED_INPUT_BYTES"
        }
        model.inputs.first().writeInt8(preprocessed.input)
        model.compiledModel.run(model.inputs, model.outputs)
        val outputArrays = model.outputs.mapIndexed { index, output ->
            output.readFloat().also {
                if (!model.loggedTensorSummary) {
                    Log.i(TAG, "Output[$index] elementCount=${it.size}")
                } else {
                    Log.v(TAG, "Output[$index] elementCount=${it.size}")
                }
            }
        }
        if (!model.loggedTensorSummary) {
            model.loggedTensorSummary = true
            Log.i(TAG, "Verified uint8/int8 inputBytes=$EXPECTED_INPUT_BYTES outputElementCounts=${outputArrays.map { it.size }}")
        }
        return postprocessor.process(outputArrays, preprocessed.metadata)
    }

    private fun createFirstAvailableModel(skipBackendsThrough: Accelerator? = null): ActiveModel {
        val backends = listOf(Accelerator.NPU, Accelerator.GPU, Accelerator.CPU)
        val startIndex = skipBackendsThrough?.let { backends.indexOf(it) + 1 } ?: 0
        val failures = ArrayList<Throwable>()

        for (backend in backends.drop(startIndex)) {
            try {
                Log.i(TAG, "Loading EfficientDet-Lite model '$modelAssetName' with backend=$backend")
                Log.i("LiteRTDebugger", "Attempting backend: $backend")
                val compiledModel = CompiledModel.create(
                    appContext.assets,
                    modelAssetName,
                    CompiledModel.Options(backend),
                )
                val inputs = compiledModel.createInputBuffers()
                val outputs = compiledModel.createOutputBuffers()
                require(inputs.size == 1) { "Expected 1 EfficientDet input, got ${inputs.size}" }
                require(outputs.size >= 4) { "Expected at least 4 EfficientDet outputs, got ${outputs.size}" }
                Log.i(TAG, "Selected LiteRT backend=$backend inputs=${inputs.size} outputs=${outputs.size}")
                Log.i("LiteRTDebugger", ">>> Inference running on: $backend (inputs=${inputs.size} outputs=${outputs.size})")
                return ActiveModel(backend, compiledModel, inputs, outputs).also {
                    activeModel = it
                }
            } catch (throwable: Throwable) {
                Log.w(TAG, "Unable to initialize LiteRT backend=$backend", throwable)
                Log.w("LiteRTDebugger", "Backend $backend unavailable: ${throwable.message}")
                failures += throwable
            }
        }

        throw IllegalStateException(
            "Unable to initialize EfficientDet-Lite model from assets/$modelAssetName with NPU, GPU, or CPU",
            failures.lastOrNull(),
        )
    }

    @Synchronized
    override fun close() {
        activeModel?.close()
        activeModel = null
    }

    private data class ActiveModel(
        val backend: Accelerator,
        val compiledModel: CompiledModel,
        val inputs: List<TensorBuffer>,
        val outputs: List<TensorBuffer>,
        var loggedTensorSummary: Boolean = false,
    ) : AutoCloseable {
        val backendName: String = backend.name

        override fun close() {
            inputs.forEach { it.close() }
            outputs.forEach { it.close() }
            compiledModel.close()
        }
    }

    private companion object {
        const val TAG = "EfficientDetDetector"
        const val MODEL_ASSET_NAME = "efficientdet_lite0_detection.tflite"
        const val EXPECTED_INPUT_BYTES = EfficientDetPreprocessor.INPUT_SIZE * EfficientDetPreprocessor.INPUT_SIZE * 3
        const val INITIAL_PERF_LOGS = 5
        const val PERF_LOG_INTERVAL = 30

        fun nsToMs(nanos: Long): String = "%.1f".format(nanos / 1_000_000f)

        fun loadLabels(context: Context): List<String> =
            runCatching {
                context.assets.open("labels.txt").bufferedReader().useLines { lines ->
                    lines.map { it.trim() }.filter { it.isNotEmpty() }.toList()
                }
            }.getOrElse { throwable ->
                Log.w(TAG, "labels.txt missing; using class ids", throwable)
                emptyList()
            }
    }
}
