package com.example.mobilenetlitert

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class MobilenetClassifier(private val context: Context) : AutoCloseable {
  private var session: ModelSession? = null
  private var labels: List<String> = emptyList()

  suspend fun initialize(policy: BackendPolicy) {
    withContext(Dispatchers.IO) {
      close()
      labels = loadLabels()

      // Force load Qualcomm libraries in order
      try {
        Log.i(TAG, "Pre-loading Qualcomm libraries...")
        System.loadLibrary("QnnSystem")
        System.loadLibrary("QnnHtp")
        Log.i(TAG, "Qualcomm libraries loaded successfully")
      } catch (e: Throwable) {
        Log.w(TAG, "Failed to pre-load Qualcomm libraries: ${e.message}")
      }

      val start = SystemClock.elapsedRealtime()
      Log.i(TAG, "Requested backend policy: $policy")

      val createdSession = when (policy) {
        BackendPolicy.CPU_EMULATOR -> initializeCpu()
        BackendPolicy.NPU_REQUIRED -> initializeNpu()
        BackendPolicy.AUTO -> {
          runCatching { initializeNpu() }
            .recoverCatching {
              Log.w(TAG, "NPU initialization failed, falling back to GPU: ${it.message}")
              initializeGpu()
            }
            .recoverCatching {
              Log.w(TAG, "GPU initialization failed, falling back to CPU: ${it.message}")
              initializeCpu()
            }
            .getOrThrow()
        }
      }

      session = createdSession
      Log.i(TAG, "Created ${createdSession.inputBuffers.size} input buffer(s)")
      Log.i(TAG, "Created ${createdSession.outputBuffers.size} output buffer(s)")

      Log.i(TAG, "Model initialized with backend=${createdSession.backendName} in ${SystemClock.elapsedRealtime() - start} ms")
    }
  }

  suspend fun classify(bitmap: Bitmap): ClassificationResult {
    return withContext(Dispatchers.Default) {
      val activeSession = requireNotNull(session) { "Model is not initialized." }

      val preprocessStart = SystemClock.elapsedRealtime()
      val input = ImagePreprocessor.preprocessBitmap(bitmap)
      val preprocessMs = SystemClock.elapsedRealtime() - preprocessStart
      Log.i(TAG, "Preprocessing time: $preprocessMs ms")

      activeSession.inputBuffers[0].writeFloat(input)

      val inferenceStart = SystemClock.elapsedRealtime()
      activeSession.model.run(activeSession.inputBuffers, activeSession.outputBuffers)
      val inferenceMs = SystemClock.elapsedRealtime() - inferenceStart
      Log.i(TAG, "CompiledModel.run() time: $inferenceMs ms on backend=${activeSession.backendName}")

      val logits = activeSession.outputBuffers[0].readFloat()

      ClassificationResult(
        predictions = ClassificationPostprocessor.topK(logits, labels, 5),
        backend = activeSession.backendName,
        preprocessMs = preprocessMs,
        inferenceMs = inferenceMs,
      )
    }
  }

  private fun initializeCpu(): ModelSession {
    val options = CompiledModel.Options(Accelerator.CPU)
    val compiledModel = CompiledModel.create(context.assets, CPU_MODEL_ASSET, options, null)
    Log.i(TAG, "Selected model source=assets/$CPU_MODEL_ASSET accelerators=[CPU]")
    return ModelSession(compiledModel, "CPU")
  }

  private fun initializeGpu(): ModelSession {
    val options = CompiledModel.Options(Accelerator.GPU)
    val compiledModel = CompiledModel.create(context.assets, CPU_MODEL_ASSET, options, null)
    Log.i(TAG, "Selected model source=assets/$CPU_MODEL_ASSET accelerators=[GPU]")
    return ModelSession(compiledModel, "GPU")
  }

  private fun initializeNpu(): ModelSession {
    val env = Environment.create(BuiltinNpuAcceleratorProvider(context))

    Log.i(TAG, "Trying JIT compilation on NPU with float model: $CPU_MODEL_ASSET")
    val options = CompiledModel.Options(setOf(Accelerator.NPU)).apply {
      qualcommOptions = CompiledModel.QualcommOptions(
        htpPerformanceMode = CompiledModel.QualcommOptions.HtpPerformanceMode.HIGH_PERFORMANCE
      )
    }

    val compiledModel = CompiledModel.create(context.assets, CPU_MODEL_ASSET, options, env)
    Log.i(TAG, "Selected model source=assets/$CPU_MODEL_ASSET accelerators=[NPU]")
    Log.i(TAG, "NPU model created successfully with JIT")
    return ModelSession(compiledModel, "NPU (JIT)")
  }

  private fun loadLabels(): List<String> {
    return context.assets.open(LABELS_ASSET).bufferedReader().useLines { lines ->
      lines.map(String::trim).filter(String::isNotEmpty).toList()
    }
  }

  override fun close() {
    runCatching {
      session?.close()
    }.onFailure {
      Log.w(TAG, "Cleanup failed: ${it.message}", it)
    }

    session = null
  }

  private class ModelSession(
    val model: CompiledModel,
    val backendName: String,
  ) : AutoCloseable {
    val inputBuffers = model.createInputBuffers()
    val outputBuffers = model.createOutputBuffers()

    override fun close() {
      inputBuffers.forEach { it.close() }
      outputBuffers.forEach { it.close() }
      model.close()
    }
  }

  companion object {
    const val TAG = "MobileNetLiteRT"
    private const val CPU_MODEL_ASSET = "model/mobilenet_v2_float.tflite"
    private const val LABELS_ASSET = "labels/imagenet_labels.txt"
  }
}
