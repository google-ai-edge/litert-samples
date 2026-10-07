// TODO: replace with the app's package.
package com.example.validation

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.core.graphics.scale
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.LiteRtException
import com.google.ai.edge.litert.TensorBuffer
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext

/**
 * Reference wrapper for the LiteRT CompiledModel API (litert 2.3.0 with litert-gpu, or litert
 * 2.2.0; kotlinx-coroutines 1.8.1 or newer; also needs androidx.core:core-ktx). [create] builds
 * the model on its own serial dispatcher, on the first option set that works (NPU, then GPU +
 * CPU, then CPU; wantGpu = false skips the GPU step, numThreads is the legacy setNumThreads), and
 * keeps the model's input/output buffers for reuse; [run] uses the same dispatcher. Replace the
 * model name and the input/output shapes; keep the shape of the class. Call [close] after the
 * last [run] has returned; a second [close] does nothing.
 *
 * One [Environment] serves every model in the process and stays open: creating it loads the
 * accelerator libraries, and an Environment created and closed per model reloads them on every
 * load (on a Galaxy S26 with litert 2.2.0 one such reload aborted the process inside the GPU
 * accelerator library). The first [create] decides whether the Environment carries the NPU
 * provider.
 *
 * [accelerator] is the step that did not throw, not proof that every op runs there: NPU-only
 * options compile as NPU + CPU, so a device without the vendor libraries still reports NPU and
 * runs on the CPU (pass wantNpu = true only with those libraries in place); GPU + CPU leaves ops
 * the GPU cannot run on the CPU, as the legacy delegate did; and with litert 2.3.0 the GPU step
 * also "works" when the app forgot the litert-gpu dependency, because the runtime logs "GPU
 * accelerator could not be loaded and registered" and runs the graph on the CPU.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class LiteRtModel private constructor(
    context: Context,
    assetName: String,
    wantNpu: Boolean,
    wantGpu: Boolean,
    numThreads: Int?,
    private val dispatcher: CoroutineDispatcher,
) : AutoCloseable {
    val model: CompiledModel
    val accelerator: Accelerator
    private val inputs: List<TensorBuffer>
    private val outputs: List<TensorBuffer>

    init {
        val useNpu = wantNpu && BuiltinNpuAcceleratorProvider(context).isDeviceSupported()
        val env = environment(context, useNpu)
        // A fresh Options per step: cpuOptions is set on it, never on the shared Options.CPU.
        fun options(vararg accelerators: Accelerator) =
            CompiledModel.Options(*accelerators).apply {
                if (numThreads != null) {
                    cpuOptions = CompiledModel.CpuOptions(numThreads = numThreads)
                }
            }
        val attempts = buildList {
            if (useNpu) {
                add(Accelerator.NPU to options(Accelerator.NPU))
            }
            if (wantGpu) {
                add(Accelerator.GPU to options(Accelerator.GPU, Accelerator.CPU))
            }
            add(Accelerator.CPU to options(Accelerator.CPU))
        }
        var created: CompiledModel? = null
        var used = Accelerator.CPU
        var lastError: LiteRtException? = null
        for ((acc, options) in attempts) {
            try {
                created = CompiledModel.create(context.assets, assetName, options, env)
                used = acc
                break
            } catch (e: LiteRtException) {
                Log.w(TAG, "$acc unavailable: ${e.message}")
                lastError = e
            }
        }
        model = created
            ?: throw IllegalStateException(
                "CompiledModel.create failed on every accelerator", lastError
            )
        accelerator = used
        // The buffers can throw after the model exists (an input or output type the buffers do not
        // take); nothing may stay open then, or every failed load keeps a model and its threads.
        var inputBuffers: List<TensorBuffer> = emptyList()
        try {
            inputBuffers = model.createInputBuffers()
            inputs = inputBuffers
            outputs = model.createOutputBuffers()
        } catch (e: Exception) {
            inputBuffers.forEach { it.close() }
            model.close()
            throw e
        }
    }

    /** Runs one inference; on the GPU, readFloat() is where the wait for the result happens. */
    suspend fun run(input: FloatArray): FloatArray =
        run({ it.writeFloat(input) }, { it.readFloat() })

    /**
     * Runs one inference with the caller filling the first input buffer and reading the first
     * output buffer. With SupportBridge:
     * `run({ bridge.write(it, image) }, { bridge.read(it, shape, DataType.FLOAT32) })`.
     */
    suspend fun <T> run(write: (TensorBuffer) -> Unit, read: (TensorBuffer) -> T): T =
        withContext(dispatcher) {
            write(inputs[0])
            model.run(inputs, outputs)
            read(outputs[0])
        }

    /** For Java callers: the same inference, blocking; call it from a background thread. */
    fun runSync(input: FloatArray): FloatArray = runBlocking { run(input) }

    override fun close() {
        inputs.forEach { it.close() }
        outputs.forEach { it.close() }
        model.close()
    }

    companion object {
        private const val TAG = "LiteRtModel"
        private var sharedEnvironment: Environment? = null

        /**
         * The process-wide Environment; with the NPU provider, Environment.create sets the dispatch
         * and compiler-plugin dirs itself.
         */
        @Synchronized
        private fun environment(context: Context, useNpu: Boolean): Environment =
            sharedEnvironment ?: run {
                val app = context.applicationContext
                val env =
                    if (useNpu) {
                        Environment.create(app, BuiltinNpuAcceleratorProvider(app))
                    } else {
                        Environment.create(app)
                    }
                sharedEnvironment = env
                env
            }

        /**
         * Creates the model on a serial dispatcher of its own; create compiles the model, so call
         * this from a coroutine. wantGpu = false leaves the GPU step out (an app whose CPU setting
         * means the CPU alone); numThreads is the legacy Interpreter.Options.setNumThreads.
         */
        suspend fun create(
            context: Context,
            assetName: String,
            wantNpu: Boolean,
            wantGpu: Boolean = true,
            numThreads: Int? = null,
        ): LiteRtModel {
            val dispatcher = Dispatchers.IO.limitedParallelism(1)
            return withContext(dispatcher) {
                LiteRtModel(context, assetName, wantNpu, wantGpu, numThreads, dispatcher)
            }
        }
    }
}

/**
 * For an app without LiteRT Support: the equivalent of TensorImage + ImageProcessor(ResizeOp,
 * NormalizeOp(127.5f, 127.5f)) for a float model that takes [1, size, size, 3] RGB pixels in
 * [-1, 1]. Adjust the normalization to the model; scale(size, size) filters like ResizeOp
 * BILINEAR, use scale(size, size, filter = false) for NEAREST_NEIGHBOR. An app that keeps LiteRT
 * Support uses its ImageProcessor and SupportBridge.kt instead.
 */
fun Bitmap.toModelInput(size: Int): FloatArray {
    val pixels = IntArray(size * size)
    scale(size, size).getPixels(pixels, 0, size, 0, 0, size, size)
    val out = FloatArray(size * size * 3)
    var i = 0
    for (p in pixels) {
        out[i++] = ((p shr 16 and 0xFF) - 127.5f) / 127.5f
        out[i++] = ((p shr 8 and 0xFF) - 127.5f) / 127.5f
        out[i++] = ((p and 0xFF) - 127.5f) / 127.5f
    }
    return out
}
