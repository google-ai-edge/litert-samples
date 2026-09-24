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
 * Reference wrapper for the LiteRT CompiledModel API (litert 2.2.0, kotlinx-coroutines 1.8.1;
 * also needs androidx.core:core-ktx). [create] builds the model on its own serial dispatcher,
 * on the first option set that works (NPU, then GPU + CPU, then CPU), and keeps the model's
 * input/output buffers for reuse; [run] uses the same dispatcher. Replace the model name and
 * the input/output shapes; keep the shape of the class. Call [close] after the last [run] has
 * returned. [accelerator] is the step that did not throw: NPU-only options compile as NPU +
 * CPU, so a failed NPU compile can still fall back to the CPU inside create (pass wantNpu =
 * true only with the vendor libraries in place); GPU + CPU leaves ops the GPU cannot run on
 * the CPU, as the legacy delegate did.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class LiteRtModel private constructor(
    context: Context,
    assetName: String,
    wantNpu: Boolean,
    private val dispatcher: CoroutineDispatcher,
) : AutoCloseable {
    private val env: Environment
    val model: CompiledModel
    val accelerator: Accelerator
    private val inputs: List<TensorBuffer>
    private val outputs: List<TensorBuffer>

    init {
        val npu = BuiltinNpuAcceleratorProvider(context)
        val useNpu = wantNpu && npu.isDeviceSupported()
        // With the provider, Environment.create sets the dispatch and compiler-plugin dirs itself.
        env = if (useNpu) Environment.create(context, npu) else Environment.create(context)
        val attempts = buildList {
            if (useNpu) add(Accelerator.NPU to CompiledModel.Options(Accelerator.NPU))
            add(Accelerator.GPU to CompiledModel.Options(Accelerator.GPU, Accelerator.CPU))
            add(Accelerator.CPU to CompiledModel.Options(Accelerator.CPU))
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
        model = created ?: run {
            env.close()
            throw IllegalStateException("CompiledModel.create failed on every accelerator", lastError)
        }
        accelerator = used
        inputs = model.createInputBuffers()
        outputs = model.createOutputBuffers()
    }

    /** Runs one inference; on the GPU, readFloat() is where the wait for the result happens. */
    suspend fun run(input: FloatArray): FloatArray = withContext(dispatcher) {
        inputs[0].writeFloat(input)
        model.run(inputs, outputs)
        outputs[0].readFloat()
    }

    /** For Java callers: the same inference, blocking; call it from a background thread. */
    fun runSync(input: FloatArray): FloatArray = runBlocking { run(input) }

    override fun close() {
        inputs.forEach { it.close() }
        outputs.forEach { it.close() }
        model.close()
        env.close()
    }

    companion object {
        private const val TAG = "LiteRtModel"

        /** Creates the model on a serial dispatcher of its own; create compiles the model, so call this from a coroutine. */
        suspend fun create(context: Context, assetName: String, wantNpu: Boolean): LiteRtModel {
            val dispatcher = Dispatchers.IO.limitedParallelism(1)
            return withContext(dispatcher) { LiteRtModel(context, assetName, wantNpu, dispatcher) }
        }
    }
}

/**
 * Replaces TensorImage + ImageProcessor(ResizeOp, NormalizeOp(127.5f, 127.5f)) for a float model
 * that takes [1, size, size, 3] RGB pixels in [-1, 1]. Adjust the normalization to the model;
 * scale(size, size) filters like ResizeOp BILINEAR, use scale(size, size, filter = false) for
 * NEAREST_NEIGHBOR.
 */
fun Bitmap.toModelInput(size: Int): FloatArray {
    val pixels = IntArray(size * size).also { scale(size, size).getPixels(it, 0, size, 0, 0, size, size) }
    val out = FloatArray(size * size * 3)
    var i = 0
    for (p in pixels) {
        out[i++] = ((p shr 16 and 0xFF) - 127.5f) / 127.5f
        out[i++] = ((p shr 8 and 0xFF) - 127.5f) / 127.5f
        out[i++] = ((p and 0xFF) - 127.5f) / 127.5f
    }
    return out
}
