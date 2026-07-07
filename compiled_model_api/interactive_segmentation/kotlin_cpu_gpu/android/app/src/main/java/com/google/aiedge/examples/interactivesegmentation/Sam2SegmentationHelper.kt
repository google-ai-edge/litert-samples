/*
 * Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.aiedge.examples.interactivesegmentation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * Promptable "tap to segment" on the LiteRT Compiled Model API, using SAM 2.1 (Hiera-Tiny).
 *
 * Two GPU-clean (LITERT_CL) models converted with litert-torch:
 *   1. Image encoder — run ONCE per image: RGB[1,3,1024,1024] -> the multi-scale feature pyramid,
 *      already projected to be decoder-ready (folds conv_s0/conv_s1 + no_memory):
 *        out 0: image_embeddings [1,256,64,64]
 *        out 1: feat_s1          [1,64,128,128]
 *        out 2: feat_s0          [1,32,256,256]
 *   2. Mask decoder — run PER TAP: takes the cached features + a sparse point embedding and emits
 *        out: pred_masks [1,3,256,256] (logits, 3 multimask candidates), iou_scores [1,3]
 *
 * The tiny point->token "prompt encoder" (a sin/cos positional encoding) is done on the host to keep
 * the decoder graph sin/cos-free; its constants are bundled as [PROMPT_CONST_FILE]. The decoder input
 * order is fixed by the converter: 0 image_embeddings, 1 sparse, 2 feat_s1, 3 feat_s0 (image_embeddings
 * and feat_s1 have the same element count, so the inputs are bound by index, never by size).
 */
class Sam2SegmentationHelper(
    private val context: Context,
    private var options: Options = Options(),
) {
    class Options(
        /** The delegate for running computationally intensive operations. */
        var delegate: AcceleratorEnum = DEFAULT_DELEGATE,
    )

    companion object {
        private const val TAG = "Sam2Segmentation"

        const val IMAGE_SIZE = 1024            // encoder input side
        const val MASK_SIZE = 256              // decoder mask side (1024 / 4)
        private const val EMBED_DIM = 256
        private const val NUM_MASKS = 3

        private const val ENCODER_FILE = "sam2_image_encoder_fp16.tflite"
        private const val DECODER_FILE = "sam2_mask_decoder_fp16.tflite"
        private const val PROMPT_CONST_FILE = "prompt_encode_const.bin"

        // ImageNet normalization, RGB, NCHW: (pixel/255 - mean) / std.
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        // Translucent overlay color drawn where a mask logit is positive.
        private const val MASK_COLOR = 0x9900C99E.toInt()  // ~60% alpha teal

        val DEFAULT_DELEGATE = AcceleratorEnum.GPU

        fun toAccelerator(acceleratorEnum: AcceleratorEnum): Accelerator {
            return when (acceleratorEnum) {
                AcceleratorEnum.CPU -> Accelerator.CPU
                AcceleratorEnum.GPU -> Accelerator.GPU
            }
        }
    }

    val segmentation: SharedFlow<SegmentationResult>
        get() = _segmentation
    private val _segmentation = MutableSharedFlow<SegmentationResult>(
        extraBufferCapacity = 8, onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    val error: SharedFlow<Throwable?>
        get() = _error
    private val _error = MutableSharedFlow<Throwable?>()

    private var encoder: CompiledModel? = null
    private var decoder: CompiledModel? = null

    // Prompt-encoder constants (host-side). Layout of PROMPT_CONST_FILE (768 little-endian floats):
    //   posmat[2,128] row-major (256), point_embed[1] (256), not_a_point (256).
    private var posMat = FloatArray(0)         // [256]  (row 0 = x, row 1 = y)
    private var pointEmbed = FloatArray(0)     // [256]
    private var notAPoint = FloatArray(0)      // [256]

    // Cached encoder outputs for the current image (re-fed to the decoder on every tap).
    private var imageEmbeddings: FloatArray? = null
    private var featS1: FloatArray? = null
    private var featS0: FloatArray? = null

    private val singleThreadDispatcher = Dispatchers.IO.limitedParallelism(1, "ModelDispatcher")

    // Reusable encoder preprocessing buffer.
    private val inputFloats = FloatArray(3 * IMAGE_SIZE * IMAGE_SIZE)
    private val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)

    /** (Re)create both [CompiledModel]s on the selected accelerator and load the prompt constants. */
    suspend fun initSegmenter() {
        cleanup()
        try {
            withContext(singleThreadDispatcher) {
                loadPromptConstants()
                // The heavy image encoder runs on the selected delegate (GPU by default — full LITERT_CL
                // residency, ~tens of ms). The small mask decoder runs on CPU: it is GPU-RESIDENT but the
                // GPU delegate's fp16 reductions corrupt its mask logits on Pixel 8a (device A/B: a GPU
                // decoder masks the background; CPU is correct) — a clean "residency != correctness" case.
                // The decoder is tiny, so the CPU cost is small.
                encoder = CompiledModel.create(
                    context.assets, ENCODER_FILE,
                    CompiledModel.Options(toAccelerator(options.delegate)), null
                )
                decoder = CompiledModel.create(
                    context.assets, DECODER_FILE,
                    CompiledModel.Options(Accelerator.CPU), null
                )
                // A new model means stale features; the caller must re-encode the current image.
                imageEmbeddings = null; featS1 = null; featS0 = null
                Log.i(TAG, "Created encoder + decoder CompiledModels on ${options.delegate}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create CompiledModels: ${e.message}")
            _error.emit(e)
        }
    }

    suspend fun cleanup() {
        withContext(singleThreadDispatcher) {
            encoder?.close()
            encoder = null
            decoder?.close()
            decoder = null
            imageEmbeddings = null
            featS1 = null
            featS0 = null
        }
    }

    fun setOptions(options: Options) {
        this.options = options
    }

    /** Run the heavy image encoder ONCE and cache the feature pyramid. Returns the encode time (ms). */
    suspend fun encodeImage(bitmap: Bitmap): Long {
        return withContext(singleThreadDispatcher) {
            val model = encoder ?: return@withContext -1L
            val start = SystemClock.uptimeMillis()
            preprocess(bitmap)

            val inputBuffers = model.createInputBuffers()
            val outputBuffers = model.createOutputBuffers()
            inputBuffers[0].writeFloat(inputFloats)
            model.run(inputBuffers, outputBuffers)
            // Output order is fixed by the converter wrapper: 0 image_embeddings, 1 feat_s1, 2 feat_s0.
            imageEmbeddings = outputBuffers[0].readFloat()
            featS1 = outputBuffers[1].readFloat()
            featS0 = outputBuffers[2].readFloat()
            inputBuffers.forEach { it.close() }
            outputBuffers.forEach { it.close() }
            SystemClock.uptimeMillis() - start
        }
    }

    /**
     * Segment around a point. [px], [py] are in 1024x1024 model space (0..1024). Emits the best
     * (highest predicted IoU) mask as a [MASK_SIZE]x[MASK_SIZE] translucent overlay bitmap.
     */
    suspend fun segmentAt(px: Float, py: Float) {
        try {
            withContext(singleThreadDispatcher) {
                val model = decoder ?: return@withContext
                val imgEmb = imageEmbeddings ?: return@withContext
                val s1 = featS1 ?: return@withContext
                val s0 = featS0 ?: return@withContext

                val start = SystemClock.uptimeMillis()
                val sparse = encodePoint(px, py)

                val inputBuffers = model.createInputBuffers()
                val outputBuffers = model.createOutputBuffers()
                // Bind by index — the converter's arg order: 0 image_embeddings, 1 sparse, 2 feat_s1, 3 feat_s0.
                inputBuffers[0].writeFloat(imgEmb)
                inputBuffers[1].writeFloat(sparse)
                inputBuffers[2].writeFloat(s1)
                inputBuffers[3].writeFloat(s0)
                model.run(inputBuffers, outputBuffers)

                // Outputs differ in size, so read both and bind by size: masks (3*256*256) > iou (3).
                val out0 = outputBuffers[0].readFloat()
                val out1 = outputBuffers[1].readFloat()
                val masks = if (out0.size >= out1.size) out0 else out1
                val iou = if (out0.size >= out1.size) out1 else out0
                inputBuffers.forEach { it.close() }
                outputBuffers.forEach { it.close() }

                var best = 0
                for (i in 1 until NUM_MASKS) if (iou[i] > iou[best]) best = i
                val maskBitmap = buildMaskBitmap(masks, best)
                val decodeTime = SystemClock.uptimeMillis() - start
                _segmentation.emit(SegmentationResult(maskBitmap, iou[best], decodeTime))
            }
        } catch (e: Exception) {
            Log.e(TAG, "Segmentation error: ${e.message}")
            _error.emit(e)
        }
    }

    /** Center the SAM resize: scale the whole bitmap to 1024x1024 and write planar NCHW floats. */
    private fun preprocess(bitmap: Bitmap) {
        val resized = if (bitmap.width == IMAGE_SIZE && bitmap.height == IMAGE_SIZE) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        }
        resized.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
        val planeSize = IMAGE_SIZE * IMAGE_SIZE
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF) / 255f
            val g = ((pixel shr 8) and 0xFF) / 255f
            val b = (pixel and 0xFF) / 255f
            inputFloats[i] = (r - MEAN[0]) / STD[0]
            inputFloats[planeSize + i] = (g - MEAN[1]) / STD[1]
            inputFloats[2 * planeSize + i] = (b - MEAN[2]) / STD[2]
        }
        if (resized !== bitmap) resized.recycle()
    }

    /**
     * Host port of the SAM 2 prompt encoder for ONE positive point (+ the implicit padding point).
     * Returns the sparse token tensor flattened as [2, 256]. Matches the upstream module to ~3.7e-7.
     */
    private fun encodePoint(px: Float, py: Float): FloatArray {
        val sparse = FloatArray(2 * EMBED_DIM)
        val cx = 2f * ((px + 0.5f) / IMAGE_SIZE) - 1f
        val cy = 2f * ((py + 0.5f) / IMAGE_SIZE) - 1f
        // token0 (the positive click): [sin(coord), cos(coord)] + point_embed[1].
        for (k in 0 until EMBED_DIM / 2) {
            val coord = (2.0 * PI * (cx * posMat[k] + cy * posMat[EMBED_DIM / 2 + k])).toFloat()
            sparse[k] = sin(coord) + pointEmbed[k]
            sparse[EMBED_DIM / 2 + k] = cos(coord) + pointEmbed[EMBED_DIM / 2 + k]
        }
        // token1 (the padding point, label -1): the learned "not a point" embedding.
        for (j in 0 until EMBED_DIM) sparse[EMBED_DIM + j] = notAPoint[j]
        return sparse
    }

    /** Build a 256x256 translucent overlay from the chosen mask's logits (positive => object). */
    private fun buildMaskBitmap(masks: FloatArray, maskIndex: Int): Bitmap {
        val offset = maskIndex * MASK_SIZE * MASK_SIZE
        val argb = IntArray(MASK_SIZE * MASK_SIZE)
        for (i in argb.indices) {
            argb[i] = if (masks[offset + i] > 0f) MASK_COLOR else Color.TRANSPARENT
        }
        return Bitmap.createBitmap(argb, MASK_SIZE, MASK_SIZE, Bitmap.Config.ARGB_8888)
    }

    private fun loadPromptConstants() {
        if (posMat.isNotEmpty()) return
        val bytes = context.assets.open(PROMPT_CONST_FILE).readBytes()
        val all = FloatArray(bytes.size / 4)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(all)
        require(all.size == 3 * EMBED_DIM) { "Unexpected prompt const size: ${all.size}" }
        posMat = all.copyOfRange(0, EMBED_DIM)
        pointEmbed = all.copyOfRange(EMBED_DIM, 2 * EMBED_DIM)
        notAPoint = all.copyOfRange(2 * EMBED_DIM, 3 * EMBED_DIM)
    }

    enum class AcceleratorEnum {
        CPU, GPU
    }

    data class SegmentationResult(
        val maskBitmap: Bitmap,
        val iou: Float,
        val decodeTimeMs: Long,
    )
}
