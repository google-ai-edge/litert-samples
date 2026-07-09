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

package com.google.ai.edge.examples.flux2_klein

import android.content.Context
import android.graphics.Bitmap
import com.google.ai.edge.litert.Environment
import java.io.Closeable
import java.io.File

/**
 * FLUX.2-klein-4B text-to-image, running entirely on the LiteRT CompiledModel GPU delegate.
 *
 * The 4B rectified-flow transformer and its 4B Qwen3 text encoder are exported as twelve int8
 * graphs, each small enough for the ML Drift shader compiler, and executed one at a time so the
 * peak footprint is a single ~912 MB graph rather than the 6.2 GB total. Everything a GPU graph
 * cannot express — tokenization, the embedding lookup, the causal/padding mask, both rotary tables,
 * the scheduler, and the two tail permutations — is precomputed by `conversion/gen_prep_klein.py`
 * and staged as little-endian `.bin` files alongside the graphs.
 *
 * klein is step-wise distilled, which makes the sampling loop unusually plain: four steps, **no
 * classifier-free guidance** (one transformer pass per step, not two), and a straight flow-matching
 * Euler update `latents += dsigma[step] * noisePrediction`.
 */
class Flux2KleinGenerator(context: Context) : Closeable {

  private val filesDir: File = requireNotNull(context.getExternalFilesDir(null))
  private val binsDir = File(filesDir, BINS_SUBDIR)
  private val environment: Environment = Environment.create()

  private val inputsEmbeds = readFloats("inputs_embeds")
  private val attentionMask = readFloats("enc_mask")
  private val encoderCos = readFloats("enc_cos")
  private val encoderSin = readFloats("enc_sin")
  private val cos = readFloats("cos")
  private val sin = readFloats("sin")
  private val timestepEmbeddings = readFloats("temb")
  private val sigmaDeltas = readFloats("dsigma")
  private val batchNormMean = readFloats("bn_mean")
  private val batchNormStd = readFloats("bn_std")
  private val unpackIndices = readInts("unpack_perm")
  private val unpatchifyIndices = readInts("unpatch_perm")
  private val initialLatents = readFloats("latents0")

  init {
    require(File(filesDir, "kc_final.tflite").exists()) {
      "Model graphs not found in ${filesDir.absolutePath}. Run install_to_device.sh first."
    }
  }

  /**
   * Runs the four-step denoising loop and decodes the latent to an image.
   *
   * @param onProgress called with a short human-readable status after each stage.
   * @return the generated 256x256 image.
   */
  fun generate(onProgress: (String) -> Unit): Bitmap {
    val startMillis = System.currentTimeMillis()
    fun elapsedSeconds() = (System.currentTimeMillis() - startMillis) / 1000f

    onProgress("Encoding the prompt…")
    val promptEmbeds = encodePrompt()

    var latents = initialLatents.copyOf()
    val timestepSize = timestepEmbeddings.size / STEPS
    for (step in 0 until STEPS) {
      val timestepEmbedding =
        timestepEmbeddings.copyOfRange(step * timestepSize, (step + 1) * timestepSize)
      val noisePrediction = denoiseStep(latents, promptEmbeds, timestepEmbedding)
      for (i in latents.indices) {
        latents[i] += sigmaDeltas[step] * noisePrediction[i]
      }
      onProgress("Step ${step + 1}/$STEPS (${elapsedSeconds()}s)")
    }

    onProgress("Decoding…")
    val latentImage = toLatentImage(latents)
    val pixels = ChunkRunner.gpu(environment, "kv_vae.tflite", filesDir, listOf(latentImage))[0]
    onProgress("Done in ${elapsedSeconds()}s")
    return toBitmap(pixels)
  }

  /**
   * Runs the three encoder chunks and interleaves their outputs into the conditioning tensor.
   *
   * klein conditions on Qwen3 hidden states from layers 9, 18 and 27, stacked channel-wise into
   * 3 x 2560 = 7680 channels per token. The tap positions are exactly the chunk boundaries, so each
   * chunk's output is both the next chunk's input and one third of the conditioning.
   */
  private fun encodePrompt(): FloatArray {
    val taps = ArrayList<FloatArray>(ENCODER_CHUNKS)
    var hidden = inputsEmbeds
    for (index in 0 until ENCODER_CHUNKS) {
      hidden =
        ChunkRunner.gpu(
          environment,
          "ke_enc$index.tflite",
          filesDir,
          listOf(hidden, attentionMask, encoderCos, encoderSin),
        )[0]
      taps.add(hidden)
    }
    val interleaved = FloatArray(TEXT_TOKENS * ENCODER_CHUNKS * ENCODER_DIM)
    for (token in 0 until TEXT_TOKENS) {
      for (tap in 0 until ENCODER_CHUNKS) {
        System.arraycopy(
          taps[tap],
          token * ENCODER_DIM,
          interleaved,
          token * ENCODER_CHUNKS * ENCODER_DIM + tap * ENCODER_DIM,
          ENCODER_DIM,
        )
      }
    }
    return interleaved
  }

  /** One transformer step: prep, the double-stream blocks, the single-stream blocks, the head. */
  private fun denoiseStep(
    latents: FloatArray,
    promptEmbeds: FloatArray,
    timestepEmbedding: FloatArray,
  ): FloatArray {
    val prep =
      ChunkRunner.gpu(
        environment,
        "kc_prep.tflite",
        filesDir,
        listOf(latents, promptEmbeds, timestepEmbedding),
      )
    var image = prep[0]
    var text = prep[1]
    val imageModulation = prep[2]
    val textModulation = prep[3]
    val singleModulation = prep[4]

    for (index in 0 until DOUBLE_CHUNKS) {
      val outputs =
        ChunkRunner.gpu(
          environment,
          "kc_double$index.tflite",
          filesDir,
          listOf(image, text, cos, sin, imageModulation, textModulation),
        )
      image = outputs[0]
      text = outputs[1]
    }

    // The single-stream blocks attend over one joint sequence: text tokens, then image tokens.
    var joint = text + image
    for (index in 0 until SINGLE_CHUNKS) {
      joint =
        ChunkRunner.gpu(
          environment,
          "kc_single$index.tflite",
          filesDir,
          listOf(joint, cos, sin, singleModulation),
        )[0]
    }
    return ChunkRunner.gpu(
      environment,
      "kc_final.tflite",
      filesDir,
      listOf(joint, timestepEmbedding),
    )[0]
  }

  /**
   * Packed latent tokens to a VAE latent: scatter by position id, denormalize, unpatchify.
   *
   * Both reorderings are pure permutations of the flat buffer, so `gen_prep_klein.py` recovers them
   * exactly with an `arange` probe and ships them as gather index maps. Only the per-channel
   * batch-norm denormalization between them is arithmetic.
   */
  private fun toLatentImage(latents: FloatArray): FloatArray {
    val unpacked = gather(latents, unpackIndices)
    val plane = PACKED_SIDE * PACKED_SIDE
    for (channel in 0 until PACKED_CHANNELS) {
      val base = channel * plane
      for (i in 0 until plane) {
        unpacked[base + i] = unpacked[base + i] * batchNormStd[channel] + batchNormMean[channel]
      }
    }
    return gather(unpacked, unpatchifyIndices)
  }

  /** Applies a precomputed index map: `out[i] = source[indices[i]]`. */
  private fun gather(source: FloatArray, indices: IntArray): FloatArray {
    val out = FloatArray(indices.size)
    for (i in indices.indices) {
      out[i] = source[indices[i]]
    }
    return out
  }

  /** Converts planar RGB in [-1, 1] to an ARGB bitmap. */
  private fun toBitmap(image: FloatArray): Bitmap {
    val plane = IMAGE_SIZE * IMAGE_SIZE
    val pixels = IntArray(plane)
    for (i in 0 until plane) {
      val red = ((image[i].coerceIn(-1f, 1f) + 1f) * HALF_BYTE_RANGE).toInt()
      val green = ((image[plane + i].coerceIn(-1f, 1f) + 1f) * HALF_BYTE_RANGE).toInt()
      val blue = ((image[2 * plane + i].coerceIn(-1f, 1f) + 1f) * HALF_BYTE_RANGE).toInt()
      pixels[i] = (OPAQUE_ALPHA shl 24) or (red shl 16) or (green shl 8) or blue
    }
    return Bitmap.createBitmap(pixels, IMAGE_SIZE, IMAGE_SIZE, Bitmap.Config.ARGB_8888)
  }

  /** Reads a little-endian float32 tensor staged by `gen_prep_klein.py`. */
  private fun readFloats(name: String): FloatArray {
    val bytes = File(binsDir, "$name.bin").readBytes()
    val out = FloatArray(bytes.size / BYTES_PER_WORD)
    for (i in out.indices) {
      out[i] = Float.fromBits(readLittleEndianInt(bytes, i * BYTES_PER_WORD))
    }
    return out
  }

  /** Reads a little-endian int32 index map staged by `gen_prep_klein.py`. */
  private fun readInts(name: String): IntArray {
    val bytes = File(binsDir, "$name.bin").readBytes()
    val out = IntArray(bytes.size / BYTES_PER_WORD)
    for (i in out.indices) {
      out[i] = readLittleEndianInt(bytes, i * BYTES_PER_WORD)
    }
    return out
  }

  private fun readLittleEndianInt(bytes: ByteArray, offset: Int): Int =
    (bytes[offset].toInt() and BYTE_MASK) or
      ((bytes[offset + 1].toInt() and BYTE_MASK) shl 8) or
      ((bytes[offset + 2].toInt() and BYTE_MASK) shl 16) or
      ((bytes[offset + 3].toInt() and BYTE_MASK) shl 24)

  override fun close() {
    environment.close()
  }

  companion object {
    /** Baked into the staged host inputs. Changing it means re-running gen_prep. */
    const val PROMPT = "a red apple on a wooden table, studio lighting"

    private const val STEPS = 4
    private const val TEXT_TOKENS = 512
    private const val ENCODER_CHUNKS = 3
    private const val ENCODER_DIM = 2560
    private const val DOUBLE_CHUNKS = 2
    private const val SINGLE_CHUNKS = 4
    private const val PACKED_CHANNELS = 128
    private const val PACKED_SIDE = 16
    private const val IMAGE_SIZE = 256
    private const val BINS_SUBDIR = "klein_bins"
    private const val BYTES_PER_WORD = 4
    private const val BYTE_MASK = 0xff
    private const val OPAQUE_ALPHA = 0xFF
    private const val HALF_BYTE_RANGE = 127.5f
  }
}
