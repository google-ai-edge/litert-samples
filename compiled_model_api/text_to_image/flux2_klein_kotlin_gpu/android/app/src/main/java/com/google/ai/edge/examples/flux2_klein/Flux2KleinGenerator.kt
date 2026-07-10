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
 * FLUX.2-klein-4B text-to-image and image editing, running entirely on the LiteRT CompiledModel GPU
 * delegate.
 *
 * The 4B rectified-flow transformer and its 4B Qwen3 text encoder are exported as twelve int8
 * graphs, each small enough for the ML Drift shader compiler, and executed one at a time so the
 * peak footprint is a single ~912 MB graph rather than the 6.2 GB total. Everything a GPU graph
 * cannot express — tokenization, the embedding lookup, the causal/padding mask, both rotary tables,
 * the scheduler, and the tail permutations — is precomputed by `conversion/gen_prep_klein.py` and
 * staged as little-endian `.bin` files alongside the graphs.
 *
 * klein is step-wise distilled, which makes the sampling loop unusually plain: four steps, **no
 * classifier-free guidance** (one transformer pass per step, not two), and a straight flow-matching
 * Euler update `latents += dsigma[step] * noisePrediction`.
 *
 * Passing a reference image switches to editing, which is the same model's `image=` path. The
 * reference is VAE-encoded on device and its latent tokens are appended to the noise tokens before
 * every step, exactly as `Flux2KleinPipeline` does:
 * ```
 * latent_model_input = cat([latents, image_latents], dim=1)
 * ```
 *
 * That doubles the image sequence (256 to 512 tokens, joint 768 to 1024), so editing runs the
 * `kce_*` graphs — the identical weights re-exported at the longer shape. Only the leading noise
 * tokens of the prediction are stepped; the reference half is discarded.
 */
class Flux2KleinGenerator(context: Context) : Closeable {

  private val filesDir: File = requireNotNull(context.getExternalFilesDir(null))
  private val environment: Environment = Environment.create()

  private val textToImageInputs = StagedInputs(File(filesDir, TEXT_TO_IMAGE_BINS))

  /** Editing inputs are only staged when the editing graphs are, so read them on demand. */
  private val editingInputs: StagedInputs by lazy { StagedInputs(File(filesDir, EDITING_BINS)) }

  init {
    require(File(filesDir, "${TEXT_TO_IMAGE_PREFIX}_final.tflite").exists()) {
      "Model graphs not found in ${filesDir.absolutePath}. Run install_to_device.sh first."
    }
  }

  /** True when the editing graphs and their staged inputs are present on the device. */
  fun isEditingAvailable(): Boolean =
    File(filesDir, "${EDITING_PREFIX}_final.tflite").exists() &&
      File(filesDir, "kv_vae_enc.tflite").exists() &&
      File(File(filesDir, EDITING_BINS), "patch_perm.bin").exists()

  /**
   * Runs the four-step denoising loop and decodes the latent to an image.
   *
   * @param reference when non-null, this image is edited with [EDIT_PROMPT] instead of a new image
   *   being generated. It is squared and resized to 256x256 first.
   * @param onProgress called with a short human-readable status after each stage.
   * @return the generated or edited 256x256 image.
   */
  fun generate(reference: Bitmap? = null, onProgress: (String) -> Unit): Bitmap {
    val editing = reference != null
    require(!editing || isEditingAvailable()) {
      "Editing graphs not found. Re-run install_to_device.sh with the editing bins directory."
    }
    val inputs = if (editing) editingInputs else textToImageInputs
    val prefix = if (editing) EDITING_PREFIX else TEXT_TO_IMAGE_PREFIX

    val startMillis = System.currentTimeMillis()
    fun elapsedSeconds() = (System.currentTimeMillis() - startMillis) / 1000f

    val referenceTokens =
      reference?.let {
        onProgress("Encoding the reference image…")
        encodeReference(it, inputs)
      }

    onProgress("Encoding the prompt…")
    val promptEmbeds = encodePrompt(inputs)

    val latents = inputs.initialLatents.copyOf()
    val timestepSize = inputs.timestepEmbeddings.size / STEPS
    for (step in 0 until STEPS) {
      val timestepEmbedding =
        inputs.timestepEmbeddings.copyOfRange(step * timestepSize, (step + 1) * timestepSize)
      val tokens = if (referenceTokens == null) latents else latents + referenceTokens
      val noisePrediction = denoiseStep(prefix, inputs, tokens, promptEmbeds, timestepEmbedding)
      // When editing, the prediction also covers the reference tokens; only the noise half moves.
      for (i in latents.indices) {
        latents[i] += inputs.sigmaDeltas[step] * noisePrediction[i]
      }
      onProgress("Step ${step + 1}/$STEPS (${elapsedSeconds()}s)")
    }

    onProgress("Decoding…")
    val latentImage = toLatentImage(latents, inputs)
    val pixels = ChunkRunner.gpu(environment, "kv_vae.tflite", filesDir, listOf(latentImage))[0]
    onProgress("Done in ${elapsedSeconds()}s")
    return toBitmap(pixels)
  }

  /**
   * VAE-encodes the reference image into the transformer's packed latent tokens.
   *
   * Mirrors `Flux2KleinPipeline._encode_vae_image`: encode to the latent mean, patchify 2x2,
   * batch-norm normalize per packed channel, then pack `[128, 16, 16]` into `[256, 128]` tokens.
   * The patchify is a pure permutation, so `gen_prep_klein.py --edit` ships it as a gather index
   * map rather than the reshape being restated here.
   */
  private fun encodeReference(reference: Bitmap, inputs: StagedInputs): FloatArray {
    val planar = listOf(toPlanarRgb(reference))
    val latent = ChunkRunner.gpu(environment, "kv_vae_enc.tflite", filesDir, planar)[0]
    val packed = gather(latent, inputs.patchifyIndices)
    val plane = PACKED_SIDE * PACKED_SIDE
    for (channel in 0 until PACKED_CHANNELS) {
      val base = channel * plane
      for (i in 0 until plane) {
        packed[base + i] =
          (packed[base + i] - inputs.batchNormMean[channel]) / inputs.batchNormStd[channel]
      }
    }
    val tokens = FloatArray(plane * PACKED_CHANNELS)
    for (channel in 0 until PACKED_CHANNELS) {
      for (i in 0 until plane) {
        tokens[i * PACKED_CHANNELS + channel] = packed[channel * plane + i]
      }
    }
    return tokens
  }

  /** Center-crops, resizes to 256x256, and converts to planar RGB in [-1, 1]. */
  private fun toPlanarRgb(source: Bitmap): FloatArray {
    val side = minOf(source.width, source.height)
    val square =
      Bitmap.createBitmap(source, (source.width - side) / 2, (source.height - side) / 2, side, side)
    val scaled = Bitmap.createScaledBitmap(square, IMAGE_SIZE, IMAGE_SIZE, true)
    val plane = IMAGE_SIZE * IMAGE_SIZE
    val pixels = IntArray(plane)
    scaled.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
    val planar = FloatArray(3 * plane)
    for (i in 0 until plane) {
      val pixel = pixels[i]
      planar[i] = ((pixel shr 16 and BYTE_MASK) / HALF_BYTE_RANGE) - 1f
      planar[plane + i] = ((pixel shr 8 and BYTE_MASK) / HALF_BYTE_RANGE) - 1f
      planar[2 * plane + i] = ((pixel and BYTE_MASK) / HALF_BYTE_RANGE) - 1f
    }
    return planar
  }

  /**
   * Runs the three encoder chunks and interleaves their outputs into the conditioning tensor.
   *
   * klein conditions on Qwen3 hidden states from layers 9, 18 and 27, stacked channel-wise into
   * 3 x 2560 = 7680 channels per token. The tap positions are exactly the chunk boundaries, so each
   * chunk's output is both the next chunk's input and one third of the conditioning.
   */
  private fun encodePrompt(inputs: StagedInputs): FloatArray {
    val taps = ArrayList<FloatArray>(ENCODER_CHUNKS)
    var hidden = inputs.inputsEmbeds
    for (index in 0 until ENCODER_CHUNKS) {
      hidden =
        ChunkRunner.gpu(
          environment,
          "ke_enc$index.tflite",
          filesDir,
          listOf(hidden, inputs.attentionMask, inputs.encoderCos, inputs.encoderSin),
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
    prefix: String,
    inputs: StagedInputs,
    tokens: FloatArray,
    promptEmbeds: FloatArray,
    timestepEmbedding: FloatArray,
  ): FloatArray {
    val prep =
      ChunkRunner.gpu(
        environment,
        "${prefix}_prep.tflite",
        filesDir,
        listOf(tokens, promptEmbeds, timestepEmbedding),
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
          "${prefix}_double$index.tflite",
          filesDir,
          listOf(image, text, inputs.cos, inputs.sin, imageModulation, textModulation),
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
          "${prefix}_single$index.tflite",
          filesDir,
          listOf(joint, inputs.cos, inputs.sin, singleModulation),
        )[0]
    }
    return ChunkRunner.gpu(
      environment,
      "${prefix}_final.tflite",
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
  private fun toLatentImage(latents: FloatArray, inputs: StagedInputs): FloatArray {
    val unpacked = gather(latents, inputs.unpackIndices)
    val plane = PACKED_SIDE * PACKED_SIDE
    for (channel in 0 until PACKED_CHANNELS) {
      val base = channel * plane
      for (i in 0 until plane) {
        unpacked[base + i] =
          unpacked[base + i] * inputs.batchNormStd[channel] + inputs.batchNormMean[channel]
      }
    }
    return gather(unpacked, inputs.unpatchifyIndices)
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

  override fun close() {
    environment.close()
  }

  /**
   * Every host tensor one mode needs, read once from its staged directory.
   *
   * Text-to-image and editing differ in the rotary tables (768 versus 1024 joint positions) and in
   * the prompt, so each mode keeps its own directory. `patchifyIndices` exists only for editing and
   * is read on first use.
   */
  private class StagedInputs(private val binsDir: File) {
    val inputsEmbeds = readFloats("inputs_embeds")
    val attentionMask = readFloats("enc_mask")
    val encoderCos = readFloats("enc_cos")
    val encoderSin = readFloats("enc_sin")
    val cos = readFloats("cos")
    val sin = readFloats("sin")
    val timestepEmbeddings = readFloats("temb")
    val sigmaDeltas = readFloats("dsigma")
    val batchNormMean = readFloats("bn_mean")
    val batchNormStd = readFloats("bn_std")
    val unpackIndices = readInts("unpack_perm")
    val unpatchifyIndices = readInts("unpatch_perm")
    val initialLatents = readFloats("latents0")
    val patchifyIndices: IntArray by lazy { readInts("patch_perm") }

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
  }

  companion object {
    /** Baked into the staged host inputs. Changing it means re-running gen_prep. */
    const val PROMPT = "a red apple on a wooden table, studio lighting"

    /** Baked into the staged editing inputs by `gen_prep_klein.py --edit`. */
    const val EDIT_PROMPT = "turn the apple into a green apple"

    private const val STEPS = 4
    private const val TEXT_TOKENS = 512
    private const val ENCODER_CHUNKS = 3
    private const val ENCODER_DIM = 2560
    private const val DOUBLE_CHUNKS = 2
    private const val SINGLE_CHUNKS = 4
    private const val PACKED_CHANNELS = 128
    private const val PACKED_SIDE = 16
    private const val IMAGE_SIZE = 256
    private const val TEXT_TO_IMAGE_BINS = "klein_bins"
    private const val EDITING_BINS = "klein_bins_edit"
    private const val TEXT_TO_IMAGE_PREFIX = "kc"
    private const val EDITING_PREFIX = "kce"
    private const val BYTES_PER_WORD = 4
    private const val BYTE_MASK = 0xff
    private const val OPAQUE_ALPHA = 0xFF
    private const val HALF_BYTE_RANGE = 127.5f
  }
}
