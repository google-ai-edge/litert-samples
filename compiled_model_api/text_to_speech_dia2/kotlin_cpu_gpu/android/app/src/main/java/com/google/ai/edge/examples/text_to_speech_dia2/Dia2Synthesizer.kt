/*
 * Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"),
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

package com.google.ai.edge.examples.text_to_speech_dia2

import android.content.Context
import android.util.Half
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.Random
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.sin

/**
* Dia2-1B (Nari Labs, Apache-2.0) on-device dialogue TTS on LiteRT CompiledModel (CPU fp32).
*
* Dia2 is a Moshi-style RQ-Transformer. Once per 12.5 Hz frame a 30-layer *temporal* transformer
* emits a word-timing action plus Mimi codebook 0, a 3-layer *depformer* then autoregressively
* fills the remaining 31 codebooks for that same frame. A [StateMachine] paces the [S1]/[S2]
* script onto two text streams according to the action head, and the 32 codebooks are decoded to
* 24 kHz audio by Mimi.
*
* Three details are easy to get wrong and are handled explicitly here:
*  * **Classifier-free guidance.** Dia2's own default is `cfg_scale = 2.0`, so every frame runs
*    twice, conditional and unconditional, each with its own KV cache. Note that guidance does
*    *not* fix the speaker: with no voice prefix the speaker identity is sampled and varies from
*    run to run at any cfg scale. What guidance does buy is steadier output levels.
*  * **The delay pattern.** Codebook `cb` lags the aligned timeline by `AUDIO_DELAYS[cb]` frames
*    and must be undelayed before Mimi decoding.
*  * **Both text streams carry real word tokens**, not just new-word/pad markers.
*
* All graphs run on CPU as fp32, because these language models collapse in fp16 on ARM. The GPU is
* not the obstacle: with a rank-4 fused-QKV rewrite and a pre-expanded attention mask the depformer
* delegates all 237 nodes and yields bit-identical audio. It is simply no faster: 21.1 ms per
* stage on the GPU against 19.7 ms on the CPU, because a 3-layer single-token step graph cannot
* amortise the dispatch and the readback synchronisation. The README has the measured breakdown.
*/
class Dia2Synthesizer(context: Context) : Closeable {

  companion object {
    /** Model width shared by the temporal transformer and the depformer. */
    const val HIDDEN = 1024
    const val HEAD_DIM = 128

    const val TEMPORAL_LAYERS = 30
    const val TEMPORAL_KV_HEADS = 8

    /** Longest temporal context the exported graph accepts, in frames. */
    const val TEMPORAL_MAX_STEPS = 256

    const val DEPFORMER_LAYERS = 3
    const val DEPFORMER_KV_HEADS = 8

    /** Depformer stages, i.e. codebooks 1..31, also its KV-cache depth. */
    const val DEPFORMER_DEPTH = 31

    /** Mimi codebooks per frame (codebook 0 plus the 31 depformer stages). */
    const val CODEBOOKS = 32

    /** Audio vocabulary, including the BOS and PAD ids at the top. */
    const val AUDIO_VOCAB = 2050

    /** Sampleable audio ids, i.e. [AUDIO_VOCAB] minus the BOS/PAD pair. */
    const val AUDIO_VOCAB_LIMIT = 2048

    /** Mimi latent width fed to the decoder. */
    const val MIMI_DIM = 512

    /** Fixed Mimi decode window, in frames, at least any undelayed utterance length. */
    const val DECODE_FRAMES = 256

    /** 24000 Hz / 12.5 Hz frame rate. */
    const val SAMPLES_PER_FRAME = 1920

    const val SAMPLE_RATE = 24000
    const val ROPE_THETA = 10000.0

    // Text-stream token ids.
    const val NEW_WORD = 2
    const val PAD = 3
    const val BOS = 1
    const val ZERO = 7
    const val SPEAKER_1 = 49152
    const val SPEAKER_2 = 49153

    // Audio-stream token ids.
    const val AUDIO_BOS = 2048
    const val AUDIO_PAD = 2049

    // StateMachine pacing (Dia2 runtime defaults).
    const val MAX_PADDING = 6
    const val SECOND_AHEAD = 2
    const val INITIAL_PADDING = 2

    /** Largest entry of [AUDIO_DELAYS], the undelayed stream is this many frames shorter. */
    const val MAX_DELAY = 18

    /** Frames to keep generating after the script ends, so every codebook flushes. */
    const val FLUSH_TAIL = MAX_DELAY + MAX_PADDING

    // Sampling and guidance (Dia2 GenerationConfig defaults).
    const val TEXT_TEMP = 0.6f
    const val AUDIO_TEMP = 0.8f
    const val CFG_SCALE = 2.0f
    const val CFG_FILTER_K = 50

    /** Per-codebook lag, in frames, of the generated grid behind the aligned timeline. */
    val AUDIO_DELAYS = IntArray(CODEBOOKS) { if (it == 0) 16 else 18 }

    /** Which of the three depformer weight sets each stage uses. */
    val WEIGHTS_SCHEDULE = IntArray(DEPFORMER_DEPTH) {
      if (it < 8) 0 else if (it < 16) 1 else 2
    }

    /** Frames generated before giving up, bounded so the undelayed length fits the decoder. */
    private const val DEFAULT_MAX_FRAMES = 250

    /** Columns of the delayed code grid, covers the warm-up, the tail write and the delay. */
    private const val AUDIO_BUFFER_FRAMES = TEMPORAL_MAX_STEPS + MAX_DELAY + 2

    /** Additive mask value standing in for negative infinity in the exported graphs. */
    private const val MASK_NEG_INF = -3.0e38f

    private const val BYTES_PER_HALF = 2
  }

  /** Generated PCM plus the frame count and wall-clock cost of producing it. */
  data class Result(val audio: FloatArray, val frames: Int, val ms: Long)

  private val modelDir = requireNotNull(context.getExternalFilesDir(null))
  private val tokenizer = BpeTokenizer(context)
  private val voicePrefix = loadVoicePrefix(context)

  private fun loadOnCpu(name: String) = CompiledModel.create(
    File(modelDir, name).absolutePath, CompiledModel.Options(Accelerator.CPU), null)

  private val temporal = loadOnCpu("dia2_temporal_fp32.tflite")
  private val depGraphs = Array(3) { loadOnCpu("dia2_depformer_wi${it}_fp32.tflite") }
  private val mimiDequant = loadOnCpu("dia2_mimi_dequant.tflite")
  private val mimiDecode = loadOnCpu("dia2_mimi_decode_t256.tflite")

  /** Memory-maps an fp16 lookup table, rows are read lazily to keep resident memory small. */
  private fun mapHalfTable(name: String): ByteBuffer {
    val channel = RandomAccessFile(File(modelDir, name), "r").channel
    return channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
      .order(ByteOrder.LITTLE_ENDIAN)
  }

  /** Reads an fp16 table fully into fp32, for weights on the per-stage matmul hot path. */
  private fun readHalfTableAsFloat(name: String): FloatArray {
    val buffer = mapHalfTable(name)
    val count = buffer.capacity() / BYTES_PER_HALF
    val out = FloatArray(count)
    var offset = 0
    for (i in 0 until count) {
      out[i] = Half.toFloat(buffer.getShort(offset))
      offset += BYTES_PER_HALF
    }
    return out
  }

  // Text embeddings are pre-multiplied by their projection, so a lookup replaces a matmul.
  private val combinedMain = mapHalfTable("dia2_combined_main.f16")     // [textVocab, HIDDEN]
  private val combinedSecond = mapHalfTable("dia2_combined_second.f16") // [textVocab, HIDDEN]
  private val temporalAudio = mapHalfTable("dia2_temporal_audio.f16")   // [32, VOCAB, HIDDEN]
  private val depAudio = mapHalfTable("dia2_dep_audio.f16")             // [31, VOCAB, HIDDEN]
  private val depIn = readHalfTableAsFloat("dia2_dep_in.f16")           // [3, HIDDEN, HIDDEN]
  private val depLogits = readHalfTableAsFloat("dia2_dep_logits.f16")   // [31, VOCAB, HIDDEN]

  private val random = Random()

  // ---- fp16 table helpers -------------------------------------------------

  private fun rowOffset(row: Long) = row * HIDDEN * BYTES_PER_HALF

  private fun addRow(table: ByteBuffer, rowStart: Long, out: FloatArray) {
    var offset = rowStart.toInt()
    for (j in 0 until HIDDEN) {
      out[j] += Half.toFloat(table.getShort(offset))
      offset += BYTES_PER_HALF
    }
  }

  private fun setRow(table: ByteBuffer, rowStart: Long, out: FloatArray) {
    var offset = rowStart.toInt()
    for (j in 0 until HIDDEN) {
      out[j] = Half.toFloat(table.getShort(offset))
      offset += BYTES_PER_HALF
    }
  }

  /** Computes `out[0 until rows] = weights[rows, HIDDEN] @ x`, row-major from `base`. */
  private fun matVec(weights: FloatArray, base: Int, rows: Int, x: FloatArray, out: FloatArray) {
    var offset = base
    for (r in 0 until rows) {
      var acc = 0f
      for (j in 0 until HIDDEN) {
        acc += weights[offset] * x[j]
        offset++
      }
      out[r] = acc
    }
  }

  // ---- rotary position embedding (host side) ------------------------------

  private val invFreq =
    FloatArray(HEAD_DIM / 2) { (1.0 / Math.pow(ROPE_THETA, 2.0 * it / HEAD_DIM)).toFloat() }

  private fun ropeCos(position: Int) = FloatArray(HEAD_DIM).also { out ->
    for (j in 0 until HEAD_DIM / 2) {
      out[j] = cos((position * invFreq[j]).toDouble()).toFloat()
      out[j + HEAD_DIM / 2] = out[j]
    }
  }

  private fun ropeSin(position: Int) = FloatArray(HEAD_DIM).also { out ->
    for (j in 0 until HEAD_DIM / 2) {
      out[j] = sin((position * invFreq[j]).toDouble()).toFloat()
      out[j + HEAD_DIM / 2] = out[j]
    }
  }

  // ---- sampling -----------------------------------------------------------

  /**
  * Draws one audio token under classifier-free guidance.
  *
  * The guided logits `uncond + CFG_SCALE * (cond - uncond)` only *select* the candidate set
  * (the top [CFG_FILTER_K]), the draw itself is a temperature softmax over the **conditional**
  * logits restricted to that set. Set [maskBosPad] for codebook 0, whose vocabulary still
  * contains the unsampleable [AUDIO_BOS] and [AUDIO_PAD] ids.
  */
  private fun sampleGuided(
    cond: FloatArray,
    uncond: FloatArray,
    temp: Float,
    vocab: Int,
    maskBosPad: Boolean,
  ): Int {
    val guided = FloatArray(vocab) { uncond[it] + CFG_SCALE * (cond[it] - uncond[it]) }
    var candidates = (0 until vocab).sortedByDescending { guided[it] }.take(CFG_FILTER_K)
    if (maskBosPad) {
      candidates = candidates.filter { it != AUDIO_BOS && it != AUDIO_PAD }
    }
    if (candidates.isEmpty()) {
      return 0
    }
    var maxLogit = Float.NEGATIVE_INFINITY
    for (i in candidates) {
      if (cond[i] > maxLogit) {
        maxLogit = cond[i]
      }
    }
    val weights = DoubleArray(candidates.size)
    var total = 0.0
    for (k in candidates.indices) {
      weights[k] = exp(((cond[candidates[k]] - maxLogit) / temp).toDouble())
      total += weights[k]
    }
    var draw = random.nextDouble() * total
    for (k in candidates.indices) {
      draw -= weights[k]
      if (draw <= 0) return candidates[k]
    }
    return candidates[0]
  }

  /** Draws from the binary action head. Index 1 means "start a new word". */
  private fun sampleAction(logits: FloatArray, temp: Float): Int {
    val maxLogit = if (logits[0] > logits[1]) logits[0] else logits[1]
    val padWeight = exp(((logits[0] - maxLogit) / temp).toDouble())
    val newWordWeight = exp(((logits[1] - maxLogit) / temp).toDouble())
    return if (random.nextDouble() * (padWeight + newWordWeight) < padWeight) 0 else 1
  }

  /**
  * Host-side packed KV cache shaped `[layers * kvHeads, maxSteps, HEAD_DIM]`.
  *
  * The exported graphs take the cache as an input and append the current step at the tail, so
  * an additive mask of length `maxSteps + 1` hides the unwritten slots (the last entry is
  * always visible and corresponds to the current token).
  */
  private class Cache(val layers: Int, val kvHeads: Int, val maxSteps: Int) {
    val keys = FloatArray(layers * kvHeads * maxSteps * HEAD_DIM)
    val values = FloatArray(layers * kvHeads * maxSteps * HEAD_DIM)
    var length = 0
      private set

    fun reset() {
      keys.fill(0f)
      values.fill(0f)
      length = 0
    }

    fun mask() = FloatArray(maxSteps + 1) { MASK_NEG_INF }.also { m ->
      for (i in 0 until length) {
        m[i] = 0f
      }
      m[maxSteps] = 0f
    }

    fun append(newKeys: FloatArray, newValues: FloatArray) {
      for (head in 0 until layers * kvHeads) {
        val src = head * HEAD_DIM
        val dst = (head * maxSteps + length) * HEAD_DIM
        System.arraycopy(newKeys, src, keys, dst, HEAD_DIM)
        System.arraycopy(newValues, src, values, dst, HEAD_DIM)
      }
      length++
    }
  }

  // ---- graph runners ------------------------------------------------------

  private val temporalIn = temporal.createInputBuffers()
  private val temporalOut = temporal.createOutputBuffers()
  private val depIns = depGraphs.map { it.createInputBuffers() }
  private val depOuts = depGraphs.map { it.createOutputBuffers() }

  /**
   * Runs one temporal step, returning (hidden, action logits, codebook-0 logits) and
   * growing [cache].
   */
  private fun runTemporal(
    embedding: FloatArray,
    position: Int,
    cache: Cache,
  ): Triple<FloatArray, FloatArray, FloatArray> {
    temporalIn[0].writeFloat(embedding)
    temporalIn[1].writeFloat(ropeCos(position))
    temporalIn[2].writeFloat(ropeSin(position))
    temporalIn[3].writeFloat(cache.mask())
    temporalIn[4].writeFloat(cache.keys)
    temporalIn[5].writeFloat(cache.values)
    temporal.run(temporalIn, temporalOut)

    var hidden = FloatArray(0)
    var action = FloatArray(0)
    var codebook0 = FloatArray(0)
    var newKeys = FloatArray(0)
    var newValues = FloatArray(0)
    val kvSize = TEMPORAL_LAYERS * TEMPORAL_KV_HEADS * HEAD_DIM
    for (buffer in temporalOut) {
      val out = buffer.readFloat()
      when (out.size) {
        HIDDEN -> hidden = out
        2 -> action = out
        AUDIO_VOCAB -> codebook0 = out
        kvSize -> if (newKeys.isEmpty()) newKeys = out else newValues = out
      }
    }
    cache.append(newKeys, newValues)
    return Triple(hidden, action, codebook0)
  }

  /** One depformer stage on weight set [weightId] that returns its hidden state. */
  private fun runDepformer(
    weightId: Int,
    input: FloatArray,
    position: Int,
    cache: Cache,
  ): FloatArray {
    val inputs = depIns[weightId]
    val outputs = depOuts[weightId]
    inputs[0].writeFloat(input)
    inputs[1].writeFloat(ropeCos(position))
    inputs[2].writeFloat(ropeSin(position))
    inputs[3].writeFloat(cache.mask())
    inputs[4].writeFloat(cache.keys)
    inputs[5].writeFloat(cache.values)
    depGraphs[weightId].run(inputs, outputs)

    var hidden = FloatArray(0)
    var newKeys = FloatArray(0)
    var newValues = FloatArray(0)
    val kvSize = DEPFORMER_LAYERS * DEPFORMER_KV_HEADS * HEAD_DIM
    for (buffer in outputs) {
      val out = buffer.readFloat()
      when (out.size) {
        HIDDEN -> hidden = out
        kvSize -> if (newKeys.isEmpty()) newKeys = out else newValues = out
      }
    }
    cache.append(newKeys, newValues)
    return hidden
  }

  /**
  * Builds this frame's input embedding for both guidance branches.
  *
  * The 32 audio channels are shared, so they are summed once. The conditional branch adds the
  * real (main, second) text embeddings -- the second stream is dropped when it is [PAD] -- and
  * the unconditional branch substitutes the [ZERO] token with no second stream.
  */
  private fun buildEmbeddings(
    audioCodes: Array<IntArray>,
    frame: Int,
    mainToken: Int,
    secondToken: Int,
    audioSum: FloatArray,
    embeddingCond: FloatArray,
    embeddingUncond: FloatArray,
  ) {
    audioSum.fill(0f)
    for (cb in 0 until CODEBOOKS) {
      val code = if (AUDIO_DELAYS[cb] > frame) AUDIO_BOS else audioCodes[cb][frame]
      addRow(temporalAudio, rowOffset(cb.toLong() * AUDIO_VOCAB + code), audioSum)
    }
    setRow(combinedMain, rowOffset(mainToken.toLong()), embeddingCond)
    if (secondToken != PAD) {
      addRow(combinedSecond, rowOffset(secondToken.toLong()), embeddingCond)
    }
    setRow(combinedMain, rowOffset(ZERO.toLong()), embeddingUncond)
    for (j in 0 until HIDDEN) {
      embeddingCond[j] += audioSum[j]
      embeddingUncond[j] += audioSum[j]
    }
  }

  /** Writes the prefix's delayed code grid into [audioCodes], mirroring `delay_frames`. */
  private fun seedPrefixCodes(audioCodes: Array<IntArray>, prefix: VoicePrefix) {
    for (cb in 0 until CODEBOOKS) {
      val delay = AUDIO_DELAYS[cb]
      for (p in 0 until delay) {
        audioCodes[cb][p] = AUDIO_PAD
      }
      for (tau in 0 until prefix.frames) {
        audioCodes[cb][delay + tau] = prefix.aligned[cb][tau]
      }
    }
  }

  /** Generates dialogue audio for a script such as `"[S1] Hello. [S2] Hi."`. */
  fun synthesize(script: String, maxFrames: Int = DEFAULT_MAX_FRAMES): Result {
    val startNanos = System.nanoTime()
    val prefix = voicePrefix
    val stateMachine = StateMachine(prefix.entries + parseScript(script))

    // Two branches for classifier-free guidance: conditional (the real text) and
    // unconditional (text forced to zero/pad). Each needs its own KV caches.
    val temporalCacheCond = Cache(TEMPORAL_LAYERS, TEMPORAL_KV_HEADS, TEMPORAL_MAX_STEPS)
    val temporalCacheUncond = Cache(TEMPORAL_LAYERS, TEMPORAL_KV_HEADS, TEMPORAL_MAX_STEPS)
    val depCacheCond = Cache(DEPFORMER_LAYERS, DEPFORMER_KV_HEADS, DEPFORMER_DEPTH)
    val depCacheUncond = Cache(DEPFORMER_LAYERS, DEPFORMER_KV_HEADS, DEPFORMER_DEPTH)

    // audioCodes[cb][t] holds the token sampled at frame t - 1, in the delayed grid.
    val audioCodes = Array(CODEBOOKS) { IntArray(AUDIO_BUFFER_FRAMES) { AUDIO_BOS } }
    seedPrefixCodes(audioCodes, prefix)
    var mainToken = BOS
    var secondToken = PAD

    val audioSum = FloatArray(HIDDEN)
    val embeddingCond = FloatArray(HIDDEN)
    val embeddingUncond = FloatArray(HIDDEN)
    val depInputCond = FloatArray(HIDDEN)
    val depInputUncond = FloatArray(HIDDEN)
    val depLogitsCond = FloatArray(AUDIO_VOCAB)
    val depLogitsUncond = FloatArray(AUDIO_VOCAB)

    // Warm-up: replay the voice prompt so both KV caches carry its speaker identity. Only
    // the temporal transformer runs, nothing is sampled and no audio is produced.
    for (t in 0 until prefix.frames) {
      buildEmbeddings(audioCodes, t, mainToken, secondToken, audioSum,
        embeddingCond, embeddingUncond)
      runTemporal(embeddingCond, t, temporalCacheCond)
      runTemporal(embeddingUncond, t, temporalCacheUncond)
      val forced = if (t in prefix.newWordSteps) NEW_WORD else PAD
      val next = stateMachine.process(t, forced, isForced = true)
      mainToken = next.first
      secondToken = next.second
    }

    // The prompt's last frame is re-entered as the first generated frame, as upstream does.
    val startStep = maxOf(prefix.frames - 1, 0)
    val frameBudget = minOf(maxFrames, TEMPORAL_MAX_STEPS - prefix.frames - 1)
    var endStep = -1
    var frame = startStep

    while (frame < startStep + frameBudget) {
      buildEmbeddings(audioCodes, frame, mainToken, secondToken, audioSum,
        embeddingCond, embeddingUncond)

      val (hiddenCond, action, codebook0Cond) =
        runTemporal(embeddingCond, frame, temporalCacheCond)
      val (hiddenUncond, _, codebook0Uncond) =
        runTemporal(embeddingUncond, frame, temporalCacheUncond)

      // Guidance is a no-op on the action head: its vocabulary of 2 is below CFG_FILTER_K.
      val nextTokens = stateMachine.process(frame, sampleAction(action, TEXT_TEMP))
      mainToken = nextTokens.first
      secondToken = nextTokens.second
      if (stateMachine.endStep >= 0 && endStep < 0) {
        endStep = stateMachine.endStep
      }

      val codebook0 =
        sampleGuided(codebook0Cond, codebook0Uncond, AUDIO_TEMP, AUDIO_VOCAB, true)
      audioCodes[0][frame + 1] = codebook0
      depCacheCond.reset()
      depCacheUncond.reset()

      var previousCode = codebook0
      for (stage in 0 until DEPFORMER_DEPTH) {
        val weightId = WEIGHTS_SCHEDULE[stage]
        val weightBase = weightId * HIDDEN * HIDDEN
        val audioRow = rowOffset(stage.toLong() * AUDIO_VOCAB + previousCode)
        matVec(depIn, weightBase, HIDDEN, hiddenCond, depInputCond)
        addRow(depAudio, audioRow, depInputCond)
        matVec(depIn, weightBase, HIDDEN, hiddenUncond, depInputUncond)
        addRow(depAudio, audioRow, depInputUncond)

        val depHiddenCond = runDepformer(weightId, depInputCond, stage, depCacheCond)
        val depHiddenUncond = runDepformer(weightId, depInputUncond, stage, depCacheUncond)
        val logitsBase = stage * AUDIO_VOCAB * HIDDEN
        matVec(depLogits, logitsBase, AUDIO_VOCAB_LIMIT, depHiddenCond, depLogitsCond)
        matVec(depLogits, logitsBase, AUDIO_VOCAB_LIMIT, depHiddenUncond, depLogitsUncond)

        previousCode = sampleGuided(
          depLogitsCond, depLogitsUncond, AUDIO_TEMP, AUDIO_VOCAB_LIMIT, false)
        audioCodes[stage + 1][frame + 1] = previousCode
      }

      frame++
      if (endStep >= 0 && frame >= endStep + FLUSH_TAIL) {
        break
      }
    }

    // With a second stream the main stream never emits NEW_WORD, so upstream's
    // `first_word_frame` stays unset and the crop is exactly the generation start: this
    // drops the voice prompt from the rendered audio.
    val audio = decodeMimi(audioCodes, frame, startStep)
    return Result(audio, frame - startStep, (System.nanoTime() - startNanos) / 1_000_000)
  }

  /**
  * Undelays the codebooks and decodes them to 24 kHz PCM in a single Mimi pass.
  *
  * Mimi's decode path (upsample, causal decoder transformer, SEANet) attends over the whole
  * prefix, so its receptive field is unbounded: decoding disjoint windows would start each one
  * with no history, costing ~13% relative error. The graph therefore spans [DECODE_FRAMES]
  * frames and the unused tail stays zeroed. Causality makes that exact for every real frame
  * (correlation 0.999999 against a torch full-sequence decode).
  */
  private fun decodeMimi(audioCodes: Array<IntArray>, frames: Int, crop: Int): FloatArray {
    // Undelay: aligned[cb][tau] = audioCodes[cb][AUDIO_DELAYS[cb] + tau], then drop `crop`
    // leading frames (the voice prompt).
    val total = (frames + 1) - MAX_DELAY
    val skip = if (crop in 0 until total) crop else 0
    val target = (total - skip).coerceIn(1, DECODE_FRAMES)

    val dequantIn = mimiDequant.createInputBuffers()
    val dequantOut = mimiDequant.createOutputBuffers()
    val latents = FloatArray(MIMI_DIM * DECODE_FRAMES)
    val frameCodes = FloatArray(CODEBOOKS)
    for (tau in 0 until target) {
      for (cb in 0 until CODEBOOKS) {
        frameCodes[cb] = audioCodes[cb][AUDIO_DELAYS[cb] + skip + tau].toFloat()
      }
      dequantIn[0].writeFloat(frameCodes)
      mimiDequant.run(dequantIn, dequantOut)
      val latent = dequantOut[0].readFloat()
      for (k in 0 until MIMI_DIM) {
        latents[k * DECODE_FRAMES + tau] = latent[k]
      }
    }

    val decodeIn = mimiDecode.createInputBuffers()
    val decodeOut = mimiDecode.createOutputBuffers()
    decodeIn[0].writeFloat(latents)
    mimiDecode.run(decodeIn, decodeOut)
    return decodeOut[0].readFloat().copyOf(target * SAMPLES_PER_FRAME)
  }

  // ---- script parsing -----------------------------------------------------

  /** One script word: its text tokens plus the padding frames that must follow it. */
  private class Entry(val tokens: IntArray, val padding: Int)

  /**
  * A precomputed two-speaker voice prompt.
  *
  * Dia2 samples the speaker when no prefix is given, so the voice changes every run.
  * Building a prefix normally needs Whisper word timings and a Mimi *encoder*, both done
  * offline by `scripts/bake_prefix.py`, leaving the device only [warmUpWithPrefix], which
  * replays the prompt through the temporal transformer to prime its KV cache.
  */
  private class VoicePrefix(
    val frames: Int,
    val aligned: Array<IntArray>,
    val newWordSteps: Set<Int>,
    val entries: List<Entry>,
  )

  private fun loadVoicePrefix(context: Context): VoicePrefix {
    val text = context.assets.open("dia2_prefix.json").bufferedReader().use { it.readText() }
    val root = org.json.JSONObject(text)
    val frames = root.getInt("frames")
    val alignedJson = root.getJSONArray("aligned")
    val aligned = Array(CODEBOOKS) { cb ->
      val row = alignedJson.getJSONArray(cb)
      IntArray(frames) { row.getInt(it) }
    }
    val stepsJson = root.getJSONArray("new_word_steps")
    val steps = HashSet<Int>(stepsJson.length())
    for (i in 0 until stepsJson.length()) {
      steps.add(stepsJson.getInt(i))
    }
    val entriesJson = root.getJSONArray("entries")
    val entries = ArrayList<Entry>(entriesJson.length())
    for (i in 0 until entriesJson.length()) {
      val entry = entriesJson.getJSONObject(i)
      val tokens = entry.getJSONArray("tokens")
      entries.add(Entry(IntArray(tokens.length()) { tokens.getInt(it) },
        entry.getInt("padding")))
    }
    return VoicePrefix(frames, aligned, steps, entries)
  }

  /**
  * Splits a script into per-word entries, mirroring Dia2's `parse_script`.
  *
  * A word introduced by `[S1]`/`[S2]` is encoded as `"[S1] word"`, so the speaker id is
  * followed by a *space-prefixed* word. A bare word carries no leading space and tokenizes
  * differently. Getting this wrong feeds the model a different sentence.
  */
  private fun parseScript(script: String): List<Entry> {
    val entries = ArrayList<Entry>()
    var pendingSpeaker = -1
    var firstContent = true
    val normalized = script.replace('’', '\'').replace(":", " ")
    for (rawWord in normalized.split(Regex("\\s+"))) {
      val word = rawWord.trim()
      if (word.isEmpty()) {
        continue
      }
      if (word == "[S1]") {
        pendingSpeaker = SPEAKER_1
        continue
      }
      if (word == "[S2]") {
        pendingSpeaker = SPEAKER_2
        continue
      }
      var speaker = pendingSpeaker
      if (speaker == -1 && firstContent) {
        speaker = SPEAKER_1
      }
      val tokens = if (speaker != -1) {
        intArrayOf(speaker) + tokenizer.encodeText(" $word")
      } else {
        tokenizer.encodeText(word)
      }
      // padding = max(0, paddingBetween + tokens - 1), with paddingBetween = 1.
      entries.add(Entry(tokens, tokens.size))
      pendingSpeaker = -1
      firstContent = false
    }
    return entries
  }

  // ---- word pacing --------------------------------------------------------

  /**
  * Paces script words onto the two text streams, following Dia2's `StateMachine`.
  *
  * The action head only says *when* a word starts. This class decides *what* each stream
  * carries: on a new word the main stream emits the word's first text token while the second
  * stream emits [NEW_WORD]. During the padding frames that follow, the main stream drains the
  * rest of that word and the second stream drains a [SECOND_AHEAD]-word lookahead. Both
  * streams therefore carry real text-token ids, never bare markers.
  */
  private inner class StateMachine(entries: List<Entry>) {
    private val queue = ArrayDeque(entries)
    private val pending = ArrayDeque<Int>()
    private val lookahead = ArrayDeque<Int>()
    private var paddingBudget = INITIAL_PADDING
    private var forcedPadding = INITIAL_PADDING

    /** Frame at which the script ran out, or -1 while words remain. */
    var endStep = -1
      private set

    /** Tokens of the `count`-th upcoming entry that still has text. */
    private fun peek(count: Int): IntArray {
      var remaining = count
      for (entry in queue) {
        if (entry.tokens.isNotEmpty()) {
          remaining--
          if (remaining == 0) return entry.tokens
        }
      }
      return IntArray(0)
    }

    /**
    * Advances one frame and returns the (main, second) tokens for the next model step.
    *
    * [action] is either a sample from the binary action head (0 = pad, 1 = new word) or,
    * during warm-up, a forced [NEW_WORD] / [PAD] id. Set [isForced] to replay a known word
    * schedule: the pacing constraints are then skipped, except that a half-emitted word
    * still holds the stream.
    */
    fun process(step: Int, action: Int, isForced: Boolean = false): Pair<Int, Int> {
      var token = when (action) {
        1, NEW_WORD -> NEW_WORD
        else -> PAD
      }
      // Mid-word and forced padding both suppress a new word, an exhausted budget forces one.
      if (pending.isNotEmpty()) {
        token = PAD
      } else if (isForced) {
        // Keep the caller's token as-is.
      } else if (forcedPadding > 0) {
        token = PAD
      } else if (paddingBudget <= 0 && token != NEW_WORD) {
        token = NEW_WORD
      }

      if (token == NEW_WORD) {
        if (queue.isNotEmpty()) {
          val entry = queue.removeFirst()
          if (entry.tokens.isNotEmpty()) {
            for (id in entry.tokens) {
              pending.addLast(id)
            }
            for (id in peek(SECOND_AHEAD)) {
              lookahead.addLast(id)
            }
            paddingBudget = MAX_PADDING
          } else {
            token = PAD
          }
          forcedPadding = entry.padding
        } else {
          // One last new_word flushes the tail, then the script is over.
          token = if (endStep < 0) NEW_WORD else PAD
          if (endStep < 0) {
            endStep = step
          }
        }
      }

      var main = when (token) {
        PAD -> {
          if (paddingBudget > 0) paddingBudget--
          if (forcedPadding > 0) forcedPadding--
          if (pending.isNotEmpty()) pending.removeFirst() else PAD
        }
        else -> token
      }

      val second: Int
      if (main == NEW_WORD) {
        second = NEW_WORD
        main = if (pending.isNotEmpty()) pending.removeFirst() else PAD
      } else if (lookahead.isNotEmpty()) {
        second = lookahead.removeFirst()
      } else {
        second = PAD
      }
      return Pair(main, second)
    }
  }

  override fun close() {
    temporal.close()
    depGraphs.forEach { it.close() }
    mimiDequant.close()
    mimiDecode.close()
  }
}
