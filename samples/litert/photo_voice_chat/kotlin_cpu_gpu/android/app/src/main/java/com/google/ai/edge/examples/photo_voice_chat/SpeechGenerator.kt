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

package com.google.ai.edge.examples.photo_voice_chat

import java.util.SplittableRandom
import kotlin.math.exp

/** The model's token ids and constants (LiquidAI/LFM3-Omni-2B: configuration and tokenizer.json). */
object Tokens {
  const val BOS = 124894
  const val IM_END = 124900
  const val VISION_START = 125009
  const val VISION_END = 125010
  const val IMAGE = 125030
  const val VIDEO = 125029
  const val AUDIO = 125031
  const val AUDIO_OUT = 125032
  const val AUDIO_START = 125012
  const val TEXT_END = 125014
  const val MIXED_START = 125015
  val EOS = intArrayOf(IM_END, TEXT_END)
  /** Placeholders never sampled in a text slot (the model's _placeholder_token_ids). */
  val NEVER_SAMPLED = intArrayOf(AUDIO, AUDIO_OUT, IMAGE, VIDEO)
  const val NUM_CODEBOOKS = 12
  const val CODEBOOK_SIZE = 2017
  const val AUDIO_EOS_CODE = 2016
  val MASKGIT_SCHEDULE = intArrayOf(4, 4, 4)
  const val INTERLEAVED_TEXT = 6
  const val INTERLEAVED_AUDIO = 12
}

/** The sampling settings of the model card (speech: temperature 0.4, repetition penalty 1.2 over the last 64 frames, no
 *  top-k; text greedy). Temperature 0 = argmax after the penalty; top-k 0 = no top-k. */
data class Sampling(
  val audioTemperature: Float = 0.4f,
  val audioTopK: Int = 0,
  val audioPenalty: Float = 1.2f,
  val audioWindow: Int = 64,
  val textTemperature: Float = 0f,
  val textPenalty: Float = 1.0f,
  val textWindow: Int = 32,
)

/** The random draws of the sampler and the maskgit order; one stream per answer, from its seed. */
class Rng(seed: Long) {
  private val r = SplittableRandom(seed)

  fun nextDouble(): Double = r.nextDouble()

  /** A permutation of 0 until n (Fisher-Yates from the end, as numpy's shuffle). */
  fun permutation(n: Int): IntArray {
    val p = IntArray(n) { it }
    for (i in n - 1 downTo 1) {
      val j = r.nextInt(i + 1)
      val t = p[i]
      p[i] = p[j]
      p[j] = t
    }
    return p
  }
}

object Sampler {
  /**
   * One draw from logits[offset until offset + n] in the vendor order: repetition penalty over the ids with a
   * positive [counts] entry (divide a positive logit, multiply a negative one) -> suppressed ids to -inf -> argmax when
   * [temperature] is 0 -> temperature -> top-k (ties with the k-th kept) -> softmax -> one draw.
   */
  fun sample(
    logits: FloatArray, offset: Int, n: Int, temperature: Float, topK: Int, penalty: Float, counts: IntArray?,
    suppress: IntArray?, rng: Rng,
  ): Int = sampleAt(logits, offset, n, temperature, topK, penalty, counts, suppress, if (temperature == 0f) 0.0 else rng.nextDouble())

  /** [sample] with its one uniform draw [u01] given (unused at temperature 0, where [sample] draws nothing). */
  fun sampleAt(
    logits: FloatArray, offset: Int, n: Int, temperature: Float, topK: Int, penalty: Float, counts: IntArray?,
    suppress: IntArray?, u01: Double,
  ): Int {
    val s = FloatArray(n) { logits[offset + it] }
    if (counts != null && penalty != 1.0f) {
      for (i in 0 until n) if (counts[i] > 0) s[i] = if (s[i] < 0f) s[i] * penalty else s[i] / penalty
    }
    if (suppress != null) for (id in suppress) if (id in 0 until n) s[id] = Float.NEGATIVE_INFINITY
    if (temperature == 0f) return argmax(s)
    for (i in 0 until n) s[i] = s[i] / temperature
    if (topK in 1 until n) {
      val sorted = s.copyOf()
      sorted.sort()
      val kth = sorted[n - topK]
      for (i in 0 until n) if (s[i] < kth) s[i] = Float.NEGATIVE_INFINITY
    }
    var max = Float.NEGATIVE_INFINITY
    for (v in s) if (v > max) max = v
    val p = DoubleArray(n)
    var sum = 0.0
    for (i in 0 until n) {
      val e = if (s[i] == Float.NEGATIVE_INFINITY) 0.0 else exp((s[i] - max).toDouble())
      p[i] = e
      sum += e
    }
    val u = u01 * sum
    var acc = 0.0
    var last = -1
    for (i in 0 until n) {
      if (p[i] == 0.0) continue
      acc += p[i]
      last = i
      if (u < acc) return i
    }
    return last
  }

  fun argmax(s: FloatArray): Int {
    var best = 0
    for (i in 1 until s.size) if (s[i] > s[best]) best = i
    return best
  }
}

/**
 * [Sampler.sampleAt]'s draw, the same id for the same inputs, without its per-call allocations and its sort of the
 * whole row: the k-th largest value by selection (in the sort's own order, so the very value [Sampler] reads at
 * sorted[n - k]), then the softmax over the kept ids only. The ids [Sampler] sets to -inf add exp(-inf) = 0 to its sum
 * and are skipped by its draw, so the sum and the walk below see the same numbers in the same order.
 * One instance per thread (scratch arrays).
 */
class FastSampler(capacity: Int) {
  private val s = FloatArray(capacity)
  private val t = FloatArray(capacity)
  private val p = DoubleArray(capacity)

  fun sampleAt(
    logits: FloatArray, offset: Int, n: Int, temperature: Float, topK: Int, penalty: Float, counts: IntArray?,
    suppress: IntArray?, u01: Double,
  ): Int {
    for (i in 0 until n) s[i] = logits[offset + i]
    if (counts != null && penalty != 1.0f) {
      for (i in 0 until n) if (counts[i] > 0) s[i] = if (s[i] < 0f) s[i] * penalty else s[i] / penalty
    }
    if (suppress != null) for (id in suppress) if (id in 0 until n) s[id] = Float.NEGATIVE_INFINITY
    if (temperature == 0f) {
      var best = 0
      for (i in 1 until n) if (s[i] > s[best]) best = i
      return best
    }
    for (i in 0 until n) s[i] = s[i] / temperature
    val cut = topK in 1 until n
    var kth = Float.NEGATIVE_INFINITY
    if (cut) {
      System.arraycopy(s, 0, t, 0, n)
      kth = select(t, n, n - topK)
    }
    var max = Float.NEGATIVE_INFINITY
    for (i in 0 until n) {
      val v = s[i]
      if (cut && v < kth) continue
      if (v > max) max = v
    }
    var sum = 0.0
    for (i in 0 until n) {
      val v = s[i]
      val e = if ((cut && v < kth) || v == Float.NEGATIVE_INFINITY) 0.0 else exp((v - max).toDouble())
      p[i] = e
      sum += e
    }
    val u = u01 * sum
    var acc = 0.0
    var last = -1
    for (i in 0 until n) {
      if (p[i] == 0.0) continue
      acc += p[i]
      last = i
      if (u < acc) return i
    }
    return last
  }

  /** The value at index [k] of a[0 until n] sorted ascending in [java.lang.Float.compare]'s order (FloatArray.sort's). */
  private fun select(a: FloatArray, n: Int, k: Int): Float {
    var lo = 0
    var hi = n - 1
    while (hi > lo) {
      val mid = (lo + hi) ushr 1
      // median of three as the pivot
      if (cmp(a[mid], a[lo]) < 0) swap(a, mid, lo)
      if (cmp(a[hi], a[lo]) < 0) swap(a, hi, lo)
      if (cmp(a[hi], a[mid]) < 0) swap(a, hi, mid)
      val pivot = a[mid]
      var i = lo
      var j = hi
      while (i <= j) {
        while (cmp(a[i], pivot) < 0) i++
        while (cmp(a[j], pivot) > 0) j--
        if (i <= j) {
          swap(a, i, j)
          i++
          j--
        }
      }
      when {
        k <= j -> hi = j
        k >= i -> lo = i
        else -> return a[k]
      }
    }
    return a[k]
  }

  private fun cmp(x: Float, y: Float): Int = java.lang.Float.compare(x, y)

  private fun swap(a: FloatArray, i: Int, j: Int) {
    val v = a[i]
    a[i] = a[j]
    a[j] = v
  }
}

/**
 * The vendor `_generate` state machine for one answer (Lfm3OmniForConditionalGeneration, LiquidAI/LFM3-Omni-2B):
 *   TEXT         one token per slot; `<|audio_start|>` turns to AUDIO, `<|mixed_start|>` to INTERLEAVED, `<|im_end|>`
 *                or `<|text_end|>` ends the answer
 *   AUDIO        one speech frame per slot: the LM hidden -> the depth decoder (3 maskgit passes) -> 12 codes; the next
 *                LM input is the frame embedding (the sum of the 12 codebook embeddings), not a token; a frame holding
 *                the end-of-audio code (2016) returns to TEXT
 *   INTERLEAVED  blocks of 6 text slots and 12 speech frames until `<|text_end|>`, then speech until end-of-audio
 * Sampling = [Sampling]; the maskgit order of a frame = a permutation of the 12 codebooks from [Rng], or the next row
 * of a recorded order (`replay`) while it lasts.
 * A conversation goes on in the language model's state ([continueState]): the next turn's prompt rows are prefilled
 * after the last answer, as the vendor's `generate` continues from `past_key_values`. The loop ends before the last
 * slot's input goes in; [Result.pending] is that input, the first row the next turn feeds.
 */
class SpeechGenerator(
  private val lm: LmRunner,
  private val depth: DepthDecoder,
  private val frame: FrameEmbed,
  /** The speech codes' sampler: fast ([FastSampler]), ref ([Sampler], sorts every row) or check (both on every draw,
   *  the reference's id kept, [samplerMismatches] counts the draws where the two differ). Text slots use [Sampler]. */
  private val samplerMode: String = "fast",
  /** The language model's text head: text = only on the steps before a text slot (a speech slot's step runs without
   *  it, LmRunner.step), all = every step (the vendor's loop computes it only for text slots too). */
  private val head: String = "text",
) {
  enum class Mode { TEXT, AUDIO, INTERLEAVED }

  private val fast = FastSampler(Tokens.CODEBOOK_SIZE)
  /** Time in the speech codes' sampling (12 draws a frame), and the check mode's count of differing draws. */
  var sampleNs = 0L
    private set
  var samplerMismatches = 0L
    private set
  var samplerDraws = 0L
    private set

  class Result(
    val tokens: List<Int>,
    /** Speech frames in order (with the end-of-audio frame when asked); 12 codes each. */
    val frames: List<IntArray>,
    /** The maskgit order of every sampled frame (the end-of-audio frame included). */
    val orders: List<IntArray>,
    val finished: Boolean,
    val stopped: Boolean,
    val prefillNs: Long,
    val firstSlotNs: Long,
    /** The input of the last slot (its token's or its frame's embedding), not fed yet; null if nothing was generated. */
    val pending: FloatArray?,
    /** The last text token sampled (-1 when the answer ended on a speech frame). */
    val lastToken: Int,
  )

  interface Listener {
    fun onToken(token: Int) {}
    fun onFrame(codes: IntArray) {}
  }

  fun generate(
    promptRows: List<FloatArray>,
    initialMode: Mode,
    explicitTextOnly: Boolean,
    maxNewTokens: Int,
    sampling: Sampling,
    rng: Rng,
    replay: List<IntArray>? = null,
    returnAudioEos: Boolean = false,
    listener: Listener? = null,
    stop: () -> Boolean = { false },
    /** Prefill [promptRows] after the conversation so far (the state as the last call left it) instead of from 0. */
    continueState: Boolean = false,
  ): Result {
    val t0 = System.nanoTime()
    var hidden = if (continueState) lm.extend(promptRows, decodeLast = true)!! else lm.prefill(promptRows)
    val prefillNs = System.nanoTime() - t0
    var firstSlotNs = -1L
    var mode = initialMode
    val suppress = Tokens.NEVER_SAMPLED
    val textOnlySuppress = Tokens.NEVER_SAMPLED + intArrayOf(Tokens.MIXED_START, Tokens.AUDIO_START)
    val tokens = ArrayList<Int>()
    val frames = ArrayList<IntArray>()
    val orders = ArrayList<IntArray>()
    val textHist = ArrayDeque<Int>()
    val textCounts = HashMap<Int, Int>()
    val audioHist = ArrayDeque<IntArray>()
    val audioCounts = Array(Tokens.NUM_CODEBOOKS) { IntArray(Tokens.CODEBOOK_SIZE) }
    var replayIndex = 0
    var slotsLeft = Tokens.INTERLEAVED_TEXT
    var speaking = false
    var textDone = false
    var finished = false
    var stopped = false
    var pending: FloatArray? = null
    var lastToken = -1
    for (step in 0 until maxNewTokens) {
      if (stop()) {
        stopped = true
        break
      }
      val interleaving = mode == Mode.INTERLEAVED
      val next: FloatArray
      if (mode == Mode.AUDIO || (interleaving && speaking)) {
        val order = if (replay != null && replayIndex < replay.size) replay[replayIndex++] else rng.permutation(Tokens.NUM_CODEBOOKS)
        var codes = sampleFrame(hidden, order, sampling, audioCounts, rng)
        orders.add(order)
        val isEos = codes.any { it == Tokens.AUDIO_EOS_CODE }
        if (isEos) {
          codes = IntArray(Tokens.NUM_CODEBOOKS) { Tokens.AUDIO_EOS_CODE }
          mode = Mode.TEXT
        } else {
          audioHist.addLast(codes)
          for (k in 0 until Tokens.NUM_CODEBOOKS) audioCounts[k][codes[k]]++
          if (audioHist.size > sampling.audioWindow) {
            val old = audioHist.removeFirst()
            for (k in 0 until Tokens.NUM_CODEBOOKS) audioCounts[k][old[k]]--
          }
          listener?.onFrame(codes)
        }
        if (!isEos || returnAudioEos) {
          frames.add(codes)
          tokens.add(Tokens.AUDIO_OUT)
        }
        next = frame.embed(codes)
        lastToken = -1
      } else {
        val logits = lm.textLogits()
        // the penalty over the last textWindow tokens (the README's 1.0 = none)
        val counts = if (sampling.textPenalty != 1.0f && textCounts.isNotEmpty()) {
          IntArray(logits.size).also { c -> for ((id, n) in textCounts) c[id] = n }
        } else null
        val token = Sampler.sample(logits, 0, logits.size, sampling.textTemperature, 0, sampling.textPenalty, counts,
          if (explicitTextOnly) textOnlySuppress else suppress, rng)
        textHist.addLast(token)
        textCounts[token] = (textCounts[token] ?: 0) + 1
        if (textHist.size > sampling.textWindow) {
          val old = textHist.removeFirst()
          val c = textCounts.getValue(old) - 1
          if (c == 0) textCounts.remove(old) else textCounts[old] = c
        }
        tokens.add(token)
        lastToken = token
        listener?.onToken(token)
        if (token == Tokens.TEXT_END) {
          if (interleaving) {
            textDone = true
            speaking = true
          } else {
            finished = true
          }
        } else if (token in Tokens.EOS) {
          finished = true
        } else if (interleaving) {
          // a text slot of the interleaved answer
        } else if (token == Tokens.AUDIO_START) {
          mode = Mode.AUDIO
        } else if (token == Tokens.MIXED_START) {
          mode = Mode.INTERLEAVED
          slotsLeft = Tokens.INTERLEAVED_TEXT
          speaking = false
          textDone = false
        }
        next = lm.embedToken(token)
      }
      if (interleaving && !textDone) {
        slotsLeft--
        if (slotsLeft == 0) {
          speaking = !speaking
          slotsLeft = if (speaking) Tokens.INTERLEAVED_AUDIO else Tokens.INTERLEAVED_TEXT
        }
      }
      if (firstSlotNs < 0) firstSlotNs = System.nanoTime() - t0
      pending = next
      if (finished || step == maxNewTokens - 1) break
      // the next slot's kind is known here (the mode and the interleaved turn are already updated)
      val nextIsSpeech = mode == Mode.AUDIO || (mode == Mode.INTERLEAVED && speaking)
      hidden = lm.step(next, logits = head != "text" || !nextIsSpeech)
      pending = null
    }
    return Result(tokens, frames, orders, finished, stopped, prefillNs, firstSlotNs, pending, lastToken)
  }

  /** One frame: the hidden state in once, then the schedule's passes, each revealing its codebooks in [order]. */
  private fun sampleFrame(hidden: FloatArray, order: IntArray, s: Sampling, counts: Array<IntArray>, rng: Rng): IntArray {
    val codes = IntArray(Tokens.NUM_CODEBOOKS)
    val masked = FloatArray(Tokens.NUM_CODEBOOKS) { 1f }
    depth.project(hidden)
    var start = 0
    for (count in Tokens.MASKGIT_SCHEDULE) {
      val logits = depth.pass(codes, masked)
      val t0 = System.nanoTime()
      for (j in start until start + count) {
        val k = order[j]
        codes[k] = draw(logits, k * Tokens.CODEBOOK_SIZE, s, counts[k], rng)
      }
      sampleNs += System.nanoTime() - t0
      for (j in start until start + count) masked[order[j]] = 0f
      start += count
    }
    return codes
  }

  /** One codebook's code; every mode takes one uniform draw per sampled code (none at temperature 0), in this order. */
  private fun draw(logits: FloatArray, offset: Int, s: Sampling, counts: IntArray, rng: Rng): Int {
    val n = Tokens.CODEBOOK_SIZE
    samplerDraws++
    if (samplerMode == "ref") return Sampler.sample(logits, offset, n, s.audioTemperature, s.audioTopK, s.audioPenalty, counts, null, rng)
    val u = if (s.audioTemperature == 0f) 0.0 else rng.nextDouble()
    val f = fast.sampleAt(logits, offset, n, s.audioTemperature, s.audioTopK, s.audioPenalty, counts, null, u)
    if (samplerMode != "check") return f
    val r = Sampler.sampleAt(logits, offset, n, s.audioTemperature, s.audioTopK, s.audioPenalty, counts, null, u)
    if (r != f) samplerMismatches++
    return r
  }
}
