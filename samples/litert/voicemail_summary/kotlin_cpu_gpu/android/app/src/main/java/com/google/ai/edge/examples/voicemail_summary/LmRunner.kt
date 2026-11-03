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

package com.google.ai.edge.examples.voicemail_summary

import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.File

/**
 * The language model of the bundle, driven step by step: the decoder section (signatures `prefill_<n>` and `decode`,
 * input `embeddings` + `input_pos` + `mask`, output `hidden` = the last norm's output and, on `decode`, `logits`) and
 * the single-token embedder section (`token_ids` -> `embeddings`). The loop feeds the graph what LiteRT-LM feeds it:
 *   input_pos  absolute positions
 *   mask       [1, 1, T, C]: -0.7 * FLT_MAX everywhere, row i open on [0, start + i]
 *   state      functional: every call returns the whole state (`kv_cache_*`). [inPlace] (LiteRT-LM's CPU policy):
 *              one buffer per state tensor, the input and the output of every call, so each cache update writes only
 *              its new row; otherwise two buffer sets swap roles each call (ping-pong, LiteRT-LM's GPU policy for
 *              this graph), and every update first copies the whole cache from the input to the output buffer (the
 *              CPU's DYNAMIC_UPDATE_SLICE copies when the two differ: 22 caches x 4 MiB a step here)
 * A prompt is prefilled in exact-cover chunks of the prefill signatures (largest first) and its last row goes through
 * `decode`, so the prompt's last hidden state and logits come from the decode signature, as in LiteRT-LM.
 */
class LmRunner(
  decoderFile: File,
  embedderFile: File,
  options: CompiledModel.Options,
  embedderOptions: CompiledModel.Options,
  env: Environment,
  /** The decoder's prefill lengths and state inputs as the bundle's section map lists them; null = look them up. */
  knownPrefill: IntArray? = null,
  knownState: List<String>? = null,
  val inPlace: Boolean = true,
) : AutoCloseable {
  private val decoder = CompiledModel.create(decoderFile.absolutePath, options, env)
  private val embedder = CompiledModel.create(embedderFile.absolutePath, embedderOptions, env)

  val prefillLengths: IntArray = knownPrefill?.sortedArrayDescending()
    ?: PREFILL_CANDIDATES.filter { hasInput("embeddings", "prefill_$it") }.toIntArray()
  val stateNames: List<String> = knownState ?: (0 until 64).flatMap { i ->
    listOf("kv_cache_c_$i", "kv_cache_k_$i", "kv_cache_v_$i").filter { hasInput(it, "decode") }
  }
  private val stateSizes: Map<String, Int> = stateNames.associateWith { elements(decoder.getInputTensorType(it, "decode")) }
  val hiddenSize: Int = dims(decoder.getInputTensorType("embeddings", "decode")).last()
  val cacheLength: Int = dims(decoder.getInputTensorType("mask", "decode")).last()

  /** The state sets (one in place, two for ping-pong); [set] ([cur]) holds the state after the latest call. */
  private val sets: Array<Map<String, TensorBuffer>> =
    Array(if (inPlace) 1 else 2) { stateNames.associateWith { decoder.createOutputBuffer(it, "decode") } }
  private var cur = 0
  private fun set(i: Int) = sets[i % sets.size]
  private val zeros = HashMap<Int, FloatArray>()

  private val decEmb = decoder.createInputBuffer("embeddings", "decode")
  private val decPos = decoder.createInputBuffer("input_pos", "decode")
  private val decMask = decoder.createInputBuffer("mask", "decode")
  private val decHidden = decoder.createOutputBuffer("hidden", "decode")
  private val decLogits = decoder.createOutputBuffer("logits", "decode")
  private val decMaskRow = FloatArray(cacheLength) { MASK_FILL }

  /** The decode call's two maps for each parity of [cur] (built once: no allocation per step). */
  private val decodeMaps = Array(2) { parity ->
    val inputs = HashMap<String, TensorBuffer>(stateNames.size + 3)
    inputs["embeddings"] = decEmb
    inputs["input_pos"] = decPos
    inputs["mask"] = decMask
    inputs.putAll(set(parity))
    val outputs = HashMap<String, TensorBuffer>(stateNames.size + 2)
    outputs["hidden"] = decHidden
    outputs["logits"] = decLogits
    outputs.putAll(set(1 - parity))
    Pair(inputs, outputs)
  }

  private class Prefill(val emb: TensorBuffer, val pos: TensorBuffer, val mask: TensorBuffer, val hidden: TensorBuffer, val maskArr: FloatArray)
  private val prefills = HashMap<Int, Prefill>()

  private val embIn = embedder.createInputBuffer("token_ids", "embedder")
  private val embOut = embedder.createOutputBuffer("embeddings", "embedder")

  /** Next position to write. */
  var pos = 0
    private set
  var decodeCalls = 0L
    private set
  var prefillCalls = 0L
    private set
  /** Wall time of the decoder's calls (write + run + read), and of the embedder's; the decode calls' parts. */
  var decodeNs = 0L
    private set
  var prefillNs = 0L
    private set
  var embedNs = 0L
    private set
  var decodeWriteNs = 0L
    private set
  var decodeRunNs = 0L
    private set
  var decodeReadNs = 0L
    private set

  init {
    check(prefillLengths.isNotEmpty() && prefillLengths.last() == 1) { "decoder: prefill signatures ${prefillLengths.toList()}" }
    check(stateNames.isNotEmpty()) { "decoder: no kv_cache_* state inputs" }
    reset()
  }

  private fun hasInput(name: String, signature: String): Boolean =
    runCatching { decoder.getInputTensorType(name, signature).layout != null }.getOrDefault(false)

  /** Back to position 0 with a zero state (every set, so the first call of any signature reads zeros). */
  fun reset() {
    pos = 0
    cur = 0
    decMaskRow.fill(MASK_FILL)
    for (set in sets) for ((name, buffer) in set) {
      val n = stateSizes.getValue(name)
      buffer.writeFloat(zeros.getOrPut(n) { FloatArray(n) })
    }
  }

  fun embedToken(id: Int): FloatArray {
    val t0 = System.nanoTime()
    embIn.writeInt(intArrayOf(id))
    embedder.run(mapOf("token_ids" to embIn), mapOf("embeddings" to embOut), "embedder")
    val row = embOut.readFloat()
    embedNs += System.nanoTime() - t0
    return row
  }

  /**
   * Prefills [rows] ([n] x hidden) from position 0 and returns the hidden state of the last row (from `decode`); the
   * text logits of that row are then [textLogits].
   */
  fun prefill(rows: List<FloatArray>): FloatArray {
    reset()
    val n = rows.size
    var i = 0
    val t0 = System.nanoTime()
    while (i < n - 1) {
      val left = n - 1 - i
      val size = prefillLengths.first { it <= left }
      prefillChunk(rows, i, size)
      i += size
    }
    prefillNs += System.nanoTime() - t0
    return step(rows[n - 1])
  }

  private fun prefillChunk(rows: List<FloatArray>, from: Int, size: Int) {
    val p = prefills.getOrPut(size) {
      val sig = "prefill_$size"
      Prefill(decoder.createInputBuffer("embeddings", sig), decoder.createInputBuffer("input_pos", sig),
        decoder.createInputBuffer("mask", sig), decoder.createOutputBuffer("hidden", sig), FloatArray(size * cacheLength))
    }
    val emb = FloatArray(size * hiddenSize)
    for (r in 0 until size) System.arraycopy(rows[from + r], 0, emb, r * hiddenSize, hiddenSize)
    p.emb.writeFloat(emb)
    p.pos.writeInt(IntArray(size) { pos + it })
    p.maskArr.fill(MASK_FILL)
    for (r in 0 until size) java.util.Arrays.fill(p.maskArr, r * cacheLength, r * cacheLength + pos + r + 1, 0f)
    p.mask.writeFloat(p.maskArr)
    val inputs = HashMap<String, TensorBuffer>(stateNames.size + 3)
    inputs["embeddings"] = p.emb
    inputs["input_pos"] = p.pos
    inputs["mask"] = p.mask
    inputs.putAll(set(cur))
    val outputs = HashMap<String, TensorBuffer>(stateNames.size + 1)
    outputs["hidden"] = p.hidden
    outputs.putAll(set(1 - cur))
    decoder.run(inputs, outputs, "prefill_$size")
    cur = 1 - cur
    pos += size
    prefillCalls++
    // the prefilled positions are open for every later row
    java.util.Arrays.fill(decMaskRow, 0, pos, 0f)
  }

  /** `prefill_1` = `decode` without its last op, the tied text head (262 MB of int8 weights a step): a speech slot's
   *  step, whose logits nobody reads, when the decoder has the signature (the same hidden state, bit for bit). */
  val hasHiddenStep: Boolean = prefillLengths.contains(1)
  var hiddenStepCalls = 0L
    private set
  private var logitsValid = false
  private val step1 by lazy {
    val p = prefills.getOrPut(1) {
      Prefill(decoder.createInputBuffer("embeddings", "prefill_1"), decoder.createInputBuffer("input_pos", "prefill_1"),
        decoder.createInputBuffer("mask", "prefill_1"), decoder.createOutputBuffer("hidden", "prefill_1"), FloatArray(cacheLength))
    }
    val maps = Array(2) { parity ->
      val inputs = HashMap<String, TensorBuffer>(stateNames.size + 3)
      inputs["embeddings"] = p.emb
      inputs["input_pos"] = p.pos
      inputs["mask"] = p.mask
      inputs.putAll(set(parity))
      val outputs = HashMap<String, TensorBuffer>(stateNames.size + 1)
      outputs["hidden"] = p.hidden
      outputs.putAll(set(1 - parity))
      Pair(inputs, outputs)
    }
    Pair(p, maps)
  }

  /**
   * One step at [pos] with input [vec]; returns its hidden state. [logits] false (the next slot is speech): the step
   * runs `prefill_1`, and [textLogits] is not valid until the next step with logits.
   */
  fun step(vec: FloatArray, logits: Boolean = true): FloatArray = if (!logits && hasHiddenStep) stepHidden(vec) else stepDecode(vec)

  private fun stepHidden(vec: FloatArray): FloatArray {
    check(pos < cacheLength) { "context full ($cacheLength)" }
    val (p, maps) = step1
    val t0 = System.nanoTime()
    p.emb.writeFloat(vec)
    p.pos.writeInt(intArrayOf(pos))
    decMaskRow[pos] = 0f
    p.mask.writeFloat(decMaskRow)
    val t1 = System.nanoTime()
    val (inputs, outputs) = maps[cur]
    decoder.run(inputs, outputs, "prefill_1")
    val t2 = System.nanoTime()
    cur = 1 - cur
    pos++
    decodeCalls++
    hiddenStepCalls++
    logitsValid = false
    val hidden = p.hidden.readFloat()
    val t3 = System.nanoTime()
    decodeWriteNs += t1 - t0
    decodeRunNs += t2 - t1
    decodeReadNs += t3 - t2
    decodeNs += t3 - t0
    return hidden
  }

  private fun stepDecode(vec: FloatArray): FloatArray {
    check(pos < cacheLength) { "context full ($cacheLength)" }
    val t0 = System.nanoTime()
    decEmb.writeFloat(vec)
    decPos.writeInt(intArrayOf(pos))
    decMaskRow[pos] = 0f
    decMask.writeFloat(decMaskRow)
    val t1 = System.nanoTime()
    val (inputs, outputs) = decodeMaps[cur]
    decoder.run(inputs, outputs, "decode")
    val t2 = System.nanoTime()
    cur = 1 - cur
    pos++
    decodeCalls++
    logitsValid = true
    val hidden = decHidden.readFloat()
    val t3 = System.nanoTime()
    decodeWriteNs += t1 - t0
    decodeRunNs += t2 - t1
    decodeReadNs += t3 - t2
    decodeNs += t3 - t0
    return hidden
  }

  /** The text logits of the latest step (the tied head inside the graph); read only for a text slot. */
  fun textLogits(): FloatArray {
    check(logitsValid) { "the last step ran without the text head (a speech slot's step)" }
    return decLogits.readFloat()
  }

  override fun close() {
    decoder.close()
    embedder.close()
  }

  companion object {
    val MASK_FILL = -0.7f * Float.MAX_VALUE
    private val PREFILL_CANDIDATES = intArrayOf(4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1)

    fun dims(t: com.google.ai.edge.litert.TensorType): List<Int> = t.layout?.dimensions ?: error("tensor without a layout")
    fun elements(t: com.google.ai.edge.litert.TensorType): Int = dims(t).fold(1) { a, b -> a * b }
  }
}
