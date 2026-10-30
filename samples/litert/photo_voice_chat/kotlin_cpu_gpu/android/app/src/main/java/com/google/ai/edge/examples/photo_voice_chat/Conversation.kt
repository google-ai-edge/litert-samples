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

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONArray
import org.json.JSONObject

/**
 * A spoken conversation about one photo, as the model card writes it (ChatPrompt): the photo's 256 vision rows are
 * prefilled with the system turn when the photo is chosen ([setPhoto]); each question (a recording through the audio
 * encoder, or typed text) is answered in text and speech, interleaved ([ask]).
 *
 * The conversation goes on in the language model's state (history "state", the default): the next question's rows are
 * prefilled after the last answer, whose text tokens, speech frames and end-of-speech frame are already in the state,
 * as the vendor's `generate` continues from `past_key_values`. History "text" instead prefills the whole conversation
 * again from the start, the earlier answers written as text (the chat template's history), without their speech.
 */
class Conversation(private val engine: Lfm3OmniEngine) {

  class Question(val clip: Clip?, val typed: String, val label: String)

  class Output(
    val turn: Int,
    val promptIds: IntArray,
    val result: SpeechGenerator.Result,
    val text: String,
    val pcm: FloatArray,
    val timing: JSONObject,
  )

  interface Listener {
    fun onAudio(pcm: FloatArray) {}
    fun onText(text: String) {}
  }

  /** The conversation is longer than the language model's context can take one more answer. */
  class Full(message: String) : Exception(message)

  private class Past(val audioRows: List<FloatArray>?, val typed: String, val instruction: String, val answer: String)

  var photo: PhotoInput? = null
    private set
  private var imageRows: List<FloatArray>? = null
  private val past = ArrayList<Past>()
  private var pending: FloatArray? = null
  private var lastToken = -1
  /** Every id prefilled or generated so far in the state (vision and audio rows as their placeholder ids). */
  val ids = ArrayList<Int>()
  val turns: Int get() = past.size
  /** The last question's audio rows, for the measuring tool's dump. */
  var lastAudioRows: List<FloatArray>? = null
    private set

  /** The photo through the vision encoder, then a new conversation about it; returns the timings. */
  fun setPhoto(p: PhotoInput): JSONObject {
    val t0 = System.nanoTime()
    val rows = engine.vision.encode(p.rgb)
    val visionMs = (System.nanoTime() - t0) / 1e6
    photo = p
    imageRows = rows
    return restart().put("vision_ms", visionMs).put("vision_rows", rows.size)
  }

  /** A new conversation about the same photo: the state from 0, the system turn and the photo prefilled. */
  fun restart(): JSONObject {
    val rows = imageRows ?: error("no photo")
    past.clear()
    pending = null
    lastToken = -1
    lastAudioRows = null
    ids.clear()
    val t0 = System.nanoTime()
    val startIds = engine.tokenizer.encode(ChatPrompt.start(ChatPrompt.SYSTEM, rows.size))
    val embedded = embed(startIds, listOf(rows), emptyList())
    engine.lm.reset()
    engine.lm.extend(embedded, decodeLast = false)
    ids.addAll(startIds.toList())
    return JSONObject().put("start_prompt_len", startIds.size).put("start_prefill_ms", (System.nanoTime() - t0) / 1e6)
  }

  fun ask(
    q: Question,
    instruction: String,
    history: String = "state",
    sampling: Sampling = Sampling(),
    seed: Long = 0,
    replay: List<IntArray>? = null,
    maxNewTokens: Int = 1024,
    listener: Listener? = null,
    stop: () -> Boolean = { false },
  ): Output {
    val image = imageRows ?: error("no photo")
    val lm = engine.lm
    val t0 = System.nanoTime()
    val before = counters()
    var audioRows: List<FloatArray>? = null
    var encodeMs = 0.0
    if (q.clip != null) {
      val te = System.nanoTime()
      audioRows = engine.audioEncoder.encode(q.clip.pcm16k)
      encodeMs = (System.nanoTime() - te) / 1e6
      if (audioRows.isEmpty()) throw UnusableAudio("No question was heard: the recording has no sound the model can read.")
    }
    val nAudio = audioRows?.size ?: 0
    // the prompt rows of this turn: the state's continuation, or the whole conversation as text from the start
    val fromStart = history == "text" && past.isNotEmpty()
    val promptIds: IntArray
    val rows: List<FloatArray>
    if (fromStart) {
      val sb = StringBuilder(ChatPrompt.start(ChatPrompt.SYSTEM, image.size))
      val audios = ArrayList<List<FloatArray>>()
      for (p in past) {
        sb.append(ChatPrompt.user(p.audioRows?.size ?: 0, p.typed, p.instruction))
        sb.append("<|im_end|>\n<|im_start|>assistant\n<think></think>").append(p.answer).append("<|im_end|>\n<|im_start|>user\n")
        p.audioRows?.let { audios.add(it) }
      }
      sb.append(ChatPrompt.ask(nAudio, q.typed, instruction))
      audioRows?.let { audios.add(it) }
      promptIds = engine.tokenizer.encode(sb.toString())
      rows = embed(promptIds, listOf(image), audios)
    } else {
      val text = if (past.isEmpty()) ChatPrompt.ask(nAudio, q.typed, instruction)
      else ChatPrompt.next(lastToken != Tokens.IM_END, nAudio, q.typed, instruction)
      promptIds = engine.tokenizer.encode(text)
      val embedded = embed(promptIds, emptyList(), listOfNotNull(audioRows))
      rows = (listOfNotNull(pending) + embedded)
    }
    val usedBefore = if (fromStart) 0 else lm.pos
    val room = lm.cacheLength - usedBefore - rows.size - 1
    if (room < MIN_ANSWER) {
      throw Full("This conversation fills the model's ${lm.cacheLength}-token context. Tap New conversation to go on.")
    }

    val codec = engine.codec
    codec.reset()
    val pcmChunks = java.util.Collections.synchronizedList(ArrayList<FloatArray>())
    val firstAudioNs = java.util.concurrent.atomic.AtomicLong(-1L)
    codec.sink = { out ->
      if (out.isNotEmpty()) {
        firstAudioNs.compareAndSet(-1L, System.nanoTime())
        pcmChunks.add(out)
        listener?.onAudio(out)
      }
    }
    val textIds = ArrayList<Int>()
    val gl = object : SpeechGenerator.Listener {
      override fun onFrame(codes: IntArray) = codec.push(codes)

      override fun onToken(token: Int) {
        if (token == Tokens.AUDIO_OUT) return
        textIds.add(token)
        listener?.onText(engine.tokenizer.decode(textIds))
      }
    }
    val tg = System.nanoTime()
    val result = try {
      engine.generator.generate(rows, SpeechGenerator.Mode.INTERLEAVED, explicitTextOnly = false,
        maxNewTokens = minOf(maxNewTokens, room), sampling = sampling, rng = Rng(seed), replay = replay, listener = gl,
        stop = stop, continueState = !fromStart)
    } catch (t: Throwable) {
      runCatching { codec.finish() }
      // the state holds part of a turn: the conversation starts over about the same photo
      runCatching { restart() }
      throw t
    }
    codec.finish()
    val wallNs = System.nanoTime() - t0
    val answer = engine.tokenizer.decode(textIds).trim()
    // the conversation so far, for the next turn
    if (fromStart) {
      ids.clear()
      ids.addAll(promptIds.toList())
    } else {
      ids.addAll(promptIds.toList())
    }
    ids.addAll(result.tokens)
    pending = result.pending
    lastToken = result.lastToken
    lastAudioRows = audioRows
    past.add(Past(audioRows, q.typed, instruction, answer))

    val pcm = FloatArray(pcmChunks.sumOf { it.size })
    var o = 0
    for (c in pcmChunks) {
      System.arraycopy(c, 0, pcm, o, c.size)
      o += c.size
    }
    val after = counters()
    val audioS = pcm.size / Lfm3OmniEngine.SAMPLE_RATE.toDouble()
    val prefillEndMs = (tg - t0) / 1e6 + result.prefillNs / 1e6
    val wallMs = wallNs / 1e6
    val timing = JSONObject().apply {
      put("run_start_ns", t0)
      put("wall_ms", wallMs)
      put("encode_ms", encodeMs)
      put("audio_tokens", nAudio)
      put("audio_tokens_host_formula", if (q.clip != null) engine.audioEncoder.lastHost else JSONObject.NULL)
      put("audio_tokens_mask", if (q.clip != null) engine.audioEncoder.lastValid else JSONObject.NULL)
      put("history", if (fromStart) "text" else "state")
      put("prompt_len", rows.size)
      put("context_used", lm.pos)
      put("prefill_ms", result.prefillNs / 1e6)
      put("prefill_end_ms", prefillEndMs)
      put("first_slot_ms", (tg - t0) / 1e6 + result.firstSlotNs / 1e6)
      put("first_audio_ns", if (firstAudioNs.get() >= 0) firstAudioNs.get() else JSONObject.NULL)
      put("first_audio_ms", if (firstAudioNs.get() >= 0) (firstAudioNs.get() - t0) / 1e6 else JSONObject.NULL)
      put("audio_s", audioS)
      put("rtf", if (audioS > 0) wallMs / 1000.0 / audioS else JSONObject.NULL)
      // the time of the generation loop (after the prompt) per second of speech
      put("speech_out_1s_ms", if (audioS > 0) (wallMs - prefillEndMs) / audioS else JSONObject.NULL)
      put("frames", result.frames.size)
      put("text_tokens", textIds.size)
      put("finished", result.finished)
      put("stopped", result.stopped)
      put("last_token", result.lastToken)
      for ((k, v) in after) {
        val d = v - (before[k] ?: 0L)
        if (k.endsWith("_ns")) put(k.removeSuffix("_ns") + "_ms", d / 1e6) else put(k, d)
      }
    }
    return Output(past.size, promptIds, result, answer, pcm, timing)
  }

  /** The measuring tool's dump (files the Mac reference reads): the vision rows, the last question's audio rows and the
   *  ids so far, as <dir>/<tag>_image.f32, _audio.f32 (float32, 2048 per row) and _ids.json. */
  fun dump(dir: File, tag: String) {
    dir.mkdirs()
    fun write(f: File, rows: List<FloatArray>) {
      val bb = ByteBuffer.allocate(rows.sumOf { it.size } * 4).order(ByteOrder.LITTLE_ENDIAN)
      for (r in rows) for (v in r) bb.putFloat(v)
      f.writeBytes(bb.array())
    }
    imageRows?.let { write(File(dir, "${tag}_image.f32"), it) }
    lastAudioRows?.let { write(File(dir, "${tag}_audio.f32"), it) }
    photo?.let { File(dir, "${tag}_photo.rgb").writeBytes(it.rgb) }
    File(dir, "${tag}_ids.json").writeText(JSONObject().put("ids", JSONArray(ids)).toString())
  }

  /** The prompt rows of [ids]: each `<|image|>` / `<|audio|>` takes the next row of its media (in order), the other ids
   *  their token embedding. */
  private fun embed(ids: IntArray, images: List<List<FloatArray>>, audios: List<List<FloatArray>>): List<FloatArray> {
    val img = images.flatten()
    val aud = audios.flatten()
    var i = 0
    var a = 0
    val out = ids.map { id ->
      when (id) {
        Tokens.IMAGE -> img[i++]
        Tokens.AUDIO -> aud[a++]
        else -> engine.lm.embedToken(id)
      }
    }
    check(i == img.size && a == aud.size) { "placeholders $i / ${img.size} image, $a / ${aud.size} audio" }
    return out
  }

  private fun counters(): Map<String, Long> = mapOf(
    "lm_decode_ns" to engine.lm.decodeNs, "lm_decode_calls" to engine.lm.decodeCalls,
    "lm_hidden_steps" to engine.lm.hiddenStepCalls,
    "lm_prefill_ns" to engine.lm.prefillNs, "lm_prefill_calls" to engine.lm.prefillCalls,
    "lm_embed_ns" to engine.lm.embedNs,
    "depth_ns" to engine.depth.stepGraph.ns, "depth_calls" to engine.depth.stepGraph.calls,
    "frame_ns" to engine.frame.graph.ns, "codec_ns" to engine.codec.graph.ns, "codec_calls" to engine.codec.graph.calls,
    "audio_encoder_ns" to engine.audioEncoder.graph.ns,
    "sample_ns" to engine.generator.sampleNs,
  )

  companion object {
    /** Fewer free context slots than this and a question is refused (an answer needs room to be spoken). */
    const val MIN_ANSWER = 160

    fun intsJson(v: List<Int>): JSONArray = JSONArray().also { a -> for (x in v) a.put(x) }
  }
}
