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
 * encoder, or typed text) is answered in text and speech, interleaved ([ask]). A conversation about a video works the
 * same way: its pictures' rows and its soundtrack's rows are prefilled when the video is read ([setVideo]).
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
    /** The whole prompt the answer was generated from: the conversation so far, then [promptIds]. */
    val fullPromptIds: IntArray,
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
  var video: VideoInput? = null
    private set
  /** How a video's questions are answered: "voice" (text and speech, interleaved) or "text" (text only, the measuring
   *  tool). The system turn differs between the two, so the form is set when the video is read ([setVideo]). */
  var form = "voice"
    private set
  private var imageRows: List<FloatArray>? = null
  /** A video's picture rows (256 per group, in order), its soundtrack's rows and the soundtrack rows of each group. */
  private var videoRows: List<FloatArray>? = null
  private var soundRows: List<FloatArray> = emptyList()
  private var soundSplit = IntArray(0)
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
    video = null
    videoRows = null
    photo = p
    imageRows = rows
    form = "voice"
    return restart().put("vision_ms", visionMs).put("vision_rows", rows.size)
  }

  /**
   * A video read by [Videos.read]: each group's picture through the vision encoder, the soundtrack through the audio
   * encoder once, its rows split between the groups at each group's first frame (Videos.bounds); then a new
   * conversation about it, answered in [form]. [progress] gets a line for the screen per step; returns the timings.
   */
  fun setVideo(v: VideoInput, form: String, progress: (String) -> Unit = {}): JSONObject {
    require(form == "voice" || form == "text") { "answer form $form (voice or text)" }
    val rec = JSONObject()
    val rows = ArrayList<FloatArray>(v.groups * VisionEncoder.ROWS)
    val each = JSONArray()
    val t0 = System.nanoTime()
    for ((g, f) in v.frames.withIndex()) {
      progress("Reading the video: picture ${g + 1} of ${v.groups}")
      val tg = System.nanoTime()
      rows.addAll(engine.vision.encode(f))
      each.put((System.nanoTime() - tg) / 1e6)
    }
    val visionMs = (System.nanoTime() - t0) / 1e6
    var sound: List<FloatArray> = emptyList()
    var encodeMs = 0.0
    if (v.pcm16k != null) {
      progress("Reading the video: its sound")
      val te = System.nanoTime()
      sound = engine.audioEncoder.encode(v.pcm16k)
      encodeMs = (System.nanoTime() - te) / 1e6
      rec.put("sound_rows_host_formula", engine.audioEncoder.lastHost).put("sound_rows_mask", engine.audioEncoder.lastValid)
    }
    val bounds = Videos.bounds(v.indices, v.fps, sound.size, v.groups)
    photo = null
    imageRows = null
    video = v
    videoRows = rows
    soundRows = sound
    soundSplit = IntArray(v.groups) { bounds[it + 1] - bounds[it] }
    this.form = form
    progress("Reading the video: the prompt")
    return restart().put("vision_ms", visionMs).put("vision_ms_each", each).put("vision_rows", rows.size)
      .put("sound_encode_ms", encodeMs).put("sound_rows", sound.size).put("sound_split", JSONArray(soundSplit.toList()))
      .put("form", form).also { r -> for (k in rec.keys()) r.put(k, rec.get(k)) }
  }

  /** A new conversation about the same photo or video: the state from 0, the system turn and the photo (or the video)
   *  prefilled. */
  fun restart(): JSONObject {
    val v = video
    val image = imageRows
    check(v != null || image != null) { "no photo or video" }
    past.clear()
    pending = null
    lastToken = -1
    lastAudioRows = null
    ids.clear()
    val t0 = System.nanoTime()
    val startIds: IntArray
    val embedded: List<FloatArray>
    if (v != null) {
      startIds = engine.tokenizer.encode(ChatPrompt.videoStart(ChatPrompt.videoSystem(form == "voice"), v.stamps,
        VisionEncoder.ROWS, soundSplit))
      embedded = embed(startIds, emptyList(), listOf(soundRows), videoRows!!)
    } else {
      startIds = engine.tokenizer.encode(ChatPrompt.start(ChatPrompt.SYSTEM, image!!.size))
      embedded = embed(startIds, listOf(image), emptyList())
    }
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
    /** The depth decoder variant of this answer (null = the engine's default one). */
    depthForm: String? = null,
  ): Output {
    val image = imageRows
    check(image != null || video != null) { "no photo or video" }
    require(history != "text" || video == null) { "history \"text\" is for a photo (a video's conversation goes on in the state)" }
    // a video's answers in its form; a photo's always in text and speech
    val voice = video == null || form == "voice"
    val lm = engine.lm
    val depth = engine.depth(depthForm)
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
    val head = if (fromStart) IntArray(0) else ids.toIntArray()
    if (fromStart) {
      val sb = StringBuilder(ChatPrompt.start(ChatPrompt.SYSTEM, image!!.size))
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
      val text = if (past.isEmpty()) ChatPrompt.ask(nAudio, q.typed, instruction, voice)
      else ChatPrompt.next(lastToken != Tokens.IM_END, nAudio, q.typed, instruction, voice)
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
      // text only: the model's text answer, the switches to speech never sampled (the vendor's text generation)
      engine.generator.generate(rows, if (voice) SpeechGenerator.Mode.INTERLEAVED else SpeechGenerator.Mode.TEXT,
        explicitTextOnly = !voice, maxNewTokens = minOf(maxNewTokens, room), sampling = sampling, rng = Rng(seed),
        replay = replay, listener = gl, stop = stop, continueState = !fromStart, depth = depth)
    } catch (t: Throwable) {
      runCatching { codec.finish() }
      // the state holds part of a turn: the conversation starts over about the same photo or video
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
      put("form", if (voice) "voice" else "text")
      put("depth_form", depth.form)
      put("depth_variant", depth.variant)
      put("depth_accel", depth.accel)
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
    return Output(past.size, promptIds, head + promptIds, result, answer, pcm, timing)
  }

  /**
   * The measuring tool's answer to a prompt given as token ids (rendered off the phone by the model's chat template): no
   * photo and no recording, the state from 0, the first slot's kind from the last id (`<|audio_start|>` = speech,
   * `<|mixed_start|>` = text and speech interleaved, else text), the speech decoded but not played, the answer not kept
   * in the conversation. A conversation about a photo starts over afterwards (the system turn and the photo again).
   */
  fun answerIds(
    promptIds: IntArray,
    sampling: Sampling = Sampling(),
    seed: Long = 0,
    maxNewTokens: Int = 1024,
    /** The depth decoder variant of this answer (null = the engine's default one). */
    depthForm: String? = null,
  ): Output {
    require(promptIds.isNotEmpty()) { "no prompt ids" }
    require(promptIds.none { it == Tokens.IMAGE || it == Tokens.AUDIO }) {
      "the prompt ids hold an image or audio placeholder"
    }
    val lm = engine.lm
    val depth = engine.depth(depthForm)
    val t0 = System.nanoTime()
    val before = counters()
    val mode = when (promptIds.last()) {
      Tokens.AUDIO_START -> SpeechGenerator.Mode.AUDIO
      Tokens.MIXED_START -> SpeechGenerator.Mode.INTERLEAVED
      else -> SpeechGenerator.Mode.TEXT
    }
    val rows = promptIds.map { lm.embedToken(it) }
    val room = lm.cacheLength - rows.size - 1
    if (room < MIN_ANSWER) throw Full("The prompt (${rows.size} ids) leaves no room for an answer in ${lm.cacheLength}.")
    val codec = engine.codec
    codec.reset()
    val pcmChunks = java.util.Collections.synchronizedList(ArrayList<FloatArray>())
    codec.sink = { out -> if (out.isNotEmpty()) pcmChunks.add(out) }
    val textIds = ArrayList<Int>()
    val gl = object : SpeechGenerator.Listener {
      override fun onFrame(codes: IntArray) = codec.push(codes)

      override fun onToken(token: Int) {
        if (token != Tokens.AUDIO_OUT) textIds.add(token)
      }
    }
    val tg = System.nanoTime()
    val result = try {
      engine.generator.generate(rows, mode, explicitTextOnly = false, maxNewTokens = minOf(maxNewTokens, room),
        sampling = sampling, rng = Rng(seed), listener = gl, continueState = false, depth = depth)
    } catch (t: Throwable) {
      runCatching { codec.finish() }
      runCatching { afterIds() }
      throw t
    }
    codec.finish()
    val wallMs = (System.nanoTime() - t0) / 1e6
    val pcm = FloatArray(pcmChunks.sumOf { it.size })
    var o = 0
    for (c in pcmChunks) {
      System.arraycopy(c, 0, pcm, o, c.size)
      o += c.size
    }
    val after = counters()
    val audioS = pcm.size / Lfm3OmniEngine.SAMPLE_RATE.toDouble()
    val prefillEndMs = (tg - t0) / 1e6 + result.prefillNs / 1e6
    val timing = JSONObject().apply {
      put("wall_ms", wallMs)
      put("mode", mode.name)
      put("depth_form", depth.form)
      put("depth_variant", depth.variant)
      put("depth_accel", depth.accel)
      put("prompt_len", rows.size)
      put("prefill_ms", result.prefillNs / 1e6)
      put("prefill_end_ms", prefillEndMs)
      put("audio_s", audioS)
      put("speech_out_1s_ms", if (audioS > 0) (wallMs - prefillEndMs) / audioS else JSONObject.NULL)
      put("frames", result.frames.size)
      put("text_tokens", textIds.size)
      put("finished", result.finished)
      put("stopped", result.stopped)
      for ((k, v) in after) {
        val d = v - (before[k] ?: 0L)
        if (k.endsWith("_ns")) put(k.removeSuffix("_ns") + "_ms", d / 1e6) else put(k, d)
      }
    }
    val answer = engine.tokenizer.decode(textIds).trim()
    afterIds()
    return Output(0, promptIds, promptIds, result, answer, pcm, timing)
  }

  /** After [answerIds]: the conversation about the photo or the video from the start again, or, without one, an empty
   *  one. */
  private fun afterIds() {
    if (imageRows != null || video != null) {
      restart()
      return
    }
    past.clear()
    pending = null
    lastToken = -1
    lastAudioRows = null
    ids.clear()
  }

  /** The measuring tool's dump (files the Mac reference reads): the vision rows (a photo's, or a video's pictures' in
   *  order), the last question's audio rows, a video's soundtrack rows and the ids so far, as <dir>/<tag>_image.f32,
   *  _video.f32, _audio.f32, _sound.f32 (float32, 2048 per row) and _ids.json. */
  fun dump(dir: File, tag: String) {
    dir.mkdirs()
    fun write(f: File, rows: List<FloatArray>) {
      val bb = ByteBuffer.allocate(rows.sumOf { it.size } * 4).order(ByteOrder.LITTLE_ENDIAN)
      for (r in rows) for (v in r) bb.putFloat(v)
      f.writeBytes(bb.array())
    }
    imageRows?.let { write(File(dir, "${tag}_image.f32"), it) }
    videoRows?.let { write(File(dir, "${tag}_video.f32"), it) }
    lastAudioRows?.let { write(File(dir, "${tag}_audio.f32"), it) }
    if (video != null && soundRows.isNotEmpty()) write(File(dir, "${tag}_sound.f32"), soundRows)
    photo?.let { File(dir, "${tag}_photo.rgb").writeBytes(it.rgb) }
    File(dir, "${tag}_ids.json").writeText(JSONObject().put("ids", JSONArray(ids)).toString())
  }

  /** The prompt rows of [ids]: each `<|image|>` / `<|audio|>` / `<|video|>` takes the next row of its media (in order),
   *  the other ids their token embedding. */
  private fun embed(ids: IntArray, images: List<List<FloatArray>>, audios: List<List<FloatArray>>,
                    video: List<FloatArray> = emptyList()): List<FloatArray> {
    val img = images.flatten()
    val aud = audios.flatten()
    var i = 0
    var a = 0
    var v = 0
    val out = ids.map { id ->
      when (id) {
        Tokens.IMAGE -> img[i++]
        Tokens.AUDIO -> aud[a++]
        Tokens.VIDEO -> video[v++]
        else -> engine.lm.embedToken(id)
      }
    }
    check(i == img.size && a == aud.size && v == video.size) {
      "placeholders $i / ${img.size} image, $a / ${aud.size} audio, $v / ${video.size} video"
    }
    return out
  }

  /** The parts' counters (an answer's record holds their change over the answer; `_ns` ones in ms). Depth = every open
   *  form's graphs (an answer runs one form), with the host -> buffer copies, the runs and the buffer -> host copies;
   *  the language model's decode steps with the same three parts. */
  private fun counters(): Map<String, Long> {
    val forms = engine.depthForms.values
    return mapOf(
      "lm_decode_ns" to engine.lm.decodeNs, "lm_decode_calls" to engine.lm.decodeCalls,
      "lm_decode_write_ns" to engine.lm.decodeWriteNs, "lm_decode_run_ns" to engine.lm.decodeRunNs,
      "lm_decode_read_ns" to engine.lm.decodeReadNs,
      "lm_hidden_steps" to engine.lm.hiddenStepCalls,
      "lm_prefill_ns" to engine.lm.prefillNs, "lm_prefill_calls" to engine.lm.prefillCalls,
      "lm_embed_ns" to engine.lm.embedNs,
      "depth_ns" to forms.sumOf { it.ns }, "depth_calls" to forms.sumOf { it.calls },
      "depth_write_ns" to forms.sumOf { it.writeNs }, "depth_run_ns" to forms.sumOf { it.runNs },
      "depth_read_ns" to forms.sumOf { it.readNs },
      "frame_ns" to engine.frame.graph.ns, "frame_calls" to engine.frame.graph.calls,
      "codec_ns" to engine.codec.graph.ns, "codec_calls" to engine.codec.graph.calls,
      "audio_encoder_ns" to engine.audioEncoder.graph.ns,
      "sample_ns" to engine.generator.sampleNs,
    )
  }

  companion object {
    /** Fewer free context slots than this and a question is refused (an answer needs room to be spoken). */
    const val MIN_ANSWER = 160

    fun intsJson(v: List<Int>): JSONArray = JSONArray().also { a -> for (x in v) a.put(x) }
  }
}
