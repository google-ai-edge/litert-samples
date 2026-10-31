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

import org.json.JSONArray
import org.json.JSONObject

/**
 * One answer: the voicemail and the instruction in, the model's text and speech out. The prompts are the model card's:
 *   interleaved  voice chat: system "Respond with interleaved text and audio.", the assistant turn opened with
 *                `<|mixed_start|>`, the user turn = the recording, then the written instruction (the card's audio
 *                understanding turn; the chat template joins the two parts with nothing between them). Text and speech
 *                come in turns from one pass.
 *   two_stage    the same model twice: (1) audio understanding, system "Answer questions about the provided audio clip.
 *                Base every answer only on what can be heard.", the same user turn, the answer opened with
 *                `<think></think>` (the card's direct text answer), text only (the card's suppress_mixed /
 *                suppress_audio_out); (2) text to speech, system "Perform TTS in <locale> using a female voice.", the
 *                user turn = the text of (1), the assistant turn opened with `<|audio_start|>`.
 */
class Summarizer(private val engine: Lfm3AudioEngine) {

  class Request(
    val voicemail: Voicemail,
    val instruction: String,
    val language: AnswerLanguage = Answer.LANGUAGES.first(),
    val path: String = "interleaved",
    val seed: Long = 0,
    val sampling: Sampling = Sampling(),
    val maxNewTokens: Int = 1024,
    val maxTextTokens: Int = 256,
  )

  class Output(
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

  fun run(r: Request, listener: Listener? = null, stop: () -> Boolean = { false }): Output {
    val lm = engine.lm
    val t0 = System.nanoTime()
    val before = counters()
    val te = System.nanoTime()
    val audioRows = engine.audioEncoder.encode(r.voicemail.pcm16k)
    val encodeMs = (System.nanoTime() - te) / 1e6
    val twoStage = r.path == "two_stage"
    val chat = if (twoStage) ChatPrompt.continued(AUDIO_QA_SYSTEM, ChatPrompt.AUDIO + r.instruction, "<think></think>")
    else ChatPrompt.continued(ChatPrompt.INTERLEAVED_SYSTEM, ChatPrompt.AUDIO + r.instruction, "<|mixed_start|>")
    val ids = promptIds(chat, audioRows.size)
    val tp = System.nanoTime()
    var next = 0
    val rows = ids.map { id -> if (id == Tokens.AUDIO) audioRows[next++] else lm.embedToken(id) }
    val promptEmbedMs = (System.nanoTime() - tp) / 1e6

    val codec = engine.codec
    codec.reset()
    // the codec's samples, in order; the codec runs on its own thread ([finish] waits for the last)
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
    var stage1Ns = 0L
    var ttsPromptLen = 0
    var ttsText: String? = null
    val result = try {
      if (!twoStage) {
        engine.generator.generate(rows, SpeechGenerator.Mode.INTERLEAVED, explicitTextOnly = false, maxNewTokens = r.maxNewTokens,
          sampling = r.sampling, rng = Rng(r.seed), listener = gl, stop = stop)
      } else {
        // (1) the answer as text only
        val first = engine.generator.generate(rows, SpeechGenerator.Mode.TEXT, explicitTextOnly = true, maxNewTokens = r.maxTextTokens,
          sampling = r.sampling, rng = Rng(r.seed), listener = gl, stop = stop)
        stage1Ns = System.nanoTime() - tg
        val answerText = engine.tokenizer.decode(textIds).trim()
        if (answerText.isEmpty() || stop()) first else {
          // (2) that text spoken (as Answer.speakable writes it); its own text tokens (the end of the turn) are not part
          // of the answer
          val spoken = Answer.speakable(answerText, r.language).also { ttsText = it }
          val tts = ChatPrompt.continued(ttsSystem(r.language), spoken, "<|audio_start|>")
          val ttsIds = promptIds(tts, 0)
          ttsPromptLen = ttsIds.size
          val ttsRows = ttsIds.map { lm.embedToken(it) }
          val speech = engine.generator.generate(ttsRows, SpeechGenerator.Mode.AUDIO, explicitTextOnly = false, maxNewTokens = r.maxNewTokens,
            sampling = r.sampling, rng = Rng(r.seed), listener = object : SpeechGenerator.Listener {
              override fun onFrame(codes: IntArray) = codec.push(codes)
            }, stop = stop)
          SpeechGenerator.Result(first.tokens + speech.tokens, speech.frames, speech.orders, speech.finished, speech.stopped,
            first.prefillNs, first.firstSlotNs)
        }
      }
    } catch (t: Throwable) {
      runCatching { codec.finish() }
      throw t
    }
    codec.finish()
    val wallNs = System.nanoTime() - t0
    val pcm = FloatArray(pcmChunks.sumOf { it.size })
    var o = 0
    for (c in pcmChunks) {
      System.arraycopy(c, 0, pcm, o, c.size)
      o += c.size
    }
    val after = counters()
    val audioS = pcm.size / Lfm3AudioEngine.SAMPLE_RATE.toDouble()
    val prefillEndMs = (tg - t0) / 1e6 + result.prefillNs / 1e6
    val wallMs = wallNs / 1e6
    val timing = JSONObject().apply {
      put("run_start_ns", t0)
      put("wall_ms", wallMs)
      put("encode_ms", encodeMs)
      put("audio_tokens", audioRows.size)
      put("prompt_len", ids.size)
      put("prompt_embed_ms", promptEmbedMs)
      put("prefill_ms", result.prefillNs / 1e6)
      put("prefill_end_ms", prefillEndMs)
      put("first_slot_ms", (tg - t0) / 1e6 + result.firstSlotNs / 1e6)
      if (twoStage) {
        put("text_stage_ms", stage1Ns / 1e6)
        put("tts_prompt_len", ttsPromptLen)
        put("tts_text", ttsText ?: JSONObject.NULL)
      }
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
      for ((k, v) in after) {
        val d = v - (before[k] ?: 0L)
        if (k.endsWith("_ns")) put(k.removeSuffix("_ns") + "_ms", d / 1e6) else put(k, d)
      }
    }
    return Output(ids, result, engine.tokenizer.decode(textIds).trim(), pcm, timing)
  }

  /** The prompt ids of a chat text in which each `<|audio|>` stands for one clip of [audioTokens] encoder tokens. */
  private fun promptIds(text: String, audioTokens: Int): IntArray {
    val ids = engine.tokenizer.encode(text)
    if (audioTokens <= 0) return ids
    val out = ArrayList<Int>(ids.size + audioTokens)
    for (id in ids) if (id == Tokens.AUDIO) repeat(audioTokens) { out.add(Tokens.AUDIO) } else out.add(id)
    return out.toIntArray()
  }

  private fun counters(): Map<String, Long> = mapOf(
    "lm_decode_ns" to engine.lm.decodeNs, "lm_decode_calls" to engine.lm.decodeCalls,
    "lm_hidden_steps" to engine.lm.hiddenStepCalls,
    "lm_prefill_ns" to engine.lm.prefillNs, "lm_prefill_calls" to engine.lm.prefillCalls,
    "depth_proj_ns" to engine.depth.projGraph.ns, "depth_pass_ns" to engine.depth.passGraph.ns,
    "depth_pass_calls" to engine.depth.passGraph.calls,
    "frame_ns" to engine.frame.graph.ns, "codec_ns" to engine.codec.graph.ns, "codec_calls" to engine.codec.graph.calls,
    "audio_encoder_ns" to engine.audioEncoder.graph.ns,
    "sample_ns" to engine.generator.sampleNs,
  )

  companion object {
    const val AUDIO_QA_SYSTEM = "Answer questions about the provided audio clip. Base every answer only on what can be heard."

    fun ttsSystem(lang: AnswerLanguage) = "Perform TTS in ${lang.ttsLocale} using a female voice."

    fun intsJson(v: List<Int>): JSONArray = JSONArray().also { a -> for (x in v) a.put(x) }
  }
}
