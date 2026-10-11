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

// Adapted from john-rocky/hfmodels-android (commit 3086d647):
// samples/voice/src/main/kotlin/io/github/johnrocky/hfmodels/samples/voice/TurnRecorder.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant

import android.os.SystemClock
import com.google.ai.edge.examples.voice_assistant.loop.VoiceLoop.Event
import com.google.ai.edge.examples.voice_assistant.ui.wholeOrNot
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONArray
import org.json.JSONObject

/**
 * The microphone's last seconds with each chunk's arrival time (elapsedRealtimeNanos), so a turn
 * can keep the utterance it heard: the loop's endpointer cuts it inside `VoiceLoop.listen` and does
 * not hand the samples out.
 */
class MicTap(val sampleRate: Int, seconds: Int = 30) {
  private val keep = sampleRate * seconds
  private val chunks = ArrayDeque<Pair<Long, FloatArray>>()
  private var held = 0

  @Synchronized
  fun add(chunk: FloatArray) {
    chunks.addLast(SystemClock.elapsedRealtimeNanos() to chunk)
    held += chunk.size
    while (chunks.size > 1 && held - chunks.first().second.size >= keep) {
      held -= chunks.removeFirst().second.size
    }
  }

  /**
   * The last [samples] samples among the chunks that arrived by [endNanos], and the arrival time of
   * the chunk that ends them; null when none did. The endpointer cuts inside a chunk, so the end is
   * within one chunk of the cut.
   */
  @Synchronized
  fun cut(endNanos: Long, samples: Int): Pair<FloatArray, Long>? {
    val upTo = chunks.filter { it.first <= endNanos }
    if (upTo.isEmpty()) {
      return null
    }
    val all = FloatArray(upTo.sumOf { it.second.size })
    var p = 0
    for ((_, c) in upTo) {
      c.copyInto(all, p)
      p += c.size
    }
    return all.copyOfRange(maxOf(0, all.size - samples), all.size) to upTo.last().first
  }
}

/**
 * Writes each turn for a screen recording's sound: `<root>/<turn>/utterance.wav` (16 kHz, the
 * microphone's turns), `reply.wav` (24 kHz, the sentences said, one after another) and
 * `events.json` (every event with System.nanoTime, SystemClock.elapsedRealtimeNanos and the wall
 * clock, from the Listening before the turn to Done).
 */
class TurnRecorder(val root: File) {
  private var listening: JSONObject? = null
  private var turn: Turn? = null
  private var next = (root.listFiles()?.mapNotNull { it.name.toIntOrNull() }?.maxOrNull() ?: 0) + 1

  private class Turn(val dir: File, val fromMic: Boolean) {
    val events = JSONArray()
    val sentences = ArrayList<String>()
    var heardElapsedNanos = 0L
    var transcribeMs = 0.0
    var audioMs = 0.0
  }

  /** Records [e] as it reached the app; at [Event.Done] the turn is complete (see [finish]). */
  fun event(e: Event, fromMic: Boolean) {
    val row = stamp(JSONObject().put("event", e.javaClass.simpleName))
    when (e) {
      Event.Listening -> {
        listening = row
        return
      }
      is Event.Heard -> {
        val dir = File(root, "%03d".format(next++)).apply { mkdirs() }
        turn =
          Turn(dir, fromMic).also { t ->
            listening?.let { t.events.put(it) }
            t.heardElapsedNanos = row.getLong("elapsed_nanos")
            t.transcribeMs = e.transcribeMs
            t.audioMs = e.audioMs
          }
        listening = null
        row.put("text", e.text).put("audio_ms", e.audioMs).put("transcribe_ms", e.transcribeMs)
      }
      Event.Thinking -> {}
      is Event.ToolCalled -> {
        val args = JSONObject()
        for ((k, v) in e.args) {
          args.put(k, json(v))
        }
        row.put("name", e.name).put("args", args).put("result", e.result).put("ms", e.ms)
      }
      is Event.Speaking -> {
        turn?.sentences?.add(e.sentence)
        row
          .put("sentence", e.sentence)
          .put("synth_ms", e.synthMs)
          .put("first_audio_ms", e.firstAudioMs ?: JSONObject.NULL)
      }
      is Event.Error -> row.put("code", e.code ?: JSONObject.NULL).put("message", e.message)
      is Event.Done -> {
        val t = e.timing
        val timing =
          JSONObject()
            .put("transcribe_ms", t.transcribeMs)
            .put("first_token_ms", t.firstTokenMs ?: JSONObject.NULL)
            .put("first_sentence_ms", t.firstSentenceMs ?: JSONObject.NULL)
            .put("first_audio_ms", t.firstAudioMs ?: JSONObject.NULL)
            .put("reply_ms", t.replyMs)
            .put("speak_ms", t.speakMs ?: JSONObject.NULL)
            .put("total_ms", t.totalMs)
            .put("tool_calls", t.toolCalls)
            .put("llm_turns", t.llmTurns)
            .put("heard", t.heard)
            .put("reply", t.reply)
            .put("spoken", t.spoken)
        row.put("timing", timing)
      }
    }
    turn?.events?.put(row)
  }

  /** The sentences said in the open turn (for [finish]). */
  fun sentences(): List<String> = turn?.sentences?.toList().orEmpty()

  /**
   * Closes the open turn: the utterance from [tap] (the microphone's turns), [replyAudio] (the
   * sentences said, synthesized again with the same text, voice and speed, in order) as reply.wav,
   * and events.json with [firstWriteNanos] (the player's first write, System.nanoTime) and where
   * each sentence sits in reply.wav. Returns the turn's directory.
   */
  fun finish(
    tap: MicTap?,
    replyAudio: List<FloatArray>,
    replyRate: Int,
    firstWriteNanos: Long,
    phoneState: String,
  ): File? {
    val t = turn ?: return null
    turn = null
    val out = JSONObject().put("input", if (t.fromMic) "mic" else "text")
    // The turn started when the utterance was cut: Heard came transcribeMs later.
    val cutNanos = t.heardElapsedNanos - (t.transcribeMs * 1e6).toLong()
    if (t.fromMic && tap != null) {
      val rate = tap.sampleRate
      tap.cut(cutNanos, (t.audioMs * rate / 1000).toInt())?.let { (pcm, endNanos) ->
        writeWav(File(t.dir, "utterance.wav"), pcm, rate)
        val utterance =
          JSONObject()
            .put("file", "utterance.wav")
            .put("samples", pcm.size)
            .put("sample_rate", rate)
            .put("end_elapsed_nanos", endNanos)
            .put("start_elapsed_nanos", endNanos - pcm.size * 1_000_000_000L / rate)
        out.put("utterance", utterance)
      }
    }
    val offset = SystemClock.elapsedRealtimeNanos() - System.nanoTime()
    val sentences = JSONArray()
    var at = 0
    for ((i, a) in replyAudio.withIndex()) {
      val sentence =
        JSONObject()
          .put("text", t.sentences.getOrNull(i) ?: "")
          .put("offset_samples", at)
          .put("samples", a.size)
      sentences.put(sentence)
      at += a.size
    }
    if (replyAudio.isNotEmpty()) {
      val all = FloatArray(at)
      var p = 0
      for (a in replyAudio) {
        a.copyInto(all, p)
        p += a.size
      }
      writeWav(File(t.dir, "reply.wav"), all, replyRate)
      val reply =
        JSONObject()
          .put("file", "reply.wav")
          .put("samples", at)
          .put("sample_rate", replyRate)
          .put("first_write_nanos", firstWriteNanos)
          .put("first_write_elapsed_nanos", firstWriteNanos + offset)
          .put("sentences", sentences)
          .put(
            "how",
            "the sentences said, synthesized again after the turn with the same text, voice and " +
              "speed; played back to back from the first write unless a sentence was not ready " +
              "in time",
          )
      out.put("reply", reply)
    }
    out
      .put("turn_start_elapsed_nanos_est", cutNanos)
      .put("phone_state", phoneState)
      .put("events", t.events)
    File(t.dir, "events.json").writeText(out.toString(2))
    return t.dir
  }

  /**
   * A tool argument for org.json, which writes a Number type it does not know (the runtime's) as
   * null.
   */
  private fun json(v: Any?): Any =
    when (v) {
      null -> JSONObject.NULL
      is Number -> wholeOrNot(v)
      is Boolean,
      is String -> v
      else -> v.toString()
    }

  private fun stamp(o: JSONObject): JSONObject =
    o.put("nano", System.nanoTime())
      .put("elapsed_nanos", SystemClock.elapsedRealtimeNanos())
      .put("elapsed_ms", SystemClock.elapsedRealtime())
      .put("wall_ms", System.currentTimeMillis())

  companion object {
    /** Mono 16-bit PCM. */
    fun writeWav(f: File, pcm: FloatArray, rate: Int) {
      val data = ByteBuffer.allocate(pcm.size * 2).order(ByteOrder.LITTLE_ENDIAN)
      for (v in pcm) {
        data.putShort((v.coerceIn(-1f, 1f) * 32767f).toInt().toShort())
      }
      val h = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
      h.put("RIFF".toByteArray()).putInt(36 + pcm.size * 2).put("WAVE".toByteArray())
      h.put("fmt ".toByteArray())
        .putInt(16)
        .putShort(1)
        .putShort(1)
        .putInt(rate)
        .putInt(rate * 2)
        .putShort(2)
        .putShort(16)
      h.put("data".toByteArray()).putInt(pcm.size * 2)
      RandomAccessFile(f, "rw").use {
        it.setLength(0)
        it.write(h.array())
        it.write(data.array())
      }
    }
  }
}
