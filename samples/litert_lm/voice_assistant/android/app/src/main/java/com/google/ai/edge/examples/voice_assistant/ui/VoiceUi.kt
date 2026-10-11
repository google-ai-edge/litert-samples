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
// samples/voice/src/main/kotlin/io/github/johnrocky/hfmodels/samples/voice/VoiceUi.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant.ui

import com.google.ai.edge.examples.voice_assistant.data.DownloadState
import com.google.ai.edge.examples.voice_assistant.data.DownloadStatus
import com.google.ai.edge.examples.voice_assistant.loop.VoiceLoop.Event
import com.google.ai.edge.examples.voice_assistant.loop.VoiceLoop.TurnTiming
import java.util.Locale

/** One tool call on the screen: what the model asked for and what the phone answered. */
data class ToolLine(val name: String, val args: Map<String, Any?>, val result: String) {
  val icon: String
    get() =
      when (name) {
        "set_alarm" -> "⏰"
        "set_timer" -> "⏲️"
        "add_calendar_event",
        "get_calendar_events" -> "📅"
        "get_current_datetime" -> "🕒"
        else -> "🔧"
      }

  /** `set_alarm(hour=7, minute=30)`: the call as the model made it. */
  val call: String
    get() = name + args.entries.joinToString(", ", "(", ")") { "${it.key}=${plain(it.value)}" }

  private fun plain(v: Any?): String =
    when (v) {
      is Number -> wholeOrNot(v).toString()
      is String -> "\"$v\""
      else -> v.toString()
    }
}

/**
 * A tool argument's number as a Long when it is whole, else a Double: the runtime hands numbers
 * over in its own Number type (7.0 for "7"), which org.json would also write as null.
 */
fun wholeOrNot(v: Number): Number =
  v.toDouble().let { d ->
    if (d == Math.floor(d) && !d.isInfinite() && Math.abs(d) < 1e15) d.toLong() else d
  }

/** One model of the loop on the screen: what it does, its size and its files' state. */
data class ModelRow(
  val id: String,
  val role: String,
  val model: String,
  val bytes: Long,
  val state: DownloadState = DownloadState(),
) {
  /** "Ready", "Missing", "Downloading 45%", "Paused at 12%". */
  val stateText: String
    get() {
      val percent = state.receivedBytes * 100 / maxOf(1L, bytes)
      return when (state.status) {
        DownloadStatus.READY -> "Ready"
        DownloadStatus.MISSING -> "Missing"
        DownloadStatus.DOWNLOADING -> "Downloading $percent%"
        DownloadStatus.PAUSED -> "Paused at $percent%"
        DownloadStatus.VERIFYING -> "Verifying"
        DownloadStatus.ERROR -> "Error: ${state.error ?: "unknown"}"
      }
    }
}

/**
 * What the screen shows. [on] folds the loop's events into it, one turn at a time: [heard] the
 * text the model got, [tools] the calls, [reply] what was said (sentence by sentence while it is
 * said, the whole of it at the end), [modelReply] the model's own words when they are not what was
 * said (an action's result was said instead). [models] are the three models' download states.
 */
data class VoiceUi(
  val status: String = "Not loaded",
  val models: List<ModelRow> = emptyList(),
  val loading: Boolean = false,
  val ready: Boolean = false,
  val listening: Boolean = false,
  val busy: Boolean = false,
  val heard: String = "",
  val tools: List<ToolLine> = emptyList(),
  val reply: String = "",
  val modelReply: String? = null,
  val replyIn: String = "",
  val breakdown: String = "",
  val totalMs: Double? = null,
  val phoneState: String = "",
  val network: String = "",
  val airplaneMode: Boolean = false,
  val error: String? = null,
  /** Bytes still to download, while the user is asked about a metered network; else null. */
  val downloadConfirmation: Long? = null,
) {
  /** While the models load, the microphone is open or a request runs (MainActivity). */
  val keepsScreenOn: Boolean
    get() = loading || listening || busy

  /** Bytes the next load still has to download: what is not on the phone yet. */
  val missingBytes: Long
    get() =
      models
        .filter { it.state.status != DownloadStatus.READY }
        .sumOf { maxOf(0L, it.bytes - it.state.receivedBytes) }
}

/**
 * The state after [e]. [hangoverMs]: the silence the endpointer waited for before it cut the
 * utterance (the microphone's turns; 0 for typed text), part of the time from the end of speech
 * to the first sound.
 */
fun VoiceUi.on(e: Event, hangoverMs: Int): VoiceUi =
  when (e) {
    Event.Listening -> copy(busy = false, status = "Listening…")
    is Event.Heard -> {
      val status = if (e.text.isBlank()) "Heard nothing" else "Thinking…"
      newTurn().copy(heard = e.text, status = status)
    }
    Event.Thinking -> copy(status = "Thinking…")
    is Event.ToolCalled -> copy(tools = tools + ToolLine(e.name, e.args, e.result))
    is Event.Speaking -> {
      copy(status = "Speaking…", reply = (reply + " " + e.sentence.removeSuffix(",")).trim())
    }
    // An Error outside a turn opens one: the transcriber failed, so no Heard comes, and the
    // previous request's lines go (an Error inside a turn keeps that turn's lines).
    is Event.Error -> {
      (if (busy) this else newTurn()).copy(error = (e.code?.let { "$it: " } ?: "") + e.message)
    }
    is Event.Done -> {
      val t = e.timing
      copy(
        busy = false,
        status = if (listening) "Listening…" else "Ready",
        reply = t.spoken.ifBlank { reply },
        modelReply = t.reply.takeIf { it.isNotBlank() && words(it) != words(t.spoken) },
        replyIn = replyIn(t, hangoverMs),
        breakdown = breakdown(t, hangoverMs),
        totalMs = t.totalMs,
      )
    }
  }

/** A turn has started: the previous request's lines cleared. */
private fun VoiceUi.newTurn(): VoiceUi =
  copy(
    busy = true,
    heard = "",
    tools = emptyList(),
    reply = "",
    modelReply = null,
    replyIn = "",
    breakdown = "",
    totalMs = null,
    error = null,
  )

/**
 * "Reply in 2.6 s": from the end of speech (the hangover before the cut, then the turn) to the
 * first sound.
 */
fun replyIn(t: TurnTiming, hangoverMs: Int): String {
  val audio = t.firstAudioMs
  if (audio != null) {
    return "Reply in " + String.format(Locale.US, "%.1f s", (hangoverMs + audio) / 1000)
  }
  return if (t.heard.isBlank()) "" else "No sound"
}

/**
 * The parts of [replyIn]: the end-of-speech wait (microphone only), hearing (the transcriber),
 * thinking (to the first sentence to say: the model's, or an action's result) and the voice (that
 * sentence synthesized, to the first write).
 */
fun breakdown(t: TurnTiming, hangoverMs: Int): String {
  val audio = t.firstAudioMs ?: return ""
  val parts = ArrayList<String>()
  if (hangoverMs > 0) {
    parts += "${ms(hangoverMs.toDouble())} end of speech"
  }
  if (t.transcribeMs > 0) {
    parts += "${ms(t.transcribeMs)} hearing"
  }
  val sentence = t.firstSentenceMs
  if (sentence != null) {
    parts += "${ms(sentence - t.transcribeMs)} thinking"
    parts += "${ms(audio - sentence)} voice"
  } else {
    parts += "${ms(audio - t.transcribeMs)} thinking and voice"
  }
  return parts.joinToString(" + ", "(", ")")
}

/** 640 -> "640 ms", 1234 -> "1.2 s". */
fun ms(v: Double): String =
  if (v < 1000) {
    String.format(Locale.US, "%.0f ms", v)
  } else {
    String.format(Locale.US, "%.1f s", v / 1000)
  }

/** 5048 -> "5 KB", 131495992 -> "131 MB", 2588147712 -> "2.59 GB". Under 1 KB it says "1 KB". */
fun size(bytes: Long): String =
  if (bytes < 1_000_000L) {
    String.format(Locale.US, "%.0f KB", maxOf(1.0, bytes / 1e3))
  } else if (bytes < 1_000_000_000L) {
    String.format(Locale.US, "%.0f MB", bytes / 1e6)
  } else {
    String.format(Locale.US, "%.2f GB", bytes / 1e9)
  }

private fun words(s: String) = s.lowercase().filter { it.isLetterOrDigit() }
