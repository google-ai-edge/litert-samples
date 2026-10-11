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
// samples/voice/src/test/kotlin/io/github/johnrocky/hfmodels/samples/voice/VoiceUiTest.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant.ui

import com.google.ai.edge.examples.voice_assistant.data.DownloadState
import com.google.ai.edge.examples.voice_assistant.data.DownloadStatus
import com.google.ai.edge.examples.voice_assistant.loop.VoiceLoop.Event
import com.google.ai.edge.examples.voice_assistant.loop.VoiceLoop.TurnTiming
import java.math.BigDecimal
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** The screen's state from the loop's events, and the milliseconds as the screen writes them. */
class VoiceUiTest {
  private fun timing(
    heard: String,
    firstSentence: Double?,
    firstAudio: Double?,
    transcribe: Double,
    reply: String,
    spoken: String,
  ) =
    TurnTiming(
      transcribeMs = transcribe,
      firstTokenMs = 900.0,
      firstSentenceMs = firstSentence,
      firstAudioMs = firstAudio,
      replyMs = 1900.0,
      speakMs = 2100.0,
      totalMs = 7000.0,
      toolCalls = 1,
      llmTurns = 2,
      heard = heard,
      reply = reply,
      spoken = spoken,
    )

  @Test
  fun aMicrophoneTurnShowsWhatWasHeardDoneAndSaidAndTheTimeFromTheEndOfSpeech() {
    var ui = VoiceUi(ready = true, listening = true)
    val args = mapOf("hour" to 16.0, "minute" to 15.0, "label" to "Wake Up")
    val events =
      listOf(
        Event.Listening,
        Event.Heard("Wake me up at six fifteen.", 2380.0, 63.0),
        Event.Thinking,
        Event.ToolCalled("set_alarm", args, "Alarm set for 16:15 (Wake Up)", 0.5),
        Event.Speaking("OK,", 120.0, 1300.0),
      )
    for (e in events) {
      ui = ui.on(e, hangoverMs = 800)
    }
    assertEquals("Wake me up at six fifteen.", ui.heard)
    val line = ui.tools.single()
    val shown = "${line.icon} ${line.call}"
    assertEquals("⏰ set_alarm(hour=16, minute=15, label=\"Wake Up\")", shown)
    assertEquals("OK", ui.reply)
    assertEquals("Speaking…", ui.status)
    ui = ui.on(Event.Speaking("Alarm set for 16:15 (Wake Up),", 300.0, null), 800)
    assertEquals("OK Alarm set for 16:15 (Wake Up)", ui.reply)
    val done =
      timing(
        "Wake me up at six fifteen.",
        1040.0,
        1224.0,
        63.0,
        "OK. I have set an alarm for 6:15.",
        "OK. Alarm set for 16:15 (Wake Up)",
      )
    ui = ui.on(Event.Done(done), 800)
    // At the end: what was said as one text, the model's own words beside it (they say 6:15; the
    // phone did 16:15).
    assertEquals("OK. Alarm set for 16:15 (Wake Up)", ui.reply)
    assertEquals("OK. I have set an alarm for 6:15.", ui.modelReply)
    assertEquals("Reply in 2.0 s", ui.replyIn)
    val parts = "(800 ms end of speech + 63 ms hearing + 977 ms thinking + 184 ms voice)"
    assertEquals(parts, ui.breakdown)
    assertEquals("Listening…", ui.status)
    // The next utterance clears the turn.
    ui = ui.on(Event.Heard("What time is it.", 1500.0, 50.0), 800)
    assertEquals(emptyList<ToolLine>(), ui.tools)
    assertEquals("", ui.reply)
    assertNull(ui.modelReply)
    assertEquals("", ui.replyIn)
  }

  @Test
  fun aTypedTurnHasNoEndOfSpeechAndASilentTurnNoReplyTime() {
    val command = "Set an alarm for seven thirty tomorrow morning."
    var ui = VoiceUi(ready = true)
    ui = ui.on(Event.Heard(command, 0.0, 0.0), 0)
    val said = "Alarm set for 07:30 (Morning Alarm)"
    ui = ui.on(Event.Done(timing(command, 1420.0, 1920.0, 0.0, "", said)), 0)
    assertEquals("Reply in 1.9 s", ui.replyIn)
    assertEquals("(1.4 s thinking + 500 ms voice)", ui.breakdown)
    // The model said nothing of its own: nothing to show beside what was said.
    assertNull(ui.modelReply)
    assertEquals("Ready", ui.status)
    // A blank transcript: no reply and no time.
    ui = ui.on(Event.Heard("", 900.0, 40.0), 800)
    ui = ui.on(Event.Done(timing("", null, null, 40.0, "", "")), 800)
    assertEquals("Heard nothing", VoiceUi().on(Event.Heard("", 900.0, 40.0), 800).status)
    assertEquals("", ui.replyIn)
    assertEquals("", ui.breakdown)
  }

  @Test
  fun aTurnWhoseTranscriberFailedShowsNothingOfThePreviousRequest() {
    var ui = VoiceUi(ready = true, listening = true)
    val clock =
      Event.ToolCalled("get_current_datetime", emptyMap(), "Sunday, 2026-10-04 21:30", 0.3)
    val previous =
      listOf(
        Event.Listening,
        Event.Heard("What time is it.", 1500.0, 50.0),
        Event.Thinking,
        clock,
        Event.Speaking("It is 21:30,", 200.0, 900.0),
        Event.Done(timing("What time is it.", 700.0, 900.0, 50.0, "It is 21:30.", "It is 21:30.")),
        Event.Listening,
      )
    for (e in previous) {
      ui = ui.on(e, 800)
    }
    // The finished request stays on the screen while the microphone waits.
    assertEquals("What time is it.", ui.heard)
    assertEquals(1, ui.tools.size)
    assertTrue(ui.keepsScreenOn)
    // The next utterance: the transcriber fails, so no Heard comes; the loop says its failure text.
    ui = ui.on(Event.Error("INFERENCE_FAILED", "transcriber: the graph failed"), 800)
    assertEquals("", ui.heard)
    assertEquals(emptyList<ToolLine>(), ui.tools)
    assertEquals("", ui.reply)
    assertEquals("", ui.replyIn)
    assertEquals("INFERENCE_FAILED: transcriber: the graph failed", ui.error)
    ui = ui.on(Event.Speaking("Sorry, I could not finish that,", 150.0, 400.0), 800)
    val failed = "Sorry, I could not finish that."
    ui = ui.on(Event.Done(timing("", 10.0, 400.0, 80.0, "", failed)), 800)
    // The TURN line reads its tools from here: none.
    assertEquals(emptyList<ToolLine>(), ui.tools)
    assertEquals(failed, ui.reply)
    assertEquals("INFERENCE_FAILED: transcriber: the graph failed", ui.error)
    // An Error inside a turn, after its Heard, keeps that turn's lines.
    ui = ui.on(Event.Heard("What time is it.", 1500.0, 50.0), 800)
    ui = ui.on(clock, 800)
    ui = ui.on(Event.Error(null, "player: IllegalStateException: closed"), 800)
    assertEquals("What time is it.", ui.heard)
    assertEquals(listOf(ToolLine(clock.name, clock.args, clock.result)), ui.tools)
    // Idle (loaded, not listening, no request) lets the screen sleep.
    assertFalse(VoiceUi(ready = true).keepsScreenOn)
  }

  @Test
  fun millisecondsUnderASecondAreMillisecondsAndAbove() {
    assertEquals("640 ms", ms(640.0))
    assertEquals("1.2 s", ms(1234.0))
    assertEquals("No sound", replyIn(timing("Hello.", 300.0, null, 0.0, "Hi.", "Hi."), 0))
    val clock = ToolLine("get_current_datetime", emptyMap(), "Saturday, 2026-10-03 17:05")
    assertEquals("get_current_datetime()", clock.call)
    // The runtime's numbers are a Number type of its own (they printed as 7.0 on the S26): whole
    // ones as whole.
    val timerArgs = mapOf("minutes" to BigDecimal("10.0"), "label" to "Tea")
    val timer = ToolLine("set_timer", timerArgs, "Timer requested: 10 min (Tea)")
    assertEquals("set_timer(minutes=10, label=\"Tea\")", timer.call)
    assertEquals(7.5, wholeOrNot(BigDecimal("7.5")))
  }

  @Test
  fun modelSizesAndDownloadStates() {
    assertEquals("131 MB", size(131495992L))
    assertEquals("2.59 GB", size(2588147712L))
    // What is left of an almost complete download is not "0 MB".
    assertEquals("5 KB", size(5048L))
    assertEquals("1 KB", size(300L))
    assertEquals("8 MB", size(8331904L))
    assertEquals("Missing", ModelRow("gemma", "Chat", "Gemma", 2588147712L).stateText)
  }

  @Test
  fun theLoadButtonCountsOnlyTheBytesStillMissing() {
    val ready = DownloadState(DownloadStatus.READY, 2588147712L, 2588147712L)
    val almost = DownloadState(DownloadStatus.PAUSED, 131490944L, 131495992L)
    val ui =
      VoiceUi(
        models =
          listOf(
            ModelRow("gemma", "Chat", "Gemma", 2588147712L, ready),
            ModelRow("zipformer", "Speech recognition", "Zipformer", 131495992L, almost),
            ModelRow("kitten", "Speech synthesis", "Kitten", 94366734L),
          )
      )
    assertEquals(5048L + 94366734L, ui.missingBytes)
    assertEquals(0L, VoiceUi(models = listOf(ModelRow("g", "Chat", "G", 10L, ready))).missingBytes)
  }
}
