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
// voice/src/test/kotlin/io/github/johnrocky/hfmodels/voice/EndpointerTest.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant.loop

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Synthetic 16 kHz signals: near-silence (LCG noise at 0.001) and a 440 Hz sine at 0.1 (RMS 0.07,
 * above the 0.02 start level).
 */
class EndpointerTest {
  private val sr = 16000
  private var seed = 7L

  private fun silence(seconds: Double) =
    FloatArray((seconds * sr).toInt()) {
      seed = (seed * 1103515245L + 12345L) % (1L shl 31)
      ((seed / (1L shl 31).toDouble() - 0.5) * 0.002).toFloat()
    }

  private fun tone(seconds: Double) =
    FloatArray((seconds * sr).toInt()) { i -> (0.1 * sin(2.0 * PI * 440.0 * i / sr)).toFloat() }

  private fun concat(vararg parts: FloatArray): FloatArray {
    val out = FloatArray(parts.sumOf { it.size })
    var o = 0
    for (p in parts) {
      System.arraycopy(p, 0, out, o, p.size)
      o += p.size
    }
    return out
  }

  /**
   * Feeds [pcm] in chunks of [chunk] samples (not a multiple of the 320-sample frame by default),
   * then flushes.
   */
  private fun stream(e: Endpointer, pcm: FloatArray, chunk: Int = 437): List<Endpointer.Event> {
    val events = ArrayList<Endpointer.Event>()
    var i = 0
    while (i < pcm.size) {
      events += e.feed(pcm.copyOfRange(i, minOf(pcm.size, i + chunk)))
      i += chunk
    }
    e.flush()?.let { events += it }
    return events
  }

  private fun utterances(events: List<Endpointer.Event>) =
    events.filterIsInstance<Endpointer.Event.Utterance>()

  @Test
  fun silenceGivesNoEvent() {
    val e = Endpointer()
    assertEquals(emptyList<Endpointer.Event>(), stream(e, silence(2.0)))
    assertNull(e.flush())
  }

  @Test
  fun oneUtteranceIsSpeechPlusHangoverWithoutPreRoll() {
    val e = Endpointer(preRollMs = 0)
    val input = concat(silence(0.5), tone(1.0), silence(1.5))
    val events = stream(e, input)
    assertEquals(2, events.size)
    assertEquals(Endpointer.Event.SpeechStart, events[0])
    val u = utterances(events).single()
    // The tone starts on a frame boundary (0.5 s = frame 25): the utterance is the input from
    // there, the hangover included.
    val end = sr / 2 + sr * (1000 + e.hangoverMs) / 1000
    assertArrayEquals(input.copyOfRange(sr / 2, end), u.pcm, 0f)
  }

  @Test
  fun preRollPutsTheAudioBeforeTheRunInFront() {
    val e = Endpointer()
    assertEquals(300, e.preRollMs)
    val input = concat(silence(0.5), tone(1.0), silence(1.5))
    val u = utterances(stream(e, input)).single()
    // 300 ms of the near-silence before the tone, then the tone and the hangover: the input slice,
    // sample for sample.
    val from = sr / 2 - sr * 300 / 1000
    val end = sr / 2 + sr * (1000 + e.hangoverMs) / 1000
    assertArrayEquals(input.copyOfRange(from, end), u.pcm, 0f)
    assertTrue((0 until sr * 300 / 1000).all { abs(u.pcm[it]) <= 0.001f })
  }

  @Test
  fun preRollIsOnlyWhatWasHeard() {
    // Speech from the first sample: nothing to put in front.
    val atOnce = concat(tone(1.0), silence(1.5))
    val first = utterances(stream(Endpointer(), atOnce)).single().pcm
    assertArrayEquals(atOnce.copyOfRange(0, sr * 1800 / 1000), first, 0f)
    // 100 ms of silence first: 100 ms of pre-roll, not 300.
    val short = concat(silence(0.1), tone(1.0), silence(1.5))
    val second = utterances(stream(Endpointer(), short)).single().pcm
    assertArrayEquals(short.copyOfRange(0, sr * 1900 / 1000), second, 0f)
  }

  @Test
  fun twoUtterancesWithAGap() {
    val input = concat(silence(0.5), tone(1.0), silence(1.5), tone(0.7), silence(1.5))
    for ((preRoll, extra) in listOf(0 to 0.0, 300 to 0.3)) {
      val events = stream(Endpointer(preRollMs = preRoll), input, chunk = 320)
      assertEquals(
        listOf("SpeechStart", "Utterance", "SpeechStart", "Utterance"),
        events.map { it.toString().substringBefore('(') },
      )
      val sizes = utterances(events).map { it.pcm.size }
      // The second pre-roll comes from the gap after the first utterance's hangover, not from the
      // first utterance.
      val firstOk = abs(sizes[0] - (1.8 + extra) * sr) <= 320
      val secondOk = abs(sizes[1] - (1.5 + extra) * sr) <= 320
      assertTrue("pre-roll $preRoll: $sizes", firstOk && secondOk)
    }
  }

  @Test
  fun longSoundIsCutAtTheMaximum() {
    val e = Endpointer()
    val events = stream(e, tone(20.0))
    val u = utterances(events)
    val max = sr * e.maxUtteranceMs / 1000
    assertEquals(max, u.first().pcm.size)
    assertTrue(u.all { it.pcm.size <= max })
    // The remaining 4 s opens a second utterance (the start rule's 100 ms included, no pre-roll
    // from the first), returned by flush().
    assertEquals(2, u.size)
    assertEquals(20 * sr - max, u[1].pcm.size)
    assertEquals(2, events.count { it == Endpointer.Event.SpeechStart })
  }

  @Test
  fun aMaximumBetweenFramesIsRoundedDownAndNothingIsDropped() {
    // 1010 ms = 16160 samples = 50.5 frames: the cut falls after frame 50, and the next utterance
    // starts at frame 51.
    val e = Endpointer(maxUtteranceMs = 1010, preRollMs = 0)
    val input = tone(2.0)
    val u = utterances(stream(e, input, chunk = 320))
    assertEquals(listOf(16000, 16000), u.map { it.pcm.size })
    assertArrayEquals(input, concat(*u.map { it.pcm }.toTypedArray()), 0f)
  }

  @Test
  fun theMaximumMustHoldThePreRollAndTheStartRun() {
    assertThrows(IllegalArgumentException::class.java) {
      Endpointer(maxUtteranceMs = 300, preRollMs = 300)
    }
    Endpointer(maxUtteranceMs = 400, preRollMs = 300)
  }
}
