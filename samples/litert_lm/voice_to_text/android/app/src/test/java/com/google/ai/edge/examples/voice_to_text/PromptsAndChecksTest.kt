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

package com.google.ai.edge.examples.voice_to_text

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PromptsAndChecksTest {
  @Test
  fun promptsAreTheModelCardsTable() {
    assertEquals(
      "transcribe the speech with proper punctuation and capitalization.",
      Prompts.prompt(Task.TRANSCRIPT, emptyList()),
    )
    assertEquals(
      "translate the speech to English with proper punctuation and capitalization.",
      Prompts.prompt(Task.ENGLISH, emptyList()),
    )
    assertEquals(
      "transcribe the speech to text. Keywords: LiteRT, Galaxy S26",
      Prompts.prompt(Task.TRANSCRIPT, listOf("LiteRT", "Galaxy S26")),
    )
    assertEquals(
      "translate the speech to English. Keywords: Quilter",
      Prompts.prompt(Task.ENGLISH, listOf("Quilter")),
    )
  }

  @Test
  fun keywordsAreTrimmedDedupedAndCapped() {
    assertEquals(listOf("LiteRT", "Galaxy S26", "Quilter"), Prompts.keywords(" LiteRT ,Galaxy S26,\n,Quilter; LiteRT"))
    assertEquals(emptyList<String>(), Prompts.keywords(" , ;\n"))
    assertTrue(Prompts.keywords("x".repeat(2_000)).single().length == Prompts.MAX_KEYWORD_CHARS)
  }

  @Test
  fun outputBudgetFitsTheContext() {
    val prompt = Prompts.prompt(Task.TRANSCRIPT, emptyList())
    // 10 s: 10 tokens per second plus 100.
    assertEquals(200, Prompts.maxOutputTokens(10.0, prompt))
    // At the cap, audio + instruction + reply stay inside 2,048 tokens.
    val atCap = Prompts.maxOutputTokens(Prompts.MAX_SECONDS.toDouble(), prompt)
    assertTrue(Prompts.MAX_SECONDS * 10 + 24 + prompt.length / 2 + atCap <= Prompts.CONTEXT_TOKENS)
    // Past the context, the budget never goes below a small floor (the runtime refuses the message).
    assertEquals(16, Prompts.maxOutputTokens(300.0, prompt))
  }

  @Test
  fun recordingChecks() {
    assertTrue(RecordingChecks.problem(0.3, 0.5f)!!.startsWith("Nothing was recorded"))
    assertTrue(RecordingChecks.problem(5.0, 0f)!!.startsWith("The recording is silent"))
    assertTrue(RecordingChecks.problem(5.0, 8f / 32768f)!!.startsWith("The recording is silent"))
    assertNull(RecordingChecks.problem(5.0, 9f / 32768f))
    assertNull(RecordingChecks.problem(0.5, 0.2f))
  }

  @Test
  fun tooLongRefusalBecomesASentence() {
    val refusal =
      IllegalStateException(
        "Status Code: 3. Message: INVALID_ARGUMENT: Input token ids are too long. Exceeding the maximum number of tokens allowed: 2100 >= 2048"
      )
    assertTrue(Messages.isTooLong(refusal.message!!))
    assertTrue(Messages.transcribeFailed(refusal).startsWith("This recording is too long"))
    assertFalse(Messages.isTooLong("Status Code: 13. Message: INTERNAL: something else"))
    assertTrue(Messages.transcribeFailed(IllegalStateException("Status Code: 13. Message: x")).contains("could not transcribe"))
  }

  @Test
  fun downloadFailureKeepsThePart() {
    assertTrue(Messages.downloadFailed(java.net.SocketTimeoutException("Read timed out")).contains("kept"))
    assertTrue(Messages.downloadFailed(IllegalStateException("SHA-256 or byte-size verification failed")).contains("deleted"))
  }

  @Test
  fun speechTrimDropsLeadingRoomNoiseAndRefusesRoomNoiseAlone() {
    val rate = Wav.SAMPLE_RATE
    val rnd = java.util.Random(3)
    fun room(seconds: Double) =
      FloatArray((seconds * rate).toInt()) { (rnd.nextGaussian() * 0.0015).toFloat() }
    fun loud(seconds: Double) =
      FloatArray((seconds * rate).toInt()) { i ->
        (0.05 * kotlin.math.sin(2 * Math.PI * 220 * i / rate)).toFloat()
      }
    assertNull(SpeechTrim.start(room(5.0)))
    assertNull(SpeechTrim.start(FloatArray(5 * rate)))
    assertNull(SpeechTrim.start(room(2.0) + loud(0.06) + room(2.0)))
    assertEquals(1.3, SpeechTrim.start(room(1.5) + loud(2.0) + room(1.0))!! / rate.toDouble(), 0.001)
    assertEquals(0, SpeechTrim.start(loud(2.0) + room(1.0)))
  }
}
