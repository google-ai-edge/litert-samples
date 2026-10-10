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

package com.google.ai.edge.examples.voice_assistant

import com.google.ai.edge.examples.voice_assistant.asr.Transcriber
import com.google.ai.edge.examples.voice_assistant.asr.TranscriberLimits
import com.google.ai.edge.examples.voice_assistant.asr.Transcript
import com.google.ai.edge.examples.voice_assistant.asr.TranscriptTiming
import com.google.ai.edge.examples.voice_assistant.loop.LoopEngine
import com.google.ai.edge.examples.voice_assistant.loop.ToolEvent
import com.google.ai.edge.examples.voice_assistant.loop.VoiceLoopConfig
import com.google.ai.edge.examples.voice_assistant.tts.Speaker
import com.google.ai.edge.examples.voice_assistant.tts.SpeechAudio
import com.google.ai.edge.examples.voice_assistant.tts.SpeechTiming
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The microphone button's rule ([stillRunning]) against the loop itself: a second tap on Stop,
 * while the first stop is still unwinding, must not start a second listen.
 */
class ListenJobTest {
  private val transcriber =
    object : Transcriber {
      override val limits = TranscriberLimits(16000, 16.0, listOf("en"))

      override suspend fun transcribe(pcm: FloatArray) =
        Transcript("SET AN ALARM", TranscriptTiming(0.0, 0.0, 0.0))

      override suspend fun closeAndJoin() {}

      override fun close() {}
    }

  private val speaker =
    object : Speaker {
      override val voices = listOf("v")
      override val sampleRate = 24000
      override val maxChars = 400

      override suspend fun synthesize(text: String, voice: String?, speed: Float) =
        SpeechAudio(FloatArray(240), 24000, SpeechTiming(0.0, 0.0, 0.0, 1))

      override fun phonemeIds(text: String) = IntArray(0)

      override suspend fun closeAndJoin() {}

      override fun close() {}
    }

  private val opened = AtomicInteger()

  /** 20 ms chunks: silence, half a second of voice, silence; then nothing until cancelled. */
  private fun mic(): Flow<FloatArray> = flow {
    opened.incrementAndGet()
    for (i in 0 until 100) {
      val level = if (i in 25 until 50) 0.1f else 0f
      emit(FloatArray(320) { k -> if (k % 2 == 0) level else -level })
    }
    awaitCancellation()
  }

  @Test
  fun aStoppedListenStillRunsUntilItsTurnHasUnwound() = runBlocking {
    val replying = CompletableDeferred<Unit>()
    val unwind = CompletableDeferred<Unit>()
    // The model has started its reply and goes on until cancelled; its stop takes until [unwind]
    // (the chat engine waits for the runtime to confirm the stop, then closes the conversation).
    fun reply(text: String): Flow<ToolEvent> = flow {
      try {
        emit(ToolEvent.Text("Setting it now. "))
        replying.complete(Unit)
        awaitCancellation()
      } finally {
        withContext(NonCancellable) { unwind.await() }
      }
    }
    val engine = LoopEngine(transcriber, speaker, ::reply, VoiceLoopConfig(), null)
    val first = launch(Dispatchers.Default) { engine.listen(mic()).collect {} }
    try {
      replying.await()
      assertTrue(stillRunning(first))
      // The first tap on Stop. What the second tap sees: the job is no longer active (the old
      // check started a second listen here), but it is still running.
      first.cancel()
      assertFalse(first.isActive)
      assertTrue(stillRunning(first))
      // The loop takes that second listen as soon as the first one's microphone has closed, while
      // the first one's turn is still unwinding: a second microphone, and a screen the first
      // one's end then shows as idle.
      withTimeout(5_000) {
        while (runCatching { engine.listen(mic()).first() }.isFailure) {
          delay(1)
        }
      }
      assertEquals(2, opened.get())
      assertTrue(stillRunning(first))
    } finally {
      unwind.complete(Unit)
    }
    first.join()
    assertFalse(stillRunning(first))
    assertFalse(stillRunning(null))
  }
}
