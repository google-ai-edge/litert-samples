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
// voice/src/main/kotlin/io/github/johnrocky/hfmodels/voice/MicSource.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant.loop

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.channels.trySendBlocking
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow

/**
 * The microphone as mono chunks in [-1, 1]: [sampleRate] Hz, [chunkMs] each, from [source]
 * (default `VOICE_RECOGNITION`, the input tuned for recognizers), for [VoiceLoop.listen] or an
 * [Endpointer]. The capture loop follows this repository's utilities/common/kotlin/AudioCapture.kt.
 *
 * Collecting [chunks] opens an [AudioRecord] and reads it on its own thread; cancelling the
 * collector stops and releases it before the collection ends. `RECORD_AUDIO` is the app's:
 * requested and granted before collecting (without it the [AudioRecord] does not initialize and
 * the flow ends with an [IllegalStateException]). A chunk that cannot be delivered at once waits on
 * the reader thread, so a slow collector loses audio in the recorder's own buffer.
 */
class MicSource(
  val sampleRate: Int = 16000,
  val chunkMs: Int = 20,
  val source: Int = MediaRecorder.AudioSource.VOICE_RECOGNITION,
) {
  private val chunkSamples = sampleRate * chunkMs / 1000

  init {
    require(chunkSamples > 0) { "sampleRate x chunkMs must give at least one sample per chunk" }
  }

  @SuppressLint("MissingPermission")
  fun chunks(): Flow<FloatArray> = callbackFlow {
    val minBytes =
      AudioRecord.getMinBufferSize(
        sampleRate,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT,
      )
    check(minBytes > 0) {
      "this device cannot record $sampleRate Hz mono 16-bit PCM (getMinBufferSize $minBytes)"
    }
    // Room for several chunks, so the reader never loses audio between two reads.
    val record =
      AudioRecord(
        source,
        sampleRate,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT,
        maxOf(minBytes, chunkSamples * 2 * 8),
      )
    if (record.state != AudioRecord.STATE_INITIALIZED) {
      record.release()
      throw IllegalStateException(
        "AudioRecord did not initialize (source $source, $sampleRate Hz; is RECORD_AUDIO granted?)"
      )
    }
    try {
      record.startRecording()
    } catch (t: Throwable) {
      record.release()
      throw t
    }
    val running = AtomicBoolean(true)
    val reader =
      thread(name = "voice-assistant-mic", isDaemon = true) {
        val shorts = ShortArray(chunkSamples)
        var filled = 0
        var failures = 0
        try {
          while (running.get()) {
            val n = record.read(shorts, filled, chunkSamples - filled)
            when {
              n > 0 -> {
                failures = 0
                filled += n
                if (filled == chunkSamples) {
                  val chunk = FloatArray(chunkSamples) { shorts[it] / 32768f }
                  if (trySendBlocking(chunk).isFailure) {
                    break
                  }
                  filled = 0
                }
              }
              // The recorder is gone (another app took the input, or it was stopped): end the
              // flow.
              n == AudioRecord.ERROR_DEAD_OBJECT || n == AudioRecord.ERROR_INVALID_OPERATION -> {
                if (running.get()) {
                  close(IllegalStateException("AudioRecord.read returned $n"))
                }
                break
              }
              // 0 or a transient error (ERROR, ERROR_BAD_VALUE): read again, after a pause so the
              // thread does not spin, until it has failed for too long in a row.
              else -> {
                failures++
                if (failures >= MAX_TRANSIENT_FAILURES) {
                  val message = "AudioRecord.read returned $n $failures times in a row"
                  close(IllegalStateException(message))
                  break
                }
                Thread.sleep(TRANSIENT_RETRY_MS)
              }
            }
          }
        } catch (t: Throwable) {
          close(t)
        } finally {
          // Released on this thread, so a read in progress never races the release.
          runCatching { record.stop() }
          record.release()
        }
      }
    awaitClose {
      running.set(false)
      // A stop() returns a read in progress; the reader then releases the recorder.
      runCatching { record.stop() }
      reader.join(READER_JOIN_MS)
    }
  }

  private companion object {
    const val READER_JOIN_MS = 1_000L
    const val TRANSIENT_RETRY_MS = 5L

    /**
     * Reads in a row that return nothing before the flow ends with an error: 400 x 5 ms is two
     * seconds without a sample, which a working recorder never takes (it delivers a 20 ms chunk
     * every 20 ms), so the screen stops saying "Listening…" over a dead microphone.
     */
    const val MAX_TRANSIENT_FAILURES = 400
  }
}
