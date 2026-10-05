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
// voice/src/main/kotlin/io/github/johnrocky/hfmodels/voice/SpeechPlayer.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant.loop

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext

/**
 * Speech out of the loudspeaker: mono float PCM in [-1, 1] at [sampleRate] (24000, the kitten
 * speaker's) through one [AudioTrack] in `MODE_STREAM` (`USAGE_ASSISTANT`, `CONTENT_TYPE_SPEECH`),
 * written in blocking slices of 200 ms so that a cancel takes effect within one slice. The
 * streaming writes follow this repository's text_to_speech_streaming sample (MainViewModel.kt).
 * Give it to [VoiceLoopConfig.player]; the loop plays each sentence while the next one is
 * synthesized.
 *
 * [play] returns once the samples are written (about one second may still be queued); [drain]
 * waits until they have been played; [stop] drops what is queued at once. A run of audio is
 * everything from the first [play] after a [drain] or [stop] (or after creation) to the next
 * [drain] or [stop]; [firstWriteAtNanos] is when the run's first write began (`System.nanoTime()`),
 * the moment the first sound left for the speaker. One caller at a time for [play] and [drain];
 * [stop] may come from anywhere. [close] releases the track.
 */
class SpeechPlayer(val sampleRate: Int = 24000) : AutoCloseable {
  private val slice = sampleRate / 5
  private val track: AudioTrack

  init {
    require(slice > 0) { "sampleRate must be at least 5" }
    val minBytes =
      AudioTrack.getMinBufferSize(
        sampleRate,
        AudioFormat.CHANNEL_OUT_MONO,
        AudioFormat.ENCODING_PCM_FLOAT,
      )
    check(minBytes > 0) {
      "this device cannot play $sampleRate Hz mono float PCM (getMinBufferSize $minBytes)"
    }
    val attributes =
      AudioAttributes.Builder()
        .setUsage(AudioAttributes.USAGE_ASSISTANT)
        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
        .build()
    val format =
      AudioFormat.Builder()
        .setSampleRate(sampleRate)
        .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
        .build()
    track =
      AudioTrack.Builder()
        .setAudioAttributes(attributes)
        .setAudioFormat(format)
        // One second of queue, so a sentence's writes rarely wait for the speaker.
        .setBufferSizeInBytes(maxOf(minBytes, sampleRate * 4))
        .setTransferMode(AudioTrack.MODE_STREAM)
        .build()
    if (track.state != AudioTrack.STATE_INITIALIZED) {
      track.release()
      throw IllegalStateException("AudioTrack did not initialize ($sampleRate Hz mono float)")
    }
    // A stream starts by default only once its whole buffer is full, so a reply shorter than a
    // second would never sound; start as soon as 25 ms are queued. A kitten chunk can be as short
    // as 1,200 samples (the speaker's min_samples after its tail trim); below a 100 ms threshold
    // such a reply alone would wait out drain's 2 s slack.
    track.setStartThresholdInFrames(minOf(sampleRate / 40, track.bufferCapacityInFrames))
  }

  /** Frames written since the track was last flushed: the target of the playback head position. */
  private val written = AtomicLong(0)

  @Volatile private var runOpen = false

  /** Bumped by [stop]: a [play] or [drain] of an earlier generation ends. */
  @Volatile private var generation = 0

  @Volatile private var closed = false

  /** When the first write of the current (or last) run began, by `System.nanoTime()`; 0 before. */
  @Volatile
  var firstWriteAtNanos: Long = 0L
    private set

  /**
   * Writes [samples] in slices of 200 ms, blocking on a full queue (on an IO thread). A cancel
   * stops the sound; a failure ends the run, so the next [play] starts a new one (and sets
   * [firstWriteAtNanos]).
   */
  suspend fun play(samples: FloatArray) {
    check(!closed) { "SpeechPlayer is closed" }
    val gen = generation
    withContext(Dispatchers.IO) {
      try {
        if (track.playState != AudioTrack.PLAYSTATE_PLAYING) {
          track.play()
        }
        var offset = 0
        while (offset < samples.size && gen == generation) {
          ensureActive()
          if (!runOpen) {
            firstWriteAtNanos = System.nanoTime()
            runOpen = true
          }
          val size = minOf(slice, samples.size - offset)
          val n = track.write(samples, offset, size, AudioTrack.WRITE_BLOCKING)
          check(n >= 0) { "AudioTrack.write returned $n" }
          offset += n
          written.addAndGet(n.toLong())
        }
      } catch (e: CancellationException) {
        stop()
        throw e
      } catch (e: Exception) {
        runOpen = false
        throw e
      }
    }
  }

  /**
   * Waits until everything written has been played (at most its length and 2 s more), then pauses
   * the track.
   */
  suspend fun drain() {
    val gen = generation
    val target = written.get()
    val queuedNs = maxOf(0L, target - head()) * 1_000_000_000L / sampleRate
    val deadline = System.nanoTime() + queuedNs + DRAIN_SLACK_NS
    try {
      while (gen == generation && head() < target && System.nanoTime() < deadline) {
        delay(DRAIN_POLL_MS)
      }
    } catch (e: CancellationException) {
      stop()
      throw e
    }
    if (gen == generation) {
      runCatching { track.pause() }
      runOpen = false
    }
  }

  /** Stops the sound now and drops what is queued; a [play] in progress returns. */
  fun stop() {
    generation++
    // A pause() returns a blocking write in progress; flush() then drops the queue and resets the
    // head position.
    runCatching { track.pause() }
    runCatching { track.flush() }
    written.set(0)
    runOpen = false
  }

  override fun close() {
    if (closed) {
      return
    }
    closed = true
    stop()
    track.release()
  }

  /** The playback head as an unsigned 32-bit frame count (it resets on flush). */
  private fun head(): Long = track.playbackHeadPosition.toLong() and 0xffffffffL

  private companion object {
    const val DRAIN_POLL_MS = 20L
    const val DRAIN_SLACK_NS = 2_000_000_000L
  }
}
