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

package com.google.ai.edge.examples.model_zoo.audio

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.Handler
import android.os.Looper

/** Blocking playback on a worker; reserve the ticket on the caller before launching that worker. */
class AudioPlayback : AutoCloseable {
  private val monitor = Object()
  private var generation = 0L
  private var active: AudioTrack? = null

  /** Cancels previous playback and gives queued work a ticket that close() can invalidate. */
  fun reserve(): Long =
    synchronized(monitor) {
      generation++
      active?.let { runCatching { it.stop() } }
      monitor.notifyAll()
      generation
    }

  fun isCurrent(ticket: Long): Boolean = synchronized(monitor) { ticket == generation }

  fun play(
    audio: FloatArray,
    sampleRate: Int,
    ticket: Long,
    onProgress: (elapsedSeconds: Float, totalSeconds: Float) -> Unit = { _, _ -> },
  ) {
    require(audio.isNotEmpty()) { "There is no audio to play." }
    require(sampleRate > 0)
    if (synchronized(monitor) { ticket != generation }) return
    val track =
      AudioTrack(
        AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).build(),
        AudioFormat.Builder()
          .setSampleRate(sampleRate)
          .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
          .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
          .build(),
        audio.size * 4,
        AudioTrack.MODE_STATIC,
        AudioManager.AUDIO_SESSION_ID_GENERATE,
      )
    try {
      var complete = false
      synchronized(monitor) {
        if (ticket != generation) return
        active = track
      }
      // MODE_STATIC is valid in STATE_NO_STATIC_DATA until the first successful write.
      check(track.state != AudioTrack.STATE_UNINITIALIZED) {
        "Audio playback could not initialize."
      }
      val written = track.write(audio, 0, audio.size, AudioTrack.WRITE_BLOCKING)
      synchronized(monitor) {
        // Stop/navigation may arrive while AudioTrack writes. Check atomically with play().
        if (ticket != generation) return
        check(written == audio.size) { "Audio playback buffer was incomplete." }
        check(track.state == AudioTrack.STATE_INITIALIZED) {
          "Audio playback buffer is unavailable."
        }
        track.setPlaybackPositionUpdateListener(
          object : AudioTrack.OnPlaybackPositionUpdateListener {
            private fun publish(track: AudioTrack, finished: Boolean) {
              synchronized(monitor) {
                if (ticket != generation || active !== track) return
                val position =
                  PlaybackPosition.fromPlaybackHead(
                    track.playbackHeadPosition,
                    audio.size,
                    sampleRate,
                  )
                onProgress(position.elapsedSeconds, position.totalSeconds)
                if (finished) {
                  complete = true
                  monitor.notifyAll()
                }
              }
            }

            override fun onPeriodicNotification(track: AudioTrack) = publish(track, false)

            override fun onMarkerReached(track: AudioTrack) = publish(track, true)
          },
          Handler(Looper.getMainLooper()),
        )
        track.setPositionNotificationPeriod(maxOf(1, sampleRate / 20))
        track.setNotificationMarkerPosition(audio.size)
        onProgress(0f, audio.size / sampleRate.toFloat())
        track.play()
        val durationMs = (audio.size * 1000L / sampleRate) + 250
        val deadline = System.nanoTime() + durationMs * 1_000_000
        while (ticket == generation && !complete) {
          val remaining = deadline - System.nanoTime()
          if (remaining <= 0) break
          monitor.wait(remaining / 1_000_000, (remaining % 1_000_000).toInt())
        }
        if (ticket == generation) {
          val position =
            PlaybackPosition.fromPlaybackHead(track.playbackHeadPosition, audio.size, sampleRate)
          onProgress(position.elapsedSeconds, position.totalSeconds)
        }
      }
    } catch (e: Exception) {
      if (synchronized(monitor) { ticket == generation }) throw e
    } finally {
      synchronized(monitor) { if (active === track) active = null }
      track.setPlaybackPositionUpdateListener(null)
      track.release()
    }
  }

  /** Invalidates even work still queued on a worker, stops audio, and wakes its completion wait. */
  override fun close() {
    synchronized(monitor) {
      generation++
      active?.let { runCatching { it.stop() } }
      monitor.notifyAll()
    }
  }
}
