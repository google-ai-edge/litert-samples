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

package com.google.ai.edge.examples.photo_voice_chat

import android.annotation.SuppressLint
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.LinkedBlockingQueue
import kotlin.concurrent.thread
import kotlin.math.abs
import kotlin.math.log10

/** A spoken question as the model hears it: 16 kHz mono floats, at most [MAX_SECONDS] (the audio encoder's window). */
class Clip(val name: String, val pcm16k: FloatArray) {
  val seconds: Double get() = pcm16k.size / 16000.0

  companion object {
    const val MAX_SECONDS = 30
  }
}

/** Why a recording cannot be a question, as the sentence the screen shows. */
class UnusableAudio(message: String) : Exception(message)

object AudioIO {
  /** Below this peak (-60 dBFS) a clip counts as silent. */
  const val SILENT_PEAK = 0.001f
  const val MIN_SECONDS = 0.5

  /** A RIFF WAV file (PCM 16-bit or float 32-bit, any channel count) as 16 kHz mono floats. */
  fun readWav16k(file: File): FloatArray = file.inputStream().use { readWav16k(it.readBytes()) }

  fun readWav16k(bytes: ByteArray): FloatArray {
    val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    require(b.remaining() >= 12 && b.getInt(0) == 0x46464952 && b.getInt(8) == 0x45564157) { "not a RIFF WAV file" }
    var p = 12
    var format = 0
    var channels = 0
    var rate = 0
    var bits = 0
    var dataAt = -1
    var dataLen = 0
    while (p + 8 <= b.limit()) {
      val id = b.getInt(p)
      val len = b.getInt(p + 4)
      if (id == 0x20746d66) { // "fmt "
        require(p + 24 <= b.limit()) { "the WAV header is cut short" }
        format = b.getShort(p + 8).toInt() and 0xffff
        channels = b.getShort(p + 10).toInt()
        rate = b.getInt(p + 12)
        bits = b.getShort(p + 22).toInt()
        if (format == 0xfffe && len >= 26 && p + 34 <= b.limit()) format = b.getShort(p + 32).toInt() and 0xffff // WAVE_FORMAT_EXTENSIBLE
      } else if (id == 0x61746164) { // "data"
        dataAt = p + 8
        dataLen = minOf(len, b.limit() - dataAt)
        break
      }
      if (len < 0) break
      p += 8 + len + (len and 1)
    }
    require(dataAt >= 0 && channels > 0 && rate > 0) { "the WAV file has no fmt or data chunk" }
    require(dataLen > 0) { "the WAV file holds no samples" }
    val frames: Int
    val mono: FloatArray
    when {
      format == 1 && bits == 16 -> {
        frames = dataLen / (2 * channels)
        mono = FloatArray(frames) { i ->
          var s = 0f
          for (c in 0 until channels) s += b.getShort(dataAt + (i * channels + c) * 2) / 32768f
          s / channels
        }
      }
      format == 3 && bits == 32 -> {
        frames = dataLen / (4 * channels)
        mono = FloatArray(frames) { i ->
          var s = 0f
          for (c in 0 until channels) s += b.getFloat(dataAt + (i * channels + c) * 4)
          s / channels
        }
      }
      else -> throw IllegalArgumentException("WAV format $format with $bits bits is not read here (PCM 16 or float 32)")
    }
    return resample(mono, rate, 16000)
  }

  /** The checks every spoken question goes through (an example, a recording): the 30 s cut, length, silence. */
  fun check(name: String, pcm16k: FloatArray): Clip {
    val max = Clip.MAX_SECONDS * 16000
    val pcm = if (pcm16k.size > max) pcm16k.copyOf(max) else pcm16k
    if (pcm.size < MIN_SECONDS * 16000) {
      throw UnusableAudio("The question was only %.1f s long. Hold the button while you speak, then let go.".format(pcm.size / 16000.0))
    }
    var peak = 0f
    for (v in pcm) peak = maxOf(peak, abs(v))
    if (peak < SILENT_PEAK) {
      val db = if (peak > 0f) "%.0f dBFS".format(20 * log10(peak.toDouble())) else "digital silence"
      throw UnusableAudio("No question was heard: $name is silent (its loudest sample is $db). Hold the button, speak, then let go.")
    }
    return Clip(name, pcm)
  }

  /** Linear interpolation (the clips this app reads are speech; 16 kHz input is returned as is). */
  fun resample(x: FloatArray, from: Int, to: Int): FloatArray {
    if (from == to || x.isEmpty()) return x
    val n = (x.size.toLong() * to / from).toInt()
    return FloatArray(n) { i ->
      val t = i.toDouble() * from / to
      val j = t.toInt()
      val f = (t - j).toFloat()
      if (j + 1 < x.size) x[j] * (1 - f) + x[j + 1] * f else x[x.size - 1]
    }
  }

  /** Mono floats as a 16-bit WAV file. */
  fun writeWav(file: File, pcm: FloatArray, rate: Int) {
    val b = ByteBuffer.allocate(44 + pcm.size * 2).order(ByteOrder.LITTLE_ENDIAN)
    b.putInt(0x46464952).putInt(36 + pcm.size * 2).putInt(0x45564157)
    b.putInt(0x20746d66).putInt(16).putShort(1).putShort(1).putInt(rate).putInt(rate * 2).putShort(2).putShort(16)
    b.putInt(0x61746164).putInt(pcm.size * 2)
    for (v in pcm) b.putShort((v.coerceIn(-1f, 1f) * 32767f).toInt().toShort())
    file.writeBytes(b.array())
  }

  /** The microphone at 16 kHz mono until [stop] or [maxSeconds]; [onTick] gets the seconds so far (recorder thread). */
  @SuppressLint("MissingPermission")
  class Recorder(private val maxSeconds: Int = Clip.MAX_SECONDS, private val onTick: (Double, Boolean) -> Unit = { _, _ -> }) {
    @Volatile private var running = false
    private val chunks = ArrayList<FloatArray>()
    private var worker: Thread? = null

    fun start() {
      val min = AudioRecord.getMinBufferSize(16000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
      val rec = AudioRecord(MediaRecorder.AudioSource.MIC, 16000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(min, 16000))
      check(rec.state == AudioRecord.STATE_INITIALIZED) { "the microphone could not be opened" }
      chunks.clear()
      running = true
      rec.startRecording()
      worker = thread(name = "recorder") {
        val buf = ShortArray(1600)
        var total = 0
        while (running && total < maxSeconds * 16000) {
          val n = rec.read(buf, 0, minOf(buf.size, maxSeconds * 16000 - total))
          if (n > 0) {
            synchronized(chunks) { chunks.add(FloatArray(n) { buf[it] / 32768f }) }
            total += n
            onTick(total / 16000.0, false)
          } else if (n < 0) {
            break
          }
        }
        rec.stop()
        rec.release()
        val reachedLimit = total >= maxSeconds * 16000
        running = false
        onTick(total / 16000.0, reachedLimit)
      }
    }

    val isRunning: Boolean get() = running

    fun stop(): FloatArray {
      running = false
      worker?.join()
      synchronized(chunks) {
        val out = FloatArray(chunks.sumOf { it.size })
        var o = 0
        for (c in chunks) {
          System.arraycopy(c, 0, out, o, c.size)
          o += c.size
        }
        return out
      }
    }
  }

  /**
   * Plays mono floats as they come, from its own thread (the generation never waits for the speaker). [onSound] is
   * called once, on the player thread, with System.nanoTime() of the first moment the playback head is past 0 (the
   * first sound out of the speaker, read every 2 ms after the first write); [onEnd] when the queued audio has played
   * (or [stopNow]); [endNs] = System.nanoTime() when the playback head reached the last written sample (-1 if stopped
   * before). [underruns] = AudioTrack's count at the end.
   */
  class Player(
    rate: Int,
    private val onSound: (Long) -> Unit = {},
    private val onEnd: (Player) -> Unit = {},
  ) {
    private val queue = LinkedBlockingQueue<FloatArray>()
    private val track = AudioTrack.Builder()
      .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
      .setAudioFormat(AudioFormat.Builder().setSampleRate(rate).setEncoding(AudioFormat.ENCODING_PCM_FLOAT).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
      .setTransferMode(AudioTrack.MODE_STREAM)
      .setBufferSizeInBytes(maxOf(AudioTrack.getMinBufferSize(rate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT), rate * 4))
      .build()
    @Volatile private var stopped = false
    @Volatile var soundNs = -1L
      private set
    @Volatile var underruns = -1
      private set
    @Volatile var written = 0L
      private set
    @Volatile var endNs = -1L
      private set
    @Volatile private var ended = false

    private val worker = thread(name = "player") {
      track.play()
      // the playback head is watched from the start: one long write (a whole spoken question) blocks until most of it
      // has played, so a watcher started after the first write would miss the first sound
      val watcher = thread(name = "player-head") {
        while (soundNs < 0 && !stopped && !ended) {
          if (track.playbackHeadPosition > 0) {
            soundNs = System.nanoTime()
            onSound(soundNs)
            break
          }
          Thread.sleep(2)
        }
      }
      while (true) {
        val chunk = queue.take()
        if (chunk.isEmpty()) break
        track.write(chunk, 0, chunk.size, AudioTrack.WRITE_BLOCKING)
        written += chunk.size
      }
      // let the written samples play out (WRITE_BLOCKING returns once they are in the buffer)
      if (!stopped) {
        val t0 = System.nanoTime()
        while (!stopped && track.playbackHeadPosition < written && System.nanoTime() - t0 < 5_000_000_000L) Thread.sleep(2)
        if (!stopped && track.playbackHeadPosition >= written) endNs = System.nanoTime()
      }
      underruns = track.underrunCount
      ended = true
      watcher.join()
      runCatching { track.stop() }
      track.release()
      onEnd(this)
    }

    fun play(pcm: FloatArray) {
      if (pcm.isNotEmpty() && !stopped) queue.put(pcm)
    }

    /** Ends after the queued audio. */
    fun close() {
      queue.put(FloatArray(0))
    }

    fun stopNow() {
      stopped = true
      queue.clear()
      runCatching {
        track.pause()
        track.flush()
      }
      queue.put(FloatArray(0))
    }
  }
}
