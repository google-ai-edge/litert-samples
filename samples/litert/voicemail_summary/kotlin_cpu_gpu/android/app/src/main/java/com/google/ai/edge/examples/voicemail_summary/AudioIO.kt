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

package com.google.ai.edge.examples.voicemail_summary

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaRecorder
import android.net.Uri
import android.util.Log
import java.io.File
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.LinkedBlockingQueue
import kotlin.concurrent.thread
import kotlin.math.abs
import kotlin.math.log10

/** A voicemail as the model hears it: 16 kHz mono floats, at most [MAX_SECONDS] (the audio encoder's window). */
class Voicemail(
  val name: String,
  val pcm16k: FloatArray,
  /** The length of the whole file; more than [MAX_SECONDS] means only the first 30 s are in [pcm16k]. */
  val totalSeconds: Double,
  /** The screen's sentence when the phone decoded less of the file than its stated length; else null. */
  val shortDecode: String? = null,
) {
  val seconds: Double get() = pcm16k.size / 16000.0
  val cut: Boolean get() = totalSeconds > MAX_SECONDS + 0.05

  companion object {
    const val MAX_SECONDS = 30
  }
}

/** Why a file or a recording cannot be summarized, as the sentence the screen shows. */
class UnusableAudio(message: String) : Exception(message)

object AudioIO {
  /** Below this peak (-60 dBFS) a clip counts as silent. */
  const val SILENT_PEAK = 0.001f
  const val MIN_SECONDS = 1.0

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

  /**
   * Any audio file the phone can decode (WAV in the app; FLAC, M4A / AAC, MP3, OGG through MediaCodec) as a
   * [Voicemail]: the first 30 s, 16 kHz mono. Throws [UnusableAudio] with the screen's sentence when the file cannot be
   * read, is shorter than [MIN_SECONDS], or is silent.
   */
  fun readVoicemail(context: Context, uri: Uri, name: String): Voicemail {
    val head = ByteArray(12)
    val headLen = try {
      context.contentResolver.openInputStream(uri)?.use { s -> readUpTo(s, head) } ?: throw UnusableAudio("Could not open $name.")
    } catch (e: UnusableAudio) {
      throw e
    } catch (e: Exception) {
      throw UnusableAudio("Could not open $name: ${e.message}")
    }
    val isWav = headLen >= 12 && String(head, 0, 4, Charsets.US_ASCII) == "RIFF" && String(head, 8, 4, Charsets.US_ASCII) == "WAVE"
    val (pcm, total) = try {
      if (isWav) {
        val all = readWav16k(context.contentResolver.openInputStream(uri)!!.use { it.readBytes() })
        all to all.size / 16000.0
      } else {
        decodeWithMediaCodec(context, uri)
      }
    } catch (e: UnusableAudio) {
      throw e
    } catch (e: Exception) {
      throw UnusableAudio("Could not read $name as audio (WAV, FLAC or M4A): ${e.message ?: e.javaClass.simpleName}")
    }
    val v = check(name, pcm, total)
    // a decoder that returns less than the file's stated length (on a Galaxy S26 a 19.1 s FLAC
    // test file came back as 18.0 s, its end missing) is said on the screen; the voicemail is still usable
    val expected = minOf(total, Voicemail.MAX_SECONDS.toDouble())
    if (isWav || total <= 0 || v.seconds >= expected - 0.5) return v
    return Voicemail(v.name, v.pcm16k, v.totalSeconds,
      "This phone decoded %.1f s of %s's %.1f s, so the end of the voicemail may be missing.".format(v.seconds, name, total))
  }

  /** The checks every input goes through (a bundled voicemail, a recording, a file): the 30 s cut, length, silence. */
  fun check(name: String, pcm16k: FloatArray, totalSeconds: Double): Voicemail {
    val max = Voicemail.MAX_SECONDS * 16000
    val pcm = if (pcm16k.size > max) pcm16k.copyOf(max) else pcm16k
    if (pcm.size < MIN_SECONDS * 16000) {
      throw UnusableAudio("$name is only %.1f s long. Record or pick a voicemail of at least 1 s.".format(pcm.size / 16000.0))
    }
    var peak = 0f
    for (v in pcm) peak = maxOf(peak, abs(v))
    if (peak < SILENT_PEAK) {
      val db = if (peak > 0f) "%.0f dBFS".format(20 * log10(peak.toDouble())) else "digital silence"
      throw UnusableAudio("$name is silent (its loudest sample is $db), so there is nothing to summarize.")
    }
    return Voicemail(name, pcm, totalSeconds)
  }

  private fun readUpTo(s: InputStream, buf: ByteArray): Int {
    var n = 0
    while (n < buf.size) {
      val r = s.read(buf, n, buf.size - n)
      if (r < 0) break
      n += r
    }
    return n
  }

  /** The first 30 s of a compressed file through MediaExtractor + MediaCodec; returns (16 kHz mono, the file's length in s). */
  private fun decodeWithMediaCodec(context: Context, uri: Uri): Pair<FloatArray, Double> {
    val ex = MediaExtractor()
    try {
      ex.setDataSource(context, uri, null)
      val track = (0 until ex.trackCount).firstOrNull { ex.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true }
        ?: throw UnusableAudio("there is no audio track in it")
      ex.selectTrack(track)
      val format = ex.getTrackFormat(track)
      val mime = format.getString(MediaFormat.KEY_MIME)!!
      val total = if (format.containsKey(MediaFormat.KEY_DURATION)) format.getLong(MediaFormat.KEY_DURATION) / 1e6 else -1.0
      var rate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
      var channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
      val codec = MediaCodec.createDecoderByType(mime)
      val out = ArrayList<FloatArray>()
      var outFrames = 0L
      var inBytes = 0L
      var inSamples = 0
      var lastTimeUs = 0L
      try {
        codec.configure(format, null, null, 0)
        codec.start()
        val info = MediaCodec.BufferInfo()
        var inputDone = false
        var floatPcm = false
        val t0 = System.nanoTime()
        while (true) {
          if (System.nanoTime() - t0 > 20_000_000_000L) throw UnusableAudio("decoding took longer than 20 s")
          if (!inputDone) {
            val i = codec.dequeueInputBuffer(10_000)
            if (i >= 0) {
              val buf = codec.getInputBuffer(i)!!
              val n = ex.readSampleData(buf, 0)
              if (n < 0) {
                // the end-of-stream buffer keeps the last sample's time (not 0)
                codec.queueInputBuffer(i, 0, 0, lastTimeUs, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                inputDone = true
              } else {
                lastTimeUs = ex.sampleTime
                inBytes += n
                inSamples++
                codec.queueInputBuffer(i, 0, n, lastTimeUs, 0)
                ex.advance()
              }
            }
          }
          val o = codec.dequeueOutputBuffer(info, 10_000)
          if (o == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
            val f = codec.outputFormat
            rate = f.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            channels = f.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            floatPcm = f.containsKey(MediaFormat.KEY_PCM_ENCODING) && f.getInteger(MediaFormat.KEY_PCM_ENCODING) == AudioFormat.ENCODING_PCM_FLOAT
          } else if (o >= 0) {
            val buf = codec.getOutputBuffer(o)!!.order(ByteOrder.LITTLE_ENDIAN)
            buf.position(info.offset)
            buf.limit(info.offset + info.size)
            val frames = if (floatPcm) info.size / (4 * channels) else info.size / (2 * channels)
            val mono = FloatArray(frames)
            for (k in 0 until frames) {
              var s = 0f
              for (c in 0 until channels) s += if (floatPcm) buf.getFloat() else buf.getShort() / 32768f
              mono[k] = s / channels
            }
            codec.releaseOutputBuffer(o, false)
            if (frames > 0) {
              out.add(mono)
              outFrames += frames
            }
            val eos = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
            // the first 30 s are all the model reads (plus a little for the resampler's edge)
            if (eos || outFrames >= (Voicemail.MAX_SECONDS + 1L) * rate) break
          }
        }
      } finally {
        Log.i(Lfm3AudioEngine.TAG, "decode $mime with ${runCatching { codec.name }.getOrDefault("?")}: in $inSamples samples / " +
          "$inBytes B (last at $lastTimeUs us), out $outFrames frames at $rate Hz, file ${"%.3f".format(total)} s")
        runCatching { codec.stop() }
        codec.release()
      }
      if (outFrames == 0L) throw UnusableAudio("the decoder returned no samples")
      val all = FloatArray(outFrames.toInt())
      var at = 0
      for (c in out) {
        System.arraycopy(c, 0, all, at, c.size)
        at += c.size
      }
      return resample(all, rate, 16000) to (if (total > 0) total else outFrames.toDouble() / rate)
    } finally {
      ex.release()
    }
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
  class Recorder(private val maxSeconds: Int = Voicemail.MAX_SECONDS, private val onTick: (Double, Boolean) -> Unit = { _, _ -> }) {
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
   * (or [stopNow]). [underruns] = AudioTrack's count at the end.
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
    @Volatile private var ended = false

    private val worker = thread(name = "player") {
      track.play()
      // the playback head is watched from the start: one long write (a whole voicemail) blocks until most of it has
      // played, so a watcher started after the first write would miss the first sound
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
        while (!stopped && track.playbackHeadPosition < written && System.nanoTime() - t0 < 5_000_000_000L) Thread.sleep(10)
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
