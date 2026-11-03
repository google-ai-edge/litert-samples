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

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import java.io.File
import java.math.BigDecimal
import java.math.RoundingMode
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONArray
import org.json.JSONObject

/**
 * A short video as the model reads it: the way the model's reference processor reads a video with its sound
 * (`use_audio_in_video`; the model card: the model sees and hears a video at the same time).
 *   frames  2 per second (4 to 384), at np.linspace(0, N - 1, n) rounded, in groups of two; each group's first frame
 *           stretched to 512 x 512 (the bundle's vision encoder takes one picture: 256 rows per group), and the
 *           group's time written before it ("<1.3 seconds>")
 *   sound   the soundtrack as 16 kHz mono, through the audio encoder once; its rows are split at each group's first
 *           frame and follow that group's picture in the prompt
 * Only the first [Videos.MAX_SECONDS] are read: 13 s of video make a prompt of about 3,650 tokens, which leaves room
 * for an answer in the language model's 4,096-token context.
 */
class VideoInput(
  val name: String,
  /** The frames of the part that was read, and the video's frame rate (its frames over its duration). */
  val frameCount: Int,
  val fps: Double,
  /** The sampled frame indices (the reference processor's sample_frames) and the groups' times as the prompt writes them. */
  val indices: IntArray,
  val stamps: List<String>,
  /** Each group's first frame stretched to 512 x 512 (row-major RGB bytes): what the model sees. */
  val frames: List<ByteArray>,
  /** The soundtrack at 16 kHz mono; null when the video has no sound track, or one that could not be decoded. */
  val pcm16k: FloatArray?,
  /** The video's length and the part that was read (seconds). */
  val seconds: Double,
  val readSeconds: Double,
  /** How it was read: the tracks, the counts, the time of each step (the measuring tool's record). */
  val record: JSONObject,
  /** The decoded frames behind [frames] (width, height, RGB bytes), kept only for the measuring tool's dump. */
  val decoded: List<Triple<Int, Int, ByteArray>> = emptyList(),
) {
  val groups: Int get() = frames.size

  /** The video is longer than the part that was read. */
  val cut: Boolean get() = readSeconds < seconds - 0.05
}

/** Why a file cannot be a video, as the sentence the screen shows. */
class UnusableVideo(message: String) : Exception(message)

object Videos {
  /** The longest part of a video that is read (seconds): the prompt of 13 s leaves room for an answer in 4,096 tokens. */
  const val MAX_SECONDS = 13.0
  /** The reference processor's sampler: 2 frames per second, 4 to 384 frames, in groups of 2. */
  const val SAMPLE_FPS = 2.0
  const val MIN_FRAMES = 4
  const val MAX_FRAMES = 384
  const val TEMPORAL = 2
  const val SAMPLE_RATE = 16000
  /** The audio encoder's rows: hop 160 samples x stride 8 at 16 kHz = 0.08 s a row. */
  private const val SECONDS_PER_ROW = 160.0 * 8 / 16000
  private const val SIZE = VisionEncoder.SIZE

  // ---- the reference processor's arithmetic -------------------------------------------------------------------

  /** The frame indices the reference processor samples from [n] frames at [fps]: n * 2 / fps frames (truncated, 4 to
   *  384, at most n) at np.linspace(0, n - 1, k), rounded half to even. */
  fun sampleIndices(n: Int, fps: Double): IntArray {
    if (n <= 0) return IntArray(0)
    var k = (n / fps * SAMPLE_FPS).toInt()
    k = minOf(maxOf(k, MIN_FRAMES), MAX_FRAMES, n)
    if (k == 1) return intArrayOf(0)
    val step = (n - 1).toDouble() / (k - 1)
    return IntArray(k) { i -> if (i == k - 1) n - 1 else Math.rint(i.toDouble() * step).toInt() }
  }

  /** Python's round(x, 6): the exact binary value rounded half to even (the reference rounds the fps so). */
  fun round6(x: Double): Double = BigDecimal(x).setScale(6, RoundingMode.HALF_EVEN).toDouble()

  /** Python's "%.1f": the exact binary value rounded half to even, one decimal. */
  fun fixed1(x: Double): String = BigDecimal(x).setScale(1, RoundingMode.HALF_EVEN).toPlainString()

  /** Each group's time: its first and last frame's mean time, at the fps rounded to 6 decimals; the last group of an odd
   *  count repeats its frame. */
  fun stamps(indices: IntArray, fps: Double): List<String> {
    val f = round6(fps)
    val idx = indices.toMutableList()
    while (idx.size % TEMPORAL != 0) idx.add(idx.last())
    return (0 until idx.size step TEMPORAL).map { i -> fixed1((idx[i] + idx[i + TEMPORAL - 1]).toDouble() / (2 * f)) }
  }

  /** Where the soundtrack's [total] rows are split between the [groups]: 0, then each later group's first frame time
   *  over 0.08 s, rounded half to even and at most [total], then [total]; a bound never goes back. */
  fun bounds(indices: IntArray, fps: Double, total: Int, groups: Int): IntArray {
    val f = round6(fps)
    val b = IntArray(groups + 1)
    for (g in 1 until groups) b[g] = minOf(total, Math.rint(indices[g * TEMPORAL] / f / SECONDS_PER_ROW).toInt())
    b[groups] = total
    for (g in 1..groups) b[g] = maxOf(b[g], b[g - 1])
    return b
  }

  // ---- reading a file -----------------------------------------------------------------------------------------

  /**
   * Reads [uri]: the frames the model looks at and the soundtrack, within the first [MAX_SECONDS]. [progress] gets a
   * line for the screen per step. [keepDecoded] keeps the decoded frames (the measuring tool's dump).
   */
  fun read(context: Context, uri: Uri, name: String, keepDecoded: Boolean = false, progress: (String) -> Unit = {}): VideoInput {
    val rec = JSONObject().put("name", name)
    val t0 = System.nanoTime()
    // the video track: every frame's time, in the order they are shown
    val ex = MediaExtractor()
    val times: LongArray
    val durationUs: Long
    val width: Int
    val height: Int
    var scannedAll = true
    try {
      try {
        ex.setDataSource(context, uri, null)
      } catch (e: Exception) {
        throw UnusableVideo("Could not open $name as a video: ${e.message ?: e.javaClass.simpleName}")
      }
      val tracks = (0 until ex.trackCount).map { ex.getTrackFormat(it) }
      rec.put("tracks", JSONArray(tracks.map { it.getString(MediaFormat.KEY_MIME) ?: "?" }))
      val v = tracks.indexOfFirst { it.getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true }
      if (v < 0) throw UnusableVideo("Could not read $name as a video: the file has no video track.")
      val fmt = tracks[v]
      width = if (fmt.containsKey(MediaFormat.KEY_WIDTH)) fmt.getInteger(MediaFormat.KEY_WIDTH) else -1
      height = if (fmt.containsKey(MediaFormat.KEY_HEIGHT)) fmt.getInteger(MediaFormat.KEY_HEIGHT) else -1
      ex.selectTrack(v)
      // every frame's time (they come in decoding order), up to 2 s past the part that is read
      val list = ArrayList<Long>()
      var low = Long.MAX_VALUE
      while (true) {
        val t = ex.sampleTime
        if (t < 0) break
        low = minOf(low, t)
        if (t > low + ((MAX_SECONDS + 2) * 1e6).toLong()) {
          scannedAll = false
          break
        }
        list.add(t)
        if (!ex.advance()) break
      }
      times = list.toLongArray().also { it.sort() }
      durationUs = if (fmt.containsKey(MediaFormat.KEY_DURATION)) fmt.getLong(MediaFormat.KEY_DURATION) else -1L
      rec.put("video_mime", fmt.getString(MediaFormat.KEY_MIME)).put("track_size", JSONArray(listOf(width, height)))
        .put("video_track_duration_us", durationUs).put("scanned_all", scannedAll)
    } finally {
      ex.release()
    }
    if (times.isEmpty()) throw UnusableVideo("Could not read $name as a video: its video track has no frames.")
    val first = times.first()
    val span = (times.last() - first) / 1e6
    // The frame rate as the reference reads it from an MP4 (frames over the track's duration). The frames' times are
    // whole microseconds, so a constant rate read from them is a little off (359 frame steps of 1/30 s = 29.999999 fps):
    // a rate within 1e-5 of a whole number is that number.
    val byTimes = if (times.size > 1 && span > 0) (times.size - 1) / span else Double.NaN
    val byDuration = if (scannedAll && durationUs > 0) times.size / (durationUs / 1e6) else Double.NaN
    var fps = when {
      !byTimes.isNaN() && !byDuration.isNaN() && Math.abs(byDuration - byTimes) <= 1e-3 * byTimes -> byDuration
      !byTimes.isNaN() -> byTimes
      !byDuration.isNaN() -> byDuration
      else -> 30.0
    }
    if (Math.abs(fps - Math.rint(fps)) < 1e-5) fps = Math.rint(fps)
    val seconds = if (durationUs > 0) durationUs / 1e6 else span + 1 / fps
    val limitUs = first + (MAX_SECONDS * 1e6).toLong()
    val n = times.count { it < limitUs }
    val readSeconds = minOf(seconds, MAX_SECONDS)
    val indices = sampleIndices(n, fps)
    val stamps = stamps(indices, fps)
    val groups = stamps.size
    rec.put("frames_scanned", times.size).put("fps_by_times", if (byTimes.isNaN()) JSONObject.NULL else byTimes)
      .put("fps_by_duration", if (byDuration.isNaN()) JSONObject.NULL else byDuration).put("fps", fps)
      .put("seconds", seconds).put("frames_read", n).put("read_seconds", readSeconds)
      .put("indices", JSONArray(indices.toList())).put("stamps", JSONArray(stamps))
    val tIndex = System.nanoTime()

    // each group's first frame, decoded at its exact time, stretched to 512 x 512
    val frames = ArrayList<ByteArray>(groups)
    val decoded = ArrayList<Triple<Int, Int, ByteArray>>()
    var decodeNs = 0L
    var resizeNs = 0L
    val mmr = MediaMetadataRetriever()
    try {
      try {
        mmr.setDataSource(context, uri)
      } catch (e: Exception) {
        throw UnusableVideo("Could not open $name to read its frames: ${e.message ?: e.javaClass.simpleName}")
      }
      for (g in 0 until groups) {
        progress("Decoding the video: frame ${g + 1} of $groups")
        val at = times[indices[g * TEMPORAL]]
        val td = System.nanoTime()
        val bitmap = frameAt(mmr, at)
          ?: throw UnusableVideo("Could not read $name as a video: the frame at %.2f s cannot be decoded on this phone.".format((at - first) / 1e6))
        val w = bitmap.width
        val h = bitmap.height
        val rgb = rgbOf(bitmap)
        bitmap.recycle()
        decodeNs += System.nanoTime() - td
        val tr = System.nanoTime()
        frames.add(resizeBicubic(rgb, w, h, SIZE, SIZE))
        resizeNs += System.nanoTime() - tr
        if (keepDecoded) decoded.add(Triple(w, h, rgb))
        if (g == 0) rec.put("frame_size", JSONArray(listOf(w, h)))
      }
    } finally {
      runCatching { mmr.release() }
    }
    rec.put("frames_decode_ms", decodeNs / 1e6).put("frames_resize_ms", resizeNs / 1e6)
    val tFrames = System.nanoTime()

    // the soundtrack
    progress("Decoding the video's sound")
    val sound = JSONObject()
    val pcm16k = try {
      decodeAudio(context, uri, readSeconds, sound)
    } catch (e: Exception) {
      sound.put("error", "${e.javaClass.simpleName}: ${e.message}".take(300))
      null
    }
    rec.put("sound", sound)
    val tSound = System.nanoTime()
    rec.put("index_ms", (tIndex - t0) / 1e6).put("frames_ms", (tFrames - tIndex) / 1e6).put("sound_ms", (tSound - tFrames) / 1e6)
      .put("read_ms", (tSound - t0) / 1e6)
    return VideoInput(name, n, fps, indices, stamps, frames, pcm16k, seconds, readSeconds, rec, decoded)
  }

  /** The frame shown at [timeUs] (its exact time: the frame itself, not the nearest key frame), as ARGB_8888. */
  private fun frameAt(mmr: MediaMetadataRetriever, timeUs: Long): Bitmap? {
    val b = if (Build.VERSION.SDK_INT >= 30) {
      val p = MediaMetadataRetriever.BitmapParams().apply { preferredConfig = Bitmap.Config.ARGB_8888 }
      mmr.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST, p)
    } else {
      mmr.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST)
    } ?: return null
    if (b.config == Bitmap.Config.ARGB_8888) return b
    return b.copy(Bitmap.Config.ARGB_8888, false).also { b.recycle() }
  }

  private fun rgbOf(b: Bitmap): ByteArray {
    val w = b.width
    val h = b.height
    val px = IntArray(w * h)
    b.getPixels(px, 0, w, 0, 0, w, h)
    val rgb = ByteArray(w * h * 3)
    for (i in px.indices) {
      val c = px[i]
      rgb[3 * i] = (c shr 16 and 0xff).toByte()
      rgb[3 * i + 1] = (c shr 8 and 0xff).toByte()
      rgb[3 * i + 2] = (c and 0xff).toByte()
    }
    return rgb
  }

  /**
   * The first [seconds] of the audio track as 16 kHz mono floats; null without an audio track or when it holds only
   * zeros. The decoder gets the track's format without its encoder delay and padding, so that it returns every decoded
   * sample; the delay and the padding the container names are then cut here, once (the reference's ffmpeg drops them
   * by the MP4 edit list). The channels are averaged, then resampled ([resample]).
   */
  private fun decodeAudio(context: Context, uri: Uri, seconds: Double, rec: JSONObject): FloatArray? {
    val ex = MediaExtractor()
    var codec: MediaCodec? = null
    try {
      ex.setDataSource(context, uri, null)
      val track = (0 until ex.trackCount).firstOrNull { ex.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true }
      if (track == null) {
        rec.put("track", JSONObject.NULL)
        return null
      }
      val fmt = ex.getTrackFormat(track)
      val mime = fmt.getString(MediaFormat.KEY_MIME)!!
      val rate = fmt.getInteger(MediaFormat.KEY_SAMPLE_RATE)
      val channels = fmt.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
      val delay = if (fmt.containsKey(KEY_ENCODER_DELAY)) fmt.getInteger(KEY_ENCODER_DELAY) else 0
      val padding = if (fmt.containsKey(KEY_ENCODER_PADDING)) fmt.getInteger(KEY_ENCODER_PADDING) else 0
      rec.put("track", mime).put("track_rate", rate).put("track_channels", channels).put("encoder_delay", delay)
        .put("encoder_padding", padding)
        .put("track_duration_us", if (fmt.containsKey(MediaFormat.KEY_DURATION)) fmt.getLong(MediaFormat.KEY_DURATION) else -1L)
      val cfg = if (Build.VERSION.SDK_INT >= 29) {
        MediaFormat(fmt).apply {
          removeKey(KEY_ENCODER_DELAY)
          removeKey(KEY_ENCODER_PADDING)
        }
      } else {
        MediaFormat.createAudioFormat(mime, rate, channels).apply {
          for (key in listOf("csd-0", "csd-1", "csd-2")) if (fmt.containsKey(key)) setByteBuffer(key, fmt.getByteBuffer(key))
          if (fmt.containsKey(MediaFormat.KEY_MAX_INPUT_SIZE)) setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, fmt.getInteger(MediaFormat.KEY_MAX_INPUT_SIZE))
          if (fmt.containsKey(MediaFormat.KEY_IS_ADTS)) setInteger(MediaFormat.KEY_IS_ADTS, fmt.getInteger(MediaFormat.KEY_IS_ADTS))
        }
      }
      ex.selectTrack(track)
      val c = MediaCodec.createDecoderByType(mime)
      codec = c
      c.configure(cfg, null, null, 0)
      c.start()
      // decode until the end, or well past the part that is read
      val chunks = ArrayList<FloatArray>()
      var total = 0L
      var outRate = rate
      var outChannels = channels
      var floatOut = false
      var firstPtsUs = Long.MIN_VALUE
      var inputDone = false
      var reachedEnd = false
      val info = MediaCodec.BufferInfo()
      val t0 = System.nanoTime()
      while (true) {
        if (System.nanoTime() - t0 > 60_000_000_000L) throw IllegalStateException("the sound track did not finish decoding in 60 s")
        if (!inputDone) {
          val i = c.dequeueInputBuffer(10_000)
          if (i >= 0) {
            val buf = c.getInputBuffer(i)!!
            val size = ex.readSampleData(buf, 0)
            // the part that is read, and a second more (the decoder's output trails its input)
            val enough = total / outChannels > (seconds + 1.0) * outRate
            if (size < 0 || enough) {
              c.queueInputBuffer(i, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
              inputDone = true
              reachedEnd = size < 0
            } else {
              c.queueInputBuffer(i, 0, size, ex.sampleTime, 0)
              ex.advance()
            }
          }
        }
        val o = c.dequeueOutputBuffer(info, 10_000)
        if (o == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
          val of = c.outputFormat
          outRate = of.getInteger(MediaFormat.KEY_SAMPLE_RATE)
          outChannels = of.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
          floatOut = of.containsKey(MediaFormat.KEY_PCM_ENCODING) && of.getInteger(MediaFormat.KEY_PCM_ENCODING) == android.media.AudioFormat.ENCODING_PCM_FLOAT
        } else if (o >= 0) {
          if (info.size > 0) {
            if (firstPtsUs == Long.MIN_VALUE) firstPtsUs = info.presentationTimeUs
            val buf = c.getOutputBuffer(o)!!.order(ByteOrder.nativeOrder())
            buf.position(info.offset)
            buf.limit(info.offset + info.size)
            val chunk = if (floatOut) FloatArray(info.size / 4).also { buf.asFloatBuffer().get(it) }
            else ShortArray(info.size / 2).also { buf.asShortBuffer().get(it) }.let { s -> FloatArray(s.size) { s[it] / 32768f } }
            chunks.add(chunk)
            total += chunk.size
          }
          c.releaseOutputBuffer(o, false)
          if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
        }
      }
      // interleaved -> one array; the delay and the padding cut (in the decoder's rate: an HE-AAC track decodes at twice
      // the rate its header names, so the cut scales with it)
      val all = FloatArray(total.toInt())
      var at = 0
      for (ch in chunks) {
        System.arraycopy(ch, 0, all, at, ch.size)
        at += ch.size
      }
      val framesAll = all.size / outChannels
      val scale = outRate.toDouble() / rate
      val cutStart = minOf(framesAll, Math.round(delay * scale).toInt())
      // the padding is cut only when the decoder ran to the end of the track (a long track is stopped early)
      val cutEnd = if (reachedEnd) minOf(framesAll - cutStart, Math.round(padding * scale).toInt()) else 0
      val frames = maxOf(0, framesAll - cutStart - cutEnd)
      val mono = FloatArray(frames) { i ->
        var s = 0f
        val base = (cutStart + i) * outChannels
        for (k in 0 until outChannels) s += all[base + k]
        s / outChannels
      }
      rec.put("decoder", c.name).put("decoded_rate", outRate).put("decoded_channels", outChannels).put("decoded_float", floatOut)
        .put("decoded_frames", framesAll).put("cut_start", cutStart).put("cut_end", cutEnd).put("first_pts_us", firstPtsUs)
      val tr = System.nanoTime()
      var pcm = resample(mono, outRate, SAMPLE_RATE)
      val max = (seconds * SAMPLE_RATE).toInt()
      if (pcm.size > max) pcm = pcm.copyOf(max)
      rec.put("resample_ms", (System.nanoTime() - tr) / 1e6).put("samples_16k", pcm.size)
      var last = -1
      for (i in pcm.indices.reversed()) if (pcm[i] != 0f) {
        last = i
        break
      }
      // up to the last sample that is not 0 (the reference's rule; a track of zeros = no sound)
      rec.put("samples_used", last + 1)
      return if (last < 0) null else pcm.copyOf(last + 1)
    } finally {
      runCatching { codec?.stop() }
      runCatching { codec?.release() }
      ex.release()
    }
  }

  private const val KEY_ENCODER_DELAY = "encoder-delay"
  private const val KEY_ENCODER_PADDING = "encoder-padding"

  // ---- the resampler and the resize -----------------------------------------------------------------------------

  /**
   * [x] at [from] Hz -> [to] Hz with the defaults of ffmpeg's resampler (the reference's soundtrack is ffmpeg's):
   * a windowed sinc (Kaiser, beta 9) with its cutoff at 0.97 of the lower rate's Nyquist frequency and 32 taps per side
   * at the output rate, its taps summing to 1, evaluated at each output sample's exact input position (the rates'
   * ratio as a fraction); past the start the signal is mirrored about its first sample, past the end it is mirrored
   * after its last sample (both as ffmpeg's output has it).
   */
  fun resample(x: FloatArray, from: Int, to: Int): FloatArray {
    if (from == to || x.isEmpty()) return x
    val g = gcd(from, to)
    val up = to / g
    val down = from / g
    val factor = minOf(1.0, to * CUTOFF / from)
    var taps = Math.ceil(FILTER_SIZE / factor).toInt()
    if (taps % 2 != 0) taps++
    val center = (taps - 1) / 2
    // one row of taps per output phase (the output sample's position between two input samples, in 1 / up steps)
    val table = Array(up) { DoubleArray(taps) }
    var norm = 0.0
    for (ph in 0 until up) {
      for (i in 0 until taps) {
        val t = (i - center).toDouble() - ph.toDouble() / up
        val xx = Math.PI * t * factor
        var y = if (xx == 0.0) 1.0 else Math.sin(xx) / xx
        val w = 2.0 * xx / (factor * taps * Math.PI)
        y *= besselI0(KAISER_BETA * Math.sqrt(maxOf(1 - w * w, 0.0)))
        table[ph][i] = y
        if (ph == 0) norm += y
      }
    }
    for (row in table) for (i in row.indices) row[i] /= norm
    val n = x.size
    val outSize = ((n.toLong() * up + down - 1) / down).toInt()
    val out = FloatArray(outSize)
    for (j in 0 until outSize) {
      val pos = j.toLong() * down
      val p = (pos / up).toInt()
      val ph = (pos % up).toInt()
      val row = table[ph]
      var s = 0.0
      val start = p - center
      for (i in 0 until taps) {
        var k = start + i
        if (k < 0) k = -k
        if (k >= n) k = 2 * n - 1 - k
        if (k < 0 || k >= n) k = if (k < 0) 0 else n - 1
        s += x[k] * row[i]
      }
      out[j] = s.toFloat()
    }
    return out
  }

  private const val FILTER_SIZE = 32.0
  private const val CUTOFF = 0.97
  private const val KAISER_BETA = 9.0

  private fun gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)

  /** The modified Bessel function of the first kind, order 0 (the Kaiser window). */
  private fun besselI0(x: Double): Double {
    var sum = 1.0
    var term = 1.0
    val q = x * x / 4
    var k = 1
    while (true) {
      term *= q / (k.toDouble() * k)
      sum += term
      if (term < sum * 1e-17) break
      k++
    }
    return sum
  }

  /**
   * Pillow's Image.resize(size, BICUBIC) of an RGB picture, bit for bit (Pillow 13's Resample.c): the reference stretches
   * each frame so. A separable convolution with the bicubic kernel (a = -0.5), its support widened by the scale when
   * shrinking, the weights in 22-bit fixed point, each pass rounded to 8 bits; the horizontal pass first, unless the
   * height shrinks by more than twice what the width does (then the vertical pass first, as Pillow does).
   */
  fun resizeBicubic(src: ByteArray, w: Int, h: Int, ow: Int, oh: Int): ByteArray {
    val horizontalFirst = !((h - oh) > 0 && (h - oh) > (w - ow) * 2)
    var cur = src
    var cw = w
    var ch = h
    if (horizontalFirst) {
      if (ow != cw) cur = passHorizontal(cur, cw, ch, ow).also { cw = ow }
      if (oh != ch) cur = passVertical(cur, cw, ch, oh).also { ch = oh }
    } else {
      if (oh != ch) cur = passVertical(cur, cw, ch, oh).also { ch = oh }
      if (ow != cw) cur = passHorizontal(cur, cw, ch, ow).also { cw = ow }
    }
    return if (cur === src) src.copyOf() else cur
  }

  /** One horizontal pass: [w] x [h] -> [ow] x [h]. */
  private fun passHorizontal(src: ByteArray, w: Int, h: Int, ow: Int): ByteArray {
    val (bounds, kk, ksize) = coeffs(w, ow)
    val out = ByteArray(ow * h * 3)
    for (y in 0 until h) {
      val row = y * w * 3
      val orow = y * ow * 3
      for (xx in 0 until ow) {
        val xmin = bounds[2 * xx]
        val xmax = bounds[2 * xx + 1]
        val k = xx * ksize
        var s0 = HALF
        var s1 = HALF
        var s2 = HALF
        for (x in 0 until xmax) {
          val p = row + (x + xmin) * 3
          val c = kk[k + x]
          s0 += (src[p].toInt() and 0xff) * c
          s1 += (src[p + 1].toInt() and 0xff) * c
          s2 += (src[p + 2].toInt() and 0xff) * c
        }
        out[orow + xx * 3] = clip8(s0)
        out[orow + xx * 3 + 1] = clip8(s1)
        out[orow + xx * 3 + 2] = clip8(s2)
      }
    }
    return out
  }

  /** One vertical pass: [w] x [h] -> [w] x [oh]. */
  private fun passVertical(src: ByteArray, w: Int, h: Int, oh: Int): ByteArray {
    val (bounds, kk, ksize) = coeffs(h, oh)
    val out = ByteArray(w * oh * 3)
    for (yy in 0 until oh) {
      val ymin = bounds[2 * yy]
      val ymax = bounds[2 * yy + 1]
      val k = yy * ksize
      val orow = yy * w * 3
      for (xx in 0 until w) {
        var s0 = HALF
        var s1 = HALF
        var s2 = HALF
        for (y in 0 until ymax) {
          val p = ((y + ymin) * w + xx) * 3
          val c = kk[k + y]
          s0 += (src[p].toInt() and 0xff) * c
          s1 += (src[p + 1].toInt() and 0xff) * c
          s2 += (src[p + 2].toInt() and 0xff) * c
        }
        out[orow + xx * 3] = clip8(s0)
        out[orow + xx * 3 + 1] = clip8(s1)
        out[orow + xx * 3 + 2] = clip8(s2)
      }
    }
    return out
  }

  private const val PRECISION_BITS = 32 - 8 - 2
  private const val HALF = 1 shl (PRECISION_BITS - 1)

  private fun clip8(v: Int): Byte {
    val x = v shr PRECISION_BITS
    return (if (x < 0) 0 else if (x > 255) 255 else x).toByte()
  }

  private fun bicubic(v: Double): Double {
    val a = -0.5
    val x = if (v < 0.0) -v else v
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a
    return 0.0
  }

  /** Pillow's precompute_coeffs + normalize_coeffs_8bpc for [inSize] -> [outSize]: (xmin, count) per output, the
   *  fixed-point weights ([ksize] per output). */
  private fun coeffs(inSize: Int, outSize: Int): Triple<IntArray, IntArray, Int> {
    val scale = inSize.toDouble() / outSize
    val filterscale = if (scale < 1.0) 1.0 else scale
    val support = 2.0 * filterscale
    val ksize = Math.ceil(support).toInt() * 2 + 1
    val bounds = IntArray(outSize * 2)
    val kk = IntArray(outSize * ksize)
    val k = DoubleArray(ksize)
    val ss = 1.0 / filterscale
    for (xx in 0 until outSize) {
      val center = (xx + 0.5) * scale
      var ww = 0.0
      var xmin = (center - support + 0.5).toInt()
      if (xmin < 0) xmin = 0
      var xmax = (center + support + 0.5).toInt()
      if (xmax > inSize) xmax = inSize
      xmax -= xmin
      for (x in 0 until xmax) {
        val wgt = bicubic(((x + xmin) - center + 0.5) * ss)
        k[x] = wgt
        ww += wgt
      }
      for (x in 0 until xmax) if (ww != 0.0) k[x] /= ww
      for (x in 0 until ksize) {
        val v = if (x < xmax) k[x] else 0.0
        kk[xx * ksize + x] = if (v < 0) (-0.5 + v * (1 shl PRECISION_BITS)).toInt() else (0.5 + v * (1 shl PRECISION_BITS)).toInt()
      }
      bounds[2 * xx] = xmin
      bounds[2 * xx + 1] = xmax
    }
    return Triple(bounds, kk, ksize)
  }

  /** The measuring tool's dump of [v]: each group's 512 x 512 frame (and, when kept, the decoded frame behind it) as
   *  RGB bytes, the soundtrack as 16 kHz float32, and how they were read (<dir>/<tag>_video.json). */
  fun dump(v: VideoInput, dir: File, tag: String, extra: JSONObject) {
    dir.mkdirs()
    val files = JSONArray()
    for ((g, f) in v.frames.withIndex()) {
      File(dir, "${tag}_frame$g.rgb").writeBytes(f)
      files.put("${tag}_frame$g.rgb")
    }
    val dec = JSONArray()
    for ((g, d) in v.decoded.withIndex()) {
      File(dir, "${tag}_decoded$g.rgb").writeBytes(d.third)
      dec.put(JSONObject().put("file", "${tag}_decoded$g.rgb").put("width", d.first).put("height", d.second))
    }
    v.pcm16k?.let { pcm ->
      val bb = ByteBuffer.allocate(pcm.size * 4).order(ByteOrder.LITTLE_ENDIAN)
      for (s in pcm) bb.putFloat(s)
      File(dir, "${tag}_audio16k.f32").writeBytes(bb.array())
    }
    File(dir, "${tag}_video.json").writeText(JSONObject().put("record", v.record).put("frames", files).put("decoded", dec)
      .put("audio16k", if (v.pcm16k != null) "${tag}_audio16k.f32" else JSONObject.NULL).put("prepare", extra).toString(1))
  }
}
