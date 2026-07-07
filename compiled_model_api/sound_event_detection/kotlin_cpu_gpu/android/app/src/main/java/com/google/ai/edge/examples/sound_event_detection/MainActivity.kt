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

package com.google.ai.edge.examples.sound_event_detection

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Typeface
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors

/**
 * PANNs CNN14 audio-tagging demo. Records 10 s of audio and lists the top AudioSet sound-event tags
 * (527 classes: speech, music, instruments, animals, vehicles, alarms, household sounds, …). This is
 * multi-label — several tags can be high at once. The CNN runs on the LiteRT CompiledModel GPU; the
 * log-mel front-end is computed on the CPU (it overflows fp16 on the GPU).
 */
class MainActivity : Activity() {

  private val tag = "SoundEventDetection"
  private val bg = Executors.newSingleThreadExecutor()
  private var tagger: AudioTagger? = null
  private lateinit var status: TextView
  private lateinit var result: TextView
  private lateinit var record: Button

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val root = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(48, 110, 48, 40)
    }
    status = TextView(this).apply {
        textSize = 14f
        text = "Loading PANNs CNN14 on GPU…"
    }
    val howto = TextView(this).apply {
      textSize = 15f
      setPadding(0, 36, 0, 8)
      text = "Tap Record, then let it listen for 10 seconds to any sound — speech, music, an " +
        "instrument, an animal, a vehicle, an alarm, an appliance…"
    }
    val note = TextView(this).apply {
      textSize = 12f
      setTextColor(Color.GRAY)
      text = "527 AudioSet sound-event classes · multi-label (several tags can be high)"
    }
    result = TextView(this).apply {
      textSize = 18f
      setPadding(0, 44, 0, 44)
      text = "—"
      typeface = Typeface.MONOSPACE
    }
    record = Button(this).apply {
      text = "🎤  Record 10 s & tag"
      isEnabled = false
      setOnClickListener { ensureMicThenRecord() }
    }
    root.addView(status)
    root.addView(howto)
    root.addView(note)
    root.addView(result)
    root.addView(record)
    setContentView(root)

    bg.execute {
      try {
        val t = AudioTagger(this)
        tagger = t
        val clip = readFloats(assets.open("test_audio.bin").readBytes())
        t.tag(clip)                            // warm up GPU
        val r = t.tag(clip)
        Log.i(tag, "bundled clip -> ${r.tags.firstOrNull()?.label} (mel ${r.melMs} ms, gpu ${r.gpuMs} ms)")
        runOnUiThread {
          status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
          status.text = "On-device GPU audio tagging ✓  (self-test → \"${r.tags.first().label}\", " +
            "mel ${r.melMs} ms + gpu ${r.gpuMs} ms)"
          record.isEnabled = true
        }
      } catch (e: Throwable) {
        Log.e(tag, "load failed", e)
        runOnUiThread {
          status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2))
          status.text = "FAIL: ${e.message}"
        }
      }
    }
  }

  private fun ensureMicThenRecord() {
    if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
      requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1)
      return
    }
    recordAndTag()
  }

  override fun onRequestPermissionsResult(rc: Int, p: Array<out String>, g: IntArray) {
    super.onRequestPermissionsResult(rc, p, g)
    if (rc == 1 && g.isNotEmpty() && g[0] == PackageManager.PERMISSION_GRANTED) recordAndTag()
  }

  private fun recordAndTag() {
    record.isEnabled = false
    bg.execute {
      try {
        val audio = recordClip { secLeft ->
          runOnUiThread {
            result.setTextColor(Color.rgb(0xD3, 0x2F, 0x2F))
            result.text = "🔴 Listening…  ${secLeft}s"
          }
        }
        val r = tagger!!.tag(audio)
        Log.i(tag, "mic -> ${r.tags.firstOrNull()?.label} (mel ${r.melMs} ms, gpu ${r.gpuMs} ms)")
        runOnUiThread {
          result.setTextColor(Color.rgb(0x1A, 0x1A, 0x1A))
          result.text = formatTags(r)
          record.isEnabled = true
        }
      } catch (e: Throwable) {
        Log.e(tag, "record failed", e)
        runOnUiThread {
            result.text = "Record failed: ${e.message}"
            record.isEnabled = true
        }
      }
    }
  }

  /** A simple text bar chart of the top tags. */
  private fun formatTags(r: AudioTagger.Result): String {
    val sb = StringBuilder()
    for (t in r.tags) {
      if (t.prob < 0.01f) continue
      val bars = (t.prob * 20).toInt().coerceIn(0, 20)
      sb.append("█".repeat(bars)).append("░".repeat(20 - bars))
      sb.append("  %4.1f%%  ".format(t.prob * 100)).append(t.label).append('\n')
    }
    sb.append("\n(mel ${r.melMs} ms + gpu ${r.gpuMs} ms)")
    return sb.toString()
  }

  /** Capture CLIP_SAMPLES of mono 32 kHz PCM from the mic, returned as float32 in [-1,1]. */
  private fun recordClip(onTick: (Int) -> Unit): FloatArray {
    val sr = AudioTagger.SAMPLES                 // 320000
    val minBuf = AudioRecord.getMinBufferSize(
      MelSpectrogram.SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
    val rec = AudioRecord(
      MediaRecorder.AudioSource.MIC, MelSpectrogram.SAMPLE_RATE,
      AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
      maxOf(minBuf, MelSpectrogram.SAMPLE_RATE * 2),
    )
    val pcm = ShortArray(sr)
    rec.startRecording()
    var off = 0
    var lastTick = -1
    while (off < sr) {
      val n = rec.read(pcm, off, sr - off)
      if (n <= 0) break
      off += n
      val secLeft = (sr - off + MelSpectrogram.SAMPLE_RATE - 1) / MelSpectrogram.SAMPLE_RATE  // ceil, 10→0
      if (secLeft != lastTick) {
          onTick(secLeft)
          lastTick = secLeft
      }
    }
    rec.stop()
    rec.release()
    val out = FloatArray(sr)
    for (i in 0 until sr) {
      out[i] = pcm[i] / 32768f   // PCM16 -> [-1,1] (PANNs uses raw scale, no peak-norm)
    }
    return out
  }

  private fun readFloats(b: ByteArray): FloatArray {
    val bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
    return FloatArray(b.size / 4) { bb.float }
  }

  override fun onDestroy() {
      super.onDestroy()
      bg.shutdown()
      tagger?.close()
  }
}
