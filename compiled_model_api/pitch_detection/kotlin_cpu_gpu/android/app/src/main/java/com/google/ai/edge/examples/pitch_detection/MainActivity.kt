/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
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

package com.google.ai.edge.examples.pitch_detection

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
import android.view.Gravity
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.sin

/**
 * CREPE real-time pitch tuner. Listens to the mic, runs CREPE on the latest 1024-sample (16 kHz)
 * window, and shows the detected note, the cents offset (flat/sharp), and the frequency. The CNN
 * runs on the LiteRT CompiledModel GPU; per-frame normalization and the bin→Hz decode are host-side.
 */
class MainActivity : Activity() {

  private val tag = "CREPE"
  private val bg = Executors.newSingleThreadExecutor()
  private var detector: PitchDetector? = null
  private val listening = AtomicBoolean(false)

  private lateinit var status: TextView
  private lateinit var noteView: TextView
  private lateinit var centsBar: TextView
  private lateinit var hzView: TextView
  private lateinit var listen: Button

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(48, 110, 48, 40) }
    status = TextView(this).apply { textSize = 14f; text = "Loading CREPE on GPU…" }
    noteView = TextView(this).apply {
      textSize = 96f; gravity = Gravity.CENTER; setPadding(0, 56, 0, 8); text = "—"
      typeface = Typeface.DEFAULT_BOLD
    }
    centsBar = TextView(this).apply {
      textSize = 22f; gravity = Gravity.CENTER; typeface = Typeface.MONOSPACE; text = " "
    }
    hzView = TextView(this).apply {
      textSize = 18f; gravity = Gravity.CENTER; setTextColor(Color.GRAY); setPadding(0, 24, 0, 56); text = " "
    }
    listen = Button(this).apply {
      text = "🎤  Start tuner"; isEnabled = false
      setOnClickListener { toggle() }
    }
    root.addView(status); root.addView(noteView); root.addView(centsBar); root.addView(hzView); root.addView(listen)
    setContentView(root)

    bg.execute {
      try {
        val d = PitchDetector(this); detector = d
        // self-test: synthesize a 440 Hz sine (A4) and confirm the pipeline (no mic needed)
        val t = FloatArray(PitchDetector.WINDOW) { sin(2.0 * Math.PI * 440.0 * it / PitchDetector.SAMPLE_RATE).toFloat() }
        d.detect(t)                                  // warm up GPU
        val p = d.detect(t)
        Log.i(tag, "self-test 440 Hz -> ${p.note}${p.octave} ${"%.1f".format(p.hz)} Hz")
        runOnUiThread {
          status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
          status.text = "On-device GPU pitch detection ✓  (self-test 440 Hz → ${p.note}${p.octave} ${"%.1f".format(p.hz)} Hz)"
          listen.isEnabled = true
        }
      } catch (e: Throwable) {
        Log.e(tag, "load failed", e)
        runOnUiThread {
          status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2)); status.text = "FAIL: ${e.message}"
        }
      }
    }
  }

  private fun toggle() {
    if (listening.get()) { listening.set(false); listen.text = "🎤  Start tuner"; return }
    if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
      requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1); return
    }
    startListening()
  }

  override fun onRequestPermissionsResult(rc: Int, p: Array<out String>, g: IntArray) {
    super.onRequestPermissionsResult(rc, p, g)
    if (rc == 1 && g.isNotEmpty() && g[0] == PackageManager.PERMISSION_GRANTED) startListening()
  }

  private fun startListening() {
    listening.set(true); listen.text = "■  Stop"
    bg.execute {
      val sr = PitchDetector.SAMPLE_RATE
      val win = PitchDetector.WINDOW
      val minBuf = AudioRecord.getMinBufferSize(sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
      // UNPROCESSED keeps AGC/NS off for accurate pitch; fall back to MIC if unavailable.
      val rec = try {
        AudioRecord(MediaRecorder.AudioSource.UNPROCESSED, sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuf, sr))
      } catch (e: Throwable) {
        AudioRecord(MediaRecorder.AudioSource.MIC, sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuf, sr))
      }
      val ring = FloatArray(win)                       // rolling latest-1024-sample window
      val chunk = ShortArray(512)
      fun shiftIn(n: Int) {
        if (n <= 0) return
        val k = minOf(n, win)
        System.arraycopy(ring, k, ring, 0, win - k)
        for (i in 0 until k) ring[win - k + i] = chunk[n - k + i] / 32768f
      }
      try {
        rec.startRecording()
        while (listening.get()) {
          shiftIn(rec.read(chunk, 0, chunk.size))                       // blocking: advance one hop
          while (true) {                                                // drain backlog → stay current
            val m = rec.read(chunk, 0, chunk.size, AudioRecord.READ_NON_BLOCKING)
            if (m <= 0) break
            shiftIn(m)
          }
          val p = detector!!.detect(ring)
          runOnUiThread { render(p) }
        }
      } catch (e: Throwable) {
        Log.e(tag, "listen failed", e)
      } finally {
        try { rec.stop() } catch (_: Throwable) {}
        rec.release()
      }
    }
  }

  private fun render(p: PitchDetector.Pitch) {
    if (p.confidence < 0.5f) {                            // no clear pitch / silence
      noteView.text = "—"; noteView.setTextColor(Color.LTGRAY)
      centsBar.text = " "; hzView.text = "listening…"
      return
    }
    noteView.text = "${p.note}${p.octave}"
    val inTune = abs(p.cents) <= 5
    noteView.setTextColor(if (inTune) Color.rgb(0x2E, 0x7D, 0x32) else Color.rgb(0x1A, 0x1A, 0x1A))
    // cents gauge: 21 cells over ±50 cents, marker at the offset
    val pos = (((p.cents + 50).coerceIn(0, 100)) * 20 / 100)
    val sb = StringBuilder("♭ ")
    for (i in 0..20) sb.append(if (i == 10) '│' else if (i == pos) '●' else '·')
    sb.append(" ♯")
    centsBar.text = sb.toString()
    centsBar.setTextColor(if (inTune) Color.rgb(0x2E, 0x7D, 0x32) else Color.rgb(0xE6, 0x7E, 0x22))
    hzView.text = "%.1f Hz   %+d cents".format(p.hz, p.cents)
  }

  override fun onDestroy() {
    super.onDestroy(); listening.set(false); bg.shutdown(); detector?.close()
  }
}
