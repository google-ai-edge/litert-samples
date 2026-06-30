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

package com.google.ai.edge.examples.parakeet

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.view.Gravity
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.concurrent.thread

/**
 * Parakeet ASR demo: record from the mic (up to 16 s) or transcribe the bundled sample, fully on the
 * LiteRT CompiledModel GPU. First launch needs the model pushed once via install_to_device.sh.
 */
class MainActivity : Activity() {

  private lateinit var status: TextView
  private lateinit var result: TextView
  private lateinit var recBtn: Button
  private lateinit var sampleBtn: Button

  private var asr: ParakeetAsr? = null
  @Volatile private var recording = false
  private var recThread: Thread? = null

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val root =
      LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(32, 48, 32, 32)
      }
    val title = TextView(this).apply {
      text = "Parakeet ASR — on-device GPU"
      textSize = 20f
    }
    status = TextView(this).apply {
      textSize = 13f
      setPadding(0, 16, 0, 16)
    }
    recBtn = Button(this).apply {
      text = "● Record"
      setOnClickListener { onRecord() }
    }
    sampleBtn = Button(this).apply {
      text = "Transcribe sample"
      setOnClickListener { onSample() }
    }
    result = TextView(this).apply {
      textSize = 18f
      setPadding(0, 24, 0, 0)
      gravity = Gravity.TOP
    }
    val scroll = ScrollView(this).apply { addView(result) }
    root.addView(title)
    root.addView(status)
    root.addView(recBtn)
    root.addView(sampleBtn)
    root.addView(scroll)
    setContentView(root)

    initAsr()
    if (
      ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) !=
        PackageManager.PERMISSION_GRANTED
    ) {
      ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), 1)
    }
  }

  private fun initAsr() {
    thread {
      try {
        val a = ParakeetAsr(this)
        asr = a
        runOnUiThread { status.text = "Model ready (GPU). Window 16 s." }
      } catch (e: Throwable) {
        runOnUiThread {
          status.text =
            "Model not loaded: ${e.message}\nPush it once with install_to_device.sh"
          recBtn.isEnabled = false
          sampleBtn.isEnabled = false
        }
      }
    }
  }

  private fun onRecord() {
    val a = asr ?: return
    if (!recording) {
      if (
        ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) !=
          PackageManager.PERMISSION_GRANTED
      ) {
        status.text = "Microphone permission needed."
        return
      }
      recording = true
      recBtn.text = "■ Stop"
      sampleBtn.isEnabled = false
      status.text = "Recording… (auto-stops at 16 s)"
      result.text = ""
      recThread = thread { recordLoop(a) }
    } else {
      recording = false // recordLoop finishes and transcribes
    }
  }

  private fun recordLoop(a: ParakeetAsr) {
    val minBuf =
      AudioRecord.getMinBufferSize(
        16000,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT,
      )
    val rec =
      AudioRecord(
        MediaRecorder.AudioSource.MIC,
        16000,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT,
        maxOf(minBuf, 4096),
      )
    val out = ByteArrayOutputStream()
    val buf = ShortArray(2048)
    val cap = ParakeetAsr.MAX_SAMPLES
    var total = 0
    rec.startRecording()
    while (recording && total < cap) {
      val r = rec.read(buf, 0, buf.size)
      if (r > 0) {
        val bb = ByteBuffer.allocate(r * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until r) bb.putShort(buf[i])
        out.write(bb.array())
        total += r
      }
    }
    rec.stop()
    rec.release()
    recording = false
    val audio = pcm16ToFloat(out.toByteArray())
    runOnUiThread {
      status.text = "Transcribing ${"%.1f".format(audio.size / 16000f)} s…"
      recBtn.text = "● Record"
    }
    runTranscribe(a, audio)
  }

  private fun onSample() {
    val a = asr ?: return
    sampleBtn.isEnabled = false
    result.text = ""
    status.text = "Transcribing sample…"
    thread {
      val audio =
        try {
          pcm16ToFloat(assets.open("sample.wav").readBytes(), skipWavHeader = true)
        } catch (e: Throwable) {
          runOnUiThread {
            status.text = "No sample.wav asset."
            sampleBtn.isEnabled = true
          }
          return@thread
        }
      runTranscribe(a, audio)
    }
  }

  private fun runTranscribe(a: ParakeetAsr, audio: FloatArray) {
    try {
      val res = a.transcribe(audio)
      android.util.Log.i("PARAKEET", "transcript=\"${res.text}\" mel=${res.melMs}ms gpu=${res.gpuMs}ms")
      runOnUiThread {
        result.text = res.text.ifBlank { "(no speech detected)" }
        status.text = "mel ${res.melMs} ms · GPU ${res.gpuMs} ms"
      }
    } catch (e: Throwable) {
      runOnUiThread { status.text = "Error: ${e.message}" }
    } finally {
      runOnUiThread {
        recBtn.isEnabled = true
        sampleBtn.isEnabled = true
      }
    }
  }

  /** PCM16 LE bytes -> float [-1,1]. Skips a standard 44-byte WAV header if requested. */
  private fun pcm16ToFloat(bytes: ByteArray, skipWavHeader: Boolean = false): FloatArray {
    val off = if (skipWavHeader && bytes.size > 44) 44 else 0
    val n = (bytes.size - off) / 2
    val bb = ByteBuffer.wrap(bytes, off, bytes.size - off).order(ByteOrder.LITTLE_ENDIAN)
    return FloatArray(n) { bb.short / 32768f }
  }

  override fun onDestroy() {
    recording = false
    recThread?.join(500)
    asr?.close()
    super.onDestroy()
  }
}
