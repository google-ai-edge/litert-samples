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

package com.google.ai.edge.examples.music_transcription

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Owns the [Transcriber] and exposes a single [UiState] for the screen. Basic Pitch reuses native
 * GPU buffers, so both model creation and every transcription run on one confined worker. Audio is
 * captured from the mic or decoded from a picked clip, transcribed to note events, then drawn onto
 * a fixed-size piano-roll [Bitmap].
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val MODEL_FILE = "basicpitch.tflite"
    private const val MAX_SECONDS = 30
    private const val ROLL_WIDTH = 1024
    private const val ROLL_HEIGHT = 512

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var transcriber: Transcriber? = null
  @Volatile private var recording = false

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    // Load the model (from filesDir; pushed by install_to_device.sh) and wait for audio.
    viewModelScope.launch(inferenceDispatcher) {
      val modelFile = File(context.filesDir, MODEL_FILE)
      if (!modelFile.exists()) {
        _uiState.update {
          it.copy(
            errorMessage =
              "Model not found. Push it first with install_to_device.sh:\n" + modelFile.absolutePath
          )
        }
        return@launch
      }
      try {
        transcriber = Transcriber(context)
        _uiState.update {
          it.copy(
            isModelReady = true,
            statusMessage = "Ready — play some notes and record, or pick a clip.",
          )
        }
      } catch (t: Throwable) {
        _uiState.update { it.copy(errorMessage = t.message ?: "Failed to load model") }
      }
    }
  }

  /**
   * Toggles microphone capture. A second call while recording stops it (the volatile flag flips on
   * the caller thread while the capture loop is blocked in [AudioRecord.read]). On stop, the buffer
   * is transcribed. Requires RECORD_AUDIO, which the Activity grants before calling this.
   */
  @SuppressLint("MissingPermission")
  fun recordAndTranscribe() {
    if (recording) {
      recording = false
      return
    }
    if (transcriber == null) return
    recording = true
    _uiState.update {
      it.copy(
        isProcessing = true,
        statusMessage = "● Recording (up to ${MAX_SECONDS}s)…",
        errorMessage = null,
      )
    }
    viewModelScope.launch(inferenceDispatcher) {
      val sr = Transcriber.SR
      val min =
        AudioRecord.getMinBufferSize(
          sr,
          AudioFormat.CHANNEL_IN_MONO,
          AudioFormat.ENCODING_PCM_FLOAT,
        )
      // Buffer size is in BYTES and must be a multiple of the 4-byte float frame;
      // reserve at least one second (sr frames * 4 bytes).
      val recd =
        AudioRecord(
          MediaRecorder.AudioSource.UNPROCESSED,
          sr,
          AudioFormat.CHANNEL_IN_MONO,
          AudioFormat.ENCODING_PCM_FLOAT,
          maxOf(min, sr * Float.SIZE_BYTES),
        )
      val out = FloatArray(sr * MAX_SECONDS)
      var total = 0
      try {
        recd.startRecording()
        val buf = FloatArray(2205)
        while (recording && total < out.size) {
          val r = recd.read(buf, 0, minOf(buf.size, out.size - total), AudioRecord.READ_BLOCKING)
          if (r > 0) {
            System.arraycopy(buf, 0, out, total, r)
            total += r
          }
        }
      } finally {
        recd.stop()
        recd.release()
        recording = false
      }
      if (total >= sr / 2) transcribe(out.copyOf(total))
      else _uiState.update { it.copy(isProcessing = false, statusMessage = "Too short.") }
    }
  }

  /** Decodes a picked audio clip and transcribes it. */
  fun transcribeUri(uri: Uri) {
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update {
        it.copy(isProcessing = true, statusMessage = "Decoding…", errorMessage = null)
      }
      try {
        val x = AudioDecoder.decode(context, uri, MAX_SECONDS)
        check(x.size >= Transcriber.SR / 2) { "Clip too short." }
        transcribe(x)
      } catch (t: Throwable) {
        _uiState.update { it.copy(isProcessing = false, errorMessage = t.message ?: "Failed") }
      }
    }
  }

  /** Runs Basic Pitch over the PCM, decodes note events, and renders the piano roll. */
  private fun transcribe(x: FloatArray) {
    val tr = transcriber ?: return
    val t0 = System.nanoTime()
    val (note, onset) =
      tr.posteriorgrams(x) { w, n ->
        _uiState.update { it.copy(statusMessage = "Transcribing on GPU… window $w/$n") }
      }
    val events = tr.decode(note, onset)
    val ms = (System.nanoTime() - t0) / 1_000_000
    val durationSec = x.size.toDouble() / Transcriber.SR
    val image = renderPianoRoll(events, durationSec)
    _uiState.update {
      it.copy(
        isModelReady = true,
        isProcessing = false,
        resultImage = image,
        statusMessage =
          "✓ ${events.size} notes from ${x.size / Transcriber.SR}s in ${ms / 1000.0}s · " +
            "Basic Pitch, CompiledModel GPU",
        errorMessage = null,
      )
    }
  }

  /**
   * Draws the note events onto a fixed-size [Bitmap] (piano roll: time -> x, pitch -> y, brightness
   * -> confidence). This is the retargeted body of the old PianoRollView.onDraw / set — identical
   * rect math, drawn to a mutable Bitmap via [Canvas] instead of the View canvas.
   */
  private fun renderPianoRoll(notes: List<Transcriber.Note>, durationSec: Double): Bitmap {
    val width = ROLL_WIDTH
    val height = ROLL_HEIGHT
    val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    val c = Canvas(bitmap)
    var loNote = 21
    var hiNote = 108
    val duration = maxOf(durationSec, 0.001)
    if (notes.isNotEmpty()) {
      loNote = maxOf(21, notes.minOf { it.midi } - 2)
      hiNote = minOf(108, notes.maxOf { it.midi } + 2)
    }
    val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    val text =
      Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.GRAY
        textSize = 24f
      }
    paint.color = Color.rgb(0xF5, 0xF5, 0xF5)
    c.drawRect(0f, 0f, width.toFloat(), height.toFloat(), paint)
    val span = (hiNote - loNote + 1).coerceAtLeast(12)
    val rowH = height.toFloat() / span
    // octave guides (C rows)
    for (m in loNote..hiNote) {
      if (m % 12 == 0) {
        val y = height - (m - loNote + 1) * rowH
        paint.color = Color.rgb(0xE0, 0xE0, 0xE0)
        c.drawRect(0f, y, width.toFloat(), y + rowH, paint)
        c.drawText("C${m / 12 - 1}", 4f, y + rowH, text)
      }
    }
    for (n in notes) {
      val x0 = (n.startSec / duration * width).toFloat()
      val x1 = (n.endSec / duration * width).toFloat().coerceAtLeast(x0 + 4f)
      val y = height - (n.midi - loNote + 1) * rowH
      val a = (80 + 175 * n.amplitude.coerceIn(0f, 1f)).toInt()
      paint.color = Color.argb(a, 0x1E, 0x88, 0xE5)
      c.drawRoundRect(x0, y + 1, x1, y + rowH - 1, 4f, 4f, paint)
    }
    return bitmap
  }

  override fun onCleared() {
    super.onCleared()
    recording = false
    transcriber?.close()
  }
}
