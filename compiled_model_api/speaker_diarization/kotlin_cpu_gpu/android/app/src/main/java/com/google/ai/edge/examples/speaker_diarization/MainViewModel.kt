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

package com.google.ai.edge.examples.speaker_diarization

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
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
 * Owns the [Diarizer] and exposes a single [UiState]. A conversation is captured from the mic or a
 * picked clip, diarized (PyanNet segmentation on CPU, WeSpeaker embeddings on the LiteRT
 * CompiledModel GPU, host-side clustering), and drawn as a per-speaker timeline. Tapping a speaker
 * plays back only their segments through an [AudioTrack]. The models reuse native buffers, so
 * loading, mic capture, and every diarize run share one confined worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val MAX_SECONDS = 120
    private const val MIN_SECONDS = 2

    /** Timeline geometry, mirroring the pixel proportions of the original custom View. */
    private const val TIMELINE_WIDTH_PX = 1000
    private const val ROW_HEIGHT_PX = 64
    private const val ROW_GAP_PX = 12
    private const val LABEL_WIDTH_PX = 130f
    private const val TIMELINE_TOP_PX = 20f

    /** Distinct per-speaker colors (shared by the timeline rows and the play buttons). */
    val SPEAKER_COLORS =
      intArrayOf(
        Color.rgb(0x42, 0x85, 0xF4),
        Color.rgb(0xEA, 0x43, 0x35),
        Color.rgb(0xFB, 0xBC, 0x05),
        Color.rgb(0x34, 0xA8, 0x53),
        Color.rgb(0xAB, 0x47, 0xBC),
        Color.rgb(0x00, 0xAC, 0xC1),
      )

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var diarizer: Diarizer? = null
  private var pcm: FloatArray? = null
  private var result: Diarizer.Result? = null
  private var player: AudioTrack? = null

  @Volatile private var recording = false

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState =
    MutableStateFlow(UiState(statusMessage = context.getString(R.string.status_loading)))
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    viewModelScope.launch(inferenceDispatcher) {
      val missing =
        listOf("wespeaker_emb_fp16.tflite", "pyannote_seg30.onnx").firstOrNull {
          !File(context.filesDir, it).exists()
        }
      if (missing != null) {
        val path = File(context.filesDir, missing).absolutePath
        val message = context.getString(R.string.error_model_missing, path)
        _uiState.update { it.copy(errorMessage = message) }
        return@launch
      }
      try {
        diarizer = Diarizer(context)
        _uiState.update {
          it.copy(isModelReady = true, statusMessage = context.getString(R.string.status_ready))
        }
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(errorMessage = t.message ?: context.getString(R.string.error_model_load))
        }
      }
    }
  }

  /** Starts a mic recording, or stops the running one so it is analyzed. */
  fun toggleRecording() {
    if (recording) {
      recording = false
      return
    }
    val recordingStatus = context.getString(R.string.status_recording)
    _uiState.update { it.copy(isRecording = true, statusMessage = recordingStatus) }
    viewModelScope.launch(inferenceDispatcher) {
      val captured = recordClip()
      _uiState.update { it.copy(isRecording = false) }
      if (captured.size >= Diarizer.SR * MIN_SECONDS) {
        diarize(captured)
      } else {
        _uiState.update { it.copy(statusMessage = context.getString(R.string.status_too_short)) }
      }
    }
  }

  /** Decodes a picked audio/video clip and diarizes it. */
  fun process(uri: Uri) {
    viewModelScope.launch(inferenceDispatcher) {
      try {
        _uiState.update { it.copy(statusMessage = context.getString(R.string.status_decoding)) }
        val audio = AudioDecoder.decode(context, uri, MAX_SECONDS)
        check(audio.size >= Diarizer.SR * MIN_SECONDS) {
          context.getString(R.string.status_too_short)
        }
        diarize(audio)
      } catch (t: Throwable) {
        _uiState.update { it.copy(errorMessage = t.message ?: "Failed to decode clip") }
      }
    }
  }

  private fun recordClip(): FloatArray {
    val sampleRate = Diarizer.SR
    val minBuffer =
      AudioRecord.getMinBufferSize(
        sampleRate,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_FLOAT,
      )
    val record =
      AudioRecord(
        MediaRecorder.AudioSource.MIC,
        sampleRate,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_FLOAT,
        maxOf(minBuffer, sampleRate * 2),
      )
    val out = FloatArray(sampleRate * MAX_SECONDS)
    var total = 0
    recording = true
    try {
      record.startRecording()
      val buffer = FloatArray(1600)
      while (recording && total < out.size) {
        val want = minOf(buffer.size, out.size - total)
        val read = record.read(buffer, 0, want, AudioRecord.READ_BLOCKING)
        if (read > 0) {
          System.arraycopy(buffer, 0, out, total, read)
          total += read
        }
      }
    } finally {
      runCatching { record.stop() }
      record.release()
      recording = false
    }
    return out.copyOf(total)
  }

  private fun diarize(audio: FloatArray) {
    val diarizer = diarizer ?: return
    pcm = audio
    _uiState.update { it.copy(isAnalyzing = true) }
    val start = System.nanoTime()
    val res =
      diarizer.diarize(audio) { message ->
        _uiState.update { it.copy(statusMessage = message) }
      }
    val elapsedMs = (System.nanoTime() - start) / 1_000_000
    result = res
    val durationSec = audio.size.toDouble() / Diarizer.SR
    _uiState.update {
      it.copy(
        isAnalyzing = false,
        timeline = drawTimeline(res.segments, durationSec),
        speakers =
          res.perSpeaker.toSortedMap().map { (spk, dur) ->
            SpeakerRow(spk, dur, SPEAKER_COLORS[spk % SPEAKER_COLORS.size])
          },
        statusMessage =
          context.getString(
            R.string.status_done,
            res.numSpeakers,
            durationSec.toInt(),
            elapsedMs / 1000.0,
          ),
      )
    }
  }

  /** Draws the colored per-speaker timeline onto a bitmap (verbatim geometry of the old View). */
  private fun drawTimeline(segments: List<Diarizer.Segment>, durationSec: Double): Bitmap {
    val speakers = segments.map { it.speaker }.distinct().sorted()
    val duration = maxOf(durationSec, 0.001)
    val height = maxOf(1, speakers.size) * ROW_HEIGHT_PX + 40
    val bitmap = Bitmap.createBitmap(TIMELINE_WIDTH_PX, height, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(bitmap)
    canvas.drawColor(Color.WHITE)
    val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { textSize = 28f }
    val trackWidth = TIMELINE_WIDTH_PX - LABEL_WIDTH_PX
    for ((row, spk) in speakers.withIndex()) {
      val y = row * ROW_HEIGHT_PX + TIMELINE_TOP_PX
      val color = SPEAKER_COLORS[spk % SPEAKER_COLORS.size]
      textPaint.color = color
      canvas.drawText("SPK ${spk + 1}", 8f, y + ROW_HEIGHT_PX / 2 + 10, textPaint)
      val barBottom = y + ROW_HEIGHT_PX - ROW_GAP_PX
      paint.color = Color.rgb(0xEE, 0xEE, 0xEE)
      canvas.drawRect(LABEL_WIDTH_PX, y, LABEL_WIDTH_PX + trackWidth, barBottom, paint)
      paint.color = color
      for (s in segments) {
        if (s.speaker != spk) continue
        val x0 = LABEL_WIDTH_PX + (s.start / duration * trackWidth).toFloat()
        val x1 = LABEL_WIDTH_PX + (s.end / duration * trackWidth).toFloat()
        canvas.drawRect(x0, y, x1, barBottom, paint)
      }
    }
    return bitmap
  }

  /** Concatenates and plays back only [speaker]'s segments. */
  fun playSpeaker(speaker: Int) {
    val audio = pcm ?: return
    val res = result ?: return
    viewModelScope.launch(inferenceDispatcher) {
      stopPlayback()
      val segments = res.segments.filter { it.speaker == speaker }
      if (segments.isEmpty()) return@launch
      val total = segments.sumOf { ((it.end - it.start) * Diarizer.SR).toInt() }
      val data = FloatArray(total)
      var pos = 0
      for (s in segments) {
        val from = (s.start * Diarizer.SR).toInt().coerceIn(0, audio.size)
        val to = (s.end * Diarizer.SR).toInt().coerceIn(0, audio.size)
        for (p in from until to) {
          if (pos < total) data[pos++] = audio[p]
        }
      }
      val track =
        AudioTrack.Builder()
          .setAudioAttributes(
            AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).build()
          )
          .setAudioFormat(
            AudioFormat.Builder()
              .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
              .setSampleRate(Diarizer.SR)
              .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
              .build()
          )
          .setBufferSizeInBytes(data.size * 4)
          .setTransferMode(AudioTrack.MODE_STATIC)
          .build()
      track.write(data, 0, data.size, AudioTrack.WRITE_BLOCKING)
      track.play()
      player = track
    }
  }

  private fun stopPlayback() {
    player?.let { runCatching { it.stop() }.also { _ -> runCatching { player?.release() } } }
    player = null
  }

  override fun onCleared() {
    super.onCleared()
    recording = false
    stopPlayback()
    diarizer?.close()
  }
}
