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

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Owns the [AudioTagger] and exposes a single [UiState]. On startup it loads the PANNs CNN14 model
 * from filesDir and self-tests on a bundled clip (a slow, one-time GPU compile). The Record button
 * captures 10 s from the mic and lists the top AudioSet tags. The tagger reuses native buffers, so
 * all model calls (and the mic capture that feeds them) run on one confined worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val TEST_AUDIO_ASSET = "test_audio.bin"

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var tagger: AudioTagger? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState(statusMessage = "Loading PANNs CNN14 on GPU…"))
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    viewModelScope.launch(inferenceDispatcher) {
      try {
        val loaded = AudioTagger(context)
        tagger = loaded
        val clip = readFloats(context.assets.open(TEST_AUDIO_ASSET).readBytes())
        loaded.tag(clip) // warm up GPU
        val r = loaded.tag(clip)
        _uiState.update {
          it.copy(
            isModelReady = true,
            statusMessage =
              "On-device GPU audio tagging ✓  (self-test → \"${r.tags.first().label}\", " +
                "mel ${r.melMs} ms + gpu ${r.gpuMs} ms)",
          )
        }
      } catch (t: Throwable) {
        _uiState.update { it.copy(errorMessage = t.message ?: "Failed to load model") }
      }
    }
  }

  /** Record a 10 s clip from the mic and tag it. Caller must hold RECORD_AUDIO permission. */
  fun record() {
    if (!_uiState.value.isModelReady || _uiState.value.isRecording) return
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update { it.copy(isRecording = true, errorMessage = null) }
      try {
        val audio =
          recordClip { secLeft -> _uiState.update { it.copy(statusMessage = "Listening… ${secLeft}s") } }
        val r = tagger!!.tag(audio)
        _uiState.update {
          it.copy(
            isRecording = false,
            statusMessage = "Tagged (mel ${r.melMs} ms + gpu ${r.gpuMs} ms)",
            resultText = formatTags(r),
          )
        }
      } catch (t: Throwable) {
        _uiState.update { it.copy(isRecording = false, errorMessage = t.message ?: "Record failed") }
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
    val sr = AudioTagger.SAMPLES // 320000
    val minBuf =
      AudioRecord.getMinBufferSize(
        MelSpectrogram.SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
    val rec =
      AudioRecord(
        MediaRecorder.AudioSource.MIC,
        MelSpectrogram.SAMPLE_RATE,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT,
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
      val secLeft =
        (sr - off + MelSpectrogram.SAMPLE_RATE - 1) / MelSpectrogram.SAMPLE_RATE // ceil, 10→0
      if (secLeft != lastTick) {
        onTick(secLeft)
        lastTick = secLeft
      }
    }
    rec.stop()
    rec.release()
    val out = FloatArray(sr)
    for (i in 0 until sr) {
      out[i] = pcm[i] / 32768f // PCM16 -> [-1,1] (PANNs uses raw scale, no peak-norm)
    }
    return out
  }

  private fun readFloats(b: ByteArray): FloatArray {
    val bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
    return FloatArray(b.size / 4) { bb.float }
  }

  override fun onCleared() {
    super.onCleared()
    tagger?.close()
  }
}
