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

package com.google.ai.edge.examples.audio_classification

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
import kotlin.math.abs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Owns the [Wav2Vec2Kws] helper and exposes a single [UiState]. On startup it loads the two GPU
 * graphs from filesDir and self-tests on a bundled clip (a slow, one-time GPU compile). The Record
 * button captures 1 s from the mic and shows which of the 10 Speech-Commands keywords was said (or
 * "_unknown_"). The helper reuses native buffers, so the model call and the mic capture that feeds
 * it both run on one confined worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val TEST_AUDIO_ASSET = "test_audio.bin"
    private const val UNKNOWN_LABEL = "_unknown_"
    private const val SILENCE_LABEL = "_silence_"

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var kws: Wav2Vec2Kws? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState =
    MutableStateFlow(UiState(statusMessage = context.getString(R.string.status_loading)))
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    viewModelScope.launch(inferenceDispatcher) {
      try {
        val loaded = Wav2Vec2Kws(context)
        kws = loaded
        val clip = readFloats(context.assets.open(TEST_AUDIO_ASSET).readBytes())
        loaded.classify(clip) // warm up GPU
        val r = loaded.classify(clip)
        _uiState.update {
          it.copy(
            isModelReady = true,
            statusMessage = context.getString(R.string.status_ready, r.label, r.ms),
          )
        }
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(errorMessage = context.getString(R.string.error_load_fail, t.message ?: ""))
        }
      }
    }
  }

  /** Record a 1 s clip from the mic and classify it. Caller must hold RECORD_AUDIO permission. */
  fun record() {
    if (!_uiState.value.isModelReady || _uiState.value.isRecording) return
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update {
        it.copy(
          isRecording = true,
          isRecognized = false,
          resultText = context.getString(R.string.result_listening),
        )
      }
      try {
        val audio = recordOneSecond()
        val r = kws!!.classify(audio)
        val recognized = r.label != UNKNOWN_LABEL && r.label != SILENCE_LABEL
        val resultText =
          if (recognized) {
            context.getString(R.string.result_recognized, r.label)
          } else {
            context.getString(R.string.result_unknown, r.label)
          }
        _uiState.update {
          it.copy(isRecording = false, isRecognized = recognized, resultText = resultText)
        }
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(
            isRecording = false,
            isRecognized = false,
            resultText = context.getString(R.string.error_record_fail, t.message ?: ""),
          )
        }
      }
    }
  }

  /** Capture 1 s of mono 16 kHz PCM from the mic, returned as float32 in [-1,1]. */
  private fun recordOneSecond(): FloatArray {
    val sr = Wav2Vec2Kws.SAMPLES
    val minBuf =
      AudioRecord.getMinBufferSize(
        sr,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT,
      )
    val rec =
      AudioRecord(
        MediaRecorder.AudioSource.VOICE_RECOGNITION,
        sr,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT,
        maxOf(minBuf, sr * 2),
      )
    val pcm = ShortArray(sr)
    rec.startRecording()
    var off = 0
    while (off < sr) {
      val n = rec.read(pcm, off, sr - off)
      if (n <= 0) break
      off += n
    }
    rec.stop()
    rec.release()
    val out = FloatArray(sr)
    var peak = 1f
    for (i in 0 until sr) {
      peak = maxOf(peak, abs(pcm[i].toFloat()))
    }
    for (i in 0 until sr) {
      out[i] = pcm[i] / peak * 0.5f // PCM16 -> peak-normalized [-0.5,0.5]
    }
    return out
  }

  private fun readFloats(b: ByteArray): FloatArray {
    val bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
    return FloatArray(b.size / 4) { bb.float }
  }

  override fun onCleared() {
    super.onCleared()
    kws?.close()
  }
}
