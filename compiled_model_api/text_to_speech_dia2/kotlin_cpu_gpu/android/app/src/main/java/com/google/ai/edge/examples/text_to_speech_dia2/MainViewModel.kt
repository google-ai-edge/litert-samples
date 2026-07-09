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

package com.google.ai.edge.examples.text_to_speech_dia2

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Owns the [Dia2Synthesizer] graphs and exposes a single [UiState]. On startup it loads the models
 * (from the external files dir, pushed via install_to_device.sh). [synthesize] runs the
 * RQ-Transformer loop — tokenization, word pacing and sampling all happen inside the
 * synthesizer — and plays the generated waveform through an [AudioTrack] as a side effect.
 * The graphs reuse native buffers, so every model call runs on one confined worker.
 *
 * Every graph runs on the CPU: the GPU delegate rejects the language models' KV-step
 * FULLY_CONNECTED shapes and fp16 collapses these deep stacks on ARM. Generation is
 * correctness-first and slow (~190 s for a short dialogue on a Pixel 8a), so the models are only
 * loaded — not warmed up — at startup.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    const val DEFAULT_TEXT = "[S1] Hello, how are you today? [S2] I'm great, thanks for asking."
    private const val TAG = "Dia2"

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var tts: Dia2Synthesizer? = null
  private var audioTrack: AudioTrack? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState(statusMessage = "Loading Dia2-1B…"))
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    viewModelScope.launch(inferenceDispatcher) {
      try {
        tts = Dia2Synthesizer(context)
        _uiState.update {
          it.copy(
            isModelReady = true,
            statusMessage =
              "On-device Dia2-1B ✓ (all graphs on CPU)\n" +
                "Type an [S1]/[S2] script and tap Speak. Generation is slow (~190 s).",
          )
        }
      } catch (t: Throwable) {
        Log.e(TAG, "load failed", t)
        _uiState.update {
          it.copy(errorMessage = "${t.message}\n\nPush models first:\n  install_to_device.sh")
        }
      }
    }
  }

  /** Synthesizes [text] on the confined worker and plays it back through an [AudioTrack]. */
  fun synthesize(text: String) {
    val t = tts ?: return
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update {
        it.copy(
          isSynthesizing = true,
          statusMessage = "Synthesizing… (this takes ~190 s)",
          errorMessage = null,
        )
      }
      try {
        val r = t.synthesize(text)
        val dur = r.audio.size.toDouble() / Dia2Synthesizer.SAMPLE_RATE
        val rtf = r.ms / 1000.0 / dur
        Log.i(TAG, "frames=${r.frames} ${r.ms}ms rtf=$rtf")
        _uiState.update {
          it.copy(
            isSynthesizing = false,
            statusMessage =
              "Spoke ${"%.2f".format(dur)} s in ${r.ms} ms · ${r.frames} frames · " +
                "RTF ${"%.2f".format(rtf)} (correctness-first, not real-time)",
          )
        }
        play(r.audio)
      } catch (e: Throwable) {
        Log.e(TAG, "synth failed", e)
        _uiState.update {
          it.copy(isSynthesizing = false, errorMessage = e.message ?: "Synthesis failed")
        }
      }
    }
  }

  private fun play(audio: FloatArray) {
    try {
      val track = AudioTrack(
        AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).build(),
        AudioFormat.Builder()
          .setSampleRate(Dia2Synthesizer.SAMPLE_RATE)
          .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
          .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
          .build(),
        audio.size * 4, AudioTrack.MODE_STATIC, AudioManager.AUDIO_SESSION_ID_GENERATE,
      )
      audioTrack = track
      track.write(audio, 0, audio.size, AudioTrack.WRITE_BLOCKING)
      track.play()
      Thread.sleep((audio.size * 1000L / Dia2Synthesizer.SAMPLE_RATE) + 250)
      track.release()
    } catch (e: Throwable) {
      Log.e(TAG, "play failed: ${e.message}")
    }
  }

  override fun onCleared() {
    super.onCleared()
    audioTrack?.let {
        runCatching { it.stop() }
        runCatching { it.release() }
    }
    tts?.close()
  }
}
