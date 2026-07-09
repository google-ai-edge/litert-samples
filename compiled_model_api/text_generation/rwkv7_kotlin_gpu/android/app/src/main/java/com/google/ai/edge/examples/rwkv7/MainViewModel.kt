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

package com.google.ai.edge.examples.rwkv7

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import java.io.ByteArrayOutputStream
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Owns the [Rwkv7Generator] + [RwkvTokenizer] and exposes a single [UiState]. On startup it loads the
 * tokenizer and the GPU step graph once; each [generate] call greedily decodes and STREAMS the growing
 * completion into [UiState.outputText] one token at a time. RWKV keeps its whole recurrent state
 * host-side and reuses native buffers, so all model calls run on one confined worker.
 *
 * The 282 MB step graph and 100 MB embedding table are NOT bundled — they are pushed to the app's
 * filesDir by `install_to_device.sh`; until then the status line asks the user to run it.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    const val DEFAULT_PROMPT = "The Eiffel tower is in the city of"
    private const val MODEL_FILE = "rwkv7_step_fp16.tflite"
    private const val EMB_FILE = "rwkv7_emb_fp16.bin"
    private const val VOCAB_ASSET = "rwkv_vocab_v20230424.txt"
    private const val MAX_NEW_TOKENS = 200

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var generator: Rwkv7Generator? = null
  private var tokenizer: RwkvTokenizer? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState(statusMessage = "Loading RWKV-7 0.1B (GPU)…"))
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    viewModelScope.launch(inferenceDispatcher) {
      try {
        val vocabLines = context.assets.open(VOCAB_ASSET).bufferedReader().use { it.readLines() }
        tokenizer = RwkvTokenizer(vocabLines)
        val modelFile = File(context.filesDir, MODEL_FILE)
        val embFile = File(context.filesDir, EMB_FILE)
        if (!modelFile.exists() || !embFile.exists()) {
          _uiState.update {
            it.copy(statusMessage = "Model files missing — run install_to_device.sh, then relaunch.")
          }
          return@launch
        }
        generator = Rwkv7Generator(modelFile.absolutePath, embFile.absolutePath)
        _uiState.update {
          it.copy(
            isModelReady = true,
            statusMessage = "RWKV-7 World 0.1B ready — full forward on GPU",
          )
        }
      } catch (t: Throwable) {
        _uiState.update { it.copy(errorMessage = t.message ?: "Failed to load model") }
      }
    }
  }

  /**
   * Greedily generates from [prompt], streaming the growing completion into [UiState.outputText] after
   * every decoded token so the UI fills in live.
   */
  fun generate(prompt: String) {
    if (prompt.isBlank()) return
    val gen = generator ?: return
    val tok = tokenizer ?: return
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update {
        it.copy(
          isGenerating = true,
          outputText = prompt,
          statusMessage = "Generating…",
          errorMessage = null,
        )
      }
      try {
        val promptIds = tok.encode(prompt)
        val bytes = ByteArrayOutputStream()
        var stepMsSum = 0f
        var count = 0
        val startNs = System.nanoTime()
        gen.generate(promptIds, MAX_NEW_TOKENS) { tokenId, stats ->
          bytes.write(tok.tokenBytes(tokenId))
          val text = bytes.toString(Charsets.UTF_8.name())
          stepMsSum += stats.stepMs
          count++
          _uiState.update { it.copy(outputText = prompt + text) }
          true
        }
        val totalS = (System.nanoTime() - startNs) / 1e9f
        val tokPerS = if (totalS > 0f) count / totalS else 0f
        _uiState.update {
          it.copy(
            isGenerating = false,
            statusMessage =
              "%d tokens, %.1f tok/s (%.1f ms/token) — prefill %d ids".format(
                count, tokPerS, if (count > 0) stepMsSum / count else 0f, promptIds.size),
          )
        }
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(isGenerating = false, errorMessage = t.message ?: "Generation failed")
        }
      }
    }
  }

  override fun onCleared() {
    super.onCleared()
    generator?.close()
  }
}
