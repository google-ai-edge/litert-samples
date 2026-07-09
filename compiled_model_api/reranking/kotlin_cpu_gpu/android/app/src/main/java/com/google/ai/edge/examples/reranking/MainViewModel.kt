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

package com.google.ai.edge.examples.reranking

import android.content.Context
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
 * Owns the [Reranker] and exposes a single [UiState]. On startup it loads a bundled candidate set once
 * (a slow, one-time GPU compile), then scores every candidate against each query by P("yes") relevance
 * and ranks them. The reranker reuses native buffers, so all model calls run on one confined worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    const val DEFAULT_QUERY = "What is the capital of China?"
    private const val DOCS_ASSET = "docs.txt"

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var reranker: Reranker? = null
  private var docs: List<String> = emptyList()

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    viewModelScope.launch(inferenceDispatcher) {
      try {
        docs =
          context.assets.open(DOCS_ASSET).bufferedReader().use { it.readLines() }
            .filter { it.isNotBlank() }
        _uiState.update {
          it.copy(
            statusMessage =
              "Loading model + reranking ${docs.size} documents " +
                "(first run compiles GPU shaders)…"
          )
        }
        reranker = Reranker(context)
        runRerank(DEFAULT_QUERY)
      } catch (t: Throwable) {
        _uiState.update { it.copy(errorMessage = t.message ?: "Failed to load model") }
      }
    }
  }

  /** Scores the candidate documents against [query] and ranks them by relevance. */
  fun rerank(query: String) {
    if (query.isBlank()) return
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update { it.copy(isSearching = true) }
      try {
        runRerank(query)
      } catch (t: Throwable) {
        _uiState.update { it.copy(isSearching = false, errorMessage = t.message ?: "Rerank failed") }
      }
    }
  }

  private fun runRerank(query: String) {
    val reranker = reranker ?: return
    val startMs = System.currentTimeMillis()
    val ranked =
      docs
        .map { RankedDocument(reranker.score(query, it), it) }
        .sortedByDescending { it.score }
    val elapsedMs = System.currentTimeMillis() - startMs
    _uiState.update {
      it.copy(
        isModelReady = true,
        isSearching = false,
        statusMessage = "Reranked ${docs.size} documents in $elapsedMs ms",
        results = ranked,
        errorMessage = null,
      )
    }
  }

  override fun onCleared() {
    super.onCleared()
    reranker?.close()
  }
}
