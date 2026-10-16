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

package android.template.ui.classify

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import com.google.ai.edge.litert.LiteRtException
import java.util.concurrent.Executors
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class ClassifyUiState(
    val ready: Boolean = false,
    val topClass: Int? = null,
    val score: Float? = null,
    val error: String? = null,
)

class ClassifyViewModel(app: Application) : AndroidViewModel(app) {
    private val executor = Executors.newSingleThreadExecutor()
    private val scope = CoroutineScope(SupervisorJob() + executor.asCoroutineDispatcher())
    private var classifier: Classifier? = null
    private val _uiState = MutableStateFlow(ClassifyUiState())
    val uiState: StateFlow<ClassifyUiState> = _uiState.asStateFlow()

    fun load() {
        scope.launch {
            if (classifier != null) {
                return@launch
            }
            try {
                classifier = Classifier(getApplication<Application>())
                _uiState.value = ClassifyUiState(ready = true)
            } catch (e: LiteRtException) {
                _uiState.value = ClassifyUiState(error = e.message)
            }
        }
    }

    fun classify(input: FloatArray) {
        scope.launch {
            val model = classifier ?: return@launch
            try {
                val scores = model.classify(input)
                val top = scores.indices.maxBy { scores[it] }
                _uiState.update { it.copy(topClass = top, score = scores[top], error = null) }
            } catch (e: LiteRtException) {
                _uiState.update { it.copy(error = e.message) }
            }
        }
    }

    override fun onCleared() {
        scope.launch { classifier?.close() }.invokeOnCompletion { scope.cancel() }
        executor.shutdown()
    }
}
