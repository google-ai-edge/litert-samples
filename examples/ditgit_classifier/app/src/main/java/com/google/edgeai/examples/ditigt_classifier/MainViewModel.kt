package com.google.edgeai.examples.ditigt_classifier

import android.content.Context
import android.graphics.Bitmap
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class MainViewModel(private val digitClassificationHelper: DigitClassificationHelper) :
    ViewModel() {
    companion object {
        fun getFactory(context: Context) = object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
                val digitClassificationHelper = DigitClassificationHelper(context)
                return MainViewModel(digitClassificationHelper) as T
            }
        }
    }

    private val drawFlow = MutableStateFlow<List<DrawOffset>>(emptyList())

    val uiState: StateFlow<UiState> = combine(
        digitClassificationHelper.classification.stateIn(
            viewModelScope,
            SharingStarted.WhileSubscribed(5_000),
            Pair("-", 0f)
        ), drawFlow
    ) { pair, drawOffsets ->
        UiState(
            digit = pair.first,
            score = pair.second,
            drawOffsets = drawOffsets
        )
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), UiState())

    fun classify(bitmap: Bitmap) {
        viewModelScope.launch {
            digitClassificationHelper.classify(bitmap)
        }
    }


    fun draw(drawOffset: DrawOffset) {
        drawFlow.update {
            it + drawOffset
        }
    }

    /* Clean the board*/
    fun cleanBoard() {
        viewModelScope.launch {
            drawFlow.emit(emptyList())
        }
    }
}
