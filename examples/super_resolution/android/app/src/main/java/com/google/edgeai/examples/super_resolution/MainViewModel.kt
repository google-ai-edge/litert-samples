package com.google.edgeai.examples.super_resolution

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class MainViewModel(private val imageSuperResolutionHelper: ImageSuperResolutionHelper) :
    ViewModel() {
    companion object {
        fun getFactory(context: Context) = object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
                val imageSuperResolutionHelper = ImageSuperResolutionHelper(context)
                return MainViewModel(imageSuperResolutionHelper) as T
            }
        }
    }

    private val uriFlow = MutableStateFlow<Uri?>(null)
    private val selectedBimapFlow = MutableStateFlow<Bitmap?>(null)
    private val bitmapsFlow = MutableStateFlow<List<Bitmap>>(emptyList())

    init {
        viewModelScope.launch {
            imageSuperResolutionHelper.superResolutionFlow
                .map {
                    val bitmap = it.bitmap
                    if (bitmap != null) {
                        val mutableList = bitmapsFlow.value.toMutableList()
                        val index = mutableList.indexOf(selectedBimapFlow.value)
                        mutableList[index] = bitmap
                        mutableList
                    } else {
                        bitmapsFlow.value
                    }
                }
                .collect(bitmapsFlow)
        }
    }

    val uiState: StateFlow<UiState> = combine(
        uriFlow,
        bitmapsFlow,
        imageSuperResolutionHelper.superResolutionFlow.stateIn(
            viewModelScope,
            SharingStarted.Lazily,
            ImageSuperResolutionHelper.Result()
        ),
    ) { uri, bitmaps, result ->
        UiState(
            selectImageUri = uri?.toString(),
            sharpenBitmap = result.bitmap,
            bitmapList = bitmaps,
            inferenceTime = result.inferenceTime.toInt(),
        )
    }.stateIn(viewModelScope, SharingStarted.Lazily, UiState())


    fun makeSharpen(bitmap: Bitmap) {
        viewModelScope.launch(Dispatchers.IO) {
            if (bitmap.config != Bitmap.Config.ARGB_8888) {
                bitmap.config = Bitmap.Config.ARGB_8888
            }
            selectedBimapFlow.update { bitmap }
            imageSuperResolutionHelper.makeSuperResolution(bitmap)
        }
    }

    fun selectImage(bitmap: Bitmap) {
        viewModelScope.launch(Dispatchers.IO) {
            val gridSize = 200

            val rowSize = bitmap.height / gridSize
            val columnSize = bitmap.width / gridSize
            val bitmaps = mutableListOf<Bitmap>()

            for (i in 0 until rowSize) {
                for (j in 0 until columnSize) {
                    val subBitmap = Bitmap.createBitmap(
                        bitmap,
                        j * gridSize,
                        i * gridSize,
                        gridSize,
                        gridSize
                    )
                    bitmaps.add(subBitmap)
                }
            }
            bitmapsFlow.update { bitmaps }
        }
    }
}
