package com.google.edgeai.examples.super_resolution.gallery

import android.content.Context
import android.graphics.Bitmap
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import com.google.edgeai.examples.super_resolution.ImageSuperResolutionHelper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class ImagePickerViewModel(private val imageSuperResolutionHelper: ImageSuperResolutionHelper) :
    ViewModel() {
    companion object {
        fun getFactory(context: Context) = object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
                val imageSuperResolutionHelper = ImageSuperResolutionHelper(context)
                return ImagePickerViewModel(imageSuperResolutionHelper) as T
            }
        }
    }

    private val originalBimapFlow = MutableStateFlow<Bitmap?>(null)
    private val subBimapIndexFlow = MutableStateFlow<Int?>(null)
    private val subBitmapsFlow = MutableStateFlow<List<Bitmap>>(emptyList())
    private val superResolutionFlow = MutableStateFlow(ImageSuperResolutionHelper.Result())
    private var sharpenJob: Job? = null

    init {
        viewModelScope.launch {
            imageSuperResolutionHelper.superResolutionFlow
                .onEach { superResolutionFlow.emit(it) }
                .map { result ->
                    val bitmap = result.bitmap
                    val index = subBimapIndexFlow.value ?: return@map subBitmapsFlow.value
                    val mutableList = subBitmapsFlow.value.toMutableList()
                    mutableList[index] = bitmap!!
                    mutableList
                }
                .collect(subBitmapsFlow)
        }
    }

    val uiState: StateFlow<ImagePickerUiState> = combine(
        originalBimapFlow,
        superResolutionFlow,
        subBitmapsFlow
    ) { originalBitmap, result, bitmaps ->
        ImagePickerUiState(
            originalBitmap = originalBitmap,
            sharpenBitmap = result.bitmap,
            bitmapList = bitmaps,
            inferenceTime = result.inferenceTime.toInt()
        )
    }.stateIn(viewModelScope, SharingStarted.Lazily, ImagePickerUiState())

    /**
     * Starts the process of sharpening the provided bitmap.
     */
    fun makeSharpen(bitmap: Bitmap) {
        if (sharpenJob?.isActive == true) return
        sharpenJob = viewModelScope.launch(Dispatchers.IO) {
            if (bitmap.config != Bitmap.Config.ARGB_8888) {
                bitmap.config = Bitmap.Config.ARGB_8888
            }
            imageSuperResolutionHelper.makeSuperResolution(bitmap)
        }

    }

    /**
     * Processes the selected image and emits the original bitmap and sub-bitmaps.
     */
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
            originalBimapFlow.emit(bitmap)
            subBitmapsFlow.update { bitmaps }
        }
    }

    /*
     * Updates the sub-bitmap index flow with the selected sub-bitmap index.
     */
    fun selectSubBitmap(index: Int) {
        subBimapIndexFlow.update { index }
    }

    /*
     * Resets the selected bitmap state.
     */
    fun resetSelectedBitmap() {
        sharpenJob?.cancel()
        subBimapIndexFlow.update { null }
        superResolutionFlow.update { ImageSuperResolutionHelper.Result() }
    }

    /** Set [ImageSuperResolutionHelper.Delegate] (CPU/NNAPI) for ImageSuperResolutionHelper*/
    fun setDelegate(delegate: ImageSuperResolutionHelper.Delegate) {
        imageSuperResolutionHelper.initClassifier(delegate)
    }
}
