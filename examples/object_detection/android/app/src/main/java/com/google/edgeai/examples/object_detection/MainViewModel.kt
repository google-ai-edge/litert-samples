package com.google.edgeai.examples.object_detection

import android.content.Context
import android.graphics.Bitmap
import androidx.camera.core.ImageProxy
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import com.google.edgeai.examples.object_detection.objectdetector.ObjectDetectorHelper
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class MainViewModel(private val objectDetectorHelper: ObjectDetectorHelper) : ViewModel() {
    companion object {
        fun getFactory(context: Context) = object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
                // To apply object detection, we use our ObjectDetectorHelper class,
                // which abstracts away the specifics of using MediaPipe  for object
                // detection from the UI elements of the app
                val objectDetectorHelper = ObjectDetectorHelper(context = context)
                return MainViewModel(objectDetectorHelper) as T
            }
        }
    }

    private var detectJob: Job? = null

    private val detectionResult =
        MutableStateFlow<ObjectDetectorHelper.DetectionResult?>(null).also {
            viewModelScope.launch {
                objectDetectorHelper.detectionResult.collect(it)
            }
        }

    private val setting = MutableStateFlow(Setting())
        .apply {
            viewModelScope.launch {
                collect {
                    objectDetectorHelper.apply {
                        model = it.model
                        delegate = it.delegate
                        maxResults = it.resultCount
                        threshold = it.threshold
                    }
                    objectDetectorHelper.setupObjectDetector()
                }
            }
        }

    private val errorMessage = MutableStateFlow<Throwable?>(null).also {
        viewModelScope.launch {
            objectDetectorHelper.error.collect(it)
        }
    }

    val uiState: StateFlow<UiState> = combine(
        detectionResult,
        setting,
        errorMessage
    ) { result, setting, error ->
        UiState(
            detectionResult = result,
            setting = setting,
            errorMessage = error?.message
        )
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), UiState())

    /**
     *  Start detect object from an image.
     *  @param bitmap Tries to make a new bitmap based on the dimensions of this bitmap,
     *  @param rotationDegrees to correct the rotationDegrees during segmentation
     */
    fun detectImageObject(bitmap: Bitmap, rotationDegrees: Int) {
        detectJob = viewModelScope.launch {
            objectDetectorHelper.detect(bitmap, rotationDegrees)
        }
    }

    fun detectImageObject(imageProxy: ImageProxy) {
        detectJob = viewModelScope.launch {
            objectDetectorHelper.detect(imageProxy)
            imageProxy.close()
        }
    }

    /** Set [ObjectDetectorHelper.Delegate] (CPU/GPU) for ObjectDetectorHelper*/
    fun setDelegate(delegate: ObjectDetectorHelper.Delegate) {
        viewModelScope.launch {
            setting.update { it.copy(delegate = delegate) }
        }
    }

    /** Set [ObjectDetectorHelper.Model] for ObjectDetectorHelper*/
    fun setModel(model: ObjectDetectorHelper.Model) {
        viewModelScope.launch {
            setting.update { it.copy(model = model) }
        }
    }

    /** Set Number of output classes of the ObjectDetectorHelper.  */
    fun setNumberOfResult(numResult: Int) {
        viewModelScope.launch {
            setting.update { it.copy(resultCount = numResult) }
        }
    }

    /** Set the threshold so the label can display score */
    fun setThreshold(threshold: Float) {
        viewModelScope.launch {
            setting.update { it.copy(threshold = threshold) }
        }
    }

    /** Stop current detection */
    fun stopDetect() {
        viewModelScope.launch {
            detectionResult.emit(null)
            detectJob?.cancel()
        }
    }

    /** Clear error message after it has been consumed*/
    fun errorMessageShown() {
        errorMessage.update { null }
    }
}