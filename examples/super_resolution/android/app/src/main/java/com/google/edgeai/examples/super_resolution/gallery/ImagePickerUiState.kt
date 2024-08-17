package com.google.edgeai.examples.super_resolution.gallery

import android.graphics.Bitmap
import androidx.compose.runtime.Immutable

@Immutable
data class ImagePickerUiState(
    val originalBitmap: Bitmap? = null,
    val sharpenBitmap: Bitmap? = null,
    val bitmapList: List<Bitmap> = emptyList(),
    val inferenceTime: Int = 0,
)
