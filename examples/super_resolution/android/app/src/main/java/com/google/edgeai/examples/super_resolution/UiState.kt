package com.google.edgeai.examples.super_resolution

import android.graphics.Bitmap
import androidx.compose.runtime.Immutable

@Immutable
data class UiState(
    val sampleUriList: List<String> = listOf(
        "lr-1.jpg", "lr-2.jpg", "lr-3.jpg"
    ),
    val selectImageUri: String? = null,
    val sharpenBitmap: Bitmap? = null,
    val bitmapList: List<Bitmap> = emptyList(),
    val delegate: String = "",
    val inferenceTime: Int = 0
)
