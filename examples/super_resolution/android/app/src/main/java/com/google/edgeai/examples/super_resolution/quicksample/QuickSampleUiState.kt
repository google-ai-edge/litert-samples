package com.google.edgeai.examples.super_resolution.quicksample

import android.graphics.Bitmap
import androidx.compose.runtime.Immutable

@Immutable
data class QuickSampleUiState(
    val sampleUriList: List<String> = listOf(
        "lr-1.jpg", "lr-2.jpg", "lr-3.jpg"
    ),
    val selectBitmap: Bitmap? = null,
    val sharpenBitmap: Bitmap? = null,
    val inferenceTime: Int = 0,
)
