package com.google.edgeai.examples.super_resolution.gallery

import android.graphics.Bitmap
import androidx.compose.runtime.Immutable
import androidx.compose.ui.geometry.Offset

@Immutable
data class ImagePickerUiState(
    val originalBitmap: Bitmap? = null,
    val sharpenBitmap: Bitmap? = null,
    val selectPoint: SelectPoint = SelectPoint(),
    val inferenceTime: Int = 0,
) {
    companion object {
        const val REQUIRE_IMAGE_SIZE = 50f
    }
}

@Immutable
data class SelectPoint(
    val offset: Offset? = null,
    val boxSize: Float = 0f,
)
