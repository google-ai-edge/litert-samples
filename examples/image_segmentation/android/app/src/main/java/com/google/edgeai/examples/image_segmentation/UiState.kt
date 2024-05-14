package com.google.edgeai.examples.image_segmentation

import android.graphics.Color
import android.net.Uri
import androidx.compose.runtime.Immutable

@Immutable
class UiState(
    val mediaUri: Uri = Uri.EMPTY,
    val overlayInfo: OverlayInfo? = null,
    val inferenceTime: Long = 0L,
    val errorMessage: String? = null,
)

class OverlayInfo(
    val pixels: IntArray,
    val width: Int,
    val height: Int,
)

@Immutable
data class ColorLabel(
    val id: Int,
    val label: String,
    val rgbColor: Int,
) {

    fun getColor(): Int {
        // Use completely transparent for the background color.
        return if (id == 0) Color.TRANSPARENT else Color.argb(
            128, Color.red(rgbColor), Color.green(rgbColor), Color.blue(rgbColor)
        )
    }
}