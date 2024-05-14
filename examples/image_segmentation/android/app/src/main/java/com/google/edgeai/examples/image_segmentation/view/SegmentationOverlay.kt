package com.google.edgeai.examples.image_segmentation.view

import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import com.google.edgeai.examples.image_segmentation.OverlayInfo

@Composable
fun SegmentationOverlay(modifier: Modifier = Modifier, overlayInfo: OverlayInfo) {
    Canvas(
        modifier = modifier
    ) {
        val imageWidth: Float = size.width
        val imageHeight: Float = size.height

        val image = Bitmap.createBitmap(
            overlayInfo.pixels, overlayInfo.width, overlayInfo.height, Bitmap.Config.ARGB_8888
        )

        val scaleBitmap =
            Bitmap.createScaledBitmap(image, imageWidth.toInt(), imageHeight.toInt(), true)
        drawImage(scaleBitmap.asImageBitmap())
    }
}