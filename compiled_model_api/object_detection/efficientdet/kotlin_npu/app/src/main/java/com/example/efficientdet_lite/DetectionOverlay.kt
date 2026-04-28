package com.example.efficientdet_lite

import android.graphics.Paint
import android.graphics.RectF
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import kotlin.math.max

@Composable
fun DetectionOverlay(
    result: EfficientDetFrameResult,
    isFrontCamera: Boolean,
    modifier: Modifier = Modifier,
) {
    Canvas(modifier = modifier) {
        if (result.frameWidth <= 0 || result.frameHeight <= 0) return@Canvas

        val scale = max(size.width / result.frameWidth, size.height / result.frameHeight)
        val dx = (size.width - result.frameWidth * scale) / 2f
        val dy = (size.height - result.frameHeight * scale) / 2f

        val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = android.graphics.Color.rgb(82, 222, 151)
            style = Paint.Style.STROKE
            strokeWidth = 4f
        }
        val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = android.graphics.Color.WHITE
            textSize = 34f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
        }
        val labelBackgroundPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = android.graphics.Color.argb(220, 24, 30, 36)
            style = Paint.Style.FILL
        }

        drawIntoCanvas { canvas ->
            val nativeCanvas = canvas.nativeCanvas
            result.detections.forEach { detection ->
                val sourceBox = if (isFrontCamera) {
                    RectF(
                        result.frameWidth - detection.box.right,
                        detection.box.top,
                        result.frameWidth - detection.box.left,
                        detection.box.bottom,
                    )
                } else {
                    detection.box
                }
                val box = RectF(
                    sourceBox.left * scale + dx,
                    sourceBox.top * scale + dy,
                    sourceBox.right * scale + dx,
                    sourceBox.bottom * scale + dy,
                )
                nativeCanvas.drawRect(box, boxPaint)

                val label = "${detection.label} ${(detection.confidence * 100).toInt()}%"
                val textWidth = labelPaint.measureText(label)
                val textHeight = labelPaint.textSize
                val labelRect = RectF(
                    box.left,
                    max(0f, box.top - textHeight - 12f),
                    box.left + textWidth + 18f,
                    max(textHeight + 12f, box.top),
                )
                nativeCanvas.drawRoundRect(labelRect, 6f, 6f, labelBackgroundPaint)
                nativeCanvas.drawText(label, labelRect.left + 9f, labelRect.bottom - 10f, labelPaint)
            }
        }

        drawRect(Color.Transparent, size = Size(size.width, size.height))
    }
}
