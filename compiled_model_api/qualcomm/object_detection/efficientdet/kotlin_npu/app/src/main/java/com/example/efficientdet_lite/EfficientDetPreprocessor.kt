package com.example.efficientdet_lite

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import kotlin.math.min

object EfficientDetPreprocessor {
    const val INPUT_SIZE = 320

    fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val nv21 = imageProxyToNv21(image)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 90, out)
        val bitmap = BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())
        val rotation = image.imageInfo.rotationDegrees
        if (rotation == 0) return bitmap
        val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    fun preprocess(bitmap: Bitmap): PreprocessedFrame {
        val inputBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(inputBitmap)
        canvas.drawColor(Color.BLACK)

        val scale = min(INPUT_SIZE / bitmap.width.toFloat(), INPUT_SIZE / bitmap.height.toFloat())
        val scaledWidth = bitmap.width * scale
        val scaledHeight = bitmap.height * scale
        val padX = (INPUT_SIZE - scaledWidth) / 2f
        val padY = (INPUT_SIZE - scaledHeight) / 2f
        canvas.drawBitmap(
            bitmap,
            null,
            Rect(padX.toInt(), padY.toInt(), (padX + scaledWidth).toInt(), (padY + scaledHeight).toInt()),
            null,
        )

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        inputBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        val input = ByteArray(INPUT_SIZE * INPUT_SIZE * 3)
        var outputIndex = 0
        for (pixel in pixels) {
            input[outputIndex++] = Color.red(pixel).toByte()
            input[outputIndex++] = Color.green(pixel).toByte()
            input[outputIndex++] = Color.blue(pixel).toByte()
        }

        inputBitmap.recycle()
        return PreprocessedFrame(
            input = input,
            metadata = FrameMetadata(
                sourceWidth = bitmap.width,
                sourceHeight = bitmap.height,
                inputSize = INPUT_SIZE,
                scale = scale,
                padX = padX,
                padY = padY,
            ),
        )
    }

    private fun imageProxyToNv21(image: ImageProxy): ByteArray {
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val width = image.width
        val height = image.height
        val output = ByteArray(width * height * 3 / 2)

        var outputOffset = 0
        copyPlane(yPlane.buffer, yPlane.rowStride, yPlane.pixelStride, width, height, output, outputOffset, 1)
        outputOffset += width * height

        val chromaWidth = width / 2
        val chromaHeight = height / 2
        val vu = ByteArray(width * height / 2)
        copyPlane(vPlane.buffer, vPlane.rowStride, vPlane.pixelStride, chromaWidth, chromaHeight, vu, 0, 2)
        copyPlane(uPlane.buffer, uPlane.rowStride, uPlane.pixelStride, chromaWidth, chromaHeight, vu, 1, 2)
        vu.copyInto(output, outputOffset)
        return output
    }

    private fun copyPlane(
        buffer: java.nio.ByteBuffer,
        rowStride: Int,
        pixelStride: Int,
        width: Int,
        height: Int,
        output: ByteArray,
        outputOffset: Int,
        outputPixelStride: Int,
    ) {
        val source = buffer.duplicate()
        val rowBytes = (width - 1) * pixelStride + 1
        val row = ByteArray(rowBytes)
        var outputIndex = outputOffset
        for (rowIndex in 0 until height) {
            source.position(rowIndex * rowStride)
            source.get(row, 0, rowBytes)
            for (column in 0 until width) {
                output[outputIndex] = row[column * pixelStride]
                outputIndex += outputPixelStride
            }
        }
    }
}
