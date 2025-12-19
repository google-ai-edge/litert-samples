package com.google.aiedge.examples.super_resolution.gallery

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.exifinterface.media.ExifInterface
import android.net.Uri
import android.util.Log
import java.io.IOException

/*
 * Represents the result of an image orientation correction operation.
 */
sealed class OrientationResult {
    /**
     * Indicates that the orientation correction was successful.
     *
     * @param bitmap The corrected bitmap.
     */
    data class Success(val bitmap: Bitmap) : OrientationResult()

    /**
     * Indicates that an error occurred during orientation correction.
     *
     * @param exception The exception that was thrown.
     */
    data class Error(val exception: Exception) : OrientationResult()
}

/**
 * Utility class for handling image orientation correction.
 */
object ImagePickerHelper {

    /**
     * Modifies the orientation of a bitmap based on the Exif information in the input stream.
     *
     * @param bitmap The bitmap to modify.
     * @param inputStream The input stream containing the image data.
     * @return The corrected bitmap.
     */
    fun modifyOrientation(
        bitmap: Bitmap,
        contentResolver: ContentResolver,
        uri: Uri
    ): OrientationResult {
        try {
            contentResolver.openInputStream(uri).use { inputStream ->
                if (inputStream == null) {
                    throw IOException(NullPointerException("Input stream is null"))
                }
                val exif: ExifInterface = ExifInterface(inputStream)
                val orientation = exif.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
                )
                return when (orientation) {
                    ExifInterface.ORIENTATION_ROTATE_90 -> OrientationResult.Success(
                        rotateImage(
                            bitmap,
                            90f
                        )
                    )

                    ExifInterface.ORIENTATION_ROTATE_180 -> OrientationResult.Success(
                        rotateImage(
                            bitmap,
                            180f
                        )
                    )

                    ExifInterface.ORIENTATION_ROTATE_270 -> OrientationResult.Success(
                        rotateImage(
                            bitmap,
                            270f
                        )
                    )

                    ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> OrientationResult.Success(
                        flipImage(
                            bitmap,
                            true,
                            false
                        )
                    )

                    ExifInterface.ORIENTATION_FLIP_VERTICAL -> OrientationResult.Success(
                        flipImage(
                            bitmap,
                            false,
                            true
                        )
                    )

                    else -> OrientationResult.Success(bitmap)
                }
            }
        } catch (e: IOException) {
            Log.e("ImagePickerHelper", "Error in modifyOrientation: ${e.message}", e)
            return OrientationResult.Error(e)
        }
    }

    /**
     * Rotates a bitmap by the specified degrees.
     *
     * @param bitmap The bitmap to rotate.
     * @param degrees The angle in degrees to rotate the bitmap.
     * @return The rotated bitmap.
     */
    fun rotateImage(bitmap: Bitmap, degrees: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(degrees)
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    /**
     * Flips a bitmap horizontally or vertically.
     *
     * @param bitmap The bitmap to flip.
     * @param horizontal Whether to flip the bitmap horizontally.
     * @param vertical Whether to flip the bitmap vertically.
     * @return The flipped bitmap.
     */
    fun flipImage(bitmap: Bitmap, horizontal: Boolean, vertical: Boolean): Bitmap {
        val matrix = Matrix()
        matrix.preScale((if (horizontal) -1 else 1).toFloat(), (if (vertical) -1 else 1).toFloat())
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
}

