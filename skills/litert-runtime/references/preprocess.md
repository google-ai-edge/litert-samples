# Preprocess a bitmap

```kotlin
fun preprocess(bitmap: Bitmap, size: Int = 224, mean: Float = 127.5f, std: Float = 127.5f): FloatArray {
    val scaled = checkNotNull(Bitmap.createScaledBitmap(bitmap, size, size, true).copy(Bitmap.Config.ARGB_8888, false))
    val pixels = IntArray(size * size).also { scaled.getPixels(it, 0, size, 0, 0, size, size) }
    val out = FloatArray(size * size * 3)
    pixels.forEachIndexed { i, p ->
        out[i * 3] = ((p shr 16 and 0xFF) - mean) / std
        out[i * 3 + 1] = ((p shr 8 and 0xFF) - mean) / std
        out[i * 3 + 2] = ((p and 0xFF) - mean) / std
    }
    return out
}
```

Use the model's own size, mean/std, channel order and layout (this example is NHWC, RGB, scaled to -1..1). The scaled bitmap is copied to `ARGB_8888` because `ImageDecoder` returns `HARDWARE` bitmaps by default on Android 9 and later (for a photo picker `Uri` too), scaling keeps that config, and `getPixels` cannot read it; the copy is of the small bitmap, not of the picture. The scale does not keep the aspect ratio; crop first if the model expects that.
