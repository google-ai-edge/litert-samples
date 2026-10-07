# Preprocess a bitmap

```kotlin
fun preprocess(bitmap: Bitmap, size: Int = 224, mean: Float = 127.5f, std: Float = 127.5f): FloatArray {
    val scaled = Bitmap.createScaledBitmap(bitmap.copy(Bitmap.Config.ARGB_8888, false), size, size, true)
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

Use the model's own size, mean/std, channel order and layout (this example is NHWC, RGB, scaled to -1..1). The copy to `ARGB_8888` is for the bitmaps `ImageDecoder` returns by default on Android 9 and later, for a photo picker `Uri` too: usually `HARDWARE` bitmaps, which `getPixels` cannot read. The scale does not keep the aspect ratio; crop first if the model expects that.
