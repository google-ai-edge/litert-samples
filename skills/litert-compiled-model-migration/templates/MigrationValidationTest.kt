package com.example.validation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.IOException
import kotlin.math.abs
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

// LiteRT CompiledModel API
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel

/**
 * Stage 2 gate: the Step 0 input, through the migrated preprocessing and the migrated model on the
 * CPU, must reproduce the golden captured from the legacy app (app/src/androidTest/assets/golden/output.txt).
 * Runs on a connected device or emulator: ./gradlew connectedDebugAndroidTest
 */
@RunWith(AndroidJUnit4::class)
class MigrationValidationTest {

    @Test
    fun verifyMigrationInference() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val golden = readGolden()

        // TODO: Replace with the actual model name in the assets folder
        val modelAssetPath = "model.tflite"

        // TODO: Build the same fixed input Step 0 used and run it through the app's migrated preprocessing
        // (the function that replaced ImageProcessor); the placeholder below is a 224x224 pattern bitmap
        // normalized to [-1, 1]. Int8 models use writeInt8 / readInt8.
        val fixedInput = fixedBitmap(224).toInput(224)

        // The like-for-like baseline: the model on the CPU, buffers from the model, closed before it.
        val outputArray = CompiledModel.create(
            context.assets,
            modelAssetPath,
            CompiledModel.Options(Accelerator.CPU)
        ).use { model ->
            val inputs = model.createInputBuffers()
            val outputs = model.createOutputBuffers()
            try {
                inputs[0].writeFloat(fixedInput)
                model.run(inputs, outputs)
                outputs[0].readFloat()
            } finally {
                (inputs + outputs).forEach { it.close() }
            }
        }

        assertTrue("Output contains NaN", outputArray.none { it.isNaN() })
        when (golden.size) {
            outputArray.size -> {
                val maxAbsDiff = outputArray.indices.maxOf { abs(outputArray[it] - golden[it]) }
                assertTrue("Output differs from the golden: max abs diff = $maxAbsDiff", maxAbsDiff < 1e-3f)
            }
            1 -> {
                val top1 = outputArray.indices.maxByOrNull { outputArray[it] }
                assertEquals("Top-1 index differs from the golden", golden[0].toInt(), top1)
            }
            else -> throw AssertionError("golden/output.txt has ${golden.size} values; expected ${outputArray.size} or 1")
        }
    }

    /** The Step 0 golden, in the test APK: one value per line, every output float or the top-1 index alone. */
    private fun readGolden(): List<Float> {
        val assets = InstrumentationRegistry.getInstrumentation().context.assets
        val lines = try {
            assets.open("golden/output.txt").bufferedReader().use { it.readLines() }
        } catch (e: IOException) {
            throw AssertionError("app/src/androidTest/assets/golden/output.txt is missing: capture it as in Step 0 of SKILL.md", e)
        }
        return lines.filter { it.isNotBlank() }.map { it.trim().toFloat() }
    }

    /** Deterministic, non-uniform RGB pattern; Step 0 must build its input with the same function. */
    private fun fixedBitmap(size: Int): Bitmap {
        val pixels = IntArray(size * size)
        for (y in 0 until size) for (x in 0 until size) {
            pixels[y * size + x] = Color.rgb((17 * x + 3 * y) and 0xFF, (5 * x + 11 * y) and 0xFF, (7 * x + 13 * y) and 0xFF)
        }
        return Bitmap.createBitmap(pixels, size, size, Bitmap.Config.ARGB_8888)
    }

    /** Placeholder preprocessing ((channel - 127.5) / 127.5, RGB); replace with the app's migrated function. */
    private fun Bitmap.toInput(size: Int): FloatArray {
        val pixels = IntArray(size * size).also { Bitmap.createScaledBitmap(this, size, size, true).getPixels(it, 0, size, 0, 0, size, size) }
        val out = FloatArray(size * size * 3)
        var i = 0
        for (p in pixels) {
            out[i++] = ((p shr 16 and 0xFF) - 127.5f) / 127.5f
            out[i++] = ((p shr 8 and 0xFF) - 127.5f) / 127.5f
            out[i++] = ((p and 0xFF) - 127.5f) / 127.5f
        }
        return out
    }
}
