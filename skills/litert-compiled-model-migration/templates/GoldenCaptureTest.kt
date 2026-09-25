package com.example.validation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Step 0 (throwaway, before any dependency changes): the legacy app's output for one fixed input,
 * written to the app's external files dir. Run it with, in the project directory:
 *   ./gradlew installDebug installDebugAndroidTest
 *   adb shell am instrument -w -e class <test package>.GoldenCaptureTest <applicationId>.test/androidx.test.runner.AndroidJUnitRunner
 *   mkdir -p app/src/androidTest/assets/golden
 *   adb pull /sdcard/Android/data/<applicationId>/files/golden.txt app/src/androidTest/assets/golden/output.txt
 * (`adb shell pm list instrumentation` prints the component; `connectedDebugAndroidTest` would uninstall the APKs and delete the file.)
 * Then delete this test.
 */
@RunWith(AndroidJUnit4::class)
class GoldenCaptureTest {
    @Test
    fun captureGolden() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val bitmap = fixedBitmap(224)

        // TODO: Run the app's existing inference entry point (the class the UI calls, with its own preprocessing)
        // on the bitmap, on the CPU with no GPU or NNAPI delegate, and return the raw output floats (or the
        // top-1 index alone when that is all the app exposes). Print interpreter.getInputTensor(0).name() too:
        // Step 4 needs the tensor name.
        val result: FloatArray = TODO("call the legacy inference on the fixed input")

        val file = File(context.getExternalFilesDir(null), "golden.txt")
        file.writeText(result.joinToString("\n") { it.toString() } + "\n")
    }

    companion object {
        /** Deterministic, non-uniform RGB pattern; MigrationValidationTest builds the same one. */
        fun fixedBitmap(size: Int): Bitmap {
            val pixels = IntArray(size * size)
            for (y in 0 until size) for (x in 0 until size) {
                pixels[y * size + x] = Color.rgb((17 * x + 3 * y) and 0xFF, (5 * x + 11 * y) and 0xFF, (7 * x + 13 * y) and 0xFF)
            }
            return Bitmap.createBitmap(pixels, size, size, Bitmap.Config.ARGB_8888)
        }
    }
}
