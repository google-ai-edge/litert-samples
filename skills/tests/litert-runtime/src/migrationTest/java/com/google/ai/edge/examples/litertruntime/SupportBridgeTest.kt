/*
 * Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.examples.litertruntime

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import com.google.ai.edge.litert.support.common.ops.NormalizeOp
import com.google.ai.edge.litert.support.image.ImageProcessor
import com.google.ai.edge.litert.support.image.TensorImage
import kotlin.math.abs
import org.junit.After
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.DataType

/**
 * Runs [SupportBridge], the skill's template with only the package changed, on the app's model on
 * the CPU: an image through LiteRT Support's `NormalizeOp` reproduces the golden output, and a
 * Support tensor of another size or type is refused before the model runs.
 */
@RunWith(AndroidJUnit4::class)
class SupportBridgeTest {
    private lateinit var model: CompiledModel
    private lateinit var inputs: List<TensorBuffer>
    private lateinit var outputs: List<TensorBuffer>
    private lateinit var bridge: SupportBridge

    @Before
    fun setUp() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        model =
            CompiledModel.create(
                context.assets,
                "model.tflite",
                CompiledModel.Options(Accelerator.CPU),
                environment,
            )
        inputs = model.createInputBuffers()
        outputs = model.createOutputBuffers()
        // The input and output names of a LiteRT-Torch export.
        bridge = SupportBridge(model, "args_0", "output_0")
    }

    @After
    fun tearDown() {
        (inputs + outputs).forEach { it.close() }
        model.close()
    }

    @Test
    fun normalizedImageReproducesTheGolden() {
        bridge.write(inputs[0], normalized(224))
        model.run(inputs, outputs)
        val output = bridge.read(outputs[0], intArrayOf(1, 10), DataType.FLOAT32).floatArray
        val golden =
            InstrumentationRegistry.getInstrumentation()
                .context
                .assets
                .open("golden/output.txt")
                .bufferedReader()
                .use { it.readLines() }
                .filter { it.isNotBlank() }
                .map { it.trim().toFloat() }
        assertTrue("${golden.size} golden values", golden.size == output.size)
        val maxAbsDiff = output.indices.maxOf { abs(output[it] - golden[it]) }
        Log.i(TAG, "max abs diff to the golden = $maxAbsDiff")
        assertTrue("max abs diff to the golden = $maxAbsDiff", maxAbsDiff < 1e-3f)
    }

    @Test
    fun imageOfAnotherSizeIsRefused() {
        val error =
            assertThrows(IllegalArgumentException::class.java) {
                bridge.write(inputs[0], normalized(32))
            }
        Log.i(TAG, "Refused: ${error.message}")
    }

    @Test
    fun uint8ImageForTheFloatInputIsRefused() {
        // 448 x 448 x 3 bytes are as many as the 224 x 224 x 3 floats of the input, so the size
        // check passes and the type check refuses the image.
        val image = TensorImage.fromBitmap(pattern(448))
        val error =
            assertThrows(IllegalArgumentException::class.java) { bridge.write(inputs[0], image) }
        Log.i(TAG, "Refused: ${error.message}")
        assertTrue(error.message, error.message.orEmpty().contains("NormalizeOp"))
    }

    @Test
    fun outputOfAnotherShapeIsRefused() {
        bridge.write(inputs[0], normalized(224))
        model.run(inputs, outputs)
        val error =
            assertThrows(IllegalArgumentException::class.java) {
                bridge.read(outputs[0], intArrayOf(1, 5), DataType.FLOAT32)
            }
        Log.i(TAG, "Refused: ${error.message}")
    }

    /** The pattern at [size] through NormalizeOp(127.5, 127.5), as the template's placeholder. */
    private fun normalized(size: Int): TensorImage =
        ImageProcessor.Builder()
            .add(NormalizeOp(127.5f, 127.5f))
            .build()
            .process(TensorImage.fromBitmap(pattern(size)))

    /** The pattern of fixedBitmap in the migration templates, at any size. */
    private fun pattern(size: Int): Bitmap {
        val pixels = IntArray(size * size)
        for (y in 0 until size) {
            for (x in 0 until size) {
                val r = (17 * x + 3 * y) and 0xFF
                val g = (5 * x + 11 * y) and 0xFF
                val b = (7 * x + 13 * y) and 0xFF
                pixels[y * size + x] = Color.rgb(r, g, b)
            }
        }
        return Bitmap.createBitmap(pixels, size, size, Bitmap.Config.ARGB_8888)
    }

    private companion object {
        const val TAG = "SupportBridgeTest"

        /** One Environment for the process, as in the skill's Step 2. */
        val environment by lazy { Environment.create() }
    }
}
