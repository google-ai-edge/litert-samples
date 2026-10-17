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
import com.google.ai.edge.litert.LiteRtException
import com.google.ai.edge.litert.TensorBuffer
import com.google.ai.edge.litert.support.common.ops.NormalizeOp
import com.google.ai.edge.litert.support.image.ImageProcessor
import com.google.ai.edge.litert.support.image.TensorImage
import java.io.File
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Runs the Kotlin of the skill's `references/migration.md`, which the build extracts into this test
 * APK, on the app's own model on the CPU. It covers the golden path through LiteRT Support's
 * `NormalizeOp` and through the skill's `preprocess` and `Classifier`, the two refusals of
 * `runWithSupportImage`, the gate on a wrong golden and on a golden of another length, the NaN that
 * the refused uint8 image would have produced, and the other snippets (`cpuOptionsWithThreads`,
 * `inputShape`, `runSignature`, `gpuWithCpuFallback` on a graph the GPU rejects, `loadLabels`, the
 * golden file round trip). The golden is `fixtures/golden/output.txt`, the model's output for the
 * pattern bitmap on the workstation (`tools/make_golden.py`).
 */
@RunWith(AndroidJUnit4::class)
class MigrationCodeTest {
    private val appContext = ApplicationProvider.getApplicationContext<Context>()
    private val testContext: Context = InstrumentationRegistry.getInstrumentation().context
    private lateinit var model: CompiledModel
    private lateinit var inputs: List<TensorBuffer>
    private lateinit var outputs: List<TensorBuffer>
    private lateinit var golden: FloatArray
    private lateinit var labels: List<String>

    @Before
    fun setUp() {
        model =
            CompiledModel.create(
                appContext.assets,
                "model.tflite",
                CompiledModel.Options(Accelerator.CPU),
                environment,
            )
        inputs = model.createInputBuffers()
        outputs = model.createOutputBuffers()
        golden = readGoldenOutput(testContext)
        labels = loadLabels(testContext)
    }

    @After
    fun tearDown() {
        (inputs + outputs).forEach { it.close() }
        model.close()
    }

    @Test
    fun supportImageMatchesTheGolden() {
        val scores =
            runWithSupportImage(model, inputs, outputs, normalized(224), INPUT_SIZE, labels)
        assertEquals(labels, scores.keys.toList())
        val diff = maxAbsDifference(golden, labels.map { scores.getValue(it) }.toFloatArray())
        Log.i(TAG, "Support image through NormalizeOp: max abs diff to the golden = $diff")
        assertTrue("max abs diff to the golden = $diff", diff < 1e-3f)
    }

    @Test
    fun preprocessAndClassifierMatchTheGolden() {
        // Path A of stage 2: the same input through the migrated preprocessing and the helper.
        val output =
            Classifier(appContext, Accelerator.CPU).use { it.infer(preprocess(pattern(224))) }
        val diff = maxAbsDifference(golden, output)
        Log.i(TAG, "preprocess and Classifier: max abs diff to the golden = $diff")
        assertTrue("max abs diff to the golden = $diff", diff < 1e-3f)
    }

    @Test
    fun uint8ImageIsRefused() {
        // An image that skipped NormalizeOp: TensorImage.fromBitmap is uint8.
        val image = TensorImage.fromBitmap(pattern(224))
        val error =
            assertThrows(IllegalStateException::class.java) {
                runWithSupportImage(model, inputs, outputs, image, INPUT_SIZE, labels)
            }
        Log.i(TAG, "Refused: ${error.message}")
        assertEquals("Normalize the image to float first", error.message)
    }

    @Test
    fun imageOfAnotherSizeIsRefused() {
        val error =
            assertThrows(IllegalStateException::class.java) {
                runWithSupportImage(model, inputs, outputs, normalized(32), INPUT_SIZE, labels)
            }
        Log.i(TAG, "Refused: ${error.message}")
        assertEquals("Input size ${32 * 32 * 3}", error.message)
    }

    @Test
    fun wrongGoldenFailsTheGate() {
        // The golden with its first value moved by 0.01, ten times the bound.
        val wrong = readGoldenOutput(testContext, "golden/output_wrong.txt")
        val diff = maxAbsDifference(wrong, infer(goldenInput))
        Log.i(TAG, "wrong golden: max abs diff = $diff")
        assertFalse("the gate passed a wrong golden: max abs diff = $diff", diff < 1e-3f)
    }

    @Test
    fun goldenOfAnotherLengthIsRefused() {
        val error =
            assertThrows(IllegalStateException::class.java) {
                maxAbsDifference(FloatArray(5), FloatArray(10))
            }
        assertEquals("Size 10 != 5", error.message)
    }

    @Test
    fun uint8BytesInAFloatInputGiveNaN() {
        // What runWithSupportImage refuses: the bytes of the uint8 image written with writeInt8
        // into the float input. The write does not throw.
        val image = TensorImage.fromBitmap(pattern(224)).tensorBuffer
        val bytes = ByteArray(image.flatSize)
        image.buffer.duplicate().apply { rewind() }.get(bytes)
        inputs[0].writeInt8(bytes)
        model.run(inputs, outputs)
        val output = outputs[0].readFloat()
        val nan = output.count { it.isNaN() }
        Log.i(TAG, "uint8 bytes in the float input: $nan of ${output.size} outputs are NaN")
        assertEquals("NaN outputs", output.size, nan)
    }

    @Test
    fun cpuOptionsWithThreadsRun() {
        val shared = CompiledModel.Options.CPU.cpuOptions
        val diff =
            CompiledModel.create(
                    appContext.assets,
                    "model.tflite",
                    cpuOptionsWithThreads(2),
                    environment,
                )
                .use { two ->
                    val ins = two.createInputBuffers()
                    val outs = two.createOutputBuffers()
                    try {
                        ins[0].writeFloat(goldenInput)
                        two.run(ins, outs)
                        maxAbsDifference(golden, outs[0].readFloat())
                    } finally {
                        (ins + outs).forEach { it.close() }
                    }
                }
        Log.i(TAG, "CpuOptions(numThreads = 2): max abs diff to the golden = $diff")
        assertTrue("max abs diff to the golden = $diff", diff < 1e-3f)
        // New options were created; the shared CompiledModel.Options.CPU is as it was.
        assertSame(shared, CompiledModel.Options.CPU.cpuOptions)
    }

    @Test
    fun inputShapeByName() {
        // The signature names of a LiteRT-Torch export.
        assertEquals(listOf(1, 224, 224, 3), inputShape(model, "args_0"))
    }

    @Test
    fun signatureRunMatchesTheListForm() {
        val listOutput = infer(goldenInput)
        val ins = model.createInputBuffers("serving_default")
        val outs = model.createOutputBuffers("serving_default")
        try {
            ins[0].writeFloat(goldenInput)
            runSignature(
                model,
                "serving_default",
                mapOf("args_0" to ins[0]),
                mapOf("output_0" to outs[0]),
            )
            val diff = maxAbsDifference(listOutput, outs[0].readFloat())
            Log.i(TAG, "runSignature: max abs diff to the list form = $diff")
            assertEquals(0f, diff, 0f)
        } finally {
            (ins + outs).forEach { it.close() }
        }
    }

    @Test
    fun gpuWithCpuFallbackLoadsAGraphTheGpuRejects() {
        // The app's model on the GPU alone says whether this device has the GPU accelerator.
        val gpu =
            try {
                CompiledModel.create(
                        appContext.assets,
                        "model.tflite",
                        CompiledModel.Options(Accelerator.GPU),
                        environment,
                    )
                    .close()
                true
            } catch (e: LiteRtException) {
                Log.i(TAG, "no GPU for the app's model: ${e.message}")
                false
            }
        assumeTrue("the GPU accelerator does not take the app's model here", gpu)
        // gpu_unsupported.tflite ends with a 6-D transpose that the GPU does not compile
        // (tools/make_fixtures.py): the GPU alone throws, the GPU with the CPU runs it.
        val alone =
            assertThrows(LiteRtException::class.java) {
                CompiledModel.create(
                        testContext.assets,
                        "gpu_unsupported.tflite",
                        CompiledModel.Options(Accelerator.GPU),
                        environment,
                    )
                    .close()
            }
        Log.i(TAG, "GPU alone on gpu_unsupported.tflite: ${alone.message}")
        val output =
            CompiledModel.create(
                    testContext.assets,
                    "gpu_unsupported.tflite",
                    gpuWithCpuFallback,
                    environment,
                )
                .use { both ->
                    val ins = both.createInputBuffers()
                    val outs = both.createOutputBuffers()
                    try {
                        ins[0].writeFloat(goldenInput)
                        both.run(ins, outs)
                        outs[0].readFloat()
                    } finally {
                        (ins + outs).forEach { it.close() }
                    }
                }
        // The transpose permutes the ten values; compare them sorted, with room for the GPU's fp16.
        val diff = maxAbsDifference(golden.sortedArray(), output.sortedArray())
        Log.i(TAG, "GPU with CPU on gpu_unsupported.tflite: sorted outputs max abs diff = $diff")
        assertTrue("NaN in the output", output.none { it.isNaN() })
        assertTrue("max abs diff of the sorted outputs = $diff", diff < 1e-2f)
    }

    @Test
    fun goldenFileRoundTrip() {
        val file = File(appContext.cacheDir, "golden.txt")
        try {
            writeGoldenOutput(file, golden)
            val back =
                file.readLines().filter { it.isNotBlank() }.map { it.toFloat() }.toFloatArray()
            assertEquals(0f, maxAbsDifference(golden, back), 0f)
        } finally {
            file.delete()
        }
    }

    /** The model's output for [input] on the CPU, through the buffers of [setUp]. */
    private fun infer(input: FloatArray): FloatArray {
        inputs[0].writeFloat(input)
        model.run(inputs, outputs)
        return outputs[0].readFloat()
    }

    /** The golden's input: the pattern at 224 through the skill's `preprocess`. */
    private val goldenInput: FloatArray by lazy { preprocess(pattern(224)) }

    /** The pattern at [size] through NormalizeOp(127.5, 127.5): the values of [goldenInput]. */
    private fun normalized(size: Int): TensorImage =
        ImageProcessor.Builder()
            .add(NormalizeOp(127.5f, 127.5f))
            .build()
            .process(TensorImage.fromBitmap(pattern(size)))

    /** The deterministic RGB pattern of tools/make_golden.py, at any size. */
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
        const val TAG = "MigrationCodeTest"

        /** One Environment for these tests, kept open, as the skill's Step 2 keeps its own. */
        val environment: Environment by lazy { Environment.create() }
    }
}
