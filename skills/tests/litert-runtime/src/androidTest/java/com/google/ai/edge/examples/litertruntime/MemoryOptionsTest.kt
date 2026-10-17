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
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.LiteRtException
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import java.util.concurrent.Callable
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Creates the app's model from a file with the skill's `memoryOptions`: the first load writes the
 * XNNPACK weight cache file, a second load with the same options reads it and gives the same
 * output, and `create` on a file that is not there throws `LiteRtException`.
 *
 * As in the skill, one thread creates, runs and closes every model, the buffers are closed before
 * their model, and every model uses the skill's own `Environment`, which serves the whole process.
 */
@RunWith(AndroidJUnit4::class)
class MemoryOptionsTest {
    private val context: Context = InstrumentationRegistry.getInstrumentation().targetContext
    private val modelThread = Executors.newSingleThreadExecutor()

    @After
    fun tearDown() {
        modelThread.shutdown()
        val ended = modelThread.awaitTermination(TIMEOUT_SECONDS, TimeUnit.SECONDS)
        assertTrue("the model's thread did not end", ended)
    }

    @Test
    fun memoryOptions_writesTheCacheOnTheFirstLoad_andTheSecondLoadReadsIt() {
        val modelFile = copyAppModel()
        val cachePath = memoryOptions(context).cpuOptions?.xnnPackWeightCachePath
        val cache = File(checkNotNull(cachePath) { "memoryOptions sets no weight cache path" })
        cache.delete()
        assertFalse("could not delete ${cache.path}", cache.exists())
        // The model averages each channel, then applies a linear layer: from all zeros its output
        // would be the bias alone, and the weights the cache holds would not reach it.
        val input = FloatArray(INPUT_SIZE) { i -> (i % 3 + 1) * 0.25f }

        val first = onModelThread { load(modelFile).use { it.run(input) } }
        assertEquals(OUTPUT_SIZE, first.size)
        val written = cache.length()
        assertTrue("the first load wrote no cache at ${cache.path}", cache.isFile && written > 0)

        val (second, cacheWhileOpen) =
            onModelThread { load(modelFile).use { it.run(input) to cache.length() } }
        assertArrayEquals("the output changed", first.rawBits(), second.rawBits())
        assertEquals("the second load changed the cache", written, cacheWhileOpen)
    }

    @Test
    fun createFromAMissingFile_throws() {
        val missing = File(context.noBackupFilesDir, "missing.tflite")
        missing.delete()
        // The create call throws before it returns a model, so no model and no buffer is left.
        assertThrows(LiteRtException::class.java) {
            onModelThread {
                CompiledModel.create(missing.path, memoryOptions(context), environment).close()
            }
        }
    }

    /** Copies the app's `assets/model.tflite` into `noBackupFilesDir`, as the skill describes. */
    private fun copyAppModel(): File {
        val file = File(context.noBackupFilesDir, "model.tflite")
        context.assets.open("model.tflite").use { input ->
            file.outputStream().use { output -> input.copyTo(output) }
        }
        return file
    }

    /** Creates the model from [file] with the skill's `memoryOptions`, then its buffers. */
    private fun load(file: File) =
        Loaded(CompiledModel.create(file.path, memoryOptions(context), environment))

    /** Runs [block] on the thread that owns the models. What [block] throws is thrown here. */
    private fun <T> onModelThread(block: () -> T): T =
        try {
            modelThread.submit(Callable { block() }).get(TIMEOUT_SECONDS, TimeUnit.SECONDS)
        } catch (e: ExecutionException) {
            throw e.cause ?: e
        }

    private fun FloatArray.rawBits() = IntArray(size) { this[it].toRawBits() }

    /**
     * A model with its buffers, closed together, the buffers first, like the skill's `Classifier`.
     */
    private class Loaded(private val model: CompiledModel) : AutoCloseable {
        private val inputs = mutableListOf<TensorBuffer>()
        private val outputs = mutableListOf<TensorBuffer>()

        init {
            try {
                inputs += model.createInputBuffers()
                outputs += model.createOutputBuffers()
            } catch (e: Exception) {
                close()
                throw e
            }
        }

        fun run(input: FloatArray): FloatArray {
            inputs[0].writeFloat(input)
            model.run(inputs, outputs)
            return outputs[0].readFloat()
        }

        override fun close() {
            (inputs + outputs).forEach { it.close() }
            model.close()
        }
    }

    private companion object {
        const val OUTPUT_SIZE = 10
        const val TIMEOUT_SECONDS = 120L

        /**
         * The skill's `environment`, which is private to the skill's code, so it is read by
         * reflection: one `Environment` serves every model of the process, the skill's included.
         */
        val environment: Environment by lazy {
            val skillCode = Class.forName("com.google.ai.edge.examples.litertruntime.SkillCodeKt")
            val delegate = skillCode.getDeclaredField("environment\$delegate")
            delegate.isAccessible = true
            (delegate.get(null) as Lazy<*>).value as Environment
        }
    }
}
