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
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.LiteRtException
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import java.util.concurrent.Callable
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Runs the memory options of step 2 of the skill, `stageModelAsset` and `memoryOptions`, with the
 * app's own model. The asset is copied once under its versioned name, a new version deletes the
 * copy of the previous one, and a copy that stopped part-way never becomes the staged file. The
 * first load from the staged file writes the XNNPACK weight cache, a second load reads it and gives
 * the same output, which is also the output of the skill's `Classifier` loading the asset, and a
 * new version deletes the outdated cache. On the GPU the same options give the output of the
 * default GPU options, and `create` on a path with no file throws `LiteRtException`.
 *
 * As in the skill, one thread creates, runs and closes every model, the buffers are closed before
 * their model, and every model uses the skill's own `Environment`, which serves the whole process.
 */
@RunWith(AndroidJUnit4::class)
class MemoryOptionsTest {
    private val context: Context = InstrumentationRegistry.getInstrumentation().targetContext
    private val modelThread = Executors.newSingleThreadExecutor()

    // The model averages each channel, then applies a linear layer: from all zeros its output
    // would be the bias alone, and the weights the cache holds would not reach it.
    private val input = FloatArray(INPUT_SIZE) { i -> (i % 3 + 1) * 0.25f }
    private val notTheAsset = "not the asset".toByteArray()

    @Before
    fun setUp() {
        // The app's files outlive a test: each test starts with no staged copy and no cache.
        stagedFiles().forEach { it.delete() }
        caches().forEach { it.delete() }
    }

    @After
    fun tearDown() {
        modelThread.shutdown()
        val ended = modelThread.awaitTermination(TIMEOUT_SECONDS, TimeUnit.SECONDS)
        assertTrue("the model's thread did not end", ended)
    }

    @Test
    fun stageModelAsset_copiesTheAssetOnce_underItsVersionedName() {
        val staged = stageModelAsset(context)
        assertEquals(File(context.noBackupFilesDir, "v1-$ASSET"), staged)
        assertArrayEquals(assetBytes(), staged.readBytes())
        // A file under that name is the copy: the next call returns it and does not copy again.
        staged.writeBytes(notTheAsset)
        assertEquals(staged, stageModelAsset(context))
        assertArrayEquals("the asset was copied again", notTheAsset, staged.readBytes())
    }

    @Test
    fun stageModelAsset_aNewVersionDeletesThePreviousCopy() {
        val previous = stageModelAsset(context, version = "v1")
        val staged = stageModelAsset(context, version = "v2")
        assertEquals(File(context.noBackupFilesDir, "v2-$ASSET"), staged)
        assertArrayEquals(assetBytes(), staged.readBytes())
        assertFalse("${previous.name} is still there", previous.exists())
    }

    @Test
    fun stageModelAsset_afterACopyThatStoppedPartWay_stagesTheWholeAsset() {
        // What a process death during the copy leaves: the first bytes, under the temporary name.
        val partial = File(context.noBackupFilesDir, "v1-$ASSET.tmp")
        partial.writeBytes(assetBytes().copyOf(PARTIAL_BYTES))
        val staged = stageModelAsset(context)
        assertArrayEquals(assetBytes(), staged.readBytes())
        assertFalse("${partial.name} is still there", partial.exists())
    }

    @Test
    fun memoryOptions_theFirstLoadWritesTheCache_andTheSecondLoadReadsIt() {
        val staged = stageModelAsset(context)
        val cache = cacheOf(memoryOptions(context))
        assertEquals(File(context.codeCacheDir, "$ASSET-v1.xnnpack_cache"), cache)

        val first = onModelThread { load(staged, memoryOptions(context)).use { it.run(input) } }
        assertEquals(OUTPUT_SIZE, first.size)
        val written = cache.length()
        assertTrue("the first load wrote no cache at ${cache.path}", cache.isFile && written > 0)
        Log.i(TAG, "weight cache: $written bytes for a model of ${staged.length()} bytes")

        val (second, cacheWhileOpen) =
            onModelThread {
                load(staged, memoryOptions(context)).use { it.run(input) to cache.length() }
            }
        assertArrayEquals("the output changed", first.rawBits(), second.rawBits())
        assertEquals("the second load changed the cache", written, cacheWhileOpen)

        // The skill's Classifier loads the same model from the asset, with the default options.
        val fromAsset =
            onModelThread { Classifier(context, Accelerator.CPU).use { it.infer(input) } }
        val diff = largestDifference(fromAsset, first)
        assertTrue("max abs diff to the asset load = $diff", diff < 1e-5f)
    }

    @Test
    fun memoryOptions_aNewVersionDeletesTheOutdatedCache() {
        val staged = stageModelAsset(context)
        onModelThread {
            load(staged, memoryOptions(context, version = "v1")).use { it.run(input) }
        }
        val outdated = File(context.codeCacheDir, "$ASSET-v1.xnnpack_cache")
        assertTrue("the load wrote no cache at ${outdated.path}", outdated.isFile)
        // The cache of another model, which a new version of this one leaves alone.
        val other = File(context.codeCacheDir, "other.tflite-v1.xnnpack_cache")
        other.writeBytes(notTheAsset)

        val cache = cacheOf(memoryOptions(context, version = "v2"))
        assertEquals(File(context.codeCacheDir, "$ASSET-v2.xnnpack_cache"), cache)
        assertFalse("${outdated.name} is still there", outdated.exists())
        assertTrue("${other.name} was deleted", other.exists())
    }

    @Test
    fun memoryOptionsOnTheGpu_giveTheOutputOfTheDefaultGpuOptions() {
        val fromAsset = onModelThread { loadOnGpuFromTheAsset()?.use { it.run(input) } }
        assumeTrue("the GPU does not take the app's model here", fromAsset != null)
        val staged = stageModelAsset(context)
        val fromPath =
            onModelThread {
                load(staged, memoryOptions(context, Accelerator.GPU)).use { it.run(input) }
            }
        // The CPU options are not used on the GPU.
        val diff = largestDifference(checkNotNull(fromAsset), fromPath)
        Log.i(TAG, "GPU: max abs diff to the default GPU options = $diff")
        assertTrue("max abs diff to the default GPU options = $diff", diff < 1e-5f)
    }

    @Test
    fun createFromAPathWithNoFile_throws() {
        val missing = File(context.noBackupFilesDir, "v1-missing.tflite")
        assertFalse("${missing.path} is there", missing.exists())
        // The create call throws before it returns a model, so no model and no buffer is left.
        assertThrows(LiteRtException::class.java) {
            onModelThread {
                CompiledModel.create(missing.path, memoryOptions(context), environment).close()
            }
        }
    }

    /** The staged copies of the app's model, with the temporary file of a copy. */
    private fun stagedFiles() =
        context.noBackupFilesDir.listFiles().orEmpty().filter { "-$ASSET" in it.name }

    /** The XNNPACK weight caches in `codeCacheDir`. */
    private fun caches() =
        context.codeCacheDir.listFiles().orEmpty().filter { it.name.endsWith(".xnnpack_cache") }

    private fun assetBytes() = context.assets.open(ASSET).use { it.readBytes() }

    private fun cacheOf(options: CompiledModel.Options) =
        File(checkNotNull(options.cpuOptions?.xnnPackWeightCachePath) { "no weight cache path" })

    /** Creates the model from [file] with [options], then its buffers. */
    private fun load(file: File, options: CompiledModel.Options) =
        Loaded(CompiledModel.create(file.path, options, environment))

    /** The app's model from its asset on the GPU alone, or null where the GPU does not take it. */
    private fun loadOnGpuFromTheAsset(): Loaded? =
        try {
            val options = CompiledModel.Options(Accelerator.GPU)
            Loaded(CompiledModel.create(context.assets, ASSET, options, environment))
        } catch (e: LiteRtException) {
            Log.i(TAG, "no GPU for the app's model: ${e.message}")
            null
        }

    /** Runs [block] on the thread that owns the models. What [block] throws is thrown here. */
    private fun <T> onModelThread(block: () -> T): T =
        try {
            modelThread.submit(Callable { block() }).get(TIMEOUT_SECONDS, TimeUnit.SECONDS)
        } catch (e: ExecutionException) {
            throw e.cause ?: e
        }

    private fun largestDifference(a: FloatArray, b: FloatArray): Float {
        assertEquals("output sizes", a.size, b.size)
        return a.indices.maxOf { abs(a[it] - b[it]) }
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
        const val TAG = "MemoryOptionsTest"
        const val ASSET = "model.tflite"
        const val OUTPUT_SIZE = 10
        const val PARTIAL_BYTES = 100
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
