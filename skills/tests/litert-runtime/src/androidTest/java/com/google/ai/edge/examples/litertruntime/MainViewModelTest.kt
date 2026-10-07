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

import android.graphics.Bitmap
import android.os.Debug
import android.util.Log
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.ViewModelStore
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.SdkSuppress
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.JniHandle
import java.io.File
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.ExecutorService
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Drives the skill's `MainViewModel` on a device through the paths where a step throws, and checks
 * three things on each: the state the screen reads, that nothing is thrown on the ViewModel's
 * thread (an exception there ends the app), and that every model that was created is closed.
 *
 * A model that is never closed is unreachable from here, so those tests repeat the failing load and
 * bound what the process keeps instead. The bounds come from a Galaxy S26 (Android 16, LiteRT
 * 2.2.0): 20 unclosed CPU models kept 13.8 MB of native heap, 20 unclosed GPU models 62.9 MB and
 * 40 threads, and 50 closed ones under 100 KB together.
 */
@RunWith(AndroidJUnit4::class)
@SdkSuppress(minSdkVersion = 30)
class MainViewModelTest {
    private val uncaught = CopyOnWriteArrayList<Throwable>()
    private val stores = LinkedHashMap<MainViewModel, ViewModelStore>()
    private val executors = ArrayList<ExecutorService>()
    private var defaultHandler: Thread.UncaughtExceptionHandler? = null
    private lateinit var assets: ModelAssets

    @Before
    fun setUp() {
        assets = ModelAssets()
        defaultHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { _, throwable -> uncaught += throwable }
    }

    @After
    fun tearDown() {
        stores.values.forEach { it.clear() }
        // The close runs on the ViewModel's thread. Wait for it before reading what was thrown.
        val ended = executors.all { it.awaitTermination(TIMEOUT_SECONDS, TimeUnit.SECONDS) }
        Thread.setDefaultUncaughtExceptionHandler(defaultHandler)
        assertTrue("the ViewModel's thread did not end", ended)
        assertTrue("thrown on the ViewModel's thread: $uncaught", uncaught.isEmpty())
    }

    @Test
    fun loadOnCpu_thenClassify() {
        val viewModel = newViewModel()
        viewModel.load(Accelerator.CPU)
        viewModel.awaitIdle()
        assertTrue(viewModel.uiState.value.ready)
        viewModel.classify(FloatArray(INPUT_SIZE))
        viewModel.awaitIdle()
        assertEquals(OUTPUT_SIZE, viewModel.uiState.value.result?.size)
    }

    @Test
    fun loadOnGpu_thenClassify() {
        val viewModel = newViewModel()
        viewModel.load(Accelerator.GPU)
        viewModel.awaitIdle()
        assumeTrue("no GPU accelerator here: ${viewModel.uiState.value.error}", viewModel.isReady())
        viewModel.classify(FloatArray(INPUT_SIZE))
        viewModel.awaitIdle()
        assertEquals(OUTPUT_SIZE, viewModel.uiState.value.result?.size)
    }

    @Test
    fun warmUpWriteThrows_onCpu_theModelIsClosed() {
        assertFailedLoadsKeepNothing("small_input", Accelerator.CPU)
    }

    @Test
    fun warmUpWriteThrows_onGpu_theModelIsClosed() {
        assumeGpu()
        assertFailedLoadsKeepNothing("small_input", Accelerator.GPU)
    }

    @Test
    fun warmUpRunThrows_theModelIsClosed() {
        assertFailedLoadsKeepNothing("float16_output", Accelerator.CPU)
    }

    @Test
    fun inputBuffersThrowAfterTheModelIsCreated_theModelIsClosed() {
        assertFailedLoadsKeepNothing("string_input", Accelerator.CPU)
    }

    @Test
    fun outputBuffersThrowAfterTheModelIsCreated_theModelIsClosed() {
        assertFailedLoadsKeepNothing("string_output", Accelerator.CPU)
    }

    @Test
    fun createThrowsOnAFileThatIsNotAModel_stateIsError() {
        assertFailedLoadsKeepNothing("not_a_model", Accelerator.CPU)
    }

    @Test
    fun gpuCreateThrows_theFailedLoadsKeepNothing() {
        assumeGpu()
        assertFailedLoadsKeepNothing("gpu_unsupported", Accelerator.GPU)
    }

    @Test
    fun gpuCreateThrows_stateIsError_thenTheCpuLoads() {
        assumeGpu()
        assets.use("gpu_unsupported")
        val viewModel = newViewModel()
        viewModel.load(Accelerator.GPU)
        viewModel.awaitIdle()
        assertNotNull(viewModel.uiState.value.error)
        assertFalse(viewModel.uiState.value.ready)
        viewModel.load(Accelerator.CPU)
        viewModel.awaitIdle()
        assertTrue(viewModel.uiState.value.ready)
        assertNull(viewModel.uiState.value.error)
        viewModel.classify(FloatArray(INPUT_SIZE))
        viewModel.awaitIdle()
        assertEquals(OUTPUT_SIZE, viewModel.uiState.value.result?.size)
    }

    @Test
    fun loadFails_thenLoadsOnceTheModelIsGood() {
        assets.use("small_input")
        val viewModel = newViewModel()
        viewModel.load(Accelerator.CPU)
        viewModel.awaitIdle()
        assertNotNull(viewModel.uiState.value.error)
        assets.use(ModelAssets.APP_MODEL)
        viewModel.load(Accelerator.CPU)
        viewModel.awaitIdle()
        assertTrue(viewModel.uiState.value.ready)
        assertNull(viewModel.uiState.value.error)
    }

    @Test
    fun loadAgain_keepsTheModel_andClassifyStillWorks() {
        val viewModel = newViewModel()
        viewModel.load(Accelerator.CPU)
        viewModel.awaitIdle()
        val first = checkNotNull(viewModel.classifier())
        // A second load would fail its warm-up on this model. The screen calls load() again after
        // every rotation, so the loaded model has to survive the call.
        assets.use("small_input")
        viewModel.load(Accelerator.GPU)
        viewModel.awaitIdle()
        assertSame(first, viewModel.classifier())
        assertFalse(first.isClosed())
        assertTrue(viewModel.uiState.value.ready)
        viewModel.classify(FloatArray(INPUT_SIZE))
        viewModel.awaitIdle()
        assertEquals(OUTPUT_SIZE, viewModel.uiState.value.result?.size)
    }

    @Test
    fun classifyThrows_stateIsError_andTheModelStaysLoaded() {
        val viewModel = newViewModel()
        viewModel.load(Accelerator.CPU)
        viewModel.awaitIdle()
        viewModel.classify(FloatArray(INPUT_SIZE + 1))
        viewModel.awaitIdle()
        assertNotNull(viewModel.uiState.value.error)
        viewModel.classify(FloatArray(INPUT_SIZE))
        viewModel.awaitIdle()
        assertEquals(OUTPUT_SIZE, viewModel.uiState.value.result?.size)
        assertNull(viewModel.uiState.value.error)
    }

    @Test
    fun onCleared_closesTheModel_andEndsTheThread() {
        val viewModel = newViewModel()
        viewModel.load(Accelerator.CPU)
        viewModel.awaitIdle()
        val classifier = checkNotNull(viewModel.classifier())
        clearAndAwait(viewModel)
        assertTrue(classifier.isClosed())
    }

    @Test
    fun onCleared_rightAfterLoad_closesTheModel() {
        val viewModel = newViewModel()
        viewModel.load(Accelerator.CPU)
        clearAndAwait(viewModel)
        assertTrue(checkNotNull(viewModel.classifier()).isClosed())
    }

    @Test
    fun newViewModelLoads_whileTheOldOneCloses() {
        var old = newViewModel()
        old.load(Accelerator.CPU)
        old.awaitIdle()
        repeat(HANDOVERS) { index ->
            val next = newViewModel()
            // The old model closes on its own thread while the next one is created on another.
            checkNotNull(stores[old]).clear()
            next.load(Accelerator.CPU)
            next.awaitIdle()
            assertTrue("handover $index: ${next.uiState.value.error}", next.isReady())
            assertTrue(old.executor().awaitTermination(TIMEOUT_SECONDS, TimeUnit.SECONDS))
            assertTrue(checkNotNull(old.classifier()).isClosed())
            old = next
        }
    }

    @Test
    fun onCleared_withoutLoad_endsTheThread() {
        clearAndAwait(newViewModel())
    }

    @Test
    fun preprocess_readsAHardwareBitmap() {
        // ImageDecoder usually returns HARDWARE bitmaps; getPixels() cannot read them.
        val software = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
        val hardware = checkNotNull(software.copy(Bitmap.Config.HARDWARE, false))
        assertEquals(Bitmap.Config.HARDWARE, hardware.config)
        val input = preprocess(hardware)
        assertEquals(INPUT_SIZE, input.size)
        assertEquals(-1f, input[0])
    }

    @Test
    fun manyViewModelLifetimes_eachLoadsAndCloses() {
        repeat(LIFETIMES) { index ->
            val viewModel = newViewModel()
            viewModel.load(Accelerator.CPU)
            viewModel.awaitIdle()
            val error = viewModel.uiState.value.error
            assertTrue("lifetime $index: $error", viewModel.isReady())
            clearAndAwait(viewModel)
            assertTrue(checkNotNull(viewModel.classifier()).isClosed())
            if (index % 25 == 24) {
                Log.i(TAG, "${index + 1} ViewModel lifetimes done")
            }
        }
    }

    @Test
    fun loadAfterOnCleared_createsNothing() {
        val viewModel = newViewModel()
        clearAndAwait(viewModel)
        val jobs = checkNotNull(viewModel.scope().coroutineContext[Job])
        val before = jobs.children.toSet()
        viewModel.load(Accelerator.CPU)
        val launched = jobs.children.filter { it !in before }.toList()
        runBlocking {
            withTimeout(TIMEOUT_SECONDS * 1000) {
                launched.forEach { it.join() }
            }
        }
        assertNull(viewModel.classifier())
    }

    /** Loads [fixture], which must fail, then repeats the load and bounds what it keeps. */
    private fun assertFailedLoadsKeepNothing(fixture: String, accelerator: Accelerator) {
        assets.use(fixture)
        val viewModel = newViewModel()
        viewModel.load(accelerator)
        viewModel.awaitIdle()
        assertNotNull(viewModel.uiState.value.error)
        assertFalse(viewModel.uiState.value.ready)
        assertNull(viewModel.classifier())
        val heap = Debug.getNativeHeapAllocatedSize()
        val threads = threadCount()
        repeat(REPEATS) {
            viewModel.load(accelerator)
        }
        viewModel.awaitIdle()
        val heapKb = (Debug.getNativeHeapAllocatedSize() - heap) / 1024
        val moreThreads = threadCount() - threads
        val kept = "$heapKb KB of native heap and $moreThreads threads"
        Log.i(TAG, "$fixture on $accelerator: $REPEATS failed loads kept $kept")
        assertTrue("$REPEATS failed loads kept $kept", heapKb < HEAP_LIMIT_KB)
        assertTrue("$REPEATS failed loads kept $kept", moreThreads < THREAD_LIMIT)
    }

    private fun assumeGpu() {
        val viewModel = newViewModel()
        viewModel.load(Accelerator.GPU)
        viewModel.awaitIdle()
        assumeTrue("no GPU accelerator here: ${viewModel.uiState.value.error}", viewModel.isReady())
    }

    private fun newViewModel(): MainViewModel {
        val store = ViewModelStore()
        val factory = ViewModelProvider.AndroidViewModelFactory(assets.application)
        val viewModel = ViewModelProvider(store, factory)[MainViewModel::class.java]
        stores[viewModel] = store
        executors += viewModel.executor()
        return viewModel
    }

    /** Calls `onCleared()` the way the framework does and waits for the ViewModel's thread. */
    private fun clearAndAwait(viewModel: MainViewModel) {
        checkNotNull(stores[viewModel]).clear()
        val ended = viewModel.executor().awaitTermination(TIMEOUT_SECONDS, TimeUnit.SECONDS)
        assertTrue("the ViewModel's thread did not end", ended)
    }

    private fun MainViewModel.isReady() = uiState.value.ready

    /** Blocks until every task already queued on the ViewModel's single thread has run. */
    private fun MainViewModel.awaitIdle() {
        val idle = executor().submit {}
        idle.get(TIMEOUT_SECONDS, TimeUnit.SECONDS)
    }

    private fun MainViewModel.executor() = checkNotNull(privateField<ExecutorService>("executor"))

    private fun MainViewModel.scope() = checkNotNull(privateField<CoroutineScope>("scope"))

    private fun MainViewModel.classifier() = privateField<Classifier>("classifier")

    /** Reads the flag LiteRT sets when a handle is closed, without calling into the model. */
    private fun Classifier.isClosed(): Boolean {
        val model = checkNotNull(privateField<CompiledModel>("model"))
        val destroyed = JniHandle::class.java.getDeclaredField("destroyed")
        destroyed.isAccessible = true
        return (destroyed.get(model) as AtomicBoolean).get()
    }

    @Suppress("UNCHECKED_CAST")
    private fun <T> Any.privateField(name: String): T? {
        val field = javaClass.getDeclaredField(name)
        field.isAccessible = true
        return field.get(this) as T?
    }

    private fun threadCount() = File("/proc/self/task").list()?.size ?: 0

    private companion object {
        const val TAG = "MainViewModelTest"
        const val OUTPUT_SIZE = 10
        const val TIMEOUT_SECONDS = 120L
        const val REPEATS = 50
        const val LIFETIMES = 200
        const val HANDOVERS = 50

        /** 20 unclosed models kept 13.8 MB; 50 closed ones keep under 100 KB together. */
        const val HEAP_LIMIT_KB = 1024L

        /** 20 unclosed GPU models kept 40 threads. */
        const val THREAD_LIMIT = 10
    }
}
