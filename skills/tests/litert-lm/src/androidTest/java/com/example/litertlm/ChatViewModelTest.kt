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

package com.example.litertlm

import android.app.Application
import android.os.Debug
import android.util.Log
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.ViewModelStore
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.Engine
import java.io.File
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutorService
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNotSame
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Drives the skill's `ChatViewModel` on a device with a real model: a second `load()`, a `load()`
 * that fails, and `load()` or `onCleared()` while a reply is streaming. Each test checks the state
 * the screen reads, that nothing is thrown on the ViewModel's thread (an exception there ends the
 * app), and that every engine and conversation that was created is closed exactly once.
 *
 * Push a model first: `adb push Qwen3-0.6B.litertlm /data/local/tmp/`. Another file or the GPU:
 * `-e modelPath <path>` and `-e backend GPU`.
 */
@RunWith(AndroidJUnit4::class)
class ChatViewModelTest {
    private val arguments = InstrumentationRegistry.getArguments()
    private val modelPath = arguments.getString("modelPath") ?: DEFAULT_MODEL
    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    /** A copy of the model, so that a second load names a different model path. */
    private val secondModelPath: String by lazy { copyOfModel().path }
    private val uncaught = CopyOnWriteArrayList<Throwable>()
    private val stores = ArrayList<ViewModelStore>()
    private val executors = ArrayList<ExecutorService>()
    private var defaultHandler: Thread.UncaughtExceptionHandler? = null

    @Before
    fun setUp() {
        assertTrue("no model at $modelPath: adb push one there", File(modelPath).canRead())
        defaultHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { _, throwable -> uncaught += throwable }
    }

    @After
    fun tearDown() {
        stores.forEach { it.clear() }
        // The close runs on the ViewModel's thread. Wait for it before reading what was thrown.
        val ended = executors.all { it.awaitTermination(TIMEOUT_SECONDS, TimeUnit.SECONDS) }
        Thread.setDefaultUncaughtExceptionHandler(defaultHandler)
        assertTrue("the ViewModel's thread did not end", ended)
        assertTrue("thrown on the ViewModel's thread: $uncaught", uncaught.isEmpty())
    }

    @Test
    fun load_thenSend_streamsAReply() {
        val viewModel = newViewModel()
        viewModel.loadAndAwait(modelPath)
        assertTrue(viewModel.state.value.ready)
        val reply = viewModel.launched { viewModel.send(SHORT_PROMPT) }
        await(reply)
        assertNull(viewModel.state.value.error)
        assertFalse(viewModel.state.value.busy)
        assertTrue(viewModel.state.value.reply.isNotEmpty())
    }

    @Test
    fun send_beforeLoad_doesNothing() {
        val viewModel = newViewModel()
        await(viewModel.launched { viewModel.send(SHORT_PROMPT) })
        assertEquals(ChatState(), viewModel.state.value)
    }

    @Test
    fun loadAnotherModel_closesTheEngineItReplaces() {
        val viewModel = newViewModel()
        viewModel.loadAndAwait(modelPath)
        val firstEngine = checkNotNull(viewModel.engine())
        val firstConversation = checkNotNull(viewModel.conversation())
        viewModel.loadAndAwait(secondModelPath)
        assertTrue(viewModel.state.value.ready)
        assertNotSame(firstEngine, viewModel.engine())
        assertFalse("the first conversation is still open", firstConversation.isAlive)
        assertFalse("the first engine is still open", firstEngine.isInitialized())
    }

    @Test
    fun loadSameModelOnTheOtherBackend_keepsTheEngine() {
        val viewModel = newViewModel()
        viewModel.loadAndAwait(modelPath)
        val firstEngine = checkNotNull(viewModel.engine())
        // The screen asks for the GPU after every rotation, also when the app fell back to the CPU.
        viewModel.loadAndAwait(modelPath, otherBackend())
        assertSame(firstEngine, viewModel.engine())
        assertTrue(firstEngine.isInitialized())
        assertTrue(viewModel.state.value.ready)
    }

    @Test
    fun sendWhileAReplyStreams_doesNothing() {
        val viewModel = newViewModel()
        viewModel.loadAndAwait(modelPath)
        val first = viewModel.launched { viewModel.send(LONG_PROMPT) }
        viewModel.await("the reply to start") { it.reply.isNotEmpty() }
        val soFar = viewModel.state.value.reply
        await(viewModel.launched { viewModel.send(SHORT_PROMPT) })
        assertFalse("the first reply ended", first.single().isCompleted)
        assertTrue("the second send reset the reply", viewModel.state.value.reply.startsWith(soFar))
        assertTrue(viewModel.state.value.busy)
    }

    @Test
    fun loadSameModelAgain_keepsTheEngine() {
        val viewModel = newViewModel()
        viewModel.loadAndAwait(modelPath)
        val firstEngine = checkNotNull(viewModel.engine())
        viewModel.loadAndAwait(modelPath)
        assertSame(firstEngine, viewModel.engine())
        assertTrue(firstEngine.isInitialized())
        assertTrue(viewModel.state.value.ready)
    }

    @Test
    fun loadAlternately_keepsOneEngine() {
        val viewModel = newViewModel()
        viewModel.loadAndAwait(modelPath)
        viewModel.loadAndAwait(secondModelPath)
        val start = pssMb()
        val seen = ArrayList<Long>()
        repeat(ALTERNATIONS) {
            viewModel.loadAndAwait(modelPath)
            viewModel.loadAndAwait(secondModelPath)
            seen += pssMb()
        }
        Log.i(TAG, "pss after two loads $start MB, after each further pair $seen MB")
        assertTrue(viewModel.state.value.ready)
        assertTrue("pss went from $start MB to $seen MB", seen.last() - start < PSS_LIMIT_MB)
    }

    @Test
    fun loadFails_stateIsError_andNothingStaysOpen() {
        val notAModel = File(context.cacheDir, "not-a-model.litertlm")
        notAModel.writeBytes(ByteArray(4096))
        val viewModel = newViewModel()
        for (path in listOf(MISSING_MODEL, notAModel.path)) {
            viewModel.loadAndAwait(path)
            assertNotNull(viewModel.state.value.error)
            assertFalse(viewModel.state.value.ready)
            assertNull(viewModel.engine())
            assertNull(viewModel.conversation())
        }
        val heap = Debug.getNativeHeapAllocatedSize()
        repeat(FAILED_LOADS) {
            viewModel.loadAndAwait(notAModel.path)
        }
        val heapKb = (Debug.getNativeHeapAllocatedSize() - heap) / 1024
        Log.i(TAG, "$FAILED_LOADS failed loads kept $heapKb KB of native heap")
        assertTrue("$FAILED_LOADS failed loads kept $heapKb KB", heapKb < HEAP_LIMIT_KB)
        viewModel.loadAndAwait(modelPath)
        assertTrue(viewModel.state.value.ready)
        assertNull(viewModel.state.value.error)
    }

    @Test
    fun loadFailsAfterAGoodLoad_thenOnCleared_closesEachOnce() {
        val viewModel = newViewModel()
        viewModel.loadAndAwait(modelPath)
        val firstEngine = checkNotNull(viewModel.engine())
        val firstConversation = checkNotNull(viewModel.conversation())
        viewModel.loadAndAwait(MISSING_MODEL)
        assertNotNull(viewModel.state.value.error)
        assertFalse(viewModel.state.value.ready)
        assertFalse("the conversation is still open", firstConversation.isAlive)
        assertFalse("the engine is still open", firstEngine.isInitialized())
        await(viewModel.launched { viewModel.send(SHORT_PROMPT) })
        assertEquals("", viewModel.state.value.reply)
        clearAndAwait(viewModel)
    }

    @Test
    fun onCleared_whileAReplyStreams_stopsIt_andClosesBoth() {
        val viewModel = newViewModel()
        viewModel.loadAndAwait(modelPath)
        val engine = checkNotNull(viewModel.engine())
        val conversation = checkNotNull(viewModel.conversation())
        val reply = viewModel.launched { viewModel.send(LONG_PROMPT) }
        viewModel.await("the reply to start") { it.reply.isNotEmpty() }
        val seconds = clearAndAwait(viewModel)
        Log.i(TAG, "onCleared while a reply streams: closed after $seconds s")
        assertStopped(reply, seconds)
        assertFalse("the conversation is still open", conversation.isAlive)
        assertFalse("the engine is still open", engine.isInitialized())
    }

    @Test
    fun onCleared_beforeTheFirstChunk_stopsTheReply_andClosesBoth() {
        val viewModel = newViewModel()
        viewModel.loadAndAwait(modelPath)
        val engine = checkNotNull(viewModel.engine())
        val conversation = checkNotNull(viewModel.conversation())
        val reply = viewModel.launched { viewModel.send(LONG_PROMPT) }
        viewModel.await("the reply to be sent") { it.busy }
        Thread.sleep(BEFORE_FIRST_CHUNK_MS)
        val stillEmpty = viewModel.state.value.reply.isEmpty()
        assumeTrue("the first chunk came within $BEFORE_FIRST_CHUNK_MS ms", stillEmpty)
        val seconds = clearAndAwait(viewModel)
        Log.i(TAG, "onCleared before the first chunk: closed after $seconds s")
        assertStopped(reply, seconds)
        assertFalse("the conversation is still open", conversation.isAlive)
        assertFalse("the engine is still open", engine.isInitialized())
    }

    @Test
    fun onCleared_rightAfterSend_stopsTheReply_andClosesBoth() {
        val viewModel = newViewModel()
        viewModel.loadAndAwait(modelPath)
        val engine = checkNotNull(viewModel.engine())
        val conversation = checkNotNull(viewModel.conversation())
        var reply = emptyList<Job>()
        val start = System.currentTimeMillis()
        viewModel.queued {
            reply = viewModel.launched { viewModel.send(LONG_PROMPT) }
            stores.forEach { it.clear() }
        }
        assertTrue(viewModel.executor().awaitTermination(TIMEOUT_SECONDS, TimeUnit.SECONDS))
        val seconds = (System.currentTimeMillis() - start) / 1000.0
        Log.i(TAG, "onCleared right after send: closed after $seconds s")
        assertStopped(reply, seconds)
        assertFalse("the conversation is still open", conversation.isAlive)
        assertFalse("the engine is still open", engine.isInitialized())
    }

    @Test
    fun loadAnotherModel_rightAfterSend_stopsTheReply_andClosesTheFirstEngine() {
        val viewModel = newViewModel()
        viewModel.loadAndAwait(modelPath)
        val firstEngine = checkNotNull(viewModel.engine())
        val firstConversation = checkNotNull(viewModel.conversation())
        val second = secondModelPath
        var reply = emptyList<Job>()
        var load = emptyList<Job>()
        val start = System.currentTimeMillis()
        viewModel.queued {
            reply = viewModel.launched { viewModel.send(LONG_PROMPT) }
            load = viewModel.launched { viewModel.load(second, backend()) }
        }
        await(reply)
        val seconds = (System.currentTimeMillis() - start) / 1000.0
        await(load)
        Log.i(TAG, "load right after send: reply stopped after $seconds s, ready after " +
            "${(System.currentTimeMillis() - start) / 1000.0} s")
        assertStopped(reply, seconds)
        assertTrue(viewModel.state.value.ready)
        assertEquals("the old reply reached the new chat", "", viewModel.state.value.reply)
        assertFalse("the first conversation is still open", firstConversation.isAlive)
        assertFalse("the first engine is still open", firstEngine.isInitialized())
    }

    @Test
    fun sendQueuedBehindALoad_goesToTheNewModel() {
        val viewModel = newViewModel()
        viewModel.loadAndAwait(modelPath)
        val second = secondModelPath
        var reply = emptyList<Job>()
        viewModel.queued {
            viewModel.launched { viewModel.load(second, backend()) }
            reply = viewModel.launched { viewModel.send(SHORT_PROMPT) }
        }
        await(reply)
        assertFalse("the reply was stopped", reply.single().isCancelled)
        assertEquals(second, viewModel.engine()?.engineConfig?.modelPath)
        assertTrue(viewModel.state.value.reply.isNotEmpty())
        assertNull(viewModel.state.value.error)
    }

    @Test
    fun loadAnotherModel_whileAReplyStreams_stopsIt_andClosesTheFirstEngine() {
        val viewModel = newViewModel()
        viewModel.loadAndAwait(modelPath)
        val firstEngine = checkNotNull(viewModel.engine())
        val firstConversation = checkNotNull(viewModel.conversation())
        val second = secondModelPath
        val reply = viewModel.launched { viewModel.send(LONG_PROMPT) }
        viewModel.await("the reply to start") { it.reply.isNotEmpty() }
        val start = System.currentTimeMillis()
        val load = viewModel.launched { viewModel.load(second, backend()) }
        await(reply)
        val seconds = (System.currentTimeMillis() - start) / 1000.0
        await(load)
        Log.i(TAG, "load while a reply streams: reply stopped after $seconds s, ready after " +
            "${(System.currentTimeMillis() - start) / 1000.0} s")
        assertStopped(reply, seconds)
        assertTrue(viewModel.state.value.ready)
        assertEquals("the old reply reached the new chat", "", viewModel.state.value.reply)
        assertFalse("the first conversation is still open", firstConversation.isAlive)
        assertFalse("the first engine is still open", firstEngine.isInitialized())
    }

    @Test
    fun onCleared_whileALoadWaitsForAReplyToStop_leavesNothingOpen() {
        val viewModel = newViewModel()
        viewModel.loadAndAwait(modelPath)
        val firstEngine = checkNotNull(viewModel.engine())
        viewModel.launched { viewModel.send(LONG_PROMPT) }
        viewModel.await("the reply to start") { it.reply.isNotEmpty() }
        viewModel.load(secondModelPath, backend())
        clearAndAwait(viewModel)
        val lastEngine = viewModel.engine()
        assertFalse("the first engine is still open", firstEngine.isInitialized())
        assertTrue("an engine is still open", lastEngine == null || !lastEngine.isInitialized())
    }

    @Test
    fun onCleared_rightAfterLoad_closesBoth() {
        val viewModel = newViewModel()
        viewModel.load(modelPath, backend())
        clearAndAwait(viewModel)
        val engine = viewModel.engine()
        assertTrue("the engine is still open", engine == null || !engine.isInitialized())
    }

    @Test
    fun onCleared_withoutLoad_endsTheThread() {
        clearAndAwait(newViewModel())
    }

    @Test
    fun loadAfterOnCleared_createsNothing() {
        val viewModel = newViewModel()
        clearAndAwait(viewModel)
        await(viewModel.launched { viewModel.load(modelPath, backend()) })
        assertNull(viewModel.engine())
    }

    /** A stopped reply ends its coroutine as cancelled. One that ran to its end does not. */
    private fun assertStopped(reply: List<Job>, seconds: Double) {
        assertEquals("send() launched", 1, reply.size)
        await(reply)
        assertTrue("the reply ran to its end ($seconds s)", reply.single().isCancelled)
        assertTrue("stopping the reply took $seconds s", seconds < STOP_LIMIT_SECONDS)
    }

    private fun newViewModel(): ChatViewModel {
        val store = ViewModelStore()
        stores += store
        val application = context.applicationContext as Application
        val factory = ViewModelProvider.AndroidViewModelFactory(application)
        val viewModel = ViewModelProvider(store, factory)[ChatViewModel::class.java]
        executors += viewModel.executor()
        return viewModel
    }

    private fun backend(): Backend =
        if (arguments.getString("backend") == "GPU") {
            Backend.GPU()
        } else {
            Backend.CPU()
        }

    private fun otherBackend(): Backend =
        if (arguments.getString("backend") == "GPU") {
            Backend.CPU()
        } else {
            Backend.GPU()
        }

    /** Starts a load and blocks until the coroutine it launched has ended. */
    private fun ChatViewModel.loadAndAwait(path: String, backend: Backend = backend()) {
        await(launched { load(path, backend) })
    }

    /** Runs [call] and returns the coroutines it launched on the ViewModel's scope. */
    private fun ChatViewModel.launched(call: () -> Unit): List<Job> {
        val jobs = checkNotNull(scope().coroutineContext[Job])
        val before = jobs.children.toSet()
        call()
        return jobs.children.filter { it !in before }.toList()
    }

    /**
     * Runs [calls] while the ViewModel's thread is held, so that what they launch is queued in
     * that order before any of it starts.
     */
    private fun ChatViewModel.queued(calls: () -> Unit) {
        val gate = CountDownLatch(1)
        executor().submit { gate.await() }
        try {
            calls()
        } finally {
            gate.countDown()
        }
    }

    /** Calls `onCleared()` the way the framework does. Returns the seconds the close took. */
    private fun clearAndAwait(viewModel: ChatViewModel): Double {
        val start = System.currentTimeMillis()
        stores.forEach { it.clear() }
        val ended = viewModel.executor().awaitTermination(TIMEOUT_SECONDS, TimeUnit.SECONDS)
        assertTrue("the ViewModel's thread did not end", ended)
        return (System.currentTimeMillis() - start) / 1000.0
    }

    private fun await(jobs: List<Job>) {
        runBlocking {
            withTimeout(TIMEOUT_SECONDS * 1000) {
                jobs.forEach { it.join() }
            }
        }
    }

    private fun ChatViewModel.await(what: String, condition: (ChatState) -> Boolean) {
        try {
            runBlocking {
                withTimeout(TIMEOUT_SECONDS * 1000) {
                    state.first { condition(it) }
                }
            }
        } catch (e: Exception) {
            throw AssertionError("timed out waiting for $what, state = ${state.value}", e)
        }
    }

    private fun copyOfModel(): File {
        val source = File(modelPath)
        val copy = File(context.cacheDir, "second-" + source.name)
        if (copy.length() != source.length()) {
            source.inputStream().use { input ->
                copy.outputStream().use { output ->
                    input.copyTo(output, 1 shl 20)
                }
            }
        }
        return copy
    }

    private fun ChatViewModel.executor() = checkNotNull(privateField<ExecutorService>("executor"))

    private fun ChatViewModel.scope() = checkNotNull(privateField<CoroutineScope>("scope"))

    private fun ChatViewModel.engine() = privateField<Engine>("engine")

    private fun ChatViewModel.conversation() = privateField<Conversation>("conversation")

    @Suppress("UNCHECKED_CAST")
    private fun <T> Any.privateField(name: String): T? {
        val field = javaClass.getDeclaredField(name)
        field.isAccessible = true
        return field.get(this) as T?
    }

    private fun pssMb() = Debug.getPss() / 1024

    private companion object {
        const val TAG = "ChatViewModelTest"
        const val DEFAULT_MODEL = "/data/local/tmp/Qwen3-0.6B.litertlm"
        const val MISSING_MODEL = "/data/local/tmp/no-such-model.litertlm"
        const val SHORT_PROMPT = "Reply with the single word: ready. /no_think"
        const val LONG_PROMPT = "Write a story of at least 500 words about a lighthouse keeper."
        const val TIMEOUT_SECONDS = 900L
        const val STOP_LIMIT_SECONDS = 20.0
        const val BEFORE_FIRST_CHUNK_MS = 300L
        const val ALTERNATIONS = 4
        const val FAILED_LOADS = 20
        const val HEAP_LIMIT_KB = 1024L

        /** One more engine of the smallest model here is about 900 MB. */
        const val PSS_LIMIT_MB = 400L
    }
}
