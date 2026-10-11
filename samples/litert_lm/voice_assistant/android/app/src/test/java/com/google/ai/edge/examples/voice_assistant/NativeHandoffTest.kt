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

package com.google.ai.edge.examples.voice_assistant

import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExecutorCoroutineDispatcher
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * A native object made on another thread while its caller is cancelled: the native call (here a
 * latch the test opens after the cancel) finishes anyway, and the object must either reach the
 * caller or be closed.
 */
class NativeHandoffTest {
  private val native: ExecutorCoroutineDispatcher =
    Executors.newSingleThreadExecutor().asCoroutineDispatcher()

  @After
  fun stop() {
    native.close()
  }

  /** The form handOver avoids: the object is made, and the cancelled caller never sees it. */
  @Test
  fun aResultCarriedBackAcrossADispatcherChangeIsLostWhenTheCallerIsCancelled() = runBlocking {
    val made = CompletableDeferred<Any>()
    val inside = CountDownLatch(1)
    val release = CountDownLatch(1)
    var returned: Any? = null
    val caller =
      launch(Dispatchers.Default) {
        returned =
          withContext(native + NonCancellable) {
            inside.countDown()
            release.await()
            Any().also { made.complete(it) }
          }
      }
    inside.await()
    caller.cancel()
    release.countDown()
    caller.join()
    assertTrue(made.isCompleted)
    assertNull(returned)
  }

  @Test
  fun handOverClosesWhatACancelledCallerCannotTake() = runBlocking {
    val made = CompletableDeferred<Any>()
    val closed = CompletableDeferred<Any>()
    val inside = CountDownLatch(1)
    val release = CountDownLatch(1)
    var returned: Any? = null
    val caller =
      launch(Dispatchers.Default) {
        returned =
          handOver(
            native,
            {
              inside.countDown()
              release.await()
              Any().also { made.complete(it) }
            },
          ) {
            closed.complete(it)
          }
      }
    inside.await()
    caller.cancel()
    release.countDown()
    caller.join()
    // handOver closes before it rethrows, so the object is closed by the time the caller is done.
    assertTrue("made and never closed", made.isCompleted && closed.isCompleted)
    assertSame(made.await(), closed.await())
    assertNull(returned)
  }

  @Test
  fun handOverGivesTheObjectToACallerStillActiveAndClosesNothing() = runBlocking {
    val closes = AtomicInteger()
    val thread = CompletableDeferred<String>()
    val got =
      handOver(native, { Thread.currentThread().name.also { thread.complete(it) } }) {
        closes.incrementAndGet()
      }
    assertEquals(thread.await(), got)
    assertTrue(got != Thread.currentThread().name)
    assertEquals(0, closes.get())
  }

  @Test
  fun aCallerAlreadyCancelledStartsNoNativeCall() = runBlocking {
    val creates = AtomicInteger()
    val caller =
      launch {
        coroutineContext[Job]!!.cancel()
        handOver(native, { creates.incrementAndGet() }) {}
      }
    caller.join()
    assertEquals(0, creates.get())
  }
}
