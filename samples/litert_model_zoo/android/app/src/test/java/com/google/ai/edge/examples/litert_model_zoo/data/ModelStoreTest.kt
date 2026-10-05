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

package com.google.ai.edge.examples.litert_model_zoo.data

import java.io.File
import java.io.IOException
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.InetAddress
import java.net.ServerSocket
import java.net.SocketException
import java.net.SocketTimeoutException
import java.net.URL
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.coroutines.CoroutineContext
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class ModelStoreTest {
  @get:Rule val temporary = TemporaryFolder()

  @Test
  fun validatesByteCountAndKnownSha256Together() {
    val file = temporary.newFile("model.tflite").apply { writeText("abc") }
    val expected = CatalogFixture.entry().files.single()
    assertTrue(ModelStore.verify(file, expected))
    file.writeText("abd")
    assertFalse("Same length cannot bypass content integrity", ModelStore.verify(file, expected))
    file.writeText("ab")
    assertFalse("A short file cannot be ready", ModelStore.verify(file, expected))
  }

  @Test
  fun inspectTrustsACommittedFileBySizeWithoutRehashingIt() = runBlocking {
    val entry = CatalogFixture.entry()
    val store = ModelStore(temporary.newFolder(), eventLogger = {})
    // Same byte count as the catalog row but different bytes: download() hashed the real file
    // before committing it, so a cold start only checks that the file is there at its size.
    val file = modelFile(store, entry).apply { writeText("abd") }
    assertTrue(ModelStore.isCommitted(file, entry.files.single()))
    assertFalse(ModelStore.verify(file, entry.files.single()))
    assertEquals(DownloadStatus.READY, store.inspect(entry).status)
    file.writeText("ab")
    assertFalse(
      "A short file is a paused download",
      ModelStore.isCommitted(file, entry.files.single()),
    )
    assertEquals(DownloadStatus.PAUSED, store.inspect(entry).status)
  }

  @Test
  fun cancellingADownloadAbortsABlockedSocketCall() = runBlocking {
    // A server that reads the request and never answers: the client blocks in responseCode. The
    // latch waits for the request, not the accept, so the cancel comes while the response is
    // awaited. On the JDK's HttpURLConnection, one just before the request goes out can wait for
    // the read timeout (see download()).
    val server = ServerSocket(0, 1, InetAddress.getLoopbackAddress())
    val requestRead = CountDownLatch(1)
    val hold =
      Thread {
          runCatching {
            server.accept().use { socket ->
              val request = socket.getInputStream().bufferedReader()
              var line = request.readLine()
              while (!line.isNullOrEmpty()) {
                line = request.readLine()
              }
              if (line != null) {
                requestRead.countDown()
              }
              Thread.sleep(60_000)
            }
          }
        }
        .apply {
          isDaemon = true
          start()
        }
    try {
      val original = CatalogFixture.entry()
      val entry =
        original.copy(
          files =
            original.files.map {
              it.copy(url = "http://127.0.0.1:${server.localPort}/${it.name}")
            }
        )
      val store = ModelStore(temporary.newFolder(), eventLogger = {})
      val job = launch(Dispatchers.IO) { store.download(entry) {} }
      assertTrue("the request reached the server", requestRead.await(10, TimeUnit.SECONDS))
      val started = System.nanoTime()
      job.cancelAndJoin()
      val waitedMs = (System.nanoTime() - started) / 1_000_000
      assertTrue(
        "cancel took $waitedMs ms; the 30 s socket timeout must not apply",
        waitedMs < 5_000,
      )
      assertTrue(job.isCancelled)
      assertEquals(DownloadStatus.MISSING, store.inspect(entry).status)
    } finally {
      hold.interrupt()
      server.close()
    }
  }

  @Test(timeout = 10_000)
  fun aCancelRightAfterTheCallBlocksStillTearsTheConnectionDown() = runBlocking {
    // Pause comes as soon as the request waits for the response, before the scheduler has run
    // anything queued since the connection was opened. The store's coroutine that disconnects on a
    // cancel must be waiting by then: one that had not started yet would skip its finally, and the
    // call would wait for its read timeout.
    lateinit var job: Job
    val scheduler = HeldDispatcher()
    val connections = ArrayList<NoAnswer>()
    val store =
      ModelStore(temporary.newFolder(), eventLogger = {}, ioDispatcher = scheduler) { url ->
        scheduler.hold()
        NoAnswer(url) { connection ->
            job.cancel()
            scheduler.runHeld()
            // A socket read fails once disconnect() closes the socket, and otherwise only at its
            // read timeout.
            if (connection.closed) {
              throw SocketException("Socket closed")
            }
            connection.timedOut = true
            throw SocketTimeoutException("Read timed out")
          }
          .also { connections += it }
      }
    val entry = CatalogFixture.entry()
    job = launch(Dispatchers.IO, start = CoroutineStart.LAZY) { store.download(entry) {} }
    job.start()
    job.join()
    assertTrue(job.isCancelled)
    assertFalse("the blocked call waited for its read timeout", connections.single().timedOut)
    assertEquals(DownloadStatus.MISSING, store.inspect(entry).status)
  }

  @Test
  fun aCancelReportsAsCancelledWhateverTheTornDownCallThrows() = runBlocking {
    // When disconnect() races the blocked call, the JDK's HttpURLConnection can throw a
    // RuntimeException around a NullPointerException instead of an IOException. A Pause must still
    // end the download with a CancellationException (MainViewModel shows it as paused), not as a
    // download error.
    lateinit var job: Job
    var thrown: Throwable? = null
    val store =
      ModelStore(temporary.newFolder(), eventLogger = {}) { url ->
        NoAnswer(url) {
          job.cancel()
          throw RuntimeException(NullPointerException("the connection was torn down"))
        }
      }
    val entry = CatalogFixture.entry()
    job =
      launch(Dispatchers.IO, start = CoroutineStart.LAZY) {
        try {
          store.download(entry) {}
        } catch (failure: Throwable) {
          thrown = failure
        }
      }
    job.start()
    job.join()
    assertTrue("the download ended with $thrown", thrown is CancellationException)
    assertEquals(DownloadStatus.MISSING, store.inspect(entry).status)
  }

  @Test
  fun aFailureWithoutACancelEndsTheDownloadAsItCame() = runBlocking {
    // Only a cancel turns a failed call into the cancellation; any other failure is the error.
    val failure = RuntimeException(NullPointerException("not a cancel"))
    val store =
      ModelStore(temporary.newFolder(), eventLogger = {}) { url -> NoAnswer(url) { throw failure } }
    val entry = CatalogFixture.entry()
    val thrown = runCatching { store.download(entry) {} }.exceptionOrNull()
    // withContext can hand back a copy of the exception, with the same type and message.
    assertEquals(failure.javaClass, thrown?.javaClass)
    assertEquals(failure.message, thrown?.message)
    assertEquals(DownloadStatus.MISSING, store.inspect(entry).status)
  }

  @Test
  fun aDisconnectThatThrowsDoesNotReplaceTheCancellation() = runBlocking {
    // The store disconnects from two coroutines, and two disconnect() calls at once can throw a
    // NullPointerException on the JDK's HttpURLConnection. A Pause must still end the download
    // with a CancellationException.
    lateinit var job: Job
    var thrown: Throwable? = null
    val store =
      ModelStore(temporary.newFolder(), eventLogger = {}) { url ->
        NoAnswer(url, onDisconnect = { throw NullPointerException("another disconnect()") }) {
          job.cancel()
          throw SocketException("Socket closed")
        }
      }
    val entry = CatalogFixture.entry()
    job =
      launch(Dispatchers.IO, start = CoroutineStart.LAZY) {
        try {
          store.download(entry) {}
        } catch (failure: Throwable) {
          thrown = failure
        }
      }
    job.start()
    job.join()
    assertTrue("the download ended with $thrown", thrown is CancellationException)
    assertEquals(DownloadStatus.MISSING, store.inspect(entry).status)
  }

  @Test
  fun aCancelBeforeTheConnectionIsMadeStopsWithoutConnecting() = runBlocking {
    // The cancel comes after the store has opened the connection but before it connects: the
    // download stops before connect(), which a cancel does not end on the JDK's HttpURLConnection
    // (see download()).
    lateinit var job: Job
    val connections = ArrayList<NoAnswer>()
    val store =
      ModelStore(temporary.newFolder(), eventLogger = {}) { url ->
        job.cancel()
        NoAnswer(url) { throw IOException("no answer") }.also { connections += it }
      }
    val entry = CatalogFixture.entry()
    job = launch(Dispatchers.IO, start = CoroutineStart.LAZY) { store.download(entry) {} }
    job.start()
    job.join()
    assertTrue(job.isCancelled)
    assertEquals("connect() calls", 0, connections.single().connectCalls)
    assertEquals(DownloadStatus.MISSING, store.inspect(entry).status)
  }

  @Test(timeout = 10_000)
  fun aCancelWhileConnectingStopsBeforeTheRequestGoesOut() = runBlocking {
    // The cancel comes while the connection is being made (Pause during the connect), when the
    // JDK's HttpURLConnection has nothing for disconnect() to close yet: the download stops once
    // connected, before it asks for the response, and closes the connection it made, instead of
    // sending the request and waiting for the read timeout.
    lateinit var job: Job
    val scheduler = HeldDispatcher()
    val connections = ArrayList<NoAnswer>()
    val store =
      ModelStore(temporary.newFolder(), eventLogger = {}, ioDispatcher = scheduler) { url ->
        scheduler.hold()
        val cancelWhileConnecting = {
          job.cancel()
          scheduler.runHeld()
        }
        NoAnswer(url, onConnect = cancelWhileConnecting) { throw IOException("no answer") }
          .also { connections += it }
      }
    val entry = CatalogFixture.entry()
    job = launch(Dispatchers.IO, start = CoroutineStart.LAZY) { store.download(entry) {} }
    job.start()
    job.join()
    assertTrue(job.isCancelled)
    val connection = connections.single()
    assertEquals(1, connection.connectCalls)
    assertFalse("the request went out", connection.requestSent)
    assertTrue("the connection was closed", connection.closed)
    assertEquals(DownloadStatus.MISSING, store.inspect(entry).status)
  }

  @Test
  fun aCancelWhileTheBodyIsReadKeepsThePartialFileForResume() = runBlocking {
    // Pause while the body is read: the download ends with a CancellationException (MainViewModel
    // shows it as paused), and the bytes read so far stay in the partial file for the resume.
    lateinit var job: Job
    var thrown: Throwable? = null
    val store =
      ModelStore(temporary.newFolder(), eventLogger = {}) { url ->
        StopsInTheBody(url, "ab".toByteArray()) {
          job.cancel()
          throw SocketException("Socket closed")
        }
      }
    val entry = CatalogFixture.entry()
    job =
      launch(Dispatchers.IO, start = CoroutineStart.LAZY) {
        try {
          store.download(entry) {}
        } catch (failure: Throwable) {
          thrown = failure
        }
      }
    job.start()
    job.join()
    assertTrue("the download ended with $thrown", thrown is CancellationException)
    val part = File(store.directory(entry), entry.files.single().name + ".part")
    assertArrayEquals("ab".toByteArray(), part.readBytes())
    val state = store.inspect(entry)
    assertEquals(DownloadStatus.PAUSED, state.status)
    assertEquals(2L, state.receivedBytes)
  }

  @Test(timeout = 10_000)
  fun aDownloadThatGetsTheWholeFileCommitsItWhenDisconnectThrows() = runBlocking {
    // A file that arrives whole ends with two disconnect() calls, the finally block's and the
    // child's, and two at once can throw a NullPointerException on the JDK's HttpURLConnection.
    // The download must still return with the file committed. It also waits for the child, which
    // ends only when the store cancels it; if the store did not, the timeout fails this case.
    val store =
      ModelStore(temporary.newFolder(), eventLogger = {}) { url ->
        AnswersInFull(url, "abc".toByteArray()) {
          throw NullPointerException("another disconnect()")
        }
      }
    val entry = CatalogFixture.entry()
    store.download(entry) {}
    val target = File(store.directory(entry), entry.files.single().name)
    assertArrayEquals("abc".toByteArray(), target.readBytes())
    assertEquals(DownloadStatus.READY, store.inspect(entry).status)
  }

  @Test
  fun missingDigestNeverMarksPresentFileReady() = runBlocking {
    val entry =
      CatalogFixture.entry().let { original ->
        original.copy(files = original.files.map { it.copy(sha256 = null) })
      }
    val store = ModelStore(temporary.newFolder(), eventLogger = {})
    val file = modelFile(store, entry).apply { writeText("abc") }
    assertFalse(ModelStore.verify(file, entry.files.single()))
    assertEquals(DownloadStatus.PAUSED, store.inspect(entry).status)
  }

  @Test
  fun inspectReportsPartialProgressWithoutDiscardingResumableBytes() = runBlocking {
    val entry = CatalogFixture.entry()
    val store = ModelStore(temporary.newFolder(), eventLogger = {})
    val part = File(modelFile(store, entry).path + ".part").apply { writeText("ab") }
    val state = store.inspect(entry)
    assertEquals(DownloadStatus.PAUSED, state.status)
    assertEquals(2L, state.receivedBytes)
    assertEquals(3L, state.totalBytes)
    assertArrayEquals("ab".toByteArray(), part.readBytes())
  }

  @Test
  fun promotesFullyDownloadedPartialFileAfterHashVerificationWithoutNetwork() = runBlocking {
    val entry = CatalogFixture.entry()
    val events = mutableListOf<String>()
    val store = ModelStore(temporary.newFolder(), eventLogger = { events.add(it) })
    val target = modelFile(store, entry)
    val part = File(target.path + ".part").apply { writeText("abc") }
    val states = mutableListOf<DownloadState>()
    store.download(entry) { states.add(it) }
    assertFalse(part.exists())
    assertArrayEquals("abc".toByteArray(), target.readBytes())
    assertEquals(listOf(DownloadStatus.VERIFYING, DownloadStatus.READY), states.map { it.status })
    assertEquals(DownloadStatus.READY, store.inspect(entry).status)
    val verified = org.json.JSONObject(events.last())
    assertEquals("sha_ok", verified.getString("event"))
    assertEquals(3L, verified.getLong("initialPartBytes"))
    assertEquals(entry.files.single().sha256, verified.getString("sha256"))
  }

  @Test
  fun corruptCompletePartialFileCannotBeCommitted() = runBlocking {
    val entry = CatalogFixture.entry()
    val store = ModelStore(temporary.newFolder(), eventLogger = {})
    val target = modelFile(store, entry)
    File(target.path + ".part").writeText("abd")
    val failed = runCatching { store.download(entry) {} }
    assertTrue(failed.exceptionOrNull() is IllegalStateException)
    assertFalse(target.exists())
    assertFalse(store.inspect(entry).status == DownloadStatus.READY)
  }

  @Test
  fun removingTaskLeavesOtherTaskFilesAndUpdatesStorageUsage() = runBlocking {
    val entry = CatalogFixture.entry()
    val other = entry.copy(taskId = "another-task")
    val store = ModelStore(temporary.newFolder(), eventLogger = {})
    modelFile(store, entry).writeText("abc")
    val retained = modelFile(store, other).apply { writeText("abc") }
    assertEquals(6L, store.storageBytes())
    store.delete(entry)
    assertFalse(store.directory(entry).exists())
    assertTrue(retained.exists())
    assertEquals(3L, store.storageBytes())
  }

  private fun modelFile(store: ModelStore, entry: ModelEntry): File {
    store.directory(entry).mkdirs()
    return File(store.directory(entry), entry.files.single().name)
  }

  /**
   * A connection to a server that never answers. Asking for the response connects first, as
   * HttpURLConnection does, then runs [awaitResponse], which throws what the waiting call would.
   * [closed] counts only a disconnect() after the connection exists, the kind that closes a socket
   * on the JDK's HttpURLConnection.
   */
  private class NoAnswer(
    url: URL,
    private val onConnect: () -> Unit = {},
    private val onDisconnect: () -> Unit = {},
    private val awaitResponse: (NoAnswer) -> Nothing,
  ) : HttpURLConnection(url) {
    var connectCalls = 0
    var requestSent = false
    @Volatile var closed = false
    @Volatile var timedOut = false

    override fun connect() {
      if (!connected) {
        connectCalls++
        onConnect()
        connected = true
      }
    }

    override fun getResponseCode(): Int {
      connect()
      requestSent = true
      awaitResponse(this)
    }

    override fun disconnect() {
      if (connected) {
        closed = true
      }
      onDisconnect()
    }

    override fun usingProxy() = false
  }

  /**
   * A connection whose server answers 200, sends [head] of the body and then nothing more: the next
   * read runs [awaitMore], which throws what the waiting read would.
   */
  private class StopsInTheBody(
    url: URL,
    private val head: ByteArray,
    private val awaitMore: () -> Nothing,
  ) : HttpURLConnection(url) {
    override fun connect() {
      connected = true
    }

    override fun getResponseCode(): Int {
      connect()
      return HTTP_OK
    }

    override fun getInputStream(): InputStream =
      object : InputStream() {
        private var sent = false

        override fun read(): Int = error("the store reads in blocks")

        override fun read(b: ByteArray, off: Int, len: Int): Int {
          if (sent) {
            awaitMore()
          }
          sent = true
          head.copyInto(b, off)
          return head.size
        }
      }

    override fun disconnect() {}

    override fun usingProxy() = false
  }

  /**
   * A connection whose server answers 200 with the whole [body]. Its disconnect() runs
   * [onDisconnect].
   */
  private class AnswersInFull(
    url: URL,
    private val body: ByteArray,
    private val onDisconnect: () -> Unit,
  ) : HttpURLConnection(url) {
    override fun connect() {
      connected = true
    }

    override fun getResponseCode(): Int {
      connect()
      return HTTP_OK
    }

    override fun getInputStream(): InputStream = body.inputStream()

    override fun disconnect() {
      onDisconnect()
    }

    override fun usingProxy() = false
  }

  /**
   * Runs dispatched work on Dispatchers.IO, except after [hold]: then the work waits for [runHeld],
   * as work the scheduler has not got to yet would. Only the test's connection calls [runHeld]: a
   * store that suspended between opening the connection and its blocking call would hang, so the
   * cases that hold work have a timeout.
   */
  private class HeldDispatcher : CoroutineDispatcher() {
    private val held = ArrayList<Runnable>()
    private var holding = false

    fun hold() {
      synchronized(held) { holding = true }
    }

    /** Stops holding and runs the held work on the calling thread, in the order it came. */
    fun runHeld() {
      val work =
        synchronized(held) {
          holding = false
          held.toList().also { held.clear() }
        }
      work.forEach { it.run() }
    }

    override fun dispatch(context: CoroutineContext, block: Runnable) {
      synchronized(held) {
        if (holding) {
          held += block
          return
        }
      }
      Dispatchers.IO.dispatch(context, block)
    }
  }
}
