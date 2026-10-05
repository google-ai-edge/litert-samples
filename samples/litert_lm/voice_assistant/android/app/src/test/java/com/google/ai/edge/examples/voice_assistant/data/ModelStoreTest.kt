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

// Adapted from this repository's model zoo app:
// samples/litert_model_zoo/android/app/src/test/java/.../data/ModelStoreTest.kt
// The sideLoad() cases are this sample's.

package com.google.ai.edge.examples.voice_assistant.data

import java.io.File
import java.io.IOException
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.InetAddress
import java.net.ServerSocket
import java.net.URL
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
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
  fun validatesByteCountAndSha256Together() {
    val file = temporary.newFile("model.tflite").apply { writeText("abc") }
    val expected = entry().files.single()
    assertTrue(ModelStore.verify(file, expected))
    file.writeText("abd")
    assertFalse("Same length cannot bypass content integrity", ModelStore.verify(file, expected))
    file.writeText("ab")
    assertFalse("A short file cannot be ready", ModelStore.verify(file, expected))
  }

  @Test
  fun inspectTrustsACommittedFileBySizeWithoutRehashingIt() = runBlocking {
    val entry = entry()
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
    // latch waits for the request, not the accept: until HttpURLConnection has its connection,
    // disconnect() has nothing to close.
    val server = ServerSocket(0, 1, InetAddress.getLoopbackAddress())
    val accepted = CountDownLatch(1)
    val hold =
      Thread {
          runCatching {
            server.accept().use { socket ->
              val request = socket.getInputStream().bufferedReader()
              var line = request.readLine()
              while (!line.isNullOrEmpty()) {
                line = request.readLine()
              }
              accepted.countDown()
              Thread.sleep(60_000)
            }
          }
        }
        .apply {
          isDaemon = true
          start()
        }
    try {
      val original = entry()
      val local =
        original.files.map { it.copy(url = "http://127.0.0.1:${server.localPort}/${it.name}") }
      val entry = original.copy(files = local)
      val store = ModelStore(temporary.newFolder(), eventLogger = {})
      val job = launch(Dispatchers.IO) { store.download(entry) {} }
      assertTrue("the download reached the server", accepted.await(10, TimeUnit.SECONDS))
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

  @Test
  fun inspectReportsPartialProgressWithoutDiscardingResumableBytes() = runBlocking {
    val entry = entry()
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
    val entry = entry()
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
    val entry = entry()
    val store = ModelStore(temporary.newFolder(), eventLogger = {})
    val target = modelFile(store, entry)
    File(target.path + ".part").writeText("abd")
    val failed = runCatching { store.download(entry) {} }
    assertTrue(failed.exceptionOrNull() is IllegalStateException)
    assertFalse(target.exists())
    assertFalse(store.inspect(entry).status == DownloadStatus.READY)
  }

  @Test
  fun sideLoadSkipsACopyOfTheRightSizeWithTheWrongHash() = runBlocking {
    val entry = entry()
    val logs = mutableListOf<String>()
    val store = ModelStore(temporary.newFolder(), eventLogger = { logs.add(it) })
    val source = temporary.newFolder()
    File(source, "model.tflite").writeText("abd")
    val result = store.sideLoad(entry, source)
    assertEquals(emptyList<String>(), result.imported)
    assertEquals(listOf("model.tflite: SHA-256 differs from the catalog"), result.skipped)
    assertFalse(File(store.directory(entry), "model.tflite").exists())
    assertEquals(DownloadStatus.MISSING, store.inspect(entry).status)
    assertTrue(logs.single(), logs.single().contains("SHA-256 differs"))
  }

  @Test
  fun sideLoadCommitsAVerifiedCopyAndLeavesTheSource() = runBlocking {
    val entry = entry()
    val store = ModelStore(temporary.newFolder(), eventLogger = {})
    val source = temporary.newFolder()
    val copy = File(source, "model.tflite").apply { writeText("abc") }
    assertEquals(listOf("model.tflite"), store.sideLoad(entry, source).imported)
    assertArrayEquals("abc".toByteArray(), File(store.directory(entry), "model.tflite").readBytes())
    assertFalse(File(store.directory(entry), "model.tflite.part").exists())
    assertTrue(copy.exists())
    assertEquals(DownloadStatus.READY, store.inspect(entry).status)
    // A committed file is not imported again.
    assertEquals(SideLoad(emptyList(), emptyList()), store.sideLoad(entry, source))
  }

  @Test
  fun aCancelWhileConnectingStopsBeforeTheRequestGoesOut() = runBlocking {
    // The cancel comes while the connection is being made (the screen closed during the connect),
    // when disconnect() has nothing to close yet: the download stops once connected, before it
    // asks for the response, instead of sending the request and waiting for the read timeout.
    val connections = ArrayList<CancelledWhileConnecting>()
    lateinit var job: Job
    val store =
      ModelStore(temporary.newFolder(), eventLogger = {}) { url ->
        CancelledWhileConnecting(url) { job.cancel() }.also { connections += it }
      }
    val entry = entry()
    job = launch(Dispatchers.IO, start = CoroutineStart.LAZY) { store.download(entry) {} }
    job.start()
    job.join()
    assertTrue(job.isCancelled)
    val connection = connections.single()
    assertEquals(1, connection.connectCalls)
    assertFalse("the request went out", connection.requestSent)
    assertTrue("disconnected", connection.disconnected)
    assertEquals(DownloadStatus.MISSING, store.inspect(entry).status)
  }

  /**
   * A connection that the download's cancel reaches while it connects, and that records whether
   * the request went out. As HttpURLConnection does, asking for the response connects first.
   */
  private class CancelledWhileConnecting(url: URL, private val cancel: () -> Unit) :
    HttpURLConnection(url) {
    var connectCalls = 0
    var requestSent = false
    var disconnected = false

    override fun connect() {
      if (!connected) {
        connectCalls++
        cancel()
        connected = true
      }
    }

    override fun getResponseCode(): Int {
      connect()
      requestSent = true
      throw IOException("the test server does not answer")
    }

    override fun getInputStream(): InputStream {
      connect()
      requestSent = true
      throw IOException("the test server does not answer")
    }

    override fun disconnect() {
      disconnected = true
    }

    override fun usingProxy() = false
  }

  private fun entry(): ModelEntry =
    ModelEntry(
      id = "example",
      role = "Example",
      model = "Example model",
      runtime = "litert",
      backend = "cpu",
      files =
        listOf(
          ModelFile(
            "model.tflite",
            "https://huggingface.co/owner/model/resolve/main/model.tflite",
            3,
            ABC_SHA256,
          )
        ),
      license = ModelLicense("MIT", "https://example.com/LICENSE"),
      modelCard = "https://huggingface.co/owner/model",
    )

  private fun modelFile(store: ModelStore, entry: ModelEntry): File {
    store.directory(entry).mkdirs()
    return File(store.directory(entry), entry.files.single().name)
  }

  private companion object {
    const val ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
  }
}
