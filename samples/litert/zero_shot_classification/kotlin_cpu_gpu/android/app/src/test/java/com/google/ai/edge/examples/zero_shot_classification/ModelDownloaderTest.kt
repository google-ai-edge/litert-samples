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

package com.google.ai.edge.examples.zero_shot_classification

import java.io.BufferedInputStream
import java.io.Closeable
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.net.SocketTimeoutException
import java.nio.file.Files
import java.security.MessageDigest
import java.util.Collections
import java.util.concurrent.CancellationException
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/** Drives [ModelDownloader] against a local HTTP server, one test per failure path. */
class ModelDownloaderTest {
  private lateinit var server: TestHttpServer
  private lateinit var directory: File
  private val requests
    get() = server.requests

  private val release = CountDownLatch(1)

  @Before
  fun setUp() {
    directory = Files.createTempDirectory("model-downloader-test").toFile()
    server = TestHttpServer()
  }

  @After
  fun tearDown() {
    release.countDown()
    server.close()
    directory.deleteRecursively()
  }

  @Test
  fun freshDownloadVerifiesFilesAndWritesRevisionMarker() {
    val large = bytes(1_000, seed = 1)
    val small = bytes(300, seed = 2)
    serve(resolvePath("a.bin"), large)
    serve(resolvePath("dir/b.json"), small)
    val loader = downloader(manifest(entry("a.bin", large), entry("dir/b.json", small)))
    val progress = mutableListOf<Pair<Long, Long>>()
    assertFalse(loader.isComplete())

    loader.download { done, total -> progress += done to total }

    assertEquals(sha256(large), sha256(File(directory, "a.bin").readBytes()))
    assertEquals(sha256(small), sha256(File(directory, "dir/b.json").readBytes()))
    assertFalse(File(directory, "a.bin.part").exists())
    assertFalse(File(directory, "dir/b.json.part").exists())
    assertEquals(REVISION, File(directory, ModelDownloader.REVISION_MARKER).readText())
    assertTrue(loader.isComplete())
    assertEquals(1_300L, loader.bytesDownloaded())
    assertEquals(1_300L to 1_300L, progress.last())
    assertTrue(requests.all { it.range == null })
  }

  @Test
  fun resumeSendsRangeFromPartLength() {
    val body = bytes(1_000)
    serve(resolvePath("a.bin"), body)
    writeRevisionMarker()
    File(directory, "a.bin.part").writeBytes(body.copyOf(400))
    val loader = downloader(manifest(entry("a.bin", body)))
    assertEquals(400L, loader.bytesDownloaded())

    loader.download()

    assertEquals(listOf("bytes=400-"), requests.map { it.range })
    assertEquals(sha256(body), sha256(File(directory, "a.bin").readBytes()))
  }

  @Test
  fun serverIgnoringRangeRestartsFromZero() {
    val body = bytes(1_000)
    serve(resolvePath("a.bin"), body, honorRange = false)
    writeRevisionMarker()
    // A wrong prefix shows that the .part was rewritten from byte 0, not appended to.
    File(directory, "a.bin.part").writeBytes(ByteArray(400) { 7 })

    downloader(manifest(entry("a.bin", body))).download()

    assertEquals(listOf("bytes=400-"), requests.map { it.range })
    assertEquals(sha256(body), sha256(File(directory, "a.bin").readBytes()))
  }

  @Test
  fun shaMismatchDeletesPartAndThrows() {
    val body = bytes(1_000)
    serve(resolvePath("a.bin"), body)
    val wrongSha = sha256(bytes(1_000, seed = 9))
    val loader = downloader(manifest(Triple("a.bin", body.size, wrongSha)))

    val failure = assertThrows(IOException::class.java) { loader.download() }

    assertTrue(failure.message!!.contains("a.bin"))
    assertFalse(File(directory, "a.bin.part").exists())
    assertFalse(File(directory, "a.bin").exists())
    assertFalse(loader.isComplete())
  }

  @Test
  fun httpErrorReportsStatusAndKeepsPart() {
    writeRevisionMarker()
    for (status in listOf(404, 500)) {
      val name = "status$status.bin"
      server.handle(resolvePath(name)) { it.respond(status) }
      val prefix = bytes(100)
      File(directory, "$name.part").writeBytes(prefix)

      val failure =
        assertThrows(IOException::class.java) {
          downloader(manifest(Triple(name, 1_000, sha256(bytes(1_000))))).download()
        }

      assertTrue(failure.message!!.contains(status.toString()))
      assertArrayEquals(prefix, File(directory, "$name.part").readBytes())
      assertFalse(File(directory, name).exists())
    }
  }

  @Test
  fun redirectsAreFollowedWithRangeAndRelativeLocations() {
    val body = bytes(1_000)
    redirect(resolvePath("a.bin"), 302, "/cdn/one/a.bin")
    redirect("/cdn/one/a.bin", 307, "../two/a.bin")
    serve("/cdn/two/a.bin", body)
    writeRevisionMarker()
    File(directory, "a.bin.part").writeBytes(body.copyOf(250))

    downloader(manifest(entry("a.bin", body))).download()

    assertEquals(
      listOf(resolvePath("a.bin"), "/cdn/one/a.bin", "/cdn/two/a.bin"),
      requests.map { it.path },
    )
    assertEquals(List(3) { "bytes=250-" }, requests.map { it.range })
    assertEquals(sha256(body), sha256(File(directory, "a.bin").readBytes()))
  }

  @Test
  fun sizeOverAndUnderFailWithoutFinalFile() {
    val expected = bytes(1_000)
    serve(resolvePath("over.bin"), bytes(1_200))
    serve(resolvePath("under.bin"), bytes(800))
    for (name in listOf("over.bin", "under.bin")) {
      val failure =
        assertThrows(IOException::class.java) {
          downloader(manifest(Triple(name, expected.size, sha256(expected)))).download()
        }

      assertTrue(failure.message!!.contains("Size mismatch"))
      assertFalse(File(directory, name).exists())
      assertFalse(File(directory, "$name.part").exists())
    }
  }

  @Test
  fun droppedConnectionKeepsPartialBytesAndNextCallResumes() {
    val body = bytes(1_000)
    server.handle(resolvePath("a.bin")) { exchange ->
      val range = exchange.range
      if (range == null) {
        // A chunked body cut after 600 bytes: the client sees a broken stream, not a clean end.
        exchange.respondCutChunked(body.copyOf(600))
      } else {
        val start = range.removeSurrounding("bytes=", "-").toInt()
        exchange.respond(206, body.copyOfRange(start, body.size))
      }
    }
    val loader = downloader(manifest(entry("a.bin", body)))

    assertThrows(IOException::class.java) { loader.download() }
    val partial = File(directory, "a.bin.part").readBytes()
    assertEquals(600, partial.size)
    assertArrayEquals(body.copyOf(600), partial)
    assertFalse(File(directory, "a.bin").exists())

    loader.download()

    assertEquals(listOf(null, "bytes=600-"), requests.map { it.range })
    assertEquals(sha256(body), sha256(File(directory, "a.bin").readBytes()))
  }

  @Test
  fun moreThanFiveRedirectsFail() {
    val body = bytes(100)
    redirectChain("near", hops = 5, body = body)
    redirectChain("far", hops = 6, body = body)

    downloader(manifest(entry("near.bin", body))).download()
    val failure =
      assertThrows(IOException::class.java) {
        downloader(manifest(entry("far.bin", body))).download()
      }

    assertEquals(sha256(body), sha256(File(directory, "near.bin").readBytes()))
    assertTrue(failure.message!!.contains("redirects"))
    assertFalse(File(directory, "far.bin").exists())
  }

  @Test
  fun partAtFullSizeOrRangeNotSatisfiableRestartsFromZero() {
    val body = bytes(1_000)
    serve(resolvePath("full.bin"), body)
    writeRevisionMarker()
    // Full size but wrong bytes: the hash rejects the .part, so the file comes again from byte 0.
    File(directory, "full.bin.part").writeBytes(ByteArray(1_000) { 3 })

    downloader(manifest(entry("full.bin", body))).download()

    assertEquals(listOf<String?>(null), requests.map { it.range })
    assertEquals(sha256(body), sha256(File(directory, "full.bin").readBytes()))

    requests.clear()
    server.handle(resolvePath("shorter.bin")) { exchange ->
      if (exchange.range != null) {
        exchange.respond(416)
      } else {
        exchange.respond(200, body)
      }
    }
    File(directory, "shorter.bin.part").writeBytes(ByteArray(400) { 3 })

    downloader(manifest(entry("shorter.bin", body))).download()

    assertEquals(listOf("bytes=400-", null), requests.map { it.range })
    assertEquals(sha256(body), sha256(File(directory, "shorter.bin").readBytes()))
  }

  @Test
  fun readTimeoutThrows() {
    server.handle(resolvePath("silent.bin")) { release.await(10, TimeUnit.SECONDS) }
    val startedNs = System.nanoTime()

    assertThrows(SocketTimeoutException::class.java) {
      downloader(manifest(Triple("silent.bin", 10, sha256(bytes(10)))), readTimeoutMs = 200)
        .download()
    }

    assertTrue(System.nanoTime() - startedNs < TimeUnit.SECONDS.toNanos(5))
    assertFalse(File(directory, "silent.bin").exists())
  }

  @Test
  fun spaceShortfallThrowsBeforeAnyRequest() {
    val body = bytes(1_000)
    serve(resolvePath("a.bin"), body)
    writeRevisionMarker()
    File(directory, "a.bin.part").writeBytes(body.copyOf(200))

    val failure =
      assertThrows(ModelDownloader.InsufficientSpaceException::class.java) {
        downloader(manifest(entry("a.bin", body)), usableSpace = { 500L }).download()
      }

    assertEquals(800L, failure.neededBytes)
    assertEquals(500L, failure.availableBytes)
    assertTrue(failure.message!!.contains("800") && failure.message!!.contains("500"))
    assertTrue(requests.isEmpty())
    assertEquals(200L, File(directory, "a.bin.part").length())
  }

  @Test
  fun changedRevisionDeletesFilesAndDownloadsAgain() {
    val large = bytes(1_000, seed = 1)
    val small = bytes(300, seed = 2)
    serve(resolvePath("a.bin"), large)
    serve(resolvePath("b.bin"), small)
    // A right-sized file and a partial file left by an older revision.
    File(directory, "a.bin").writeBytes(ByteArray(1_000))
    File(directory, "b.bin.part").writeBytes(ByteArray(100))
    File(directory, ModelDownloader.REVISION_MARKER).writeText("older-revision")
    val loader = downloader(manifest(entry("a.bin", large), entry("b.bin", small)))
    assertFalse(loader.isComplete())
    assertEquals(0L, loader.bytesDownloaded())

    loader.download()

    assertEquals(listOf(resolvePath("a.bin"), resolvePath("b.bin")), requests.map { it.path })
    assertTrue(requests.all { it.range == null })
    assertEquals(sha256(large), sha256(File(directory, "a.bin").readBytes()))
    assertEquals(sha256(small), sha256(File(directory, "b.bin").readBytes()))
    assertEquals(REVISION, File(directory, ModelDownloader.REVISION_MARKER).readText())
    assertTrue(loader.isComplete())
  }

  @Test
  fun leftoversWithoutMarkerAreDiscarded() {
    val large = bytes(1_000, seed = 1)
    val small = bytes(300, seed = 2)
    serve(resolvePath("a.bin"), large)
    serve(resolvePath("b.bin"), small)
    // A right-sized file with wrong bytes and a partial file, and no revision marker.
    File(directory, "a.bin").writeBytes(ByteArray(1_000))
    File(directory, "b.bin.part").writeBytes(ByteArray(100))
    val loader = downloader(manifest(entry("a.bin", large), entry("b.bin", small)))
    assertEquals(0L, loader.bytesDownloaded())

    loader.download()

    // Both leftovers were deleted: a.bin is fetched again, and b.bin starts at byte 0.
    assertEquals(listOf(resolvePath("a.bin"), resolvePath("b.bin")), requests.map { it.path })
    assertTrue(requests.all { it.range == null })
    assertEquals(sha256(large), sha256(File(directory, "a.bin").readBytes()))
    assertEquals(sha256(small), sha256(File(directory, "b.bin").readBytes()))
    assertEquals(REVISION, File(directory, ModelDownloader.REVISION_MARKER).readText())
  }

  @Test
  fun fullSizePartWithMatchingShaIsRenamedWithoutRequest() {
    val body = bytes(1_000)
    serve(resolvePath("a.bin"), body)
    writeRevisionMarker()
    // Every byte arrived, but the rename to the final name did not happen.
    File(directory, "a.bin.part").writeBytes(body)

    downloader(manifest(entry("a.bin", body))).download()

    assertTrue(requests.isEmpty())
    assertEquals(sha256(body), sha256(File(directory, "a.bin").readBytes()))
    assertFalse(File(directory, "a.bin.part").exists())
  }

  @Test
  fun progressExceptionStopsTransferAndKeepsPart() {
    // Larger than one 64 KiB read, so the first progress call follows a partial write.
    val body = bytes(200_000)
    serve(resolvePath("a.bin"), body)
    val loader = downloader(manifest(entry("a.bin", body)))

    assertThrows(CancellationException::class.java) {
      loader.download { done, _ -> if (done > 0) throw CancellationException("stop") }
    }
    val partLength = File(directory, "a.bin.part").length()
    assertTrue(partLength > 0 && partLength < body.size)
    assertFalse(File(directory, "a.bin").exists())

    loader.download()

    assertEquals(listOf(null, "bytes=$partLength-"), requests.map { it.range })
    assertEquals(sha256(body), sha256(File(directory, "a.bin").readBytes()))
  }

  /** Marks the files on disk as leftovers of [REVISION], as a started download does. */
  private fun writeRevisionMarker() {
    File(directory, ModelDownloader.REVISION_MARKER).writeText(REVISION)
  }

  private fun downloader(
    manifestJson: String,
    readTimeoutMs: Int = 5_000,
    usableSpace: (File) -> Long = { Long.MAX_VALUE },
  ) =
    ModelDownloader(
      manifestJson,
      directory,
      baseUrl = server.baseUrl,
      connectTimeoutMs = 5_000,
      readTimeoutMs = readTimeoutMs,
      usableSpace = usableSpace,
    )

  private fun manifest(vararg files: Triple<String, Int, String>): String =
    LayaJson.stringify(
      linkedMapOf(
        "repository" to REPOSITORY,
        "revision" to REVISION,
        "files" to
          files.map { (name, size, sha) ->
            linkedMapOf("name" to name, "size_bytes" to size, "sha256" to sha)
          },
      )
    )

  private fun entry(name: String, body: ByteArray) = Triple(name, body.size, sha256(body))

  private fun resolvePath(name: String) = "/$REPOSITORY/resolve/$REVISION/$name"

  /** Serves [body] at [path], answering "Range: bytes=<n>-" with 206 unless [honorRange] is off. */
  private fun serve(path: String, body: ByteArray, honorRange: Boolean = true) {
    server.handle(path) { exchange ->
      val range = exchange.range
      if (honorRange && range != null) {
        val start = range.removeSurrounding("bytes=", "-").toInt()
        exchange.respond(
          206,
          body.copyOfRange(start, body.size),
          mapOf("Content-Range" to "bytes $start-${body.size - 1}/${body.size}"),
        )
      } else {
        exchange.respond(200, body)
      }
    }
  }

  private fun redirect(path: String, status: Int, location: String) {
    server.handle(path) { it.respond(status, headers = mapOf("Location" to location)) }
  }

  private fun redirectChain(label: String, hops: Int, body: ByteArray) {
    var from = resolvePath("$label.bin")
    for (hop in 1..hops) {
      val to = "/$label/hop$hop.bin"
      redirect(from, 302, to)
      from = to
    }
    serve(from, body)
  }

  private fun bytes(size: Int, seed: Int = 0) = ByteArray(size) { (it * 31 + seed).toByte() }

  private fun sha256(body: ByteArray): String =
    MessageDigest.getInstance("SHA-256").digest(body).joinToString("") { "%02x".format(it) }

  private companion object {
    const val REPOSITORY = "test-owner/test-model"
    const val REVISION = "0123456789abcdef0123456789abcdef01234567"
  }
}

/** One request line as the server saw it: the path and the Range header, if any. */
private data class Request(val path: String, val range: String?)

/**
 * A small HTTP/1.1 server on a loopback socket, one connection per request. Handlers write the
 * whole response, and the socket closes when the handler returns.
 */
private class TestHttpServer : Closeable {
  class Exchange(val range: String?, private val output: OutputStream) {
    fun respond(
      status: Int,
      body: ByteArray = ByteArray(0),
      headers: Map<String, String> = emptyMap(),
    ) {
      val head = StringBuilder("HTTP/1.1 $status Test\r\n")
      headers.forEach { (name, value) -> head.append("$name: $value\r\n") }
      head.append("Content-Length: ${body.size}\r\nConnection: close\r\n\r\n")
      output.write(head.toString().toByteArray(Charsets.US_ASCII))
      output.write(body)
      output.flush()
    }

    /** Sends a chunked 200 response with one chunk and no final chunk, like a dropped link. */
    fun respondCutChunked(chunk: ByteArray) {
      val head =
        "HTTP/1.1 200 Test\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n" +
          Integer.toHexString(chunk.size) +
          "\r\n"
      output.write(head.toByteArray(Charsets.US_ASCII))
      output.write(chunk)
      output.write("\r\n".toByteArray(Charsets.US_ASCII))
      output.flush()
    }
  }

  private val socket = ServerSocket(0, 50, InetAddress.getByName("127.0.0.1"))
  private val executor = Executors.newCachedThreadPool()
  private val handlers = ConcurrentHashMap<String, (Exchange) -> Unit>()

  /** Every request received, in arrival order. */
  val requests: MutableList<Request> = Collections.synchronizedList(mutableListOf())

  /** Scheme, host and port for [ModelDownloader]. */
  val baseUrl = "http://127.0.0.1:${socket.localPort}"

  init {
    executor.execute {
      while (!socket.isClosed) {
        val connection =
          try {
            socket.accept()
          } catch (closed: IOException) {
            break
          }
        executor.execute { serve(connection) }
      }
    }
  }

  /** Registers [handler] for requests whose path equals [path]. */
  fun handle(path: String, handler: (Exchange) -> Unit) {
    handlers[path] = handler
  }

  private fun serve(connection: Socket) {
    connection.use {
      val input = BufferedInputStream(it.getInputStream())
      val requestLine = readLine(input) ?: return
      var range: String? = null
      while (true) {
        val line = readLine(input) ?: return
        if (line.isEmpty()) {
          break
        }
        if (line.startsWith("Range:", ignoreCase = true)) {
          range = line.substringAfter(':').trim()
        }
      }
      val path = requestLine.split(' ')[1]
      requests += Request(path, range)
      val exchange = Exchange(range, it.getOutputStream())
      val handler = handlers[path]
      if (handler == null) {
        exchange.respond(404)
      } else {
        handler(exchange)
      }
    }
  }

  private fun readLine(input: InputStream): String? {
    val line = StringBuilder()
    while (true) {
      val next = input.read()
      if (next < 0) {
        return null
      }
      if (next == '\n'.code) {
        return line.toString().trimEnd('\r')
      }
      line.append(next.toChar())
    }
  }

  override fun close() {
    socket.close()
    executor.shutdownNow()
  }
}
