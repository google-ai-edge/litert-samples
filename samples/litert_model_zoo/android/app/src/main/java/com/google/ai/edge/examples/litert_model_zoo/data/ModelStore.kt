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

import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

enum class DownloadStatus {
  MISSING,
  STARTING,
  DOWNLOADING,
  PAUSED,
  VERIFYING,
  READY,
  ERROR,
}

data class DownloadState(
  val status: DownloadStatus = DownloadStatus.MISSING,
  val receivedBytes: Long = 0,
  val totalBytes: Long = 0,
  val error: String? = null,
)

/**
 * All model bytes stay outside the APK, under filesDir/models/<taskId>. [ioDispatcher] runs the
 * file and network calls, and [openConnection] opens a file's URL. A test passes its own of both.
 * [ioDispatcher] must be able to run other work while a download blocks one of its threads, since
 * the disconnect() on a cancel runs on it: Dispatchers.IO can, a single-thread dispatcher cannot.
 */
class ModelStore(
  private val root: File,
  private val eventLogger: (String) -> Unit = { Log.i("ModelZooDownload", it) },
  private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO,
  private val openConnection: (URL) -> HttpURLConnection = {
    it.openConnection() as HttpURLConnection
  },
) {
  fun directory(entry: ModelEntry): File = File(root, entry.taskId)

  suspend fun inspect(entry: ModelEntry): DownloadState =
    withContext(ioDispatcher) {
      val dir = directory(entry)
      val present =
        entry.files.sumOf { file ->
          val finalFile = File(dir, file.name)
          if (finalFile.isFile) finalFile.length() else File(dir, "${file.name}.part").length()
        }
      // A committed file was hashed before download() renamed it into place, so a cold start only
      // checks that every file is there at its catalog size instead of re-reading gigabytes.
      val ready = entry.canDownload && entry.files.all { isCommitted(File(dir, it.name), it) }
      DownloadState(
        if (ready) {
          DownloadStatus.READY
        } else if (present > 0) {
          DownloadStatus.PAUSED
        } else {
          DownloadStatus.MISSING
        },
        present,
        entry.totalBytes,
      )
    }

  /**
   * Downloads the entry's missing files (a partial file resumes), checks each file's size and
   * SHA-256 before it is committed, and reports [progress]. A cancel is checked before the
   * connection is made and again before the request goes out. When the download is cancelled, a
   * child coroutine calls disconnect() on the connection, and what that ends depends on the
   * HttpURLConnection in use. On the JDK's HttpURLConnection, disconnect() ends a read that waits
   * for the response but not a connect in progress, so a cancel while connecting is seen only when
   * the connect ends. The 30 s connect timeout does not bound that wait, since the JDK resolves the
   * host name before the timed connect starts. On the JDK, that wait comes in connect(), in the
   * connect responseCode makes to a redirect's target, and in the connect responseCode makes again
   * when a cancel closes the connection after the last check but before responseCode uses it. In
   * the last two, once that connect succeeds, responseCode also sends the request and waits for the
   * response or the 30 s read timeout before the cancel is seen. On Android 16, disconnect() also
   * closes a socket that is still connecting.
   */
  suspend fun download(entry: ModelEntry, progress: (DownloadState) -> Unit) =
    withContext(ioDispatcher) {
      require(entry.canDownload) { "This model is awaiting verified download metadata" }
      val dir = directory(entry)
      check(dir.isDirectory || dir.mkdirs()) { "Cannot create model directory" }
      var completed = 0L
      entry.files.forEach { modelFile ->
        currentCoroutineContext().ensureActive()
        val fileStartedNanos = System.nanoTime()
        val target = File(dir, modelFile.name)
        if (isCommitted(target, modelFile)) {
          completed += modelFile.bytes
        } else {
          val part = File(dir, "${modelFile.name}.part")
          if (part.length() > modelFile.bytes) {
            check(part.delete()) { "Cannot discard oversized partial file" }
          }
          var offset = part.length()
          val initialBytes = offset
          downloadEvent("start", entry, modelFile, offset, fileStartedNanos)
          if (offset < modelFile.bytes) {
            val connection = openConnection(URL(modelFile.url))
            connection.connectTimeout = 30_000
            connection.readTimeout = 30_000
            connection.instanceFollowRedirects = true
            connection.setRequestProperty("Accept-Encoding", "identity")
            if (offset > 0) {
              connection.setRequestProperty("Range", "bytes=$offset-")
            }
            // Pause and Delete cancel this coroutine, but the socket calls below block without
            // seeing the cancel. The child coroutine is cancelled with the download and calls the
            // connection's disconnect(), whose effect the KDoc of download() describes. It starts
            // undispatched: a child cancelled before it ever ran would skip its finally.
            coroutineScope {
              val abortOnCancel =
                launch(start = CoroutineStart.UNDISPATCHED) {
                  try {
                    awaitCancellation()
                  } finally {
                    runCatching { connection.disconnect() }
                  }
                }
              try {
                currentCoroutineContext().ensureActive()
                connection.connect()
                // On the JDK's HttpURLConnection, disconnect() closes nothing until the connection
                // exists: a cancel that came while it was being made is seen here, before the
                // request goes out.
                currentCoroutineContext().ensureActive()
                val code = connection.responseCode
                downloadEvent("response", entry, modelFile, offset, fileStartedNanos, code)
                check(connection.url.protocol == "https") { "Insecure download redirect" }
                check(code == 200 || code == 206) { "Download returned HTTP $code" }
                if (code == 206) {
                  val range = connection.getHeaderField("Content-Range") ?: ""
                  check(
                    range.startsWith("bytes $offset-") && range.endsWith("/${modelFile.bytes}")
                  ) {
                    "Invalid resume response"
                  }
                } else {
                  offset = 0
                }
                connection.inputStream.use { input ->
                  FileOutputStream(part, offset > 0).use { output ->
                    val buffer = ByteArray(64 * 1024)
                    var lastReport = 0L
                    var lastLog = 0L
                    while (true) {
                      currentCoroutineContext().ensureActive()
                      val count = input.read(buffer)
                      if (count < 0) break
                      check(offset + count <= modelFile.bytes) { "Download exceeds catalog size" }
                      output.write(buffer, 0, count)
                      offset += count
                      val now = System.nanoTime()
                      if (now - lastLog >= 1_000_000_000L) {
                        downloadEvent("progress", entry, modelFile, offset, fileStartedNanos, code)
                        lastLog = now
                      }
                      if (now - lastReport >= 100_000_000L) {
                        progress(
                          DownloadState(
                            DownloadStatus.DOWNLOADING,
                            completed + offset,
                            entry.totalBytes,
                          )
                        )
                        lastReport = now
                      }
                    }
                  }
                }
                // A connection torn down by cancellation can also end the stream early instead of
                // throwing; that is a pause, not a short download.
                currentCoroutineContext().ensureActive()
              } catch (failure: Exception) {
                // A call torn down by cancellation reports as cancelled, not as a download error.
                // The JDK's HttpURLConnection does not always throw an IOException then: it can
                // throw a RuntimeException around a NullPointerException when disconnect() races
                // the blocked call.
                currentCoroutineContext().ensureActive()
                throw failure
              } finally {
                abortOnCancel.cancel()
                // Also closes a connection made after the child's disconnect(). The child can be
                // in disconnect() still, and two at once can throw on the JDK's HttpURLConnection.
                runCatching { connection.disconnect() }
              }
            }
          }
          check(part.length() >= modelFile.bytes) {
            "Download ended early; the partial file is saved for resume"
          }
          progress(
            DownloadState(DownloadStatus.VERIFYING, completed + part.length(), entry.totalBytes)
          )
          if (!verify(part, modelFile)) {
            part.delete()
            error(
              "SHA-256 or byte-size verification failed for ${modelFile.name}; " +
                "retry starts this file again"
            )
          }
          check(!target.exists() || target.delete()) { "Cannot replace invalid model file" }
          check(part.renameTo(target)) { "Cannot commit verified model file" }
          downloadEvent(
            "sha_ok",
            entry,
            modelFile,
            modelFile.bytes,
            fileStartedNanos,
            initialBytes = initialBytes,
          )
          completed += modelFile.bytes
        }
      }
      progress(DownloadState(DownloadStatus.READY, entry.totalBytes, entry.totalBytes))
    }

  suspend fun delete(entry: ModelEntry) =
    withContext(ioDispatcher) {
      val dir = directory(entry)
      check(!dir.exists() || dir.deleteRecursively()) { "Could not delete all model files" }
    }

  suspend fun storageBytes(): Long =
    withContext(ioDispatcher) {
      if (root.exists()) root.walkTopDown().filter { it.isFile }.sumOf { it.length() } else 0L
    }

  private fun downloadEvent(
    event: String,
    entry: ModelEntry,
    file: ModelFile,
    bytes: Long,
    started: Long,
    http: Int? = null,
    initialBytes: Long? = null,
  ) {
    // Model delivery metadata only; never log user images, audio, text, or identifiers.
    val value =
      JSONObject()
        .put("event", event)
        .put("taskId", entry.taskId)
        .put("file", file.name)
        .put("bytes", bytes)
        .put("totalBytes", file.bytes)
        .put("elapsedMs", (System.nanoTime() - started) / 1_000_000.0)
    if (http != null) {
      value.put("httpStatus", http)
    }
    if (initialBytes != null) {
      value.put("initialPartBytes", initialBytes)
    }
    if (event == "sha_ok") {
      value.put("sha256", file.sha256)
    }
    eventLogger(value.toString())
  }

  companion object {
    /**
     * True for a file that [download] committed: present at its catalog size, for an entry whose
     * digest is known. The content was verified once, before the atomic rename; the app's private
     * storage is the only writer afterwards.
     */
    fun isCommitted(file: File, expected: ModelFile): Boolean =
      expected.sha256 != null && file.isFile && file.length() == expected.bytes

    /** Byte size and SHA-256 of [file] against the catalog; run once per downloaded file. */
    fun verify(file: File, expected: ModelFile): Boolean {
      if (!isCommitted(file, expected)) return false
      val digest = MessageDigest.getInstance("SHA-256")
      file.inputStream().use { input ->
        val buffer = ByteArray(64 * 1024)
        while (true) {
          val n = input.read(buffer)
          if (n < 0) break
          digest.update(buffer, 0, n)
        }
      }
      return digest.digest().joinToString("") { "%02x".format(it) } == expected.sha256
    }
  }
}
