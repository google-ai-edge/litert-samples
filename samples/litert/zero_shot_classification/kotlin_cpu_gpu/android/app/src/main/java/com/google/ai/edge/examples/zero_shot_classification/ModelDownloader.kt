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

import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * Downloads the files of a model manifest from Hugging Face at a pinned revision and checks each
 * one against the manifest size and SHA-256 before it gets its final name.
 *
 * Plain Kotlin on [HttpURLConnection]: the manifest text and the target directory are passed in,
 * so the app, the instrumented parity test and the JVM unit tests share this class. Every file is
 * written to `<name>.part` before it is renamed. A later call resumes it with an HTTP Range
 * request, so closing the app pauses a download instead of discarding it. The revision marker is
 * written before the first transfer, so files on disk are kept only under a marker that names
 * this revision. Files under another revision's marker, or with no marker, are deleted first.
 */
class ModelDownloader(
  manifestJson: String,
  private val directory: File,
  private val baseUrl: String = "https://huggingface.co",
  private val connectTimeoutMs: Int = 15_000,
  private val readTimeoutMs: Int = 30_000,
  private val usableSpace: (File) -> Long = { it.usableSpace },
  private val revisionMarkerName: String = REVISION_MARKER,
) {
  /** One manifest entry. [name] is also the file's path inside the repository revision. */
  data class ModelFile(val name: String, val sizeBytes: Long, val sha256: String)

  /** Thrown before any transfer when the directory cannot hold the bytes still needed. */
  class InsufficientSpaceException(val neededBytes: Long, val availableBytes: Long) :
    IOException("Needs $neededBytes bytes free, $availableBytes bytes available")

  /** Hugging Face repository id. */
  val repository: String

  /** Pinned commit. The manifest's SHA-256 values describe the files at this revision. */
  val revision: String

  /** Files in manifest order. */
  val files: List<ModelFile>

  /** Sum of the manifest sizes. */
  val totalBytes: Long

  init {
    val manifest = LayaJson.asObject(LayaJson.parse(manifestJson))
    repository = manifest["repository"] as? String ?: error("Manifest needs a repository")
    revision = manifest["revision"] as? String ?: error("Manifest needs a revision")
    files =
      LayaJson.asArray(manifest["files"]).map { value ->
        val entry = LayaJson.asObject(value)
        ModelFile(
          name = entry["name"] as? String ?: error("Manifest entry needs a name"),
          sizeBytes = (entry["size_bytes"] as Number).toLong(),
          sha256 = (entry["sha256"] as String).lowercase(),
        )
      }
    require(files.isNotEmpty()) { "Manifest lists no files" }
    totalBytes = files.sumOf { it.sizeBytes }
  }

  /**
   * True when the revision marker names this revision and every file has its final name at the
   * manifest size. The marker alone only shows that a download started.
   */
  fun isComplete(): Boolean = markerRevision() == revision && files.all { isFinal(it) }

  /**
   * Bytes on disk for this revision: finished files plus resumable `.part` files. Zero when the
   * marker names another revision or is missing while files exist, since [download] deletes them.
   */
  fun bytesDownloaded(): Long =
    if (isStale()) {
      0L
    } else {
      files.sumOf { onDisk(it) }
    }

  /**
   * Deletes files left under another revision or without a marker, checks free space, writes the
   * revision marker, then downloads every missing file. It blocks, so call it off the main thread.
   * [onProgress] receives (bytes on disk, total bytes) after every write. An exception thrown from
   * [onProgress], such as a coroutine cancellation, stops the transfer and keeps the `.part` file
   * for the next call.
   */
  fun download(onProgress: (Long, Long) -> Unit = { _, _ -> }) {
    directory.mkdirs()
    if (isStale()) {
      files.forEach {
        finalFile(it).delete()
        partFile(it).delete()
      }
      markerFile().delete()
    }
    val needed = totalBytes - bytesDownloaded()
    val available = usableSpace(directory)
    if (needed > available) {
      throw InsufficientSpaceException(needed, available)
    }
    if (markerRevision() != revision) {
      writeMarker()
    }
    files.forEach { file ->
      if (!isFinal(file)) {
        finalFile(file).delete()
        val otherBytes = files.filter { it != file }.sumOf { onDisk(it) }
        fetch(file, otherBytes, onProgress)
      }
    }
    onProgress(totalBytes, totalBytes)
  }

  /** Writes the marker through a temporary file, so a crash never leaves a partial revision. */
  private fun writeMarker() {
    val pending = File(directory, "$revisionMarkerName.tmp")
    pending.writeText(revision)
    if (!pending.renameTo(markerFile())) {
      throw IOException("Could not rename ${pending.name} to $revisionMarkerName")
    }
  }

  private fun fetch(file: ModelFile, otherBytes: Long, onProgress: (Long, Long) -> Unit) {
    val part = partFile(file)
    part.parentFile?.mkdirs()
    // A full-size .part is finished bytes that missed the rename, or wrong bytes: the hash decides.
    if (part.length() == file.sizeBytes) {
      val digest = MessageDigest.getInstance("SHA-256")
      digestExisting(part, digest)
      if (hex(digest.digest()) == file.sha256) {
        renameToFinal(file, part)
        return
      }
    }
    if (part.length() >= file.sizeBytes) {
      part.delete()
    }
    var connection = connect(file, part.length())
    if (connection.responseCode == HTTP_RANGE_NOT_SATISFIABLE && part.length() > 0) {
      connection.disconnect()
      part.delete()
      connection = connect(file, 0L)
    }
    try {
      receive(file, part, connection, otherBytes, onProgress)
    } finally {
      connection.disconnect()
    }
  }

  private fun receive(
    file: ModelFile,
    part: File,
    connection: HttpURLConnection,
    otherBytes: Long,
    onProgress: (Long, Long) -> Unit,
  ) {
    val status = connection.responseCode
    if (status != HttpURLConnection.HTTP_OK && status != HttpURLConnection.HTTP_PARTIAL) {
      throw IOException("HTTP $status for ${file.name}")
    }
    val digest = MessageDigest.getInstance("SHA-256")
    // 206 continues the bytes already on disk. 200 is the whole file, so the .part starts over.
    val resumed = status == HttpURLConnection.HTTP_PARTIAL
    var received = if (resumed) digestExisting(part, digest) else 0L
    var oversized = false
    FileOutputStream(part, resumed).use { output ->
      connection.inputStream.use { input ->
        val buffer = ByteArray(BUFFER_BYTES)
        while (!oversized) {
          val count = input.read(buffer)
          if (count < 0) {
            break
          }
          if (received + count > file.sizeBytes) {
            oversized = true
          } else {
            output.write(buffer, 0, count)
            digest.update(buffer, 0, count)
            received += count
            onProgress(otherBytes + received, totalBytes)
          }
        }
      }
    }
    if (oversized || received != file.sizeBytes) {
      part.delete()
      val actual = if (oversized) "more" else "$received"
      throw IOException(
        "Size mismatch for ${file.name}: expected ${file.sizeBytes} bytes, received $actual"
      )
    }
    if (hex(digest.digest()) != file.sha256) {
      part.delete()
      throw IOException("SHA-256 mismatch for ${file.name}")
    }
    renameToFinal(file, part)
  }

  /** Gives a verified `.part` its final name. */
  private fun renameToFinal(file: ModelFile, part: File) {
    val target = finalFile(file)
    target.delete()
    if (!part.renameTo(target)) {
      throw IOException("Could not rename ${part.name} to ${target.name}")
    }
  }

  /** Opens the file URL and follows redirects by hand so every hop carries the Range header. */
  private fun connect(file: ModelFile, offset: Long): HttpURLConnection {
    var url = URL("$baseUrl/$repository/resolve/$revision/${file.name}")
    var hops = 0
    while (true) {
      val connection = url.openConnection() as HttpURLConnection
      connection.instanceFollowRedirects = false
      connection.connectTimeout = connectTimeoutMs
      connection.readTimeout = readTimeoutMs
      connection.setRequestProperty("Accept-Encoding", "identity")
      if (offset > 0) {
        connection.setRequestProperty("Range", "bytes=$offset-")
      }
      val status =
        try {
          connection.responseCode
        } catch (failure: IOException) {
          connection.disconnect()
          throw failure
        }
      if (status !in REDIRECT_CODES) {
        return connection
      }
      val location = connection.getHeaderField("Location")
      connection.disconnect()
      if (location == null) {
        throw IOException("HTTP $status without a Location header for ${file.name}")
      }
      if (hops == MAX_REDIRECTS) {
        throw IOException("More than $MAX_REDIRECTS redirects for ${file.name}")
      }
      url = URL(url, location)
      hops++
    }
  }

  private fun digestExisting(part: File, digest: MessageDigest): Long {
    if (!part.isFile) {
      return 0L
    }
    var total = 0L
    FileInputStream(part).use { input ->
      val buffer = ByteArray(BUFFER_BYTES)
      while (true) {
        val count = input.read(buffer)
        if (count < 0) {
          break
        }
        digest.update(buffer, 0, count)
        total += count
      }
    }
    return total
  }

  private fun onDisk(file: ModelFile): Long {
    val partLength = partFile(file).length()
    return when {
      isFinal(file) -> file.sizeBytes
      partLength < file.sizeBytes -> partLength
      else -> 0L
    }
  }

  private fun isFinal(file: ModelFile): Boolean {
    val target = finalFile(file)
    return target.isFile && target.length() == file.sizeBytes
  }

  /** True when the marker names another revision, or when listed files exist with no marker. */
  private fun isStale(): Boolean {
    val marker = markerRevision()
    return if (marker == null) {
      files.any { finalFile(it).exists() || partFile(it).exists() }
    } else {
      marker != revision
    }
  }

  private fun markerRevision(): String? = markerFile().takeIf { it.isFile }?.readText()?.trim()

  private fun markerFile() = File(directory, revisionMarkerName)

  private fun finalFile(file: ModelFile) = File(directory, file.name)

  private fun partFile(file: ModelFile) = File(directory, file.name + PART_SUFFIX)

  private fun hex(bytes: ByteArray): String = bytes.joinToString("") { "%02x".format(it) }

  companion object {
    /** File in the target directory that records the revision the files were verified against. */
    const val REVISION_MARKER = "model_manifest.revision"
    private const val PART_SUFFIX = ".part"
    private const val MAX_REDIRECTS = 5
    private const val BUFFER_BYTES = 64 * 1024
    private const val HTTP_RANGE_NOT_SATISFIABLE = 416
    private val REDIRECT_CODES = setOf(301, 302, 303, 307, 308)
  }
}
