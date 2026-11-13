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

// Adapted from this repository's math_to_latex sample (MemoryCheck.kt).

package com.google.ai.edge.examples.voice_to_text

import java.io.File

/** The phone has too little free memory to make an Engine; nothing was loaded. */
class NotEnoughMemoryException(message: String) : IllegalStateException(message)

/**
 * Checks the phone's available memory (`MemAvailable` in /proc/meminfo) before the Engine is made,
 * so a phone short of memory gets a message instead of having other apps (or this one) killed.
 */
class MemoryCheck(
  private val neededBytes: Long = NEEDED_BYTES,
  private val read: () -> Long? = ::memAvailableBytes,
) {
  fun require() {
    val available = read() ?: return
    if (available < neededBytes) {
      throw NotEnoughMemoryException(
        "About ${gb(neededBytes)} GB of free memory is needed and ${gb(available)} GB is free. " +
          "Close other apps and tap Load. Nothing was loaded."
      )
    }
  }

  companion object {
    /**
     * What the phones lost in available memory (MemAvailable) from the Engine's start through the
     * first transcriptions: up to 4.78 GB on a Galaxy S26, 4.71 GB on a Pixel 11 Pro.
     */
    const val NEEDED_BYTES = 4_800_000_000L

    private fun gb(bytes: Long) = "%.1f".format(bytes / 1e9)

    /** MemAvailable in bytes, or null where /proc/meminfo cannot be read. */
    fun memAvailableBytes(): Long? =
      runCatching {
          File("/proc/meminfo").useLines { lines ->
            lines
              .firstOrNull { it.startsWith("MemAvailable:") }
              ?.split(Regex("\\s+"))
              ?.getOrNull(1)
              ?.toLong()
              ?.times(1024)
          }
        }
        .getOrNull()
  }
}
