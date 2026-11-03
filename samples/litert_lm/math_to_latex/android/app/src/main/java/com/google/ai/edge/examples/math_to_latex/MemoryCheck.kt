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

package com.google.ai.edge.examples.math_to_latex

import java.io.File

/** The phone has too little free memory to make an Engine; nothing was loaded. */
class NotEnoughMemoryException(message: String) : IllegalStateException(message)

/**
 * Checks the phone's available memory (`MemAvailable` in /proc/meminfo) before an Engine is made,
 * so a phone short of memory gets a message instead of having other apps (or this one) killed.
 * [neededBytes] is what the phone the README lists (a Galaxy S26) lost in available memory over a
 * run of reads, at most 1.4 GB, plus a reserve.
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
          "Close other apps and try again. Nothing was loaded."
      )
    }
  }

  companion object {
    /**
     * At most 1.4 GB taken on the Galaxy S26 (text decoder and vision encoder on the CPU), plus a
     * reserve.
     */
    const val NEEDED_BYTES = 2_500_000_000L

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
