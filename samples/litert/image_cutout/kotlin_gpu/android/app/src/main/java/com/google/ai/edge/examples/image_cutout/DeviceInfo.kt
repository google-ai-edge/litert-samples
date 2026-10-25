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

package com.google.ai.edge.examples.image_cutout

import android.content.Context
import android.os.Build
import android.os.PowerManager
import java.io.File

/** Device facts for the status line and the run JSON. */
object DeviceInfo {
  /** `MemAvailable` of `/proc/meminfo` in kB (what the memory checks compare with), or -1. */
  fun memAvailableKb(): Long = meminfo("MemAvailable:")

  fun memTotalKb(): Long = meminfo("MemTotal:")

  private fun meminfo(key: String): Long = runCatching {
    File("/proc/meminfo").useLines { lines ->
      lines.first { it.startsWith(key) }.trim().split(Regex("\\s+"))[1].toLong()
    }
  }
    .getOrDefault(-1L)

  /** VmHWM / VmRSS / VmSwap of this process in kB (`/proc/self/status`). */
  fun selfStatusKb(): Map<String, Long> = runCatching {
    File("/proc/self/status")
      .readLines()
      .mapNotNull { line ->
        val key = line.substringBefore(':')
        if (key in setOf("VmHWM", "VmRSS", "VmSwap")) {
          key to line.substringAfter(':').trim().split(' ')[0].toLong()
        } else {
          null
        }
      }
      .toMap()
  }
    .getOrDefault(emptyMap())

  /** "Galaxy S26 (SM-S942Q)" for the phones this sample was measured on, else `Build.MODEL`. */
  fun displayName(): String {
    val model = Build.MODEL
    val market = MARKET_NAMES.entries.firstOrNull { model.startsWith(it.key) }?.value
    return if (market == null || market == model) model else "$market ($model)"
  }

  fun thermalStatus(context: Context): Int =
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
      context.getSystemService(PowerManager::class.java).currentThermalStatus
    } else {
      -1
    }

  /** The cpuset of this process (`/top-app` while the app is in front). */
  fun cpuset(): String = runCatching {
    File("/proc/self/cgroup").readLines().firstOrNull { ":cpuset:" in it } ?: "none"
  }
    .getOrDefault("unreadable")

  private val MARKET_NAMES =
    mapOf("SM-S942" to "Galaxy S26", "akita" to "Pixel 8a", "Pixel 8a" to "Pixel 8a")
}
