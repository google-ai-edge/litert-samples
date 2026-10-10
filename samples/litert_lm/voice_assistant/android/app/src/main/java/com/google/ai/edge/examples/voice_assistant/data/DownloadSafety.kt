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
// samples/litert_model_zoo/android/app/src/main/java/.../litert_model_zoo/data/DownloadSafety.kt
// The metered-network confirmation is not used here.

package com.google.ai.edge.examples.voice_assistant.data

/**
 * Reserve space for the bytes still to be written plus verification: twice what is missing, as
 * the model zoo app does for a download.
 */
object DownloadSafety {
  fun requiredFreeBytes(modelBytes: Long): Long {
    require(modelBytes > 0 && modelBytes <= Long.MAX_VALUE / 2) { "Invalid model size" }
    return modelBytes * 2
  }

  fun hasSpace(modelBytes: Long, availableBytes: Long, reservedBytes: Long = 0): Boolean {
    val required = requiredFreeBytes(modelBytes)
    return reservedBytes >= 0 &&
      availableBytes >= reservedBytes &&
      availableBytes - reservedBytes >= required
  }
}
