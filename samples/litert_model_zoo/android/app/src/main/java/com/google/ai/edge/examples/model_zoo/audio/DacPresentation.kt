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

package com.google.ai.edge.examples.model_zoo.audio

import java.util.Locale

/** Raw unaligned SNR is a diagnostic, not a listener-facing codec quality measure. */
internal fun audioCodecSummary(samples: Int, sampleRate: Int, codeCount: Int): String {
  require(samples >= 0 && sampleRate > 0 && codeCount >= 0)
  return String.format(
    Locale.US,
    "Decoded %.2f s · %d codes",
    samples.toDouble() / sampleRate,
    codeCount,
  )
}
