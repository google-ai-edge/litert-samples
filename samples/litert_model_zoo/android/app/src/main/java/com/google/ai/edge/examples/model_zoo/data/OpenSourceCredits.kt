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

package com.google.ai.edge.examples.model_zoo.data

data class OpenSourceCredit(
  val name: String,
  val license: String,
  val licenseUrl: String,
  val projectUrl: String,
)

object OpenSourceCredits {
  private const val apache = "https://www.apache.org/licenses/LICENSE-2.0"
  val libraries =
    listOf(
      OpenSourceCredit(
        "AndroidX (Activity, Core, Lifecycle, CameraX, WorkManager and related Android libraries)",
        "Apache-2.0",
        apache,
        "https://github.com/androidx/androidx",
      ),
      OpenSourceCredit(
        "Jetpack Compose",
        "Apache-2.0",
        apache,
        "https://github.com/androidx/androidx/tree/androidx-main/compose",
      ),
      OpenSourceCredit(
        "Kotlin standard library",
        "Apache-2.0",
        apache,
        "https://github.com/JetBrains/kotlin",
      ),
      OpenSourceCredit(
        "kotlinx.coroutines",
        "Apache-2.0",
        apache,
        "https://github.com/Kotlin/kotlinx.coroutines",
      ),
      OpenSourceCredit("LiteRT", "Apache-2.0", apache, "https://github.com/google-ai-edge/LiteRT"),
      OpenSourceCredit(
        "Guava (including ListenableFuture and failureaccess)",
        "Apache-2.0",
        apache,
        "https://github.com/google/guava",
      ),
      // License names and URLs are from the exact resolved Maven POMs, not inferred from LiteRT.
      OpenSourceCredit(
        "Additional SDK terms: Google Play AI Delivery 0.1.1-alpha01, Asset Delivery 2.3.0 and Core Common 2.0.4",
        "Play Core Software Development Kit Terms of Service",
        "https://developer.android.com/guide/playcore/license",
        "https://developer.android.com/guide/playcore",
      ),
      OpenSourceCredit(
        "Additional SDK terms: Google Play services Basement 18.4.0 and Tasks 18.2.0",
        "Android Software Development Kit License",
        "https://developer.android.com/studio/terms.html",
        "https://developers.google.com/android/guides/overview",
      ),
    )

  // Attribution source: https://zenodo.org/records/3987831; conversion and ontology credits:
  // https://huggingface.co/litert-community/PANNs-CNN14-AudioSet-LiteRT (read 2026-09-20).
  const val pannsAttribution =
    "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition — " +
      "Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang and Mark D. Plumbley. " +
      "Pretrained CNN14 weights are licensed under Creative Commons Attribution 4.0 International " +
      "(CC-BY-4.0). The litert-community conversion uses an FP16 CNN graph with the log-mel " +
      "frontend on the host. This app uses those converted model files unchanged. " +
      "AudioSet class labels and ontology: © Google, CC-BY-4.0. No endorsement is implied."
  const val ccByLicenseUrl = "https://creativecommons.org/licenses/by/4.0/"
  const val pannsSourceUrl = "https://zenodo.org/records/3987831"
  const val audioSetSourceUrl = "https://research.google.com/audioset/ontology/index.html"
}
