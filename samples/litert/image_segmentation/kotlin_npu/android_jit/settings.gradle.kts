/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
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

pluginManagement {
  repositories {
    google {
      content {
        includeGroupByRegex("com\\.android.*")
        includeGroupByRegex("com\\.google.*")
        includeGroupByRegex("androidx.*")
      }
    }
    mavenCentral()
    gradlePluginPortal()
  }
}

dependencyResolutionManagement {
  repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
  repositories {
    google()
    mavenCentral()
  }
}

rootProject.name = "Image Segmentation"

include(":app")

// NPU runtime libraries
include(":litert_npu_runtime_libraries:runtime_strings")

val intelRuntimeBuild =
  providers.gradleProperty("intelRuntimeBuild").orNull == "true"

if (!intelRuntimeBuild) {
  include(":litert_npu_runtime_libraries:mediatek_runtime_common")
  include(":litert_npu_runtime_libraries:mediatek_runtime_v8")
  include(":litert_npu_runtime_libraries:mediatek_runtime_v9")

  include(":litert_npu_runtime_libraries:google_tensor_runtime")

  // include(":litert_npu_runtime_libraries:samsung_runtime")

  include(":litert_npu_runtime_libraries:qualcomm_runtime_common")
  include(":litert_npu_runtime_libraries:qualcomm_runtime_dsp")
  include(":litert_npu_runtime_libraries:qualcomm_runtime_htp")

  include(":litert_npu_runtime_libraries:qualcomm_runtime_v65")
  include(":litert_npu_runtime_libraries:qualcomm_runtime_v66")
  include(":litert_npu_runtime_libraries:qualcomm_runtime_v68")
  include(":litert_npu_runtime_libraries:qualcomm_runtime_v69")

  include(":litert_npu_runtime_libraries:qualcomm_runtime_v73")

  include(":litert_npu_runtime_libraries:qualcomm_runtime_v75")

  include(":litert_npu_runtime_libraries:qualcomm_runtime_v79")

  include(":litert_npu_runtime_libraries:qualcomm_runtime_v81")
} else {
  include(":litert_npu_runtime_libraries:intel_runtime")
}
