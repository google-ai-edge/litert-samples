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

plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.jetbrains.kotlin.android)
  alias(libs.plugins.compose.compiler)
}

android {
  namespace = "com.google.ai.edge.examples.zero_shot_classification"
  compileSdk = 36

  defaultConfig {
    applicationId = "com.google.ai.edge.examples.zero_shot_classification"
    minSdk = 31
    targetSdk = 35
    versionCode = 1
    versionName = "1.0"
    ndk { abiFilters += setOf("arm64-v8a") }
    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
  }

  buildTypes {
    release {
      // Every runtime module ships the same LiteRT dispatch library and compiler plugin, so their
      // debug symbol tables would collide in the bundle.
      ndk { debugSymbolLevel = "NONE" }
    }
  }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }
  kotlinOptions { jvmTarget = "17" }
  buildFeatures {
    compose = true
    buildConfig = true
  }

  packaging {
    jniLibs {
      // Needed for Qualcomm NPU runtimes: LiteRT loads them from the app's native library dir.
      useLegacyPackaging = true
      pickFirsts +=
        setOf(
          "**/libc++_shared.so",
          "**/libtensorflowlite_jni.so",
          "**/libtensorflowlite_gpu_jni.so",
        )
    }
  }

  androidResources { noCompress += listOf("json") }

  // NPU runtime libraries, delivered at install time to the matching Qualcomm SoC only.
  dynamicFeatures.add(":litert_npu_runtime_libraries:qualcomm_runtime_v73")
  dynamicFeatures.add(":litert_npu_runtime_libraries:qualcomm_runtime_v75")
  dynamicFeatures.add(":litert_npu_runtime_libraries:qualcomm_runtime_v79")
  dynamicFeatures.add(":litert_npu_runtime_libraries:qualcomm_runtime_v81")

  bundle {
    deviceTargetingConfig = file("device_targeting_configuration.xml")
    deviceGroup {
      enableSplit = true // one split per device group
      defaultGroup = "other" // group used for standalone APKs
    }
  }
}

dependencies {
  // Strings for NPU runtime libraries
  implementation(project(":litert_npu_runtime_libraries:runtime_strings"))

  implementation(libs.litert)
  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.activity.compose)
  implementation(libs.androidx.lifecycle.runtime.compose)
  implementation(libs.androidx.lifecycle.viewmodel.ktx)
  implementation(platform(libs.androidx.compose.bom))
  implementation(libs.androidx.ui)
  implementation(libs.androidx.ui.tooling.preview)
  implementation(libs.androidx.material2)
  debugImplementation(libs.androidx.ui.tooling)

  testImplementation(libs.junit)
  androidTestImplementation(libs.androidx.test.ext.junit)
  androidTestImplementation(libs.androidx.test.runner)
}
