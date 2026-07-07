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
  id("com.android.application")
  id("org.jetbrains.kotlin.android")
}

android {
  namespace = "com.google.ai.edge.examples.inpainting"
  compileSdk = 35

  defaultConfig {
    applicationId = "com.google.ai.edge.examples.inpainting"
    minSdk = 26
    targetSdk = 35
    versionCode = 1
    versionName = "1.0"
    ndk { abiFilters += setOf("arm64-v8a") }
  }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }
  kotlinOptions { jvmTarget = "17" }

  packaging {
    jniLibs {
      pickFirsts += setOf(
        "**/libc++_shared.so",
        "**/libtensorflowlite_jni.so",
        "**/libtensorflowlite_gpu_jni.so",
      )
    }
  }

  androidResources { noCompress += listOf("tflite") }
}

dependencies {
  implementation("com.google.ai.edge.litert:litert:2.1.5")
  implementation("androidx.core:core-ktx:1.15.0")
}
