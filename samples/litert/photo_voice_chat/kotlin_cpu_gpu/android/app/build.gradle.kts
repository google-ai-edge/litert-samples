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

// Photo voice chat: LFM3-Omni-2B on LiteRT looks at a photo, hears a spoken question and answers in text and speech.
// The language model, the audio encoder and the vision encoder are read from the .litertlm bundle's sections, the
// depth decoder, the frame embedding and the codec decoder are LiteRT graphs, and the host loop (sampling, the
// interleaved text / speech turns, overlap-add) is Kotlin.
plugins {
  id("com.android.application")
  id("org.jetbrains.kotlin.android")
}

val litertVersion: String = providers.gradleProperty("litert.version").get()
val litertAtLeast23 = litertVersion.split('.').let { v -> v[0].toInt() > 2 || (v[0].toInt() == 2 && v[1].toInt() >= 3) }

android {
  namespace = "com.google.ai.edge.examples.photo_voice_chat"
  compileSdk = 35

  defaultConfig {
    applicationId = "com.google.ai.edge.examples.photo_voice_chat"
    minSdk = 26
    targetSdk = 35
    versionCode = 1
    versionName = "1.0"
    ndk { abiFilters += setOf("arm64-v8a") }
    buildConfigField("String", "LITERT_VERSION", "\"$litertVersion\"")
  }

  buildFeatures { buildConfig = true }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }

  packaging {
    jniLibs {
      // The GPU accelerator is opened by path from the extracted native library directory.
      useLegacyPackaging = true
    }
  }

  androidResources { noCompress += listOf("json", "bin", "png", "wav") }
}

kotlin {
  compilerOptions {
    jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    // LiteRT 2.3.0's Kotlin API carries Kotlin 2.4 metadata; Kotlin 2.2.21 reads it with this flag.
    if (litertAtLeast23) freeCompilerArgs.add("-Xskip-metadata-version-check")
  }
}

dependencies {
  implementation("com.google.ai.edge.litert:litert:$litertVersion")
  // From 2.3.0 the GPU accelerator is its own artifact (the measuring tool can move a graph to the GPU).
  if (litertAtLeast23) implementation("com.google.ai.edge.litert:litert-gpu:$litertVersion")
  implementation("androidx.core:core-ktx:1.15.0")
}
