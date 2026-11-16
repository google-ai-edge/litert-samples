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

// Optional: `-Pnpu.libs=/path/to/qairt_arm64_libs` bundles QNN HTP .so files into the APK.
val npuLibs = providers.gradleProperty("npu.libs").orElse(providers.systemProperty("npu.libs"))
val stagedNpuJniLibs = layout.buildDirectory.dir("generated/npuJniLibs")

val stageNpuLibs =
  tasks.register<Sync>("stageNpuLibs") {
    into(stagedNpuJniLibs)
    npuLibs.orNull
      ?.takeIf { it.isNotBlank() }
      ?.let { dir ->
        val src = rootProject.file(dir)
        require(src.isDirectory) { "npu.libs is not a directory: $src" }
        from(src) {
          include("libQnn*.so")
          into("arm64-v8a")
        }
      }
  }

android {
  namespace = "com.google.ai.edge.examples.text_to_speech_lm"
  compileSdk = 35

  defaultConfig {
    applicationId = "com.google.ai.edge.examples.text_to_speech_lm"
    minSdk = 31
    targetSdk = 35
    versionCode = 1
    versionName = "1.0"
    buildConfigField("boolean", "NPU_LIBS", "${npuLibs.isPresent}")
    ndk { abiFilters += setOf("arm64-v8a") }
  }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }
  kotlinOptions {
    jvmTarget = "17"
    freeCompilerArgs += "-Xskip-metadata-version-check"
  }
  buildFeatures {
    buildConfig = true
  }

  packaging {
    jniLibs {
      useLegacyPackaging = true
      pickFirsts += setOf(
        "**/libc++_shared.so",
        "**/libtensorflowlite_jni.so",
        "**/libtensorflowlite_gpu_jni.so",
      )
    }
  }

  sourceSets.named("main") { jniLibs.srcDir(stagedNpuJniLibs) }

  androidResources { noCompress += listOf("json") }
}

tasks.named("preBuild") { dependsOn(stageNpuLibs) }

dependencies {
  implementation("com.google.ai.edge.litert:litert:2.3.0")
  implementation("com.google.ai.edge.litert:litert-gpu:2.3.0")
  implementation("com.google.ai.edge.litert:litert-npu-runtime-qualcomm:2.3.0")
  implementation("androidx.core:core-ktx:1.15.0")
  testImplementation("junit:junit:4.13.2")
}
