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

// Voicemail summary: LFM3-Audio-2B on LiteRT hears a voicemail and answers in text and speech. The language model and
// the audio encoder are read from the .litertlm bundle's sections, the depth decoder, the frame embedding and the codec
// decoder are LiteRT graphs, and the host loop (sampling, the interleaved text / speech turns, overlap-add) is Kotlin.
plugins {
  id("com.android.application")
  id("org.jetbrains.kotlin.android")
}

val litertVersion: String = providers.gradleProperty("litert.version").get()
val litertAtLeast23 = litertVersion.split('.').let { v -> v[0].toInt() > 2 || (v[0].toInt() == 2 && v[1].toInt() >= 3) }
// -Pnpu.libs=<dir>: QAIRT's QNN libraries for the Qualcomm NPU (libQnn*.so, searched in the dir's subdirs too) of the
// QAIRT release LiteRT pins (2.3.0: QAIRT 2.50; 2.2.0: QAIRT 2.47). Qualcomm's license lets them ship only inside an
// application, so they are never part of the sources; the build copies them into the APK (README.md, NPU). LiteRT's
// own dispatch library comes from Maven (litert-npu-runtime-qualcomm, 2.3.0 on) or, for 2.2.0, from the same dir.
val npuLibs: String? = providers.gradleProperty("npu.libs").orNull?.takeIf { it.isNotBlank() }
val npuStage = layout.buildDirectory.dir("npuJniLibs")

android {
  namespace = "com.google.ai.edge.examples.voicemail_summary"
  compileSdk = 35

  defaultConfig {
    applicationId = "com.google.ai.edge.examples.voicemail_summary"
    minSdk = 26
    targetSdk = 35
    versionCode = 1
    versionName = "1.0"
    ndk { abiFilters += setOf("arm64-v8a") }
    buildConfigField("String", "LITERT_VERSION", "\"$litertVersion\"")
    buildConfigField("boolean", "NPU_LIBS", if (npuLibs != null) "true" else "false")
  }

  buildFeatures { buildConfig = true }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }

  packaging {
    jniLibs {
      // The GPU accelerator and the NPU libraries are opened by path from the extracted native library directory;
      // kept byte for byte (the strip step changes some of Qualcomm's libraries).
      useLegacyPackaging = true
      keepDebugSymbols += "**/*.so"
    }
  }

  if (npuLibs != null) {
    sourceSets["main"].jniLibs.srcDir(npuStage)
  }

  androidResources { noCompress += listOf("json", "bin") }
}

kotlin {
  compilerOptions {
    jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    // LiteRT 2.3.0's Kotlin API carries Kotlin 2.4 metadata; Kotlin 2.2.21 reads it with this flag.
    if (litertAtLeast23) freeCompilerArgs.add("-Xskip-metadata-version-check")
  }
}

if (npuLibs != null) {
  // Sync, not Copy: the stage keeps nothing from an earlier build with other libraries (a 2.2.0 dispatch library left
  // there went into a 2.3.0 APK and crashed in LiteRtDispatchDelegateGetHooks)
  val stageNpuLibs by tasks.registering(Sync::class) {
    from(npuLibs) {
      include("**/libQnn*.so")
      if (!litertAtLeast23) include("**/libLiteRtDispatch_Qualcomm.so", "**/libLiteRtCompilerPlugin_Qualcomm.so")
      eachFile { relativePath = RelativePath(true, name) }
    }
    includeEmptyDirs = false
    into(npuStage.map { it.dir("arm64-v8a") })
  }
  tasks.named("preBuild") { dependsOn(stageNpuLibs) }
}

dependencies {
  implementation("com.google.ai.edge.litert:litert:$litertVersion")
  // From 2.3.0 the GPU accelerator is its own artifact.
  if (litertAtLeast23) implementation("com.google.ai.edge.litert:litert-gpu:$litertVersion")
  // LiteRT's Qualcomm dispatch library and JIT compiler plugin (no QNN library inside): only in the NPU build.
  if (litertAtLeast23 && npuLibs != null) implementation("com.google.ai.edge.litert:litert-npu-runtime-qualcomm:$litertVersion")
  implementation("androidx.core:core-ktx:1.15.0")
}
