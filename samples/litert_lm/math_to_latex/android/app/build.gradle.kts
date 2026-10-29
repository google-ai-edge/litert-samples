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

import java.net.URI
import java.security.MessageDigest

plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.kotlin.android)
  alias(libs.plugins.kotlin.compose)
}

/**
 * Fetches MathJax's single-file TeX-to-SVG build (Apache-2.0), which draws the model's LaTeX on the
 * phone without the network. The file is checked against its SHA-256 and never committed; once it
 * is in the build directory, builds run offline.
 */
abstract class FetchMathJax : DefaultTask() {
  @get:Input abstract val url: Property<String>
  @get:Input abstract val sha256: Property<String>
  @get:OutputDirectory abstract val outputDir: DirectoryProperty

  @TaskAction
  fun fetch() {
    val target = outputDir.file("mathjax/tex-svg-full.js").get().asFile
    if (target.isFile && digest(target.readBytes()) == sha256.get()) {
      return
    }
    target.parentFile.mkdirs()
    val bytes = URI(url.get()).toURL().openStream().use { it.readBytes() }
    val got = digest(bytes)
    check(got == sha256.get()) { "MathJax from ${url.get()} has SHA-256 $got, not ${sha256.get()}" }
    target.writeBytes(bytes)
  }

  private fun digest(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
}

val fetchMathJax =
  tasks.register<FetchMathJax>("fetchMathJax") {
    url.set("https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg-full.js")
    sha256.set("a4354ff94fd868aea0cc6eaaa79a57fda0588646fc46ee3700a349ee0a11cbe6")
    outputDir.set(layout.buildDirectory.dir("generated/mathjax"))
  }

android {
  namespace = "com.google.ai.edge.examples.math_to_latex"
  compileSdk = 36
  defaultConfig {
    applicationId = "com.google.ai.edge.examples.math_to_latex"
    minSdk = 31
    targetSdk = 36
    versionCode = 1
    versionName = "1.0.0"
    ndk { abiFilters += "arm64-v8a" }
  }
  buildTypes {
    getByName("release") {
      isMinifyEnabled = true
      isShrinkResources = true
      proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
    }
  }
  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }
  buildFeatures {
    compose = true
    buildConfig = true
  }
  // The JVM tests read the packaged model catalog straight from the app assets.
  sourceSets["test"].resources.srcDir("src/main/assets")
  packaging { resources.excludes += setOf("META-INF/AL2.0", "META-INF/LGPL2.1") }
}

androidComponents {
  onVariants { variant ->
    variant.sources.assets?.addGeneratedSourceDirectory(fetchMathJax, FetchMathJax::outputDir)
  }
}

kotlin {
  compilerOptions {
    jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    // litertlm-android 0.18.1 brings kotlin-reflect 2.4.0 (and with it kotlin-stdlib 2.4.0), whose
    // metadata is newer than this compiler's.
    freeCompilerArgs.add("-Xskip-metadata-version-check")
  }
}

dependencies {
  implementation(libs.litertlm)
  implementation(libs.coroutines)
  implementation(libs.exifinterface)
  implementation(libs.activity.compose)
  implementation(libs.lifecycle.viewmodel.compose)
  implementation(platform(libs.compose.bom))
  implementation(libs.compose.ui)
  implementation(libs.compose.material3)
  testImplementation(libs.junit)
  testImplementation(libs.json)
}
