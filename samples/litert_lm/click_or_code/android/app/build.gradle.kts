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
  alias(libs.plugins.kotlin.android)
  alias(libs.plugins.kotlin.compose)
}

android {
  namespace = "com.google.ai.edge.examples.click_or_code"
  compileSdk = 36
  defaultConfig {
    applicationId = "com.google.ai.edge.examples.click_or_code"
    minSdk = 31
    targetSdk = 36
    versionCode = 1
    versionName = "1.0.0"
    buildConfigField("String", "LITERTLM_VERSION", "\"${libs.versions.litertlm.get()}\"")
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

kotlin {
  compilerOptions {
    jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    // litertlm-android 0.19.0 brings kotlin-reflect 2.4.0 (and with it kotlin-stdlib 2.4.0), whose
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
