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
  namespace = "com.google.ai.edge.examples.model_zoo"
  compileSdk = 36
  defaultConfig {
    // The id of the published app; the code lives in the samples namespace above.
    applicationId = "com.mlboydaisuke.edgezoo"
    minSdk = 26
    targetSdk = 36
    versionCode = 2
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
  packaging {
    resources.excludes += setOf("META-INF/AL2.0", "META-INF/LGPL2.1")
    jniLibs.pickFirsts += "**/libc++_shared.so"
  }
}

kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }

dependencies {
  implementation(libs.litert)
  implementation(libs.core)
  implementation(libs.activity.compose)
  implementation(libs.lifecycle.viewmodel)
  implementation(libs.lifecycle.compose)
  implementation(platform(libs.compose.bom))
  implementation(libs.compose.ui)
  implementation(libs.compose.tooling.preview)
  implementation(libs.compose.material3)
  implementation(libs.camera.core)
  implementation(libs.camera.camera2)
  implementation(libs.camera.lifecycle)
  implementation(libs.camera.view)
  implementation(libs.coroutines)
  testImplementation(libs.junit)
  testImplementation(libs.json)
}
