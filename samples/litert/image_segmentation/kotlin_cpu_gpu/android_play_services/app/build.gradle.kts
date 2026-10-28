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

// Image segmentation sample that runs on the LiteRT runtime provided by Google Play services.
//
// Example:
//   ./gradlew build

plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.jetbrains.kotlin.android)
  alias(libs.plugins.undercouch.download)
  alias(libs.plugins.compose.compiler)
}

android {
  namespace = "com.google.ai.edge.examples.image_segmentation"
  compileSdk = 36

  defaultConfig {
    // A distinct application ID, so this sample can be installed next to the bundled-runtime one.
    applicationId = "com.google.ai.edge.examples.image_segmentation.playservices"
    minSdk = 24
    targetSdk = 33
    versionCode = 1
    versionName = "1.0"

    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    vectorDrawables { useSupportLibrary = true }
  }

  buildTypes {
    release {
      // Sign release builds with the debug key so that `./gradlew :app:installRelease` works out of
      // the box, which makes it easy to measure the optimized build. Real apps must sign release
      // builds with their own key.
      signingConfig = signingConfigs.getByName("debug")
      isMinifyEnabled = false
      proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
    }
  }
  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_1_8
    targetCompatibility = JavaVersion.VERSION_1_8
  }
  buildFeatures { compose = true }
  packaging { resources { excludes += "/META-INF/{AL2.0,LGPL2.1}" } }

  // Disable lint analysis to avoid build failures due to lint errors.
  lint {
    disable.add("CoroutineCreationDuringComposition")
    disable.add("FlowOperatorInvokedInComposition")
    disable.add("Aligned16KB")
  }
}

kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_1_8) } }

// Import DownloadModels task
project.extensions.extraProperties["ASSET_DIR"] = "$projectDir/src/main/assets"
apply(from = "download_model.gradle")

dependencies {
  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.lifecycle.runtime.ktx)
  implementation(libs.androidx.lifecycle.runtime.compose)
  implementation(libs.androidx.lifecycle.viewmodel.compose)
  implementation(libs.androidx.activity.compose)
  implementation(platform(libs.androidx.compose.bom))
  implementation(libs.androidx.ui)
  implementation(libs.androidx.ui.graphics)
  implementation(libs.androidx.ui.tooling.preview)
  implementation(libs.androidx.material.icons.core)
  implementation(libs.androidx.material.icons.extended)
  implementation(libs.androidx.material2)
  // LiteRT CompiledModel API, backed by the runtime in Google Play services.
  implementation(libs.play.services.litert)
  implementation(libs.kotlinx.coroutines.play.services)
  implementation(libs.androidx.camera.core)
  implementation(libs.androidx.camera.lifecycle)
  implementation(libs.androidx.camera.view)
  implementation(libs.androidx.camera.camera2)
  implementation(libs.coil.compose)
  implementation(libs.androidx.compose.runtime.livedata)
  implementation(platform("org.jetbrains.kotlinx:kotlinx-coroutines-bom:1.10.2"))
  implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android")

  testImplementation(libs.junit)
  androidTestImplementation(libs.androidx.junit)
  androidTestImplementation(libs.androidx.espresso.core)
  androidTestImplementation(platform(libs.androidx.compose.bom))
  androidTestImplementation(libs.androidx.ui.test.junit4)
  debugImplementation(libs.androidx.ui.tooling)
  debugImplementation(libs.androidx.ui.test.manifest)
}
