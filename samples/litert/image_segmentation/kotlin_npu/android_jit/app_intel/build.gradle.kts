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
    applicationId = "com.google.ai.edge.examples.image_segmentation.intel"
    minSdk = 31
    targetSdk = 33
    versionCode = 1
    versionName = "1.0"

    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    vectorDrawables { useSupportLibrary = true }
    ndk { abiFilters.add("x86_64") }
    packaging { jniLibs { useLegacyPackaging = true } }
  }

  buildTypes {
    release {
      isMinifyEnabled = false
      proguardFiles(
        getDefaultProguardFile("proguard-android-optimize.txt"),
        file("../app/proguard-rules.pro"),
      )
    }
  }
  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_1_8
    targetCompatibility = JavaVersion.VERSION_1_8
  }
  buildFeatures { compose = true }
  packaging { resources { excludes += "/META-INF/{AL2.0,LGPL2.1}" } }

  sourceSets {
    getByName("main") {
      java.srcDirs("../app/src/main/java")
      res.srcDirs("../app/src/main/res")
      assets.srcDirs("../app/src/main/assets")
      // let gradle pack the shared library into apk
      jniLibs.srcDirs("src/main/jni")
    }
  }

  lint {
    disable.add("CoroutineCreationDuringComposition")
    disable.add("FlowOperatorInvokedInComposition")
    disable.add("Aligned16KB")
  }
}

project.extensions.extraProperties["ASSET_DIR"] = "$rootDir/app/src/main/assets"
apply(from = "../app/download_model.gradle")

dependencies {
  implementation(libs.litert) {
    exclude(group = "com.google.ai.edge.litert", module = "litert-support")
    exclude(group = "com.google.ai.edge.litert", module = "litert-support-api")
  }
  implementation(libs.litert.support) {
    exclude(group = "com.google.ai.edge.litert", module = "litert-api")
  }
  implementation(libs.litert.gpu)
  implementation(
    "com.google.ai.edge.litert:litert-npu-runtime-intel-openvino:${libs.versions.litert.get()}"
  )

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
  implementation(libs.androidx.camera.core)
  implementation(libs.androidx.camera.lifecycle)
  implementation(libs.androidx.camera.view)
  implementation(libs.androidx.camera.camera2)
  implementation(libs.coil.compose)
  implementation(libs.androidx.compose.runtime.livedata)
  implementation(libs.android.play.ai.delivery)
  implementation(platform(libs.kotlinx.coroutines.bom))
  implementation(libs.kotlinx.coroutines.android)

  testImplementation(libs.junit)
  androidTestImplementation(libs.androidx.junit)
  androidTestImplementation(libs.androidx.espresso.core)
  androidTestImplementation(platform(libs.androidx.compose.bom))
  androidTestImplementation(libs.androidx.ui.test.junit4)
  debugImplementation(libs.androidx.ui.tooling)
  debugImplementation(libs.androidx.ui.test.manifest)
}

tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile>().configureEach {
  compilerOptions {
    jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_1_8)
  }
}