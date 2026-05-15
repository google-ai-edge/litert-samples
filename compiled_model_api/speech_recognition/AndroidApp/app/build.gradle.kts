plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.kotlin.android)
  alias(libs.plugins.download)
  id("org.jetbrains.kotlin.plugin.serialization") version "2.2.10"
}

android {
  namespace = "com.google.ai.edge.examples.asr"
  compileSdk = 35

  testOptions { unitTests.isReturnDefaultValues = true }

  defaultConfig {
    applicationId = "com.google.ai.edge.examples.asr"
    minSdk = 31
    targetSdk = 35
    versionCode = 1
    versionName = "1.0"

    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

    ndk {
      // Chaquopy requires specific ABIs
      abiFilters += listOf("arm64-v8a")
    }
    // Needed for Qualcomm NPU runtimes
    packaging { jniLibs { useLegacyPackaging = true } }
  }

  buildTypes {
    release {
      isMinifyEnabled = false
      proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
    }
  }
  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_11
    targetCompatibility = JavaVersion.VERSION_11
  }
  kotlinOptions { jvmTarget = "11" }
  buildFeatures { viewBinding = true }

  externalNativeBuild {
    cmake {
      path = file("src/main/cpp/CMakeLists.txt")
      version = "3.22.1"
    }
  }

  // NPU runtime libraries. Uncomment the ones you need.
  // dynamicFeatures.add(":litert_npu_runtime_libraries:google_tensor_runtime")
  // dynamicFeatures.add(":litert_npu_runtime_libraries:mediatek_runtime_v8")
  // dynamicFeatures.add(":litert_npu_runtime_libraries:mediatek_runtime_v9")
  // dynamicFeatures.add(":litert_npu_runtime_libraries:qualcomm_runtime_v69")
  // dynamicFeatures.add(":litert_npu_runtime_libraries:qualcomm_runtime_v73")
  // dynamicFeatures.add(":litert_npu_runtime_libraries:qualcomm_runtime_v75")
  // dynamicFeatures.add(":litert_npu_runtime_libraries:qualcomm_runtime_v79")
  // dynamicFeatures.add(":litert_npu_runtime_libraries:qualcomm_runtime_v81")

  bundle {
    deviceTargetingConfig = file("device_targeting_configuration.xml")
    deviceGroup {
      enableSplit = true // split bundle by #group
      defaultGroup = "other" // group used for standalone APKs
    }
  }
}

tasks.register<de.undercouch.gradle.tasks.download.Download>("downloadJLibrosa") {
  src(
    "https://github.com/Subtitle-Synchronizer/jlibrosa/raw/master/binaries/jlibrosa-1.1.8-SNAPSHOT-jar-with-dependencies.jar"
  )
  dest(file("libs/jlibrosa-1.1.8-SNAPSHOT-jar-with-dependencies.jar"))
  overwrite(false)
}

tasks.named("preBuild") { dependsOn("downloadJLibrosa") }

dependencies {
  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.appcompat)
  implementation(libs.material)
  implementation(libs.androidx.constraintlayout)
  implementation(libs.androidx.navigation.fragment.ktx)
  implementation(libs.androidx.navigation.ui.ktx)
  implementation("com.google.ai.edge.litert:litert:2.1.5")
  implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")
  implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
  implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.8.0")
  implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.6.2")
  implementation(files("libs/jlibrosa-1.1.8-SNAPSHOT-jar-with-dependencies.jar"))
  implementation(project(":litert_npu_runtime_libraries:runtime_strings"))
  testImplementation(libs.junit)
  testImplementation("org.json:json:20231013")
  androidTestImplementation(libs.androidx.junit)
  androidTestImplementation(libs.androidx.espresso.core)
}
