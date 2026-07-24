import de.undercouch.gradle.tasks.download.Download
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.jetbrains.kotlin.android)
  alias(libs.plugins.compose.compiler)
  alias(libs.plugins.undercouch.download)
}

android {
  namespace = "com.example.mobilenetlitert"
  compileSdk = 36

  defaultConfig {
    applicationId = "com.example.mobilenetlitert"
    minSdk = 31
    targetSdk = 36
    versionCode = 1
    versionName = "1.0"

    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    vectorDrawables { useSupportLibrary = true }
  }

  buildTypes {
    release {
      isMinifyEnabled = false
      proguardFiles(
        getDefaultProguardFile("proguard-android-optimize.txt"),
        "proguard-rules.pro",
      )
    }
  }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_1_8
    targetCompatibility = JavaVersion.VERSION_1_8
  }
  kotlin {
    compilerOptions { jvmTarget = JvmTarget.JVM_1_8 }
  }

  buildFeatures {
    buildConfig = true
    compose = true
  }

  packaging {
    jniLibs { useLegacyPackaging = true }
    resources { excludes += "/META-INF/{AL2.0,LGPL2.1}" }
  }
}

val modelZip = layout.buildDirectory.file("downloads/mobilenet_v2-tflite-float.zip")
val assetDir = layout.projectDirectory.dir("src/main/assets")
val sampleImage = layout.projectDirectory.file("src/main/assets/sample/grace_hopper.jpg")

val downloadMobileNet by
  tasks.registering(Download::class) {
    src(
      "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/" +
        "qai-hub-models/models/mobilenet_v2/releases/v0.51.0/" +
        "mobilenet_v2-tflite-float.zip"
    )
    dest(modelZip)
    overwrite(false)
  }

val extractMobileNetAssets by
  tasks.registering(Copy::class) {
    dependsOn(downloadMobileNet)
    from(zipTree(modelZip)) {
      include("mobilenet_v2-tflite-float/mobilenet_v2.tflite")
      eachFile { path = "model/mobilenet_v2_float.tflite" }
      includeEmptyDirs = false
    }
    from(zipTree(modelZip)) {
      include("mobilenet_v2-tflite-float/labels.txt")
      eachFile { path = "labels/imagenet_labels.txt" }
      includeEmptyDirs = false
    }
    into(assetDir)
  }

val downloadSampleImage by
  tasks.registering(Download::class) {
    src("https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
    dest(sampleImage)
    overwrite(false)
  }

tasks.named("preBuild") {
  dependsOn(extractMobileNetAssets)
  dependsOn(downloadSampleImage)
}

dependencies {
  implementation(libs.androidx.activity.compose)
  implementation(platform(libs.androidx.compose.bom))
  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.lifecycle.runtime.compose)
  implementation(libs.androidx.lifecycle.runtime.ktx)
  implementation(libs.androidx.material.icons.extended)
  implementation(libs.androidx.material3)
  implementation(libs.androidx.ui)
  implementation(libs.androidx.ui.graphics)
  implementation(libs.androidx.ui.tooling.preview)
  implementation(libs.kotlinx.coroutines.android)
  implementation(libs.litert)

  testImplementation(libs.junit)
  debugImplementation(libs.androidx.ui.tooling)
  debugImplementation(libs.androidx.ui.test.manifest)
}
