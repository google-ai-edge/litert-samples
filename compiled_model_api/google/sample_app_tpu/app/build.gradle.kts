import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.kotlin.android)
}

android {
  namespace = "com.google.edgetpu.edgeTPUApp"
  compileSdk = 36

  defaultConfig {
    applicationId = "com.google.edgetpu.edgeTPUApp"
    minSdk = 26
    targetSdk = 36
    versionCode = 1
    versionName = "1.0"

    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    ndk {
      abiFilters += "arm64-v8a"
    }
  }

  packaging {
    jniLibs {
      useLegacyPackaging = false
      pickFirst("**/libLiteRt.so")
      pickFirst("**/libLiteRtDispatch_GoogleTensor.so")
      pickFirst("**/libLiteRtClGlAccelerator.so")
    }
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

  
  testOptions {
    unitTests {
      isIncludeAndroidResources = true
    }
  }
}
configurations.all {
    exclude(group = "com.google.ai.edge.litert", module = "litert-api")
}

// Migrate to the compilerOptions DSL as required by Kotlin 2.0+
tasks.withType<KotlinCompile>().configureEach {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_11)
    }
}

dependencies {
  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.appcompat)
  implementation(libs.material)
  implementation(libs.litertlm.android)
  implementation(libs.litert)
  implementation(libs.androidx.activity.ktx) // Add this line
  implementation(libs.androidx.camera.core)
  implementation(libs.androidx.camera.camera2)
  implementation(libs.androidx.camera.lifecycle)
  implementation(libs.androidx.camera.view)

  testImplementation(libs.junit)
  testImplementation("org.mockito:mockito-core:5.11.0")
  testImplementation("org.mockito.kotlin:mockito-kotlin:5.2.1")
  testImplementation("org.robolectric:robolectric:4.12.1")
  testImplementation("androidx.test:core:1.5.0")
  testImplementation("org.json:json:20231013")
}