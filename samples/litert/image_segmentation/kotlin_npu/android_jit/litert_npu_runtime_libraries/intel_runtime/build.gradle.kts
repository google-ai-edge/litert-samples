plugins { id("com.android.dynamic-feature") }

android {
  namespace = "com.google.ai.edge.litert.intel_runtime"
  compileSdk = 35

  defaultConfig { minSdk = 31 }

  sourceSets {
    getByName("main") {
      // let gradle pack the shared library into apk
      jniLibs.srcDirs("src/main/jni")
    }
  }
}

dependencies {
  implementation(project(":app"))
  // Provides the Intel LiteRT plugins; the fetch script supplies their native dependencies.
  // Keep the plugin version aligned with LiteRT.
  implementation(
    "com.google.ai.edge.litert:litert-npu-runtime-intel-openvino:${libs.versions.litert.get()}"
  )
}
