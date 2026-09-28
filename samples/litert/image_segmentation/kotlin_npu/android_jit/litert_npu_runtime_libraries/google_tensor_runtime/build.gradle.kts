plugins { id("com.android.dynamic-feature") }

android {
  namespace = "com.google.ai.edge.litert.google_tensor_runtime"
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
  // Provides libLiteRtDispatch_GoogleTensor.so and libLiteRtCompilerPlugin_google_tensor.so.
  // The version must match the LiteRT runtime, so it reuses the `litert` version alias from the
  // app's `gradle/libs.versions.toml`, e.g. `[versions] litert = "2.3.0"`.
  implementation(
    "com.google.ai.edge.litert:litert-npu-runtime-google-tensor:${libs.versions.litert.get()}"
  )
}
