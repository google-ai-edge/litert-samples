plugins { id("com.android.dynamic-feature") }

android {
  namespace = "com.google.ai.edge.litert.qualcomm_runtime.common"
  compileSdk = 35

  defaultConfig {
    minSdk = 31
    ndk {
      abiFilters.add("arm64-v8a")
      abiFilters.add("x86_64")
    }
    externalNativeBuild {
      cmake { arguments += "-DLITERT_ABI_STUB_NAME=litert_${project.name}_abi_stub" }
    }
  }

  externalNativeBuild { cmake { path = file("../abi_stub/CMakeLists.txt") } }

  sourceSets {
    getByName("main") {
      // let gradle pack the shared library into apk
      jniLibs.srcDirs("src/main/jni")
    }
  }
}

dependencies {
  implementation(project(":app"))
  // Provides libLiteRtDispatch_Qualcomm.so and libLiteRtCompilerPlugin_Qualcomm.so, which are
  // shared by every `qualcomm_runtime_v*` module. They live in this dedicated module because an
  // Android App Bundle rejects the same library being packaged by more than one feature module.
  // The version must match the LiteRT runtime, so it reuses the `litert` version alias from the
  // app's `gradle/libs.versions.toml`, e.g. `[versions] litert = "2.3.0"`.
  implementation(
    "com.google.ai.edge.litert:litert-npu-runtime-qualcomm:${libs.versions.litert.get()}"
  )
}
