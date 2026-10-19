plugins { id("com.android.dynamic-feature") }

android {
  namespace = "com.google.ai.edge.litert.intel_runtime"
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
  // Provides the Intel LiteRT plugins; the fetch script supplies their native dependencies.
  // Keep the plugin version aligned with LiteRT.
  implementation(
    "com.google.ai.edge.litert:litert-npu-runtime-intel-openvino:${libs.versions.litert.get()}"
  )
}