plugins { id("com.android.dynamic-feature") }

android {
  namespace = "com.google.ai.edge.litert.qualcomm_runtime.v75"
  compileSdk = 35

  defaultConfig { minSdk = 31 }

  sourceSets {
    getByName("main") {
      // let gradle pack the shared library into apk
      jniLibs.srcDirs("src/main/jni")
    }
  }
}

dependencies { implementation(project(":app")) }
