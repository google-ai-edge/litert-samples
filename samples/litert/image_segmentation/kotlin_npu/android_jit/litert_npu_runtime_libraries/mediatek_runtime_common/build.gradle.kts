plugins { id("com.android.dynamic-feature") }

android {
  namespace = "com.google.ai.edge.litert.mediatek_runtime.common"
  compileSdk = 35

  defaultConfig { minSdk = 31 }
}

dependencies {
  implementation(project(":app"))
  // Provides libLiteRtDispatch_MediaTek.so and libLiteRtCompilerPlugin_MediaTek.so, which are
  // shared by every `mediatek_runtime_v*` module. They live in this dedicated module because an
  // Android App Bundle rejects the same library being packaged by more than one feature module.
  // The version must match the LiteRT runtime, so it reuses the `litert` version alias from the
  // app's `gradle/libs.versions.toml`, e.g. `[versions] litert = "2.3.0"`.
  implementation(
    "com.google.ai.edge.litert:litert-npu-runtime-mediatek:${libs.versions.litert.get()}"
  )
}
