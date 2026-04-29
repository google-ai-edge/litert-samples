package com.example.mobilenetlitert

import android.os.Build

enum class BackendPolicy {
  CPU_EMULATOR,
  NPU_REQUIRED,
  AUTO,
}

object DevicePolicy {
  fun defaultBackendPolicy(): BackendPolicy {
    return if (isProbablyEmulator()) {
      BackendPolicy.CPU_EMULATOR
    } else {
      BackendPolicy.AUTO
    }
  }

  fun isProbablyEmulator(): Boolean {
    val fingerprint = Build.FINGERPRINT.lowercase()
    val model = Build.MODEL.lowercase()
    val manufacturer = Build.MANUFACTURER.lowercase()
    val brand = Build.BRAND.lowercase()
    val device = Build.DEVICE.lowercase()
    val product = Build.PRODUCT.lowercase()
    val hardware = Build.HARDWARE.lowercase()

    return fingerprint.startsWith("generic") ||
      fingerprint.contains("emulator") ||
      model.contains("sdk") ||
      model.contains("emulator") ||
      model.contains("android sdk built for") ||
      manufacturer.contains("genymotion") ||
      hardware.contains("goldfish") ||
      hardware.contains("ranchu") ||
      (brand.startsWith("generic") && device.startsWith("generic")) ||
      product.contains("sdk") ||
      product.contains("emulator")
  }
}
