plugins {
  id("com.android.asset-pack")
}

assetPack {
  packName.set("selfie_multiclass")
  dynamicDelivery {
    deliveryType.set("install-time")
  }
}
