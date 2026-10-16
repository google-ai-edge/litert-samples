/*
 * Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.examples.depth_estimation

import android.graphics.Bitmap

/**
 * Colorizes depth maps into ARGB bitmaps. MiDaS inverse-depth uses an `inferno` LUT (larger =
 * nearer = brighter); Depth Anything 3 uses disparity + a `Spectral` LUT (the model's official
 * visualization).
 */
object DepthColorMap {

  /** MiDaS: min-max normalize inverse depth, map through the `inferno` LUT. */
  fun inverseDepthInferno(depth: FloatArray, width: Int, height: Int): Bitmap {
    var min = Float.MAX_VALUE
    var max = -Float.MAX_VALUE
    for (v in depth) {
      if (v < min) {
        min = v
      }
      if (v > max) {
        max = v
      }
    }
    val range = (max - min).coerceAtLeast(1e-8f)
    val pixels = IntArray(width * height)
    for (i in depth.indices) {
      val t = (depth[i] - min) / range
      val idx = (t * 255f).toInt().coerceIn(0, 255)
      pixels[i] = INFERNO[idx]
    }
    return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
  }

  /**
   * Depth Anything 3: crop to the content region, convert depth -> disparity (1/depth), robustly
   * normalize on the 2nd/98th percentiles (avoids outlier wash-out), invert, then map through the
   * `Spectral` LUT — matching the official DA3 `visualize.py`.
   */
  fun disparitySpectral(
    depth: FloatArray,
    width: Int,
    height: Int,
    cropL: Int,
    cropT: Int,
    cropR: Int,
    cropB: Int,
  ): Bitmap {
    val cw = (cropR - cropL).coerceAtLeast(1)
    val ch = (cropB - cropT).coerceAtLeast(1)
    val disp = FloatArray(cw * ch)
    var k = 0
    for (y in cropT until cropB) {
      for (x in cropL until cropR) {
        val d = depth[y * width + x]
        disp[k++] = if (d > 0f) 1f / d else 0f
      }
    }
    val sorted = disp.copyOf()
    sorted.sort()
    val lo = sorted[(0.02f * (sorted.size - 1)).toInt()]
    val hi = sorted[(0.98f * (sorted.size - 1)).toInt()]
    val range = if (hi > lo) hi - lo else 1e-6f
    val pixels = IntArray(cw * ch)
    for (i in disp.indices) {
      val n = 1f - ((disp[i] - lo) / range).coerceIn(0f, 1f)
      pixels[i] = SPECTRAL[(n * 255f).toInt().coerceIn(0, 255)]
    }
    return Bitmap.createBitmap(pixels, cw, ch, Bitmap.Config.ARGB_8888)
  }

  /** matplotlib "Spectral" colormap, 256 RGB entries (the DA3 default depth colormap). */
  private const val SPECTRAL_HEX =
    "9e0142a00343a20643a40844a70b44a90d45ab0f45ad1246" +
      "af1446b11747b41947b61b48b81e48ba2049bc2249be254a" +
      "c1274ac32a4bc52c4bc72e4cc9314ccb334dcd364dd0384e" +
      "d23a4ed43d4fd63f4fd7414ed8434ed9444dda464ddc484c" +
      "dd4a4cde4c4bdf4e4be1504be2514ae3534ae45549e55749" +
      "e75948e85b48e95c47ea5e47eb6046ed6246ee6445ef6645" +
      "f06744f26944f36b43f46d43f47044f57245f57547f57748" +
      "f67a49f67c4af67f4bf7814cf7844ef8864ff88950f88c51" +
      "f98e52f99153f99355fa9656fa9857fa9b58fb9d59fba05b" +
      "fba35cfca55dfca85efcaa5ffdad60fdaf62fdb163fdb365" +
      "fdb567fdb768fdb96afdbb6cfdbd6dfdbf6ffdc171fdc372" +
      "fdc574fdc776fec877feca79fecc7bfece7cfed07efed27f" +
      "fed481fed683fed884feda86fedc88fede89fee08bfee18d" +
      "fee28ffee491fee593fee695fee797fee999feea9bfeeb9d" +
      "feec9ffeeda1feefa3fff0a6fff1a8fff2aafff3acfff5ae" +
      "fff6b0fff7b2fff8b4fffab6fffbb8fffcbafffdbcfffebe" +
      "ffffbefefebdfdfebbfcfebafbfdb8fafdb7f9fcb5f8fcb4" +
      "f7fcb2f6fbb0f5fbaff4faadf3faacf2faaaf1f9a9f0f9a7" +
      "eff9a6eef8a4edf8a3ecf7a1ebf7a0eaf79ee9f69de8f69b" +
      "e7f59ae6f598e4f498e1f399dff299ddf19adaf09ad8ef9b" +
      "d6ee9bd3ed9cd1ed9ccfec9dcdeb9dcaea9ec8e99ec6e89f" +
      "c3e79fc1e6a0bfe5a0bce4a0bae3a1b8e2a1b5e1a2b3e0a2" +
      "b1dfa3aedea3acdda4aadca4a7dba4a4daa4a2d9a49fd8a4" +
      "9cd7a499d6a497d5a494d4a491d3a48fd2a48cd1a489d0a4" +
      "86cfa584cea581cda57ecca57ccaa579c9a576c8a574c7a5" +
      "71c6a56ec5a56bc4a569c3a566c2a564c0a662bda760bba8" +
      "5eb9a95cb7aa5ab4ab58b2ac56b0ad54aead52abae50a9af" +
      "4ea7b04ba4b149a2b247a0b3459eb4439bb54199b63f97b7" +
      "3d95b83b92b93990ba378ebb358bbc3389bd3387bc3585bb" +
      "3682ba3880b93a7eb83b7cb73d79b63f77b54175b44273b3" +
      "4471b2466eb1486cb0496aaf4b68ae4d65ad4e63ac5061aa" +
      "525fa9545ca8555aa75758a65956a55b53a45c51a35e4fa2"

  private val SPECTRAL =
    IntArray(256) { i ->
      val r = SPECTRAL_HEX.substring(i * 6, i * 6 + 2).toInt(16)
      val g = SPECTRAL_HEX.substring(i * 6 + 2, i * 6 + 4).toInt(16)
      val b = SPECTRAL_HEX.substring(i * 6 + 4, i * 6 + 6).toInt(16)
      (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }

  /** 256-entry inferno color LUT (ARGB). */
  private val INFERNO =
    intArrayOf(
      0xFF000004.toInt(), 0xFF010005.toInt(), 0xFF010106.toInt(), 0xFF010108.toInt(),
      0xFF02010A.toInt(), 0xFF02020C.toInt(), 0xFF02020E.toInt(), 0xFF030210.toInt(),
      0xFF040312.toInt(), 0xFF040314.toInt(), 0xFF050417.toInt(), 0xFF060419.toInt(),
      0xFF07051B.toInt(), 0xFF08051D.toInt(), 0xFF09061F.toInt(), 0xFF0A0722.toInt(),
      0xFF0B0724.toInt(), 0xFF0C0826.toInt(), 0xFF0D0829.toInt(), 0xFF0E092B.toInt(),
      0xFF10092D.toInt(), 0xFF110A30.toInt(), 0xFF120A32.toInt(), 0xFF140B34.toInt(),
      0xFF150B37.toInt(), 0xFF160B39.toInt(), 0xFF180C3C.toInt(), 0xFF190C3E.toInt(),
      0xFF1B0C41.toInt(), 0xFF1C0C43.toInt(), 0xFF1E0C45.toInt(), 0xFF1F0C48.toInt(),
      0xFF210C4A.toInt(), 0xFF230C4C.toInt(), 0xFF240C4F.toInt(), 0xFF260C51.toInt(),
      0xFF280B53.toInt(), 0xFF290B55.toInt(), 0xFF2B0B57.toInt(), 0xFF2D0B59.toInt(),
      0xFF2F0A5B.toInt(), 0xFF310A5C.toInt(), 0xFF320A5E.toInt(), 0xFF340A5F.toInt(),
      0xFF360961.toInt(), 0xFF380962.toInt(), 0xFF390963.toInt(), 0xFF3B0964.toInt(),
      0xFF3D0965.toInt(), 0xFF3E0966.toInt(), 0xFF400A67.toInt(), 0xFF420A68.toInt(),
      0xFF440A68.toInt(), 0xFF450A69.toInt(), 0xFF470B6A.toInt(), 0xFF490B6A.toInt(),
      0xFF4A0C6B.toInt(), 0xFF4C0C6B.toInt(), 0xFF4D0D6C.toInt(), 0xFF4F0D6C.toInt(),
      0xFF510E6C.toInt(), 0xFF520E6D.toInt(), 0xFF540F6D.toInt(), 0xFF550F6D.toInt(),
      0xFF57106E.toInt(), 0xFF59106E.toInt(), 0xFF5A116E.toInt(), 0xFF5C126E.toInt(),
      0xFF5D126E.toInt(), 0xFF5F136E.toInt(), 0xFF61136E.toInt(), 0xFF62146E.toInt(),
      0xFF64156E.toInt(), 0xFF65156E.toInt(), 0xFF67166E.toInt(), 0xFF69166E.toInt(),
      0xFF6A176E.toInt(), 0xFF6C186E.toInt(), 0xFF6D186E.toInt(), 0xFF6F196E.toInt(),
      0xFF71196E.toInt(), 0xFF721A6E.toInt(), 0xFF741A6E.toInt(), 0xFF751B6E.toInt(),
      0xFF771C6D.toInt(), 0xFF781C6D.toInt(), 0xFF7A1D6D.toInt(), 0xFF7C1D6D.toInt(),
      0xFF7D1E6D.toInt(), 0xFF7F1E6C.toInt(), 0xFF801F6C.toInt(), 0xFF82206C.toInt(),
      0xFF84206B.toInt(), 0xFF85216B.toInt(), 0xFF87216B.toInt(), 0xFF88226A.toInt(),
      0xFF8A226A.toInt(), 0xFF8C2369.toInt(), 0xFF8D2369.toInt(), 0xFF8F2469.toInt(),
      0xFF902568.toInt(), 0xFF922568.toInt(), 0xFF932667.toInt(), 0xFF952667.toInt(),
      0xFF972766.toInt(), 0xFF982766.toInt(), 0xFF9A2865.toInt(), 0xFF9B2964.toInt(),
      0xFF9D2964.toInt(), 0xFF9F2A63.toInt(), 0xFFA02A63.toInt(), 0xFFA22B62.toInt(),
      0xFFA32C61.toInt(), 0xFFA52C60.toInt(), 0xFFA62D60.toInt(), 0xFFA82E5F.toInt(),
      0xFFA92E5E.toInt(), 0xFFAB2F5E.toInt(), 0xFFAD305D.toInt(), 0xFFAE305C.toInt(),
      0xFFB0315B.toInt(), 0xFFB1325A.toInt(), 0xFFB3325A.toInt(), 0xFFB43359.toInt(),
      0xFFB63458.toInt(), 0xFFB73557.toInt(), 0xFFB93556.toInt(), 0xFFBA3655.toInt(),
      0xFFBC3754.toInt(), 0xFFBD3853.toInt(), 0xFFBF3952.toInt(), 0xFFC03A51.toInt(),
      0xFFC13A50.toInt(), 0xFFC33B4F.toInt(), 0xFFC43C4E.toInt(), 0xFFC63D4D.toInt(),
      0xFFC73E4C.toInt(), 0xFFC83F4B.toInt(), 0xFFCA404A.toInt(), 0xFFCB4149.toInt(),
      0xFFCC4248.toInt(), 0xFFCE4347.toInt(), 0xFFCF4446.toInt(), 0xFFD04545.toInt(),
      0xFFD24644.toInt(), 0xFFD34743.toInt(), 0xFFD44842.toInt(), 0xFFD54A41.toInt(),
      0xFFD74B3F.toInt(), 0xFFD84C3E.toInt(), 0xFFD94D3D.toInt(), 0xFFDA4E3C.toInt(),
      0xFFDB503B.toInt(), 0xFFDD513A.toInt(), 0xFFDE5238.toInt(), 0xFFDF5337.toInt(),
      0xFFE05536.toInt(), 0xFFE15635.toInt(), 0xFFE25734.toInt(), 0xFFE35933.toInt(),
      0xFFE45A31.toInt(), 0xFFE55C30.toInt(), 0xFFE65D2F.toInt(), 0xFFE75E2E.toInt(),
      0xFFE8602D.toInt(), 0xFFE9612B.toInt(), 0xFFEA632A.toInt(), 0xFFEB6429.toInt(),
      0xFFEB6628.toInt(), 0xFFEC6726.toInt(), 0xFFED6925.toInt(), 0xFFEE6A24.toInt(),
      0xFFEF6C23.toInt(), 0xFFEF6E21.toInt(), 0xFFF06F20.toInt(), 0xFFF1711F.toInt(),
      0xFFF1731D.toInt(), 0xFFF2741C.toInt(), 0xFFF3761B.toInt(), 0xFFF37819.toInt(),
      0xFFF47918.toInt(), 0xFFF57B17.toInt(), 0xFFF57D15.toInt(), 0xFFF67E14.toInt(),
      0xFFF68013.toInt(), 0xFFF78212.toInt(), 0xFFF78410.toInt(), 0xFFF8850F.toInt(),
      0xFFF8870E.toInt(), 0xFFF8890C.toInt(), 0xFFF98B0B.toInt(), 0xFFF98C0A.toInt(),
      0xFFF98E09.toInt(), 0xFFFA9008.toInt(), 0xFFFA9207.toInt(), 0xFFFA9407.toInt(),
      0xFFFB9606.toInt(), 0xFFFB9706.toInt(), 0xFFFB9906.toInt(), 0xFFFB9B06.toInt(),
      0xFFFB9D07.toInt(), 0xFFFC9F07.toInt(), 0xFFFCA108.toInt(), 0xFFFCA309.toInt(),
      0xFFFCA50A.toInt(), 0xFFFCA60C.toInt(), 0xFFFCA80D.toInt(), 0xFFFCAA0F.toInt(),
      0xFFFCAC11.toInt(), 0xFFFCAE12.toInt(), 0xFFFCB014.toInt(), 0xFFFCB216.toInt(),
      0xFFFCB418.toInt(), 0xFFFBB61A.toInt(), 0xFFFBB81D.toInt(), 0xFFFBBA1F.toInt(),
      0xFFFBBC21.toInt(), 0xFFFBBE23.toInt(), 0xFFFAC026.toInt(), 0xFFFAC228.toInt(),
      0xFFFAC42A.toInt(), 0xFFFAC62D.toInt(), 0xFFF9C72F.toInt(), 0xFFF9C932.toInt(),
      0xFFF9CB35.toInt(), 0xFFF8CD37.toInt(), 0xFFF8CF3A.toInt(), 0xFFF7D13D.toInt(),
      0xFFF7D340.toInt(), 0xFFF6D543.toInt(), 0xFFF6D746.toInt(), 0xFFF5D949.toInt(),
      0xFFF5DB4C.toInt(), 0xFFF4DD4F.toInt(), 0xFFF4DF53.toInt(), 0xFFF4E156.toInt(),
      0xFFF3E35A.toInt(), 0xFFF3E55D.toInt(), 0xFFF2E661.toInt(), 0xFFF2E865.toInt(),
      0xFFF2EA69.toInt(), 0xFFF1EC6D.toInt(), 0xFFF1ED71.toInt(), 0xFFF1EF75.toInt(),
      0xFFF1F179.toInt(), 0xFFF2F27D.toInt(), 0xFFF2F482.toInt(), 0xFFF3F586.toInt(),
      0xFFF3F68A.toInt(), 0xFFF4F88E.toInt(), 0xFFF5F992.toInt(), 0xFFF6FA96.toInt(),
      0xFFF8FB9A.toInt(), 0xFFF9FC9D.toInt(), 0xFFFAFDA1.toInt(), 0xFFFCFFA4.toInt(),
    )
}
