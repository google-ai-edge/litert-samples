/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
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
 * Turns a MiDaS inverse-depth map into a colorized ARGB bitmap using an `inferno`
 * lookup table (matches the Python conversion preview). Larger depth = nearer =
 * brighter.
 */
object DepthColorMap {

  fun toBitmap(depth: FloatArray, width: Int, height: Int): Bitmap {
    var min = Float.MAX_VALUE
    var max = -Float.MAX_VALUE
    for (v in depth) {
      if (v < min) min = v
      if (v > max) max = v
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

  /** 256-entry inferno color LUT (ARGB). */
  private val INFERNO =
    intArrayOf(
    0xFF000004.toInt(), 0xFF010005.toInt(), 0xFF010106.toInt(), 0xFF010108.toInt(), 0xFF02010A.toInt(), 0xFF02020C.toInt(), 0xFF02020E.toInt(), 0xFF030210.toInt(),
    0xFF040312.toInt(), 0xFF040314.toInt(), 0xFF050417.toInt(), 0xFF060419.toInt(), 0xFF07051B.toInt(), 0xFF08051D.toInt(), 0xFF09061F.toInt(), 0xFF0A0722.toInt(),
    0xFF0B0724.toInt(), 0xFF0C0826.toInt(), 0xFF0D0829.toInt(), 0xFF0E092B.toInt(), 0xFF10092D.toInt(), 0xFF110A30.toInt(), 0xFF120A32.toInt(), 0xFF140B34.toInt(),
    0xFF150B37.toInt(), 0xFF160B39.toInt(), 0xFF180C3C.toInt(), 0xFF190C3E.toInt(), 0xFF1B0C41.toInt(), 0xFF1C0C43.toInt(), 0xFF1E0C45.toInt(), 0xFF1F0C48.toInt(),
    0xFF210C4A.toInt(), 0xFF230C4C.toInt(), 0xFF240C4F.toInt(), 0xFF260C51.toInt(), 0xFF280B53.toInt(), 0xFF290B55.toInt(), 0xFF2B0B57.toInt(), 0xFF2D0B59.toInt(),
    0xFF2F0A5B.toInt(), 0xFF310A5C.toInt(), 0xFF320A5E.toInt(), 0xFF340A5F.toInt(), 0xFF360961.toInt(), 0xFF380962.toInt(), 0xFF390963.toInt(), 0xFF3B0964.toInt(),
    0xFF3D0965.toInt(), 0xFF3E0966.toInt(), 0xFF400A67.toInt(), 0xFF420A68.toInt(), 0xFF440A68.toInt(), 0xFF450A69.toInt(), 0xFF470B6A.toInt(), 0xFF490B6A.toInt(),
    0xFF4A0C6B.toInt(), 0xFF4C0C6B.toInt(), 0xFF4D0D6C.toInt(), 0xFF4F0D6C.toInt(), 0xFF510E6C.toInt(), 0xFF520E6D.toInt(), 0xFF540F6D.toInt(), 0xFF550F6D.toInt(),
    0xFF57106E.toInt(), 0xFF59106E.toInt(), 0xFF5A116E.toInt(), 0xFF5C126E.toInt(), 0xFF5D126E.toInt(), 0xFF5F136E.toInt(), 0xFF61136E.toInt(), 0xFF62146E.toInt(),
    0xFF64156E.toInt(), 0xFF65156E.toInt(), 0xFF67166E.toInt(), 0xFF69166E.toInt(), 0xFF6A176E.toInt(), 0xFF6C186E.toInt(), 0xFF6D186E.toInt(), 0xFF6F196E.toInt(),
    0xFF71196E.toInt(), 0xFF721A6E.toInt(), 0xFF741A6E.toInt(), 0xFF751B6E.toInt(), 0xFF771C6D.toInt(), 0xFF781C6D.toInt(), 0xFF7A1D6D.toInt(), 0xFF7C1D6D.toInt(),
    0xFF7D1E6D.toInt(), 0xFF7F1E6C.toInt(), 0xFF801F6C.toInt(), 0xFF82206C.toInt(), 0xFF84206B.toInt(), 0xFF85216B.toInt(), 0xFF87216B.toInt(), 0xFF88226A.toInt(),
    0xFF8A226A.toInt(), 0xFF8C2369.toInt(), 0xFF8D2369.toInt(), 0xFF8F2469.toInt(), 0xFF902568.toInt(), 0xFF922568.toInt(), 0xFF932667.toInt(), 0xFF952667.toInt(),
    0xFF972766.toInt(), 0xFF982766.toInt(), 0xFF9A2865.toInt(), 0xFF9B2964.toInt(), 0xFF9D2964.toInt(), 0xFF9F2A63.toInt(), 0xFFA02A63.toInt(), 0xFFA22B62.toInt(),
    0xFFA32C61.toInt(), 0xFFA52C60.toInt(), 0xFFA62D60.toInt(), 0xFFA82E5F.toInt(), 0xFFA92E5E.toInt(), 0xFFAB2F5E.toInt(), 0xFFAD305D.toInt(), 0xFFAE305C.toInt(),
    0xFFB0315B.toInt(), 0xFFB1325A.toInt(), 0xFFB3325A.toInt(), 0xFFB43359.toInt(), 0xFFB63458.toInt(), 0xFFB73557.toInt(), 0xFFB93556.toInt(), 0xFFBA3655.toInt(),
    0xFFBC3754.toInt(), 0xFFBD3853.toInt(), 0xFFBF3952.toInt(), 0xFFC03A51.toInt(), 0xFFC13A50.toInt(), 0xFFC33B4F.toInt(), 0xFFC43C4E.toInt(), 0xFFC63D4D.toInt(),
    0xFFC73E4C.toInt(), 0xFFC83F4B.toInt(), 0xFFCA404A.toInt(), 0xFFCB4149.toInt(), 0xFFCC4248.toInt(), 0xFFCE4347.toInt(), 0xFFCF4446.toInt(), 0xFFD04545.toInt(),
    0xFFD24644.toInt(), 0xFFD34743.toInt(), 0xFFD44842.toInt(), 0xFFD54A41.toInt(), 0xFFD74B3F.toInt(), 0xFFD84C3E.toInt(), 0xFFD94D3D.toInt(), 0xFFDA4E3C.toInt(),
    0xFFDB503B.toInt(), 0xFFDD513A.toInt(), 0xFFDE5238.toInt(), 0xFFDF5337.toInt(), 0xFFE05536.toInt(), 0xFFE15635.toInt(), 0xFFE25734.toInt(), 0xFFE35933.toInt(),
    0xFFE45A31.toInt(), 0xFFE55C30.toInt(), 0xFFE65D2F.toInt(), 0xFFE75E2E.toInt(), 0xFFE8602D.toInt(), 0xFFE9612B.toInt(), 0xFFEA632A.toInt(), 0xFFEB6429.toInt(),
    0xFFEB6628.toInt(), 0xFFEC6726.toInt(), 0xFFED6925.toInt(), 0xFFEE6A24.toInt(), 0xFFEF6C23.toInt(), 0xFFEF6E21.toInt(), 0xFFF06F20.toInt(), 0xFFF1711F.toInt(),
    0xFFF1731D.toInt(), 0xFFF2741C.toInt(), 0xFFF3761B.toInt(), 0xFFF37819.toInt(), 0xFFF47918.toInt(), 0xFFF57B17.toInt(), 0xFFF57D15.toInt(), 0xFFF67E14.toInt(),
    0xFFF68013.toInt(), 0xFFF78212.toInt(), 0xFFF78410.toInt(), 0xFFF8850F.toInt(), 0xFFF8870E.toInt(), 0xFFF8890C.toInt(), 0xFFF98B0B.toInt(), 0xFFF98C0A.toInt(),
    0xFFF98E09.toInt(), 0xFFFA9008.toInt(), 0xFFFA9207.toInt(), 0xFFFA9407.toInt(), 0xFFFB9606.toInt(), 0xFFFB9706.toInt(), 0xFFFB9906.toInt(), 0xFFFB9B06.toInt(),
    0xFFFB9D07.toInt(), 0xFFFC9F07.toInt(), 0xFFFCA108.toInt(), 0xFFFCA309.toInt(), 0xFFFCA50A.toInt(), 0xFFFCA60C.toInt(), 0xFFFCA80D.toInt(), 0xFFFCAA0F.toInt(),
    0xFFFCAC11.toInt(), 0xFFFCAE12.toInt(), 0xFFFCB014.toInt(), 0xFFFCB216.toInt(), 0xFFFCB418.toInt(), 0xFFFBB61A.toInt(), 0xFFFBB81D.toInt(), 0xFFFBBA1F.toInt(),
    0xFFFBBC21.toInt(), 0xFFFBBE23.toInt(), 0xFFFAC026.toInt(), 0xFFFAC228.toInt(), 0xFFFAC42A.toInt(), 0xFFFAC62D.toInt(), 0xFFF9C72F.toInt(), 0xFFF9C932.toInt(),
    0xFFF9CB35.toInt(), 0xFFF8CD37.toInt(), 0xFFF8CF3A.toInt(), 0xFFF7D13D.toInt(), 0xFFF7D340.toInt(), 0xFFF6D543.toInt(), 0xFFF6D746.toInt(), 0xFFF5D949.toInt(),
    0xFFF5DB4C.toInt(), 0xFFF4DD4F.toInt(), 0xFFF4DF53.toInt(), 0xFFF4E156.toInt(), 0xFFF3E35A.toInt(), 0xFFF3E55D.toInt(), 0xFFF2E661.toInt(), 0xFFF2E865.toInt(),
    0xFFF2EA69.toInt(), 0xFFF1EC6D.toInt(), 0xFFF1ED71.toInt(), 0xFFF1EF75.toInt(), 0xFFF1F179.toInt(), 0xFFF2F27D.toInt(), 0xFFF2F482.toInt(), 0xFFF3F586.toInt(),
    0xFFF3F68A.toInt(), 0xFFF4F88E.toInt(), 0xFFF5F992.toInt(), 0xFFF6FA96.toInt(), 0xFFF8FB9A.toInt(), 0xFFF9FC9D.toInt(), 0xFFFAFDA1.toInt(), 0xFFFCFFA4.toInt(),
    )
}
