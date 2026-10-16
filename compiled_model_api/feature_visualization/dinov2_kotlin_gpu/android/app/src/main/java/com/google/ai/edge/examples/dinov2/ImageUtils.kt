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

package com.google.ai.edge.examples.dinov2

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri

/** Decodes a bitmap bundled in the app's assets. */
fun Context.decodeAssetBitmap(assetName: String): Bitmap =
  assets.open(assetName).use { BitmapFactory.decodeStream(it) }
    ?: error("Cannot decode asset image: $assetName")

/** Decodes a gallery image picked by the user. */
fun Context.decodeUriBitmap(uri: Uri): Bitmap =
  contentResolver.openInputStream(uri).use { BitmapFactory.decodeStream(it) }
    ?: error("Cannot decode image: $uri")
