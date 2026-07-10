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

package com.google.ai.edge.examples.flux2_klein

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.ai.edge.examples.flux2_klein.view.ApplicationTheme
import com.google.ai.edge.examples.flux2_klein.view.Flux2KleinScreen

/**
 * On-device FLUX.2-klein-4B text-to-image and image editing on the LiteRT CompiledModel GPU. The 4B
 * transformer and its 4B text encoder run as sequentially-resident int8 chunks (see
 * [Flux2KleinGenerator]). This is a thin Compose host over [MainViewModel].
 */
class MainActivity : ComponentActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }
    setContent {
      ApplicationTheme {
        val uiState by viewModel.uiState.collectAsStateWithLifecycle()
        val picker =
          rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
            decodeBitmap(uri)?.let { viewModel.generate(it) }
          }
        Flux2KleinScreen(
          uiState = uiState,
          onGenerate = { viewModel.generate() },
          onPickImage = {
            picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
          },
        )
      }
    }
  }

  /**
   * Decodes the picked image into a software bitmap.
   *
   * `BitmapFactory` rather than `ImageDecoder`: the generator reads the pixels back with
   * `getPixels`, which a hardware bitmap does not support.
   */
  private fun decodeBitmap(uri: Uri?): Bitmap? =
    uri?.let { contentResolver.openInputStream(it)?.use(BitmapFactory::decodeStream) }
}
