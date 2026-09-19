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

package com.google.ai.edge.examples.face_restoration.view

import android.graphics.Bitmap
import androidx.annotation.StringRes
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.material.Button
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Scaffold
import androidx.compose.material.Text
import androidx.compose.material.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.google.ai.edge.examples.face_restoration.R
import com.google.ai.edge.examples.face_restoration.UiState

/** Top-level restoration screen: a status header, an image picker, and a before/after pair. */
@Composable
fun FaceRestorationScreen(
  uiState: UiState,
  onPickImage: () -> Unit,
  modifier: Modifier = Modifier,
) {
  Scaffold(
    modifier = modifier.statusBarsPadding(),
    topBar = {
      TopAppBar(
        backgroundColor = MaterialTheme.colors.secondary,
        title = { Text(text = stringResource(R.string.app_name), color = Color.White) },
      )
    },
  ) { padding ->
    Column(
      modifier =
        Modifier.fillMaxSize().padding(padding).padding(16.dp)
    ) {
      StatusHeader(uiState)
      Spacer(modifier = Modifier.height(12.dp))
      Button(onClick = onPickImage, enabled = uiState.isModelReady && !uiState.isProcessing) {
        Text(text = stringResource(R.string.action_pick_image))
      }
      Spacer(modifier = Modifier.height(12.dp))
      LabelledImage(R.string.label_original, uiState.beforeImage)
      LabelledImage(R.string.label_restored, uiState.afterImage)
    }
  }
}

@Composable
private fun StatusHeader(uiState: UiState) {
  val statusText =
    when {
      uiState.errorMessage != null -> uiState.errorMessage
      !uiState.isModelReady -> stringResource(R.string.status_loading)
      uiState.isProcessing -> stringResource(R.string.status_processing)
      uiState.afterImage == null -> stringResource(R.string.status_ready)
      else -> stringResource(R.string.status_result, uiState.inferenceTimeMs)
    }
  Text(text = statusText, fontSize = 15.sp, fontWeight = FontWeight.Medium)
}

/** Renders one captioned bitmap, or nothing at all while [bitmap] is still null. */
@Composable
private fun ColumnScope.LabelledImage(@StringRes labelRes: Int, bitmap: Bitmap?) {
  if (bitmap == null) {
    return
  }
  Text(text = stringResource(labelRes), fontSize = 13.sp, fontWeight = FontWeight.Medium)
  Spacer(modifier = Modifier.height(4.dp))
  Image(
    bitmap = bitmap.asImageBitmap(),
    contentDescription = null,
    contentScale = ContentScale.Fit,
    modifier = Modifier.fillMaxWidth().weight(1f),
  )
  Spacer(modifier = Modifier.height(12.dp))
}
