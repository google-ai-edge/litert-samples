/*
 * Copyright 2024 The Google AI Edge Authors. All Rights Reserved.
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

package com.google.edgeai.examples.super_resolution.quicksample

import android.graphics.BitmapFactory
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.Button
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.AsyncImage
import com.google.edgeai.examples.super_resolution.ImageSuperResolutionHelper
import java.io.InputStream

@Composable
fun QuickSampleScreen(
    modifier: Modifier = Modifier,
    delegate: ImageSuperResolutionHelper.Delegate,
    onInferenceTimeCallback: (Int) -> Unit,
) {
    val context = LocalContext.current
    val viewModel: QuickSampleViewModel =
        viewModel(factory = QuickSampleViewModel.getFactory(context))

    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val selectBitmap = uiState.selectBitmap
    val sharpenBitmap = uiState.sharpenBitmap

    LaunchedEffect(key1 = uiState.inferenceTime) {
        onInferenceTimeCallback(uiState.inferenceTime)
    }

    LaunchedEffect(key1 = delegate) {
        viewModel.setDelegate(delegate)
    }

    Column(modifier.padding(8.dp)) {
        Text(text = "Choose a low resolution image below:")
        Spacer(modifier = Modifier.height(5.dp))

        Row(modifier) {
            uiState.sampleUriList.forEach {
                AsyncImage(
                    modifier = Modifier
                        .size(70.dp)
                        .clickable {
                            val assetManager = context.assets
                            val inputStream: InputStream =
                                assetManager.open(it)
                            val bitmap = BitmapFactory.decodeStream(inputStream)
                            viewModel.selectImage(bitmap)
                        },
                    model = "file:///android_asset/$it",
                    contentDescription = null
                )
            }
        }

        Text(
            text = "Click UPSAMPLE button below will we use TFLite to generate a corresponding " +
                    "high resolution image based n your chosen image"
        )
        Spacer(modifier = Modifier.height(5.dp))

        Row {
            if (selectBitmap != null) {
                AsyncImage(
                    modifier = Modifier.size(150.dp),
                    model = selectBitmap,
                    contentDescription = null,
                )
            }

            Spacer(modifier = Modifier.width(10.dp))

            if (sharpenBitmap != null) {
                AsyncImage(
                    modifier = Modifier.size(150.dp),
                    model = sharpenBitmap,
                    contentDescription = null,
                )
            }
        }
        if (selectBitmap != null) {
            Button(onClick = {
                viewModel.makeSharpen()
            }) {
                Text(text = "UPSAMPLE")
            }
        }
    }
}