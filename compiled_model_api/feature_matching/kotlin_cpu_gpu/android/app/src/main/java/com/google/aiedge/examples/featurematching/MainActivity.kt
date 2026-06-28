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

package com.google.aiedge.examples.featurematching

import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.Button
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Text
import androidx.compose.material.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.aiedge.examples.featurematching.view.ApplicationTheme
import kotlin.math.min
import kotlin.math.roundToInt

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }
        setContent {
            val uiState by viewModel.uiState.collectAsStateWithLifecycle()

            val pickA = rememberLauncherForActivityResult(
                ActivityResultContracts.PickVisualMedia()
            ) { uri: Uri? ->
                if (uri != null) contentResolver.openInputStream(uri)?.use {
                    BitmapFactory.decodeStream(it)?.let { b -> viewModel.setImage(true, b) }
                }
            }
            val pickB = rememberLauncherForActivityResult(
                ActivityResultContracts.PickVisualMedia()
            ) { uri: Uri? ->
                if (uri != null) contentResolver.openInputStream(uri)?.use {
                    BitmapFactory.decodeStream(it)?.let { b -> viewModel.setImage(false, b) }
                }
            }

            LaunchedEffect(uiState.errorMessage) {
                uiState.errorMessage?.let {
                    Toast.makeText(this@MainActivity, it, Toast.LENGTH_SHORT).show()
                    viewModel.errorMessageShown()
                }
            }

            ApplicationTheme {
                Column(Modifier.fillMaxSize()) {
                    TopAppBar(
                        backgroundColor = MaterialTheme.colors.secondary,
                        title = { Text("XFeat Feature Matching") },
                    )
                    Row(
                        Modifier.fillMaxWidth().padding(8.dp),
                        horizontalArrangement = Arrangement.SpaceEvenly,
                    ) {
                        val req = PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                        Button(onClick = { pickA.launch(req) }) { Text("Image A") }
                        Button(onClick = { pickB.launch(req) }) { Text("Image B") }
                    }
                    MatchView(uiState, Modifier.fillMaxWidth().weight(1f))
                    Row(
                        Modifier.fillMaxWidth().padding(12.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                    ) {
                        Text("${uiState.matches.size} matches · ${uiState.inferenceTime} ms")
                        Row {
                            XFeatHelper.AcceleratorEnum.entries.forEach { d ->
                                Button(
                                    onClick = { viewModel.setDelegate(d) },
                                    modifier = Modifier.padding(start = 6.dp),
                                ) { Text(if (d == uiState.delegate) "[${d.name}]" else d.name) }
                            }
                        }
                    }
                }
            }
        }
    }

    /** Draws image A (left) and image B (right) and connects matched keypoints with colored lines. */
    @Composable
    fun MatchView(uiState: UiState, modifier: Modifier = Modifier) {
        val a = uiState.imageA
        val b = uiState.imageB
        Canvas(modifier) {
            if (a == null || b == null) return@Canvas
            val halfW = size.width / 2f
            val sa = min(halfW / a.width, size.height / a.height)
            val sb = min(halfW / b.width, size.height / b.height)
            val aw = a.width * sa; val ah = a.height * sa
            val bw = b.width * sb; val bh = b.height * sb
            val aox = (halfW - aw) / 2f; val aoy = (size.height - ah) / 2f
            val box = halfW + (halfW - bw) / 2f; val boy = (size.height - bh) / 2f
            drawImage(
                a.asImageBitmap(),
                dstOffset = IntOffset(aox.roundToInt(), aoy.roundToInt()),
                dstSize = IntSize(aw.roundToInt(), ah.roundToInt()),
            )
            drawImage(
                b.asImageBitmap(),
                dstOffset = IntOffset(box.roundToInt(), boy.roundToInt()),
                dstSize = IntSize(bw.roundToInt(), bh.roundToInt()),
            )
            uiState.matches.forEachIndexed { idx, mt ->
                val col = Color.hsv((idx * 37 % 360).toFloat(), 0.9f, 1f)
                drawLine(
                    col,
                    Offset(aox + mt.ax * sa, aoy + mt.ay * sa),
                    Offset(box + mt.bx * sb, boy + mt.by * sb),
                    strokeWidth = 1.5f,
                )
            }
        }
    }
}
