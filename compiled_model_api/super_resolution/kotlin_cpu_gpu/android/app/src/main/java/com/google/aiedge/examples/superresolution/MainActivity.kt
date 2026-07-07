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

package com.google.aiedge.examples.superresolution

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
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.BottomSheetScaffold
import androidx.compose.material.DropdownMenu
import androidx.compose.material.DropdownMenuItem
import androidx.compose.material.ExperimentalMaterialApi
import androidx.compose.material.FloatingActionButton
import androidx.compose.material.Icon
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Text
import androidx.compose.material.TopAppBar
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.aiedge.examples.superresolution.view.ApplicationTheme

class MainActivity : ComponentActivity() {
    @OptIn(ExperimentalMaterialApi::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }
        setContent {
            val uiState by viewModel.uiState.collectAsStateWithLifecycle()
            val galleryLauncher = rememberLauncherForActivityResult(
                ActivityResultContracts.PickVisualMedia()
            ) { uri: Uri? ->
                if (uri != null) {
                    contentResolver.openInputStream(uri)?.use {
                        val bmp = BitmapFactory.decodeStream(it)
                        if (bmp != null) viewModel.superResolve(bmp)
                    }
                }
            }

            LaunchedEffect(uiState.errorMessage) {
                if (uiState.errorMessage != null) {
                    Toast.makeText(this@MainActivity, "${uiState.errorMessage}", Toast.LENGTH_SHORT).show()
                    viewModel.errorMessageShown()
                }
            }

            ApplicationTheme {
                BottomSheetScaffold(
                    sheetPeekHeight = 130.dp,
                    sheetContent = { BottomSheet(uiState, onDelegate = { viewModel.setAccelerator(it) }) },
                    floatingActionButton = {
                        FloatingActionButton(
                            backgroundColor = MaterialTheme.colors.secondary, shape = CircleShape,
                            onClick = {
                                galleryLauncher.launch(
                                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                                )
                            },
                        ) { Icon(Icons.Filled.Add, contentDescription = "Pick image") }
                    },
                ) {
                    Column(Modifier.fillMaxSize()) {
                        Header()
                        val sr = uiState.superResolved
                        val base = uiState.baseline
                        if (sr != null && base != null && sr.width > 1) {
                            CompareSlider(base, sr, Modifier.fillMaxWidth().padding(8.dp))
                        } else {
                            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                                Text("Tap + to pick an image to ×4 upscale", fontSize = 16.sp)
                            }
                        }
                    }
                }
            }
        }
    }

    @Composable
    fun Header() {
        TopAppBar(
            backgroundColor = MaterialTheme.colors.secondary,
            title = {
                Image(
                    modifier = Modifier.size(120.dp),
                    alignment = Alignment.CenterStart,
                    painter = painterResource(id = R.drawable.logo),
                    contentDescription = null,
                )
            },
        )
    }

    /** Drag the divider to compare bicubic (left) vs Real-ESRGAN (right), both shown at ×4 size. */
    @Composable
    fun CompareSlider(
        before: android.graphics.Bitmap,
        after: android.graphics.Bitmap,
        modifier: Modifier = Modifier,
    ) {
        var frac by remember { mutableFloatStateOf(0.5f) }
        Box(
            modifier
                .aspectRatio(1f)
                .pointerInput(Unit) {
                    detectHorizontalDragGestures { change, _ ->
                        frac = (change.position.x / size.width).coerceIn(0f, 1f)
                    }
                }
        ) {
            Image(before.asImageBitmap(), "bicubic", Modifier.fillMaxSize(), contentScale = ContentScale.Fit)
            Image(
                after.asImageBitmap(), "Real-ESRGAN",
                Modifier
                    .fillMaxSize()
                    .drawWithContent {
                        clipRect(left = size.width * frac) { this@drawWithContent.drawContent() }
                    },
                contentScale = ContentScale.Fit,
            )
            Box(
                Modifier
                    .fillMaxSize()
                    .drawWithContent {
                        drawContent()
                        val x = size.width * frac
                        drawLine(Color.White, Offset(x, 0f), Offset(x, size.height), strokeWidth = 4f)
                    }
            )
            Text("bicubic", Modifier.align(Alignment.TopStart).padding(8.dp), color = Color.White, fontSize = 13.sp)
            Text("Real-ESRGAN", Modifier.align(Alignment.TopEnd).padding(8.dp), color = Color.White, fontSize = 13.sp)
        }
    }

    @Composable
    fun BottomSheet(uiState: UiState, onDelegate: (SuperResolutionHelper.AcceleratorEnum) -> Unit) {
        Column(Modifier.padding(horizontal = 20.dp, vertical = 8.dp)) {
            Text("Real-ESRGAN ×4 — drag to compare", fontWeight = FontWeight.SemiBold, fontSize = 15.sp)
            Spacer(Modifier.height(12.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Inference Time", Modifier.weight(0.5f))
                Text("${uiState.inferenceTime} ms")
            }
            Spacer(Modifier.height(8.dp))
            OptionMenu(
                "Delegate",
                SuperResolutionHelper.AcceleratorEnum.entries.map { it.name },
                uiState.setting.delegate.name,
            ) { onDelegate(SuperResolutionHelper.AcceleratorEnum.valueOf(it)) }
        }
    }

    @Composable
    fun OptionMenu(label: String, options: List<String>, selected: String, onSelected: (String) -> Unit) {
        var expanded by remember { mutableStateOf(false) }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(label, Modifier.weight(0.5f), fontSize = 15.sp)
            Box {
                Row(Modifier.clickable { expanded = true }, verticalAlignment = Alignment.CenterVertically) {
                    Text(selected, fontSize = 15.sp)
                    Icon(Icons.Default.ArrowDropDown, contentDescription = null)
                }
                DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                    options.forEach {
                        DropdownMenuItem(content = { Text(it, fontSize = 15.sp) }, onClick = {
                            onSelected(it)
                            expanded = false
                        })
                    }
                }
            }
        }
    }
}
