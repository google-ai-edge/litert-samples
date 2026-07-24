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

package com.google.aiedge.examples.object_detection

import android.graphics.Bitmap
import android.os.Build
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.annotation.RequiresApi
import androidx.camera.core.ImageProxy
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material3.BottomSheetScaffold
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.aiedge.examples.object_detection.home.camera.CameraScreen
import com.google.aiedge.examples.object_detection.home.gallery.GalleryScreen
import com.google.aiedge.examples.object_detection.objectdetector.ObjectDetectorHelper
import com.google.aiedge.examples.object_detection.ui.ApplicationTheme
import com.google.aiedge.examples.object_detection.ui.darkBlue
import com.google.aiedge.examples.object_detection.ui.teal
import java.util.Locale

class MainActivity : ComponentActivity() {
    @RequiresApi(Build.VERSION_CODES.P)
    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }
        setContent {
            val uiState by viewModel.uiState.collectAsStateWithLifecycle()

            var tabState by remember { mutableStateOf(Tab.Camera) }

            LaunchedEffect(uiState.errorMessage) {
                if (uiState.errorMessage != null) {
                    Toast.makeText(
                        this@MainActivity, "${uiState.errorMessage}", Toast.LENGTH_SHORT
                    ).show()
                    viewModel.errorMessageShown()
                }
            }

            ApplicationTheme {
                BottomSheetScaffold(
                    sheetPeekHeight = 70.dp,
                    sheetContent = {
                        BottomSheet(uiState = uiState,
                            onDelegateSelected = {
                                viewModel.setDelegate(it)
                            },
                            onThresholdSet = {
                                viewModel.setThreshold(it)
                            },
                            onMaxResultSet = {
                                viewModel.setNumberOfResult(it)
                            })
                    }) {
                    Column {
                        Header()
                        Content(
                            uiState = uiState, tab = tabState,
                            onTabChanged = {
                                tabState = it
                                viewModel.stopDetect()
                            },
                            onMediaPicked = {
                                viewModel.stopDetect()
                            },
                            onImageProxyAnalyzed = { imageProxy ->
                                viewModel.detectImageObject(imageProxy)
                            },
                            onImageBitMapAnalyzed = { bitmap, degrees ->
                                viewModel.detectImageObject(bitmap, degrees)
                            },
                        )
                    }
                }
            }
        }
    }

    @Composable
    fun BottomSheet(
        uiState: UiState,
        modifier: Modifier = Modifier,
        onDelegateSelected: (ObjectDetectorHelper.Delegate) -> Unit,
        onThresholdSet: (value: Float) -> Unit,
        onMaxResultSet: (value: Int) -> Unit,
    ) {
        val inferenceTime = uiState.detectionResult?.inferenceTime
        val threshold = uiState.setting.threshold
        val resultCount = uiState.setting.resultCount
        Column(modifier = modifier.padding(horizontal = 20.dp)) {
            Row {
                Text(modifier = Modifier.weight(0.5f), text = "Inference Time")
                Text(text = inferenceTime?.toString() ?: "-")
            }
            Spacer(modifier = Modifier.height(20.dp))
            AdjustItem(
                name = "Threshold",
                value = threshold,
                onMinusClicked = {
                    if (threshold > 0.3f) {
                        val newThreshold = (threshold - 0.1f).coerceAtLeast(0.3f)
                        onThresholdSet(newThreshold)
                    }
                },
                onPlusClicked = {
                    if (threshold < 0.8f) {
                        val newThreshold = threshold + 0.1f.coerceAtMost(0.8f)
                        onThresholdSet(newThreshold)
                    }
                },
            )

            Spacer(modifier = Modifier.height(20.dp))

            AdjustItem(
                name = "Max Results",
                value = uiState.setting.resultCount,
                onMinusClicked = {
                    if (resultCount >= 2) {
                        val count = resultCount - 1
                        onMaxResultSet(count)
                    }
                },
                onPlusClicked = {
                    if (resultCount < 5) {
                        val count = resultCount + 1
                        onMaxResultSet(count)
                    }
                },
            )

            Spacer(modifier = Modifier.height(20.dp))

            OptionMenu(
                label = "Delegate",
                options = ObjectDetectorHelper.Delegate.entries.map { it.name }) {
                onDelegateSelected(ObjectDetectorHelper.Delegate.valueOf(it))
            }

            Spacer(modifier = Modifier.height(20.dp))
        }
    }

    @Composable
    fun OptionMenu(
        label: String,
        modifier: Modifier = Modifier,
        options: List<String>,
        onOptionSelected: (option: String) -> Unit,
    ) {
        var expanded by remember { mutableStateOf(false) }
        var option by remember { mutableStateOf(options.first()) }
        Row(
            modifier = modifier, verticalAlignment = Alignment.CenterVertically
        ) {
            Text(modifier = Modifier.weight(0.5f), text = label, fontSize = 15.sp)
            Box {
                Row(
                    modifier = Modifier.clickable {
                        expanded = true
                    }, verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(text = option, fontSize = 15.sp)
                    Spacer(modifier = Modifier.width(5.dp))
                    Icon(
                        imageVector = Icons.Default.ArrowDropDown,
                        contentDescription = "Localized description"
                    )
                }

                DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                    options.forEach {
                        DropdownMenuItem(
                            text = {
                                Text(it, fontSize = 15.sp)
                            },
                            onClick = {
                                option = it
                                onOptionSelected(option)
                                expanded = false
                            },
                        )
                    }
                }
            }
        }
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun Header() {
        TopAppBar(
            colors = TopAppBarDefaults.topAppBarColors(
                containerColor = teal,
            ),
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

    @RequiresApi(Build.VERSION_CODES.P)
    @Composable
    fun Content(
        uiState: UiState,
        tab: Tab,
        modifier: Modifier = Modifier,
        onTabChanged: (Tab) -> Unit,
        onMediaPicked: () -> Unit,
        onImageProxyAnalyzed: (ImageProxy) -> Unit,
        onImageBitMapAnalyzed: (Bitmap, Int) -> Unit,
    ) {
        val tabs = Tab.entries
        Column(modifier) {
            TabRow(containerColor = darkBlue, selectedTabIndex = tab.ordinal) {
                tabs.forEach { t ->
                    Tab(
                        text = { Text(t.name, color = Color.White) },
                        selected = tab == t,
                        onClick = { onTabChanged(t) },
                    )
                }
            }

            when (tab) {
                Tab.Camera -> CameraScreen(
                    uiState = uiState,
                    onImageAnalyzed = {
                        onImageProxyAnalyzed(it)
                    },
                )

                Tab.Gallery -> GalleryScreen(
                    modifier = Modifier.fillMaxSize(),
                    uiState = uiState,
                    onMediaPicked = onMediaPicked,
                    onImageAnalyzed = {
                        onImageBitMapAnalyzed(it, 0)
                    },
                )
            }
        }
    }

    @Composable
    fun AdjustItem(
        name: String,
        value: Number,
        modifier: Modifier = Modifier,
        onMinusClicked: () -> Unit,
        onPlusClicked: () -> Unit,
    ) {
        Row(
            modifier = modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                modifier = Modifier.weight(0.5f),
                text = name,
                fontSize = 15.sp,
            )
            Row(
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Button(
                    onClick = {
                        onMinusClicked()
                    }) {
                    Text(text = "-", fontSize = 15.sp)
                }
                Spacer(modifier = Modifier.width(10.dp))
                Text(
                    modifier = Modifier.width(30.dp),
                    textAlign = TextAlign.Center,
                    text = if (value is Float) String.format(
                        Locale.US, "%.1f", value
                    ) else value.toString(),
                    fontSize = 15.sp,
                )
                Spacer(modifier = Modifier.width(10.dp))
                Button(
                    onClick = {
                        onPlusClicked()
                    }) {
                    Text(text = "+", fontSize = 15.sp)
                }
            }
        }
    }

    enum class Tab {
        Camera, Gallery,
    }
}
