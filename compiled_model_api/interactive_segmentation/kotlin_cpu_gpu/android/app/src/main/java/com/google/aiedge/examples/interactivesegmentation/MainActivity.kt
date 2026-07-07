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

package com.google.aiedge.examples.interactivesegmentation

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
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
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.aiedge.examples.interactivesegmentation.view.ApplicationTheme
import com.google.aiedge.examples.interactivesegmentation.view.SegmentationScreen
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.Locale

class MainActivity : ComponentActivity() {
    @OptIn(ExperimentalMaterialApi::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }
        setContent {
            var mediaUri: Uri by remember { mutableStateOf(Uri.EMPTY) }
            val galleryLauncher =
                rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
                    if (uri != null) {
                        mediaUri = uri
                    }
                }

            val uiState by viewModel.uiState.collectAsStateWithLifecycle()

            // Decode the picked image off the main thread, then kick off the one-time encode.
            LaunchedEffect(mediaUri) {
                if (mediaUri != Uri.EMPTY) {
                    val bitmap = withContext(Dispatchers.IO) { loadBitmap(mediaUri) }
                    if (bitmap != null) {
                        viewModel.onImagePicked(bitmap)
                    }
                }
            }

            LaunchedEffect(uiState.errorMessage) {
                if (uiState.errorMessage != null) {
                    Toast.makeText(this@MainActivity, "${uiState.errorMessage}", Toast.LENGTH_SHORT)
                        .show()
                    viewModel.errorMessageShown()
                }
            }

            ApplicationTheme {
                BottomSheetScaffold(
                    sheetPeekHeight = 180.dp,
                    sheetContent = {
                        BottomSheet(uiState = uiState, onDelegateSelected = {
                            viewModel.setAccelerator(it)
                        })
                    },
                    floatingActionButton = {
                        FloatingActionButton(
                            backgroundColor = MaterialTheme.colors.secondary,
                            shape = CircleShape,
                            onClick = {
                                galleryLauncher.launch(
                                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                                )
                            }) {
                            Icon(Icons.Filled.Add, contentDescription = stringResource(R.string.pick_image))
                        }
                    }) {
                    Column {
                        Header()
                        Text(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 8.dp),
                            text = stringResource(R.string.tap_hint),
                            textAlign = TextAlign.Center,
                            fontSize = 14.sp,
                        )
                        SegmentationScreen(
                            uiState = uiState,
                            modifier = Modifier.fillMaxWidth(),
                            onTap = { px, py -> viewModel.onTap(px, py) },
                        )
                    }
                }
            }
        }
    }

    /** Load a content Uri into a software ARGB_8888 bitmap (so its pixels can be read). */
    private fun loadBitmap(uri: Uri): Bitmap? {
        return try {
            val raw = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                val source = ImageDecoder.createSource(contentResolver, uri)
                ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                    decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                    decoder.isMutableRequired = true
                }
            } else {
                @Suppress("DEPRECATION")
                MediaStore.Images.Media.getBitmap(contentResolver, uri)
            }
            if (raw.config == Bitmap.Config.ARGB_8888) raw
            else raw.copy(Bitmap.Config.ARGB_8888, false)
        } catch (e: Exception) {
            null
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

    @Composable
    fun BottomSheet(
        uiState: UiState,
        modifier: Modifier = Modifier,
        onDelegateSelected: (Sam2SegmentationHelper.AcceleratorEnum) -> Unit,
    ) {
        Column(modifier = modifier.padding(horizontal = 20.dp, vertical = 10.dp)) {
            Row {
                Text(
                    modifier = Modifier.weight(0.5f),
                    text = stringResource(id = R.string.encode_title),
                )
                Text(text = stringResource(id = R.string.inference_value, uiState.encodeTimeMs))
            }
            Spacer(modifier = Modifier.height(6.dp))
            Row {
                Text(
                    modifier = Modifier.weight(0.5f),
                    text = stringResource(id = R.string.inference_title),
                )
                Text(text = stringResource(id = R.string.inference_value, uiState.decodeTimeMs))
            }
            Spacer(modifier = Modifier.height(6.dp))
            Row {
                Text(
                    modifier = Modifier.weight(0.5f),
                    text = stringResource(id = R.string.iou_title),
                )
                Text(
                    text = if (uiState.maskIou <= 0f) "--"
                    else String.format(Locale.US, "%.3f", uiState.maskIou)
                )
            }
            Spacer(modifier = Modifier.height(16.dp))
            OptionMenu(
                label = stringResource(id = R.string.delegate),
                options = Sam2SegmentationHelper.AcceleratorEnum.entries.map { it.name },
                selected = uiState.setting.delegate.name,
            ) {
                onDelegateSelected(Sam2SegmentationHelper.AcceleratorEnum.valueOf(it))
            }
        }
    }

    @Composable
    fun OptionMenu(
        label: String,
        options: List<String>,
        selected: String,
        modifier: Modifier = Modifier,
        onOptionSelected: (option: String) -> Unit,
    ) {
        var expanded by remember { mutableStateOf(false) }
        Row(modifier = modifier, verticalAlignment = Alignment.CenterVertically) {
            Text(modifier = Modifier.weight(0.5f), text = label, fontSize = 15.sp)
            Box {
                Row(
                    modifier = Modifier.clickable { expanded = true },
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(text = selected, fontSize = 15.sp)
                    Spacer(modifier = Modifier.width(5.dp))
                    Icon(imageVector = Icons.Default.ArrowDropDown, contentDescription = null)
                }
                DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                    options.forEach {
                        DropdownMenuItem(
                            content = { Text(it, fontSize = 15.sp) },
                            onClick = {
                                onOptionSelected(it)
                                expanded = false
                            },
                        )
                    }
                }
            }
        }
    }
}
