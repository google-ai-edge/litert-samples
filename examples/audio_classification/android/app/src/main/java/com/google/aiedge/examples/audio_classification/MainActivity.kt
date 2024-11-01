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

package com.google.aiedge.examples.audio_classification

import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material3.BottomSheetScaffold
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.integerArrayResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.google.aiedge.examples.audio_classification.ui.ApplicationTheme
import java.util.Locale

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            AudioClassificationScreen()
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AudioClassificationScreen(
    viewModel: MainViewModel = viewModel(
        factory = MainViewModel.getFactory(LocalContext.current.applicationContext)
    )
) {
    val context = LocalContext.current
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            viewModel.startAudioClassification()
        } else {
            // Permission Denied
            Toast.makeText(context, "Audio permission is denied", Toast.LENGTH_SHORT).show()
        }
    }

    LaunchedEffect(key1 = uiState.errorMessage) {
        if (uiState.errorMessage != null) {
            Toast.makeText(
                context, "${uiState.errorMessage}", Toast.LENGTH_SHORT
            ).show()
            viewModel.errorMessageShown()
        }
    }

    DisposableEffect(Unit) {
        if (ContextCompat.checkSelfPermission(
                context, android.Manifest.permission.RECORD_AUDIO
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            viewModel.startAudioClassification()
        } else {
            launcher.launch(android.Manifest.permission.RECORD_AUDIO)
        }

        onDispose {
            viewModel.stopClassifier()
        }
    }

    ApplicationTheme {
        BottomSheetScaffold(
            sheetDragHandle = {
                Image(
                    modifier = Modifier
                        .size(40.dp)
                        .padding(top = 2.dp, bottom = 5.dp),
                    painter = painterResource(id = R.drawable.ic_chevron_up),
                    colorFilter = ColorFilter.tint(MaterialTheme.colorScheme.secondary),
                    contentDescription = ""
                )
            },
            sheetPeekHeight = 70.dp,
            sheetContent = {
                BottomSheet(
                    uiState = uiState,
                    onModelSelected = {
                        viewModel.setModel(it)
                    },
                    onDelegateSelected = {
                        if (it == AudioClassificationHelper.Delegate.NNAPI) {
                            viewModel.throwError(IllegalArgumentException("Cannot use NNAPI"))
                        } else {
                            viewModel.setDelegate(it)
                        }
                    },
                    onMaxResultSet = {
                        viewModel.setMaxResults(it)
                    },
                    onThresholdSet = {
                        viewModel.setThreshold(it)
                    },
                    onThreadsCountSet = {
                        viewModel.setThreadCount(it)
                    },
                )
            }) {
            ClassificationBody(uiState = uiState)
        }
    }
}


@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun Header() {
    TopAppBar(
        colors = TopAppBarDefaults.topAppBarColors(
            containerColor = MaterialTheme.colorScheme.secondary,
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

@Composable
fun BottomSheet(
    uiState: UiState,
    modifier: Modifier = Modifier,
    onModelSelected: (AudioClassificationHelper.TFLiteModel) -> Unit,
    onDelegateSelected: (option: AudioClassificationHelper.Delegate) -> Unit,
    onMaxResultSet: (value: Int) -> Unit,
    onThresholdSet: (value: Float) -> Unit,
    onThreadsCountSet: (value: Int) -> Unit,
) {
    val maxResults = uiState.setting.resultCount
    val threshold = uiState.setting.threshold
    val threadCount = uiState.setting.threadCount
    val model = uiState.setting.model
    val delegate = uiState.setting.delegate
    Column(modifier = modifier.padding(horizontal = 20.dp, vertical = 5.dp)) {
        Row {
            Text(
                modifier = Modifier.weight(0.5f),
                text = stringResource(id = R.string.inference_title)
            )
            Text(
                text = stringResource(
                    id = R.string.inference_value,
                    uiState.setting.inferenceTime
                )
            )
        }
        Spacer(modifier = Modifier.height(20.dp))
        ModelSelection(
            model = model,
            onModelSelected = onModelSelected,
        )
        OptionMenu(label = stringResource(id = R.string.delegate),
            options = AudioClassificationHelper.Delegate.entries.map { it.name }.toList(),
            currentOption = delegate.name,
            onOptionSelected = {
                onDelegateSelected(AudioClassificationHelper.Delegate.valueOf(it))
            })
        Spacer(modifier = Modifier.height(20.dp))

        Spacer(modifier = Modifier.height(10.dp))

        AdjustItem(
            name = stringResource(id = R.string.max_result_),
            value = maxResults,
            onMinusClicked = {
                if (maxResults > 1) {
                    val max = maxResults - 1
                    onMaxResultSet(max)
                }
            },
            onPlusClicked = {
                if (maxResults < 5) {
                    val max = maxResults + 1
                    onMaxResultSet(max)
                }
            },
        )
        AdjustItem(
            name = stringResource(id = R.string.thresh_hold),
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
        AdjustItem(
            name = stringResource(id = R.string.threads),
            value = threadCount,
            onMinusClicked = {
                if (threadCount >= 2) {
                    val count = threadCount - 1
                    onThreadsCountSet(count)
                }
            },
            onPlusClicked = {
                if (threadCount < 5) {
                    val count = threadCount + 1
                    onThreadsCountSet(count)
                }
            },
        )
    }
}

@Composable
fun ClassificationBody(uiState: UiState, modifier: Modifier = Modifier) {
    val primaryProgressColorList = integerArrayResource(id = R.array.colors_progress_primary)
    val backgroundProgressColorList = integerArrayResource(id = R.array.colors_progress_background)
    Column(modifier = modifier) {
        Header()
        LazyColumn(
            contentPadding = PaddingValues(10.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            val categories = uiState.classifications
            items(count = categories.size) { index ->
                val category = categories[index]
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        modifier = Modifier.weight(0.4f),
                        text = category.label,
                        fontSize = 15.sp,
                        fontWeight = FontWeight.Bold
                    )
                    LinearProgressIndicator(
                        modifier = Modifier
                            .weight(0.6f)
                            .height(25.dp)
                            .clip(RoundedCornerShape(7.dp)),
                        progress = category.score,
                        trackColor = Color(
                            backgroundProgressColorList[index % backgroundProgressColorList.size]
                        ),
                        color = Color(
                            primaryProgressColorList[index % backgroundProgressColorList.size]
                        ),
                    )
                }
            }
        }

    }
}

@Composable
fun OptionMenu(
    label: String,
    options: List<String>,
    currentOption: String,
    modifier: Modifier = Modifier,
    onOptionSelected: (option: String) -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }
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
                Text(text = currentOption, fontSize = 15.sp)
                Spacer(modifier = Modifier.width(5.dp))
                Icon(
                    imageVector = Icons.Default.ArrowDropDown,
                    contentDescription = "Localized description"
                )
            }

            DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                options.forEach {
                    DropdownMenuItem(text = {
                        Text(it, fontSize = 15.sp)
                    }, onClick = {
                        onOptionSelected(it)
                        expanded = false
                    })
                }
            }
        }
    }

}

@Composable
fun ModelSelection(
    model: AudioClassificationHelper.TFLiteModel,
    modifier: Modifier = Modifier,
    onModelSelected: (AudioClassificationHelper.TFLiteModel) -> Unit,
) {
    val radioOptions = AudioClassificationHelper.TFLiteModel.entries

    Column(modifier = modifier) {
        radioOptions.forEach { option ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                RadioButton(
                    selected = (option == model),
                    onClick = {
                        if (option == model) return@RadioButton
                        onModelSelected(option)
                    }, // Recommended for accessibility with screen readers
                )
                Text(
                    modifier = Modifier.padding(start = 16.dp), text = option.name, fontSize = 15.sp
                )
            }
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
