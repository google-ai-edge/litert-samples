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

package com.google.aiedge.examples.textclassification

import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.BottomSheetScaffold
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
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
import androidx.compose.ui.graphics.painter.ColorPainter
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.google.aiedge.examples.textclassification.R

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

    LaunchedEffect(key1 = uiState.errorMessage) {
        if (uiState.errorMessage != null) {
            Toast.makeText(
                context,
                "${uiState.errorMessage}",
                Toast.LENGTH_SHORT
            ).show()
            viewModel.errorMessageShown()
        }
    }

    BottomSheetScaffold(
        sheetDragHandle = {
            Image(
                modifier = Modifier
                    .size(40.dp)
                    .padding(top = 2.dp, bottom = 5.dp),
                painter = painterResource(id = R.drawable.ic_chevron_up),
                contentDescription = ""
            )
        },
        sheetPeekHeight = 70.dp,
        topBar = {
            Header()
        },
        sheetContent = {
            BottomSheetContent(
                inferenceTime = uiState.inferenceTime,
                onModelSelected = {
                    viewModel.setModel(it)
                },
            )
        }) {
        ClassificationBody(
            positivePercentage = uiState.positivePercentage,
            negativePercentage = uiState.negativePercentage,
            onSubmitted = {
                if (it.isNotBlank()) {
                    viewModel.runClassification(it)
                }
            })
    }
}


@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun Header(modifier: Modifier = Modifier) {
    TopAppBar(
        colors = TopAppBarDefaults.topAppBarColors(
            containerColor = Color.LightGray,
        ),
        title = {
            Row(
                modifier = Modifier.fillMaxSize(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Image(
                    modifier = Modifier.size(50.dp),
                    painter = ColorPainter(color = Color.White),
                    contentDescription = null,
                )
                Spacer(modifier = modifier.width(10.dp))
                Text(text = "LiteRT", color = Color.Blue, fontWeight = FontWeight.SemiBold)
            }
        },
    )
}

@Composable
fun BottomSheetContent(
    inferenceTime: Long,
    modifier: Modifier = Modifier,
    onModelSelected: (TextClassificationHelper.Model) -> Unit,
) {
    Column(modifier = modifier.padding(horizontal = 20.dp, vertical = 5.dp)) {
        Row {
            Text(modifier = Modifier.weight(0.5f), text = "Inference Time", fontSize = 16.sp)
            Text(text = "$inferenceTime ms", fontSize = 16.sp)
        }
        Spacer(modifier = Modifier.height(20.dp))
        ModelSelection(
            onModelSelected = onModelSelected,
        )
    }
}

@Composable
fun ClassificationBody(
    positivePercentage: Float,
    negativePercentage: Float,
    modifier: Modifier = Modifier,
    onSubmitted: (String) -> Unit
) {
    var text by remember { mutableStateOf("") }
    val focusManager = LocalFocusManager.current

    Column(modifier = modifier.padding(horizontal = 20.dp)) {
        Spacer(modifier = Modifier.height(20.dp))
        TextField(
            modifier = Modifier
                .fillMaxWidth()
                .height(100.dp),
            value = text,
            onValueChange = {
                text = it
            }, placeholder = {
                Text(text = "Enter text to classify")
            })
        Spacer(modifier = Modifier.height(10.dp))
        Button(
            onClick = {
                focusManager.clearFocus()
                onSubmitted(text)
            }) {
            Text(text = "Classify")
        }
        Spacer(modifier = Modifier.height(20.dp))
        Text(text = "Positive: ($positivePercentage)", fontWeight = FontWeight.Bold)
        Text(text = "Negative: ($negativePercentage)", fontWeight = FontWeight.Bold)
    }
}

@Composable
fun ModelSelection(
    modifier: Modifier = Modifier,
    onModelSelected: (TextClassificationHelper.Model) -> Unit,
) {
    val radioOptions = TextClassificationHelper.Model.entries.map { it.name }.toList()
    var selectedOption by remember { mutableStateOf(radioOptions.first()) }

    Column(modifier = modifier) {
        radioOptions.forEach { option ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                RadioButton(
                    selected = (option == selectedOption),
                    onClick = {
                        if (selectedOption == option) return@RadioButton
                        onModelSelected(TextClassificationHelper.Model.valueOf(option))
                        selectedOption = option
                    }, // Recommended for accessibility with screen readers
                )
                Text(modifier = Modifier.padding(start = 16.dp), text = option, fontSize = 15.sp)
            }
        }
    }
}
