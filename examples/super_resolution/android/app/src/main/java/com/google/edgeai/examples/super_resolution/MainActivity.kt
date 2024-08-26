package com.google.edgeai.examples.super_resolution

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.BottomSheetScaffold
import androidx.compose.material.DropdownMenu
import androidx.compose.material.DropdownMenuItem
import androidx.compose.material.ExperimentalMaterialApi
import androidx.compose.material.Icon
import androidx.compose.material.Tab
import androidx.compose.material.TabRow
import androidx.compose.material.Text
import androidx.compose.material.TopAppBar
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.google.edgeai.examples.super_resolution.gallery.ImagePickerScreen
import com.google.edgeai.examples.super_resolution.quicksample.QuickSampleScreen

class MainActivity : ComponentActivity() {
    @OptIn(ExperimentalMaterialApi::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            var tabState by remember { mutableStateOf(MenuTab.QuickSample) }
            var inferenceTimeState by remember {
                mutableStateOf("--")
            }
            var delegateState by remember {
                mutableStateOf(ImageSuperResolutionHelper.Delegate.CPU)
            }

            BottomSheetScaffold(
                sheetShape = RoundedCornerShape(topStart = 15.dp, topEnd = 15.dp),
                sheetPeekHeight = 0.dp,
                sheetContent = {
                    BottomSheet(inferenceTime = inferenceTimeState, onDelegateSelected = {
                        delegateState = it
                    })
                },
            ) {
                Column {
                    Header()
                    Content(
                        tab = tabState,
                        delegate = delegateState,
                        onTabChanged = {
                            tabState = it
                        },
                        onInferenceTimeCallback = {
                            inferenceTimeState = it.toString()
                        }
                    )
                }
            }
        }
    }

    @Composable
    fun Content(
        tab: MenuTab,
        delegate: ImageSuperResolutionHelper.Delegate,
        modifier: Modifier = Modifier,
        onTabChanged: (MenuTab) -> Unit,
        onInferenceTimeCallback: (Int) -> Unit,
    ) {
        val tabs = MenuTab.entries
        Column(modifier) {
            TabRow(backgroundColor = Color.LightGray, selectedTabIndex = tab.ordinal) {
                tabs.forEach { t ->
                    Tab(
                        text = { Text(t.name) },
                        selected = tab == t,
                        onClick = { onTabChanged(t) },
                    )
                }
            }

            when (tab) {
                MenuTab.QuickSample -> QuickSampleScreen(
                    delegate = delegate,
                    onInferenceTimeCallback = onInferenceTimeCallback
                )

                MenuTab.Gallery -> ImagePickerScreen(
                    delegate = delegate,
                    onInferenceTimeCallback = onInferenceTimeCallback
                )
            }
        }
    }

    @Composable
    fun Header(modifier: Modifier = Modifier) {
        TopAppBar(
            modifier = modifier,
            backgroundColor = Color.LightGray,
            title = {
                Image(
                    modifier = Modifier.fillMaxSize(),
                    painter = painterResource(id = R.drawable.tfl_logo),
                    contentDescription = null,
                )
            },
        )
    }

    @Composable
    fun BottomSheet(
        inferenceTime: String,
        modifier: Modifier = Modifier,
        onDelegateSelected: (ImageSuperResolutionHelper.Delegate) -> Unit,
    ) {
        Column(modifier = modifier.padding(horizontal = 20.dp, vertical = 5.dp)) {
            Image(
                modifier = Modifier
                    .size(40.dp)
                    .padding(top = 2.dp, bottom = 5.dp)
                    .align(Alignment.CenterHorizontally),
                painter = painterResource(id = R.drawable.ic_chevron_up),
                contentDescription = ""
            )
            Row {
                Text(modifier = Modifier.weight(0.5f), text = "Inference Time")
                Text(text = inferenceTime)
            }
            
            Spacer(modifier = Modifier.height(10.dp))

            OptionMenu(
                label = "Delegate",
                options = ImageSuperResolutionHelper.Delegate.entries.map { it.name }) {
                onDelegateSelected(ImageSuperResolutionHelper.Delegate.valueOf(it))
            }
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
                    Text(text = option)
                    Spacer(modifier = Modifier.width(5.dp))
                    Icon(
                        imageVector = Icons.Default.ArrowDropDown,
                        contentDescription = "Localized description"
                    )
                }

                DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                    options.forEach {
                        DropdownMenuItem(
                            content = {
                                Text(it)
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
}

enum class MenuTab {
    QuickSample, Gallery,
}
