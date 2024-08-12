package com.google.edgeai.examples.super_resolution

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.BottomSheetScaffold
import androidx.compose.material.ExperimentalMaterialApi
import androidx.compose.material.FloatingActionButton
import androidx.compose.material.Icon
import androidx.compose.material.Tab
import androidx.compose.material.TabRow
import androidx.compose.material.Text
import androidx.compose.material.TopAppBar
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.edgeai.examples.super_resolution.tab.ImagePickerScreen
import com.google.edgeai.examples.super_resolution.tab.MenuTab
import com.google.edgeai.examples.super_resolution.tab.QuickSampleScreen
import java.io.InputStream

class MainActivity : ComponentActivity() {
    @OptIn(ExperimentalMaterialApi::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }
        setContent {
            var tabState by remember { mutableStateOf(MenuTab.QuickSample) }
            val context = LocalContext.current

            // Register ActivityResult handler
            val galleryLauncher =
                rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
                    if (uri != null) {
                        val contentResolver = context.contentResolver
                        val inputStream: InputStream? = contentResolver.openInputStream(uri)
                        val bitmap = BitmapFactory.decodeStream(inputStream)
                        viewModel.selectImage(bitmap)
                    }
                }

            val uiState by viewModel.uiState.collectAsStateWithLifecycle()

//            LaunchedEffect(uiState.errorMessage) {
//                if (uiState.errorMessage != null) {
//                    Toast.makeText(
//                        this@MainActivity, "${uiState.errorMessage}", Toast.LENGTH_SHORT
//                    ).show()
//                    viewModel.errorMessageShown()
//                }
//            }

            BottomSheetScaffold(sheetShape = RoundedCornerShape(topStart = 15.dp, topEnd = 15.dp),
                sheetPeekHeight = 70.dp,
                sheetContent = {
                    BottomSheet(inferenceTime = uiState.inferenceTime)
                },
                floatingActionButton = {
                    if (tabState == MenuTab.Gallery) {
                        FloatingActionButton(shape = CircleShape, onClick = {
                            val request =
                                PickVisualMediaRequest(mediaType = ActivityResultContracts.PickVisualMedia.ImageOnly)
                            galleryLauncher.launch(request)
                        }) {
                            Icon(Icons.Filled.Add, contentDescription = null)
                        }
                    }
                }) {
                Column {
                    Header()
                    Content(
                        uiState = uiState,
                        tab = tabState,
                        onTabChanged = {
                            tabState = it
//                            viewModel.stopSegment()
                        },
                        onMakeSharpen = {
                            viewModel.makeSharpen(it)
                        }
                    )
                }
            }
        }
    }

    @Composable
    fun Content(
        uiState: UiState,
        tab: MenuTab,
        modifier: Modifier = Modifier,
        onTabChanged: (MenuTab) -> Unit,
        onMakeSharpen: (Bitmap) -> Unit,
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
                    uiState = uiState,
                    onMakeSharpen = onMakeSharpen,
                )

                MenuTab.Gallery -> ImagePickerScreen(
                    uiState = uiState,
                    onMakeSharpen = onMakeSharpen,
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
        inferenceTime: Int,
        modifier: Modifier = Modifier,
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
                Text(text = inferenceTime.toString())
            }

        }
    }
}
