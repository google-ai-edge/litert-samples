---
name: litert-runtime
description: Creates an Android app that runs a .tflite model on the CPU or the GPU with the LiteRT CompiledModel API in Kotlin. Use this skill to build a new app around a vision, audio or embedding model, or to add on-device inference to an existing app - the dependency, where the model file goes, the inference class, the ViewModel and screen, and checking the output on a device.
license: Apache-2.0
metadata:
  last-updated: '2026-10-07'
  keywords: [LiteRT, CompiledModel, tflite, Android app, GPU]
---

This skill provides step-by-step guidance for building an Android app that runs a `.tflite` model with the LiteRT CompiledModel API (`com.google.ai.edge.litert:litert` 2.x; overview: https://ai.google.dev/edge/litert/android, sources: https://github.com/google-ai-edge/litert) on the CPU or the GPU. The Interpreter API is not covered. For language models, use the `litert-lm` skill from https://github.com/google-ai-edge/litert-samples/tree/main/skills (LiteRT-LM: https://github.com/google-ai-edge/LiteRT-LM).

## Prerequisites

- A Kotlin Android project (Android Studio's Empty Activity template is enough). The LiteRT 2.2.0 AAR declares `minSdk` 24.
- The dependency in the app-level `build.gradle.kts`: `implementation("com.google.ai.edge.litert:litert:2.2.0")` from Google Maven. The 2.2.0 AAR includes the GPU accelerator and declares the GPU driver libraries in its own manifest; no second artifact and no manifest entry are needed. From 2.3.0 the GPU accelerator is its own artifact, `com.google.ai.edge.litert:litert-gpu`, added next to `litert` with the same version.
- A `.tflite` model and its input requirements (input size, mean/std, channel order). Models with a LiteRT recipe: https://github.com/google-ai-edge/litert-samples/tree/main/models

## Detailed steps

### 1. Set up the project and place the model

In the app-level `build.gradle.kts`, set `minSdk = 24` in `defaultConfig`. Put the model at `app/src/main/assets/model.tflite` (AGP stores `.tflite` files uncompressed by default). A model too large for the APK is delivered by Play for On-device AI (beta) as an AI pack, or downloaded by the app into `context.filesDir`: https://developer.android.com/google/play/on-device-ai (its bundletool `--local-testing` installs the packs without the store). An install-time AI pack is read through the `AssetManager`, as that page shows. A fast-follow or on-demand pack and a downloaded file are loaded from their path with `CompiledModel.create(filePath, CompiledModel.Options(accelerator), environment)` in place of the asset call in step 2. The pack's directory comes from the AI Delivery library (`com.google.android.play:ai-delivery`): `AiPackManagerFactory.getInstance(context).getPackLocation(name)?.assetsPath()`, null until the pack has downloaded.

With AGP 9.x the build stops at `processDebugMainManifest`: `litert` 2.2.0 and its dependency `litert-api` 2.2.0 both declare the namespace `com.google.ai.edge.litert` (https://github.com/google-ai-edge/LiteRT/issues/8474); AGP 8.x reports it as a warning and builds. For 2.2.0, add `android.uniquePackageNames=false` to `gradle.properties` before the first build. Excluding `litert-api` does not work: `CompiledModel` and `Accelerator` are in it, and the build then fails at `compileDebugKotlin`. The setting also hides the same clash between any other two libraries, so remove it with 2.3.0, whose two libraries have different namespaces.

### 2. Write the inference class

```kotlin
import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer

const val INPUT_SIZE = 224 * 224 * 3
private val environment by lazy { Environment.create() }

class Classifier(context: Context, accelerator: Accelerator) : AutoCloseable {
    private val model = CompiledModel.create(context.assets, "model.tflite", CompiledModel.Options(accelerator), environment)
    private val inputs = mutableListOf<TensorBuffer>()
    private val outputs = mutableListOf<TensorBuffer>()

    init {
        try {
            inputs += model.createInputBuffers()
            outputs += model.createOutputBuffers()
            infer(FloatArray(INPUT_SIZE))
        } catch (e: Exception) {
            close()
            throw e
        }
    }

    fun infer(input: FloatArray): FloatArray {
        inputs[0].writeFloat(input)
        model.run(inputs, outputs)
        return outputs[0].readFloat()
    }

    override fun close() {
        (inputs + outputs).forEach { it.close() }
        model.close()
    }
}
```

`Accelerator.CPU` needs nothing from the device. `Accelerator.GPU` compiles the graph for the GPU; when the GPU cannot take the model, the constructor throws `LiteRtException`: from `create` for an op the GPU does not support, and on the Android emulator (API 36, arm64) from `create` or from the buffers. A missing asset and a file that is not a model also throw `LiteRtException` from `create`. The buffers are created once with the model, reused for every inference and closed before the model. One `Environment` serves every model in the process and stays open, so LiteRT loads the GPU accelerator library once and not on every `create`. The constructor ends with one inference as the warm-up (the first GPU run includes shader compilation) and closes the model and its buffers if any step throws. `INPUT_SIZE` is the model's input element count: `writeFloat` throws on a longer array and writes a shorter one without an error, so a wrong size shows up as a wrong result.

### 3. Wire a ViewModel and the screen

```kotlin
data class UiState(val ready: Boolean = false, val result: List<Float>? = null, val error: String? = null)

class MainViewModel(app: Application) : AndroidViewModel(app) {
    private val executor = Executors.newSingleThreadExecutor()
    private val scope = CoroutineScope(SupervisorJob() + executor.asCoroutineDispatcher())
    private var classifier: Classifier? = null
    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    fun load(accelerator: Accelerator = Accelerator.CPU) {
        scope.launch {
            if (classifier != null) return@launch
            try {
                classifier = Classifier(getApplication<Application>(), accelerator)
                _uiState.value = UiState(ready = true)
            } catch (e: LiteRtException) {
                _uiState.value = UiState(error = e.message)
            }
        }
    }

    fun classify(input: FloatArray) {
        scope.launch {
            try {
                classifier?.infer(input)?.let { result -> _uiState.update { it.copy(result = result.toList(), error = null) } }
            } catch (e: LiteRtException) {
                _uiState.update { it.copy(error = e.message) }
            }
        }
    }

    override fun onCleared() {
        scope.launch { classifier?.close() }.invokeOnCompletion { scope.cancel() }
        executor.shutdown()
    }
}
```

One single-thread executor owns the model: create, run and close happen only on it, never on the main thread. The ViewModel keeps its own scope because `viewModelScope` is cancelled before `onCleared()` runs, so a close launched there would not run; `onCleared()` cancels the scope once the close is done. `load()` does nothing when a model is already loaded, so the screen can call it again after a rotation; a `load()` that fails leaves `classifier` `null`. `classify()` reports a `LiteRtException` (package `com.google.ai.edge.litert`) in `error` and keeps the model. The screen below takes the picture (`bitmap`) from the photo picker or CameraX and runs `preprocess()` in the click handler (a few milliseconds for 224 by 224; move it onto the executor for bigger pictures); it does not fall back on its own: if `load(Accelerator.GPU)` ends in `error`, call `load(Accelerator.CPU)`. `viewModel()` and `collectAsStateWithLifecycle()` come from `androidx.lifecycle:lifecycle-viewmodel-compose` and `androidx.lifecycle:lifecycle-runtime-compose` (2.10.0); every import the code blocks need is listed in [imports](references/imports.md):

```kotlin
@Composable
fun MainScreen(bitmap: Bitmap, viewModel: MainViewModel = viewModel()) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    LaunchedEffect(Unit) { viewModel.load(Accelerator.GPU) }
    Column {
        state.error?.let { Text(it) }
        Button(onClick = { viewModel.classify(preprocess(bitmap)) }, enabled = state.ready) { Text("Run") }
        state.result?.let { Text("Top class: ${it.indices.maxBy { i -> it[i] }}") }
    }
}
```

### 4. Preprocess by the model's input requirements

Turn the bitmap into the model's input with [`preprocess()`](references/preprocess.md). Use the model's own size, mean/std, channel order and layout (that example is NHWC, RGB, scaled to -1..1); a wrong mean/std looks exactly like a broken model. An audio or embedding model replaces `preprocess()` with its own input encoding and keeps the inference class.

### 5. Run on a device and check the output

Run the app on a physical device from Android Studio. Compare the app's output with the model's reference output for one fixed input, on the CPU first and then on the GPU; follow [verification](references/verify.md). Moving an existing Interpreter-API app to the CompiledModel API is a separate skill: https://github.com/google-ai-edge/litert-samples/tree/main/skills/litert-compiled-model-migration
