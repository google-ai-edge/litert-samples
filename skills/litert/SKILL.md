---
name: litert
description: Creates an Android app that runs a .tflite model on the CPU or the GPU with the LiteRT CompiledModel API in Kotlin. Use this skill to build a new app around a vision, audio or embedding model, or to add on-device inference to an existing app - the dependency, where the model file goes, the inference class, the ViewModel and screen, and checking the output on a device.
license: Apache-2.0
metadata:
  last-updated: '2026-09-23'
  keywords: [LiteRT, CompiledModel, tflite, Android app, GPU]
---

This skill provides step-by-step guidance for building an Android app that runs a `.tflite` model with the LiteRT CompiledModel API (`com.google.ai.edge.litert:litert` 2.x; overview: https://ai.google.dev/edge/litert/android, sources: https://github.com/google-ai-edge/litert) on the CPU or the GPU. The Interpreter API is not covered. For language models, use the LiteRT-LM skill (https://github.com/google-ai-edge/LiteRT-LM).

## Prerequisites

- A Kotlin Android project (Android Studio's Empty Activity template is enough). The LiteRT 2.2.0 AAR declares `minSdk` 24.
- The dependency in the app-level `build.gradle.kts`: `implementation("com.google.ai.edge.litert:litert:2.2.0")` from Google Maven. The AAR includes the GPU accelerator and declares the GPU driver libraries in its own manifest; no second artifact and no manifest entry are needed.
- A `.tflite` model and its input requirements (input size, mean/std, channel order). Models with a LiteRT recipe: https://github.com/google-ai-edge/litert-samples/tree/main/models

## Detailed steps

### 1. Set up the project and place the model

```kotlin
android { defaultConfig { minSdk = 24 }; androidResources { noCompress += "tflite" } }
dependencies { implementation("com.google.ai.edge.litert:litert:2.2.0") }
```

Put the model at `app/src/main/assets/model.tflite` (`noCompress` keeps it memory-mappable); a model too large to bundle is downloaded into `context.filesDir` and loaded with `CompiledModel.create(filePath, options)`.

### 2. Write the inference class

```kotlin
import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel

const val INPUT_SIZE = 224 * 224 * 3

class Classifier(context: Context, accelerator: Accelerator) : AutoCloseable {
    private val model = CompiledModel.create(context.assets, "model.tflite", CompiledModel.Options(accelerator))
    private val inputs = model.createInputBuffers()
    private val outputs = model.createOutputBuffers()

    fun infer(input: FloatArray): FloatArray {
        inputs[0].writeFloat(input)
        model.run(inputs, outputs)
        return outputs[0].readFloat()
    }

    override fun close() {
        inputs.forEach { it.close() }
        outputs.forEach { it.close() }
        model.close()
    }
}
```

`Accelerator.CPU` runs everywhere. `Accelerator.GPU` compiles the graph for the GPU; if `create` throws `LiteRtException` (an op the GPU does not support), create with `Accelerator.CPU`. The buffers are created once with the model, reused for every inference and closed before the model.

### 3. Wire a ViewModel and the screen

```kotlin
data class UiState(val ready: Boolean = false, val result: FloatArray? = null, val error: String? = null)

class MainViewModel(app: Application) : AndroidViewModel(app) {
    private val executor = Executors.newSingleThreadExecutor()
    private val scope = CoroutineScope(SupervisorJob() + executor.asCoroutineDispatcher())
    private var classifier: Classifier? = null
    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState

    fun load(accelerator: Accelerator = Accelerator.CPU) {
        scope.launch {
            try {
                classifier?.close()
                classifier = Classifier(getApplication<Application>(), accelerator).also { it.infer(FloatArray(INPUT_SIZE)) }
                _uiState.value = UiState(ready = true)
            } catch (e: LiteRtException) {
                _uiState.value = UiState(error = e.message)
            }
        }
    }

    fun classify(input: FloatArray) {
        scope.launch { classifier?.let { _uiState.value = _uiState.value.copy(result = it.infer(input)) } }
    }

    override fun onCleared() {
        scope.launch { classifier?.close() }
        executor.shutdown()
    }
}
```

One single-thread executor owns the model: create, run and close happen only on it, never on the main thread. The inference in `load()` is the warm-up (the first GPU run includes shader compilation). The screen:

```kotlin
@Composable
fun MainScreen(viewModel: MainViewModel = viewModel()) {
    val state by viewModel.uiState.collectAsState()
    LaunchedEffect(Unit) { viewModel.load(Accelerator.GPU) }
    Column {
        state.error?.let { Text("GPU unavailable: $it") }
        Button(onClick = { viewModel.classify(preprocess(bitmap)) }, enabled = state.ready) { Text("Run") }
        state.result?.let { Text("Top class: ${it.indices.maxBy { i -> it[i] }}") }
    }
}
```

On `error`, call `load(Accelerator.CPU)`; `bitmap` comes from the photo picker or CameraX.

### 4. Preprocess by the model's input requirements

```kotlin
fun preprocess(bitmap: Bitmap, size: Int = 224, mean: Float = 127.5f, std: Float = 127.5f): FloatArray {
    val scaled = Bitmap.createScaledBitmap(bitmap, size, size, true)
    val pixels = IntArray(size * size).also { scaled.getPixels(it, 0, size, 0, 0, size, size) }
    val out = FloatArray(size * size * 3)
    pixels.forEachIndexed { i, p ->
        out[i * 3] = ((p shr 16 and 0xFF) - mean) / std
        out[i * 3 + 1] = ((p shr 8 and 0xFF) - mean) / std
        out[i * 3 + 2] = ((p and 0xFF) - mean) / std
    }
    return out
}
```

Use the model's own size, mean/std, channel order and layout (this example is NHWC, RGB, scaled to -1..1); a wrong mean/std looks exactly like a broken model.

### 5. Run on a device and check the output

Run the app on a physical device from Android Studio. Compare the app's output with the model's reference output for one fixed input, on the CPU first and then on the GPU; follow [verification](references/verify.md). A complete app in this shape: https://github.com/google-ai-edge/litert-samples/tree/main/samples/litert/image_segmentation/kotlin_cpu_gpu/android. Moving an existing Interpreter-API app to the CompiledModel API is a separate skill: https://github.com/google-ai-edge/litert-samples/tree/main/skills/litert-compiled-model-migration
