---
name: litert-lm
description: Creates an Android app that runs an open text LLM on the device, on the CPU or the GPU, with the LiteRT-LM Kotlin API. Use this skill to build a new chat, summarization or extraction app with a .litertlm model from Hugging Face litert-community (Gemma, Qwen, Llama, Phi) - the dependency and manifest entries, getting the model file onto the device, engine initialization, a streamed multi-turn conversation, the ViewModel and screen.
license: Apache-2.0
metadata:
  last-updated: '2026-10-07'
  keywords: [LiteRT-LM, litertlm, Gemma, on-device LLM, Android app, GPU]
---

This skill provides step-by-step guidance for building an Android app that runs an open text LLM through the LiteRT-LM Kotlin API (`com.google.ai.edge.litertlm`; guide: https://ai.google.dev/edge/litert-lm/android, sources: https://github.com/google-ai-edge/LiteRT-LM) on the CPU or the GPU. The model is one `.litertlm` file. Images, audio, tool calling and NPU backends are not covered. For vision and audio models, use the `litert-runtime` skill from https://github.com/google-ai-edge/litert-samples/tree/main/skills (LiteRT: https://github.com/google-ai-edge/litert).

## Prerequisites

- A Kotlin Android project (Android Studio's Empty Activity template is enough). The litertlm-android 0.18.0 AAR declares `minSdk` 24.
- The dependency in the app-level `build.gradle.kts`: `implementation("com.google.ai.edge.litertlm:litertlm-android:0.18.0")` from Google Maven (0.18.0 or the latest release).
- For the GPU backend, both lines inside `<application>` in `AndroidManifest.xml`, and `<uses-permission android:name="android.permission.INTERNET"/>` if the app downloads the model:

```xml
<uses-native-library android:name="libvndksupport.so" android:required="false"/>
<uses-native-library android:name="libOpenCL.so" android:required="false"/>
```

- The model file lives on the device's filesystem (hundreds of MB to a few GB), never in assets or in the APK.

## Detailed steps

### 1. Pick a model from litert-community

Choose a `.litertlm` file at https://huggingface.co/litert-community. Start with a powerful model such as `litert-community/gemma-4-E2B-it-litert-lm` or a smaller model like `litert-community/Qwen3-0.6B`. The file to download is `gemma-4-E2B-it.litertlm` (2.6 GB) from the first and `Qwen3-0.6B.litertlm` (0.6 GB) from the second; both run on the CPU and the GPU. A SoC suffix such as `_Google_Tensor_G5` or `.mediatek.mt6993` marks an NPU build (not covered here). The model card names the tested backends and the size. Converting a model yourself is a separate step: [get a model](references/get-a-model.md).

### 2. Put the model file on the device

For the first run, copy the file into the app's private storage with adb: `adb push model.litertlm /data/local/tmp/`, then `adb shell run-as <package> cp /data/local/tmp/model.litertlm files/` (a debuggable build). For users, download it inside the app with `DownloadManager` or your HTTP client into `filesDir`, show the progress, and check the size against the model card before loading. The engine takes the absolute path.

### 3. Initialize the engine off the main thread

`Engine(config).initialize()`, with `EngineConfig(modelPath = …, backend = …, cacheDir = …)` as below, loads the weights and blocks for seconds, so it runs on a background thread. `Backend.CPU()` is the default and runs everywhere; `Backend.GPU()` needs the two manifest lines: without them the engine initializes, and the first reply ends in the error `Can not find OpenCL library on this device`. `cacheDir` speeds up the second load. `Engine` and `Conversation` are `AutoCloseable`. A second `close()` on either throws `IllegalStateException`, and so does `close()` on an engine whose `initialize()` threw: that engine holds nothing, so the code below stores it only after `initialize()` returns.

### 4. Conversation, streaming and the screen

```kotlin
data class ChatState(val ready: Boolean = false, val busy: Boolean = false, val reply: String = "", val error: String? = null)

class ChatViewModel(app: Application) : AndroidViewModel(app) {
    private val executor = Executors.newSingleThreadExecutor()
    private val scope = CoroutineScope(SupervisorJob() + executor.asCoroutineDispatcher())
    private val mutex = Mutex()
    private var engine: Engine? = null
    private var conversation: Conversation? = null
    private val _state = MutableStateFlow(ChatState())
    val state: StateFlow<ChatState> = _state.asStateFlow()

    fun load(modelPath: String, backend: Backend = Backend.CPU()) {
        scope.launch {
            mutex.withLock {
                if (engine?.engineConfig?.modelPath == modelPath) return@launch
                release()
                try {
                    val config = EngineConfig(modelPath = modelPath, backend = backend, cacheDir = getApplication<Application>().cacheDir.path)
                    engine = Engine(config).also { it.initialize() }
                    conversation = engine?.createConversation(ConversationConfig(systemInstruction = Contents.of("You are a helpful assistant.")))
                    _state.value = ChatState(ready = true)
                } catch (e: Exception) {
                    release()
                    _state.value = ChatState(error = e.message)
                }
            }
        }
    }

    fun send(text: String) {
        scope.launch {
            if (!_state.value.ready || _state.value.busy) return@launch
            _state.update { it.copy(busy = true, reply = "", error = null) }
            try {
                checkNotNull(conversation).sendMessageAsync(text).collect { message -> _state.update { it.copy(reply = it.reply + message) } }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                _state.update { it.copy(error = e.message) }
            } finally {
                _state.update { it.copy(busy = false) }
            }
        }
    }

    private suspend fun release() {
        _state.update { it.copy(ready = false) }
        while (_state.value.busy) {
            conversation?.cancelProcess()
            delay(100)
        }
        conversation?.close()
        conversation = null
        engine?.close()
        engine = null
        _state.value = ChatState()
    }

    override fun onCleared() {
        scope.launch {
            mutex.withLock { release() }
            executor.shutdown()
        }
    }
}
```

`sendMessageAsync(text)` returns a `Flow<Message>` of chunks (`toString()` gives a chunk's text); `sendMessage(text)` blocks and returns the whole reply. The conversation keeps its history, so the next `send()` continues the chat, and `busy` keeps the button disabled until the flow completes. `SamplerConfig(topK, topP, temperature)` in `ConversationConfig` sets the sampling. `load()` with the model path that is already loaded does nothing, so the screen can call it again after a rotation, also once the app has fallen back to another backend; `send()` runs only when the state is `ready` and not `busy` (the button's own test), so a tap queued behind a `load()` or `onCleared()` does nothing. The screen (`viewModel()` and `collectAsStateWithLifecycle()` come from `androidx.lifecycle:lifecycle-viewmodel-compose` and `androidx.lifecycle:lifecycle-runtime-compose`):

```kotlin
@Composable
fun ChatScreen(modelPath: String, viewModel: ChatViewModel = viewModel()) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    var input by remember { mutableStateOf("") }
    LaunchedEffect(modelPath) { viewModel.load(modelPath, Backend.GPU()) }
    Column {
        Text(state.reply)
        state.error?.let { Text(it) }
        TextField(value = input, onValueChange = { input = it })
        Button(onClick = { viewModel.send(input); input = "" }, enabled = state.ready && !state.busy) { Text("Send") }
    }
}
```

### 5. Lifecycle

- Keep the engine for the app's lifetime in the `ViewModel` (or an application-scoped holder). `release()` turns `ready` off, calls `cancelProcess()` until the streaming reply's flow has ended, then closes the conversation and the engine, each once; `onCleared()` and a `load()` that replaces the model both go through it, one at a time.
- Create a new conversation to start a fresh chat. The reference app for this API is Google AI Edge Gallery: https://github.com/google-ai-edge/gallery

## Troubleshooting

- First reply is slow: `initialize()` ran on first use; load at app start and set `cacheDir`.
