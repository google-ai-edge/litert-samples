---
name: litert-lm
description: Creates an Android app that runs an open text LLM on the device, on the CPU or the GPU, with the LiteRT-LM Kotlin API. Use this skill to build a new chat, summarization or extraction app with a .litertlm model from Hugging Face litert-community (Gemma, Qwen, Llama, Phi) - the dependency and manifest entries, getting the model file onto the device, engine initialization, a streamed multi-turn conversation, the ViewModel and screen.
license: Apache-2.0
metadata:
  last-updated: '2026-09-23'
  keywords: [LiteRT-LM, litertlm, Gemma, on-device LLM, Android app, GPU]
---

This skill provides step-by-step guidance for building an Android app that runs an open text LLM through the LiteRT-LM Kotlin API (`com.google.ai.edge.litertlm`; guide: https://ai.google.dev/edge/litert-lm/android, sources: https://github.com/google-ai-edge/LiteRT-LM) on the CPU or the GPU. The model is one `.litertlm` file. Images, audio, tool calling and NPU backends are not covered. For classic models (vision, audio), use the LiteRT skill (https://github.com/google-ai-edge/litert).

## Prerequisites

- A Kotlin Android project (Android Studio's Empty Activity template is enough). The litertlm-android 0.17.1 AAR declares `minSdk` 24.
- The dependency in the app-level `build.gradle.kts`: `implementation("com.google.ai.edge.litertlm:litertlm-android:0.17.1")` from Google Maven (0.17.1 or the latest release).
- For the GPU backend, both lines inside `<application>` in `AndroidManifest.xml`, and `<uses-permission android:name="android.permission.INTERNET"/>` if the app downloads the model:

```xml
<uses-native-library android:name="libvndksupport.so" android:required="false"/>
<uses-native-library android:name="libOpenCL.so" android:required="false"/>
```

- The model file lives on the device's filesystem (hundreds of MB to a few GB), never in assets or in the APK.

## Detailed steps

### 1. Pick a model from litert-community

Choose a `.litertlm` file at https://huggingface.co/litert-community. The file name carries the quantization (`q8`, `q4`, `wi4b32`) and the KV cache size (`ekv1280`, `ekv4096`); the model card names the tested backend. A small model is the right start: `litert-community/Qwen3-0.6B` downloads without a login, while the Gemma repos are gated (an accepted license and a Hugging Face token are needed to download). Converting a model yourself is a separate step: [get a model](references/get-a-model.md).

### 2. Put the model file on the device

For the first run, copy the file into the app's private storage with adb: `adb push model.litertlm /data/local/tmp/`, then `adb shell run-as <package> cp /data/local/tmp/model.litertlm files/` (a debuggable build). For users, download it inside the app with `DownloadManager` or your HTTP client into `filesDir`, show the progress, and check the size against the model card before loading. The engine takes the absolute path.

### 3. Initialize the engine off the main thread

`Engine(EngineConfig(modelPath, backend, cacheDir)).initialize()` loads the weights and blocks for seconds, so it runs on a background thread. `Backend.CPU()` is the default and runs everywhere; `Backend.GPU()` needs the two manifest lines. `cacheDir` speeds up the second load. `Engine` and `Conversation` are `AutoCloseable`.

### 4. Conversation, streaming and the screen

```kotlin
data class ChatState(val ready: Boolean = false, val busy: Boolean = false, val reply: String = "", val error: String? = null)

class ChatViewModel(app: Application) : AndroidViewModel(app) {
    private val executor = Executors.newSingleThreadExecutor()
    private val scope = CoroutineScope(SupervisorJob() + executor.asCoroutineDispatcher())
    private var engine: Engine? = null
    private var conversation: Conversation? = null
    private val _state = MutableStateFlow(ChatState())
    val state: StateFlow<ChatState> = _state

    fun load(modelPath: String, backend: Backend = Backend.CPU()) {
        scope.launch {
            try {
                val config = EngineConfig(modelPath = modelPath, backend = backend, cacheDir = getApplication<Application>().cacheDir.path)
                engine = Engine(config).also { it.initialize() }
                conversation = engine?.createConversation(ConversationConfig(systemInstruction = Contents.of("You are a helpful assistant.")))
                _state.value = ChatState(ready = true)
            } catch (e: Exception) {
                _state.value = ChatState(error = e.message)
            }
        }
    }

    fun send(text: String) {
        val conversation = conversation ?: return
        scope.launch {
            _state.value = _state.value.copy(busy = true, reply = "")
            try {
                conversation.sendMessageAsync(text).collect { message -> _state.value = _state.value.copy(reply = _state.value.reply + message) }
            } catch (e: Exception) {
                _state.value = _state.value.copy(error = e.message)
            }
            _state.value = _state.value.copy(busy = false)
        }
    }

    override fun onCleared() {
        scope.launch { conversation?.close(); engine?.close() }
        executor.shutdown()
    }
}
```

`sendMessageAsync(text)` returns a `Flow<Message>` of chunks; `sendMessage(text)` blocks and returns the whole reply. The conversation keeps its history, so the next `send()` continues the chat, and `busy` keeps the button disabled until the flow completes. `SamplerConfig(topK, topP, temperature)` in `ConversationConfig` sets the sampling. The screen:

```kotlin
@Composable
fun ChatScreen(modelPath: String, viewModel: ChatViewModel = viewModel()) {
    val state by viewModel.state.collectAsState()
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

- Keep the engine for the app's lifetime in the `ViewModel` (or an application-scoped holder); close the conversation, then the engine, after the last reply has finished.
- Create a new conversation to start a fresh chat. The reference app for this API is Google AI Edge Gallery: https://github.com/google-ai-edge/gallery

## Troubleshooting

- Engine creation fails with `Backend.GPU()`: the two manifest lines are missing. Confirm the file with `Backend.CPU()` first.
- First reply is slow: `initialize()` ran on first use; load at app start and set `cacheDir`.
