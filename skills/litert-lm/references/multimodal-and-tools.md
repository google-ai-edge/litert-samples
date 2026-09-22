# Images, audio and tools

## Multimodal input

Only with a model that supports it (Gemma 3n). Give the vision and audio graphs a backend, then send mixed content:

```kotlin
val engineConfig = EngineConfig(modelPath = path, backend = Backend.CPU(), visionBackend = Backend.GPU(), audioBackend = Backend.CPU())
conversation.sendMessage(Contents.of(Content.ImageFile("/path/to/image"), Content.AudioBytes(audioBytes), Content.Text("Describe this image and audio.")))
```

Content types: `Text`, `ImageBytes`, `ImageFile`, `AudioBytes`, `AudioFile`.

## Tools (function calling)

Only with a model trained for it (FunctionGemma). Implement `ToolSet`, annotate functions with `@Tool` and parameters with `@ToolParam` (`String`, `Int`, `Boolean`, `Float`, `Double`, or a `List` of these; a nullable type or a default value marks the parameter optional). The return value becomes JSON.

```kotlin
class WeatherTools : ToolSet {
    @Tool(description = "Get the current weather for a city")
    fun getCurrentWeather(@ToolParam(description = "The city name") city: String): Map<String, Any> =
        mapOf("temperature" to 25, "condition" to "Sunny")
}

val conversation = engine.createConversation(ConversationConfig(tools = listOf(tool(WeatherTools()))))
```

With `automaticToolCalling` (the default) the engine runs the function and feeds the result back to the model. `OpenApiTool` with a JSON schema is the alternative when the schema already exists.

## Faster decoding on the GPU

Multi-token prediction through speculative decoding, set before the engine is created:

```kotlin
@OptIn(ExperimentalApi::class)
ExperimentalFlags.enableSpeculativeDecoding = true
```

Guide with the full API: https://ai.google.dev/edge/litert-lm/android
