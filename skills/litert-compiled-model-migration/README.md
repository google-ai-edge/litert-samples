# LiteRT CompiledModel Migration Skill

This directory contains a shareable "Skill" (Agent Specification) to automate the migration of Android applications from legacy TensorFlow Lite (TFLite) to the modern LiteRT CompiledModel API.

It is designed to be loaded by coding agents (like Gemini, Windsurf, Cursor, or custom AI coding assistants) using Andrej Karpathy's agentic engineering concepts.

## Structure
*   `SKILL.md`: The core specification containing the step-by-step instructions, package mappings (including the TensorFlow Lite Support Library's move to LiteRT Support), code refactoring examples (Kotlin, plus the CMake link line for native modules), and the verification feedback loop.
*   `templates/LiteRtModel.kt`: A reference implementation of the CompiledModel API (accelerator fallback cascade, buffer reuse, one serial dispatcher, Bitmap preprocessing for apps without the Support library) that compiles against the LiteRT 2.3.0 and 2.2.0 artifacts and closes what it created when a step after `create` throws; agents copy it into the target project and adapt the model name and shapes.
*   `templates/SupportBridge.kt`: For apps that keep the TensorFlow Lite Support Library, now LiteRT Support (`litert-support` 2.3.0): moves a `TensorImage` or Support `TensorBuffer` into the model's buffers and back, checking the tensor sizes once; copied only when the app uses the Support library.
*   `templates/GoldenCaptureTest.kt`: A throwaway instrumented test for Step 0 that captures the legacy app's output for one fixed input on the device, with the commands to run it in its header.
*   `templates/MigrationValidationTest.kt`: A boilerplate Kotlin instrumented test (runs on a device or emulator) that agents can inject into target projects to verify that the compiled model reproduces the golden output captured from the legacy app in Step 0.

## How to Use with an Agent
When invoking a coding agent on a repository that requires migration, point the agent to the `SKILL.md` file in this directory.

### Example Prompt to Agent:
```
Migrate the TFLite code in this Android project to the LiteRT CompiledModel API. 
Use the instructions and mappings defined in this skill: https://github.com/google-ai-edge/litert-samples/blob/main/skills/litert-compiled-model-migration/SKILL.md

Ensure you follow the verification loop: capture the golden output first, compile the code, and inject the validation test to confirm the output matches the golden.
```
