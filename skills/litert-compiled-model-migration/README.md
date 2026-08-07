# LiteRT CompiledModel Migration Skill

This repository contains a shareable "Skill" (Agent Specification) to automate the migration of Android applications from legacy TensorFlow Lite (TFLite) to the modern LiteRT CompiledModel API.

It is designed to be loaded by coding agents (like Gemini, Windsurf, Cursor, or custom AI coding assistants) using Andrej Karpathy's agentic engineering concepts.

## Structure
*   `SKILL.md`: The core specification containing the step-by-step instructions, package mappings, code refactoring examples (Kotlin and C++ JNI), and the verification feedback loop.
*   `templates/MigrationValidationTest.kt`: A boilerplate Kotlin JUnit test that agents can inject into target projects to verify that the compiled model initializes and executes inference successfully.

## How to Use with an Agent
When invoking a coding agent on a repository that requires migration, point the agent to the `SKILL.md` file in this directory.

### Example Prompt to Agent:
```
Migrate the TFLite code in this Android project to the LiteRT CompiledModel API. 
Use the instructions and mappings defined in this skill: https://github.com/<username>/<repo>/blob/main/SKILL.md

Ensure you follow the verification loop: compile the code and inject a validation test to confirm inference succeeds.
```
