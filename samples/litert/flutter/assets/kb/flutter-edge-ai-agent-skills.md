---
title: On-device agent skills with flutter_edge_ai_agent
source: https://pub.dev/packages/flutter_edge_ai_agent/versions/0.2.7 (README.md, CHANGELOG.md, lib/src/agent_tools.dart, skill_registry.dart, skill_md_parser.dart, skill.dart, skill_executor.dart, agent_session.dart, agent_loop.dart, agent_event.dart, skill_result.dart, executors/, sources/) ; https://github.com/DenisovAV/flutter_edge_ai/tree/main/packages/flutter_edge_ai_agent
license: MIT ; Apache-2.0 (bundled starter skills and the SKILL.md format, from google-ai-edge/gallery)
---

# On-device agent skills with flutter_edge_ai_agent

This document explains flutter_edge_ai_agent (formerly flutter_gemma_agent), the opt-in package that turns flutter_edge_ai into an on-device agent driven by SKILL.md skills. It covers the skill types, the SKILL.md format, two-stage skill discovery, the built-in agent tools, executors, AgentSession and its event stream, the bundled starter skills and the platform setup.

## What flutter_edge_ai_agent does

flutter_edge_ai_agent provides on-device agentic skills for flutter_edge_ai. This opt-in satellite package turns the inference core into an on-device agent: the model is given a set of skills (SKILL.md), decides which to invoke via flutter_edge_ai's existing function calling, runs them, and feeds the results back — fully offline.

The package contains the Skill and SkillType types and the parseSkillMd parser (YAML frontmatter plus markdown body); a SkillRegistry that holds available and selected skills and builds the cheap name-plus-description discovery string for the system prompt; a SkillExecutor probe chain that mirrors flutter_edge_ai's engine registry, with a sealed SkillResult (TextResult, ImageResult, WidgetResult, WebviewResult, ErrorResult); concrete executors for text, JavaScript, native intents and MCP; an AgentLoop and AgentSession orchestrator that emits a Stream of AgentEvent; and cross-platform UI such as AgentChatView, SkillManagerView, McpManagerView, SecretEditorDialog and SkillTesterView. The recommended model is a function-calling model, Gemma 4 E2B or E4B.

## Compatibility with the Google AI Edge Gallery and licensing

flutter_edge_ai_agent is reverse-engineered from google-ai-edge/gallery (Apache-2.0) and is Gallery-compatible: their SKILL.md catalog parses unmodified, and their JavaScript skills run as-is, because the window.ai_edge_gallery_get_result contract is preserved.

The built-in tool names and parameter schemas mirror Gallery's AgentTools.kt (load_skill, runJs, runIntent, runMcpTool) so small function-calling models call them reliably; only the Dart-side names differ. The bundled SKILL.md files, copied verbatim from Gallery, instruct the model to "Call the run_js / run_intent tool", while the tool declarations are named runSkill, runIntent and runMcp. Native decoders take the name from the structured declaration, but the web decoder follows the SKILL.md text literally and calls run_js, so the agent loop accepts the SKILL.md spelling as an alias and both paths dispatch to the same executor.

flutter_edge_ai_agent is licensed under MIT. The bundled starter skills (calculate-hash, qr-code, query-wikipedia, interactive-map, send-email, create-calendar-event, get-current-time, kitchen-adventure) and the SKILL.md format are derived from google-ai-edge/gallery, licensed under the Apache License 2.0.

## Platform support for each skill type

Text-only, MCP (Streamable HTTP) and native-intent skills run on Android, iOS, macOS, Windows and Linux. JavaScript skills, which run in a webview, work on Android, iOS, macOS and Windows, but not on Linux. None of the skill types run on Web.

Windows JS skills need the WebView2 Runtime, which is pre-installed on Windows 11; on Windows 10 ship the bootstrapper. Linux has no embeddable webview, so JS skills return an ErrorResult there (isAvailable is false), while text, native-intent and MCP skills work on Linux.

The agent is not supported on Web: it has never been verified end to end in a browser, and the package's example app disables it there. Some pieces the loop needs are in place — the web .litertlm path emits well-formed tool calls and survives the call, result, continue round-trip — but sizeInTokens is approximate there, which the loop's context balancing depends on. The agent is verified on Android, iOS, macOS and Windows.

## Text-only skills

A skill's SkillType is one of four values — textOnly, js, intent or mcp — and it decides which executor runs the skill. A text-only skill is a pure prompt or persona — no tool, no code, no risk. It is just instructions fed to the model, for example Gallery's kitchen-adventure, a text-adventure dungeon-master persona. TextSkillExecutor runs it and has zero dependencies.

## JavaScript skills in a sandboxed webview

A JS skill is JavaScript run in a sandboxed webview via the run_js tool. The skill ships scripts/index.html exposing window.ai_edge_gallery_get_result(data, secret), which returns a JSON string with one of result, image, webview or error. JsSkillExecutor loads the skill's HTML into a headless, sandboxed webview — flutter_inappwebview on native, a package:web iframe on web — and posts the returned JSON back over a single result bridge, the only native callback exposed to the foreign page. There is no file access and no arbitrary native bridge, and cross-origin or file navigations away from the loaded page are blocked.

To grant a secure context, so skills using crypto.subtle and other secure-context Web APIs work, the package serves each skill's assets over a loopback HTTP server (http://127.0.0.1, a W3C "potentially trustworthy" origin). This one mechanism works identically across WebView2, WKWebView and Android WebView.

## Native-intent skills and the intent whitelist

A native-intent skill fires an OS action such as email, calendar or notification via the run_intent tool. There is no foreign code: NativeIntentExecutor can only ever run six Gallery-parity intents — send_email, send_text, create_calendar_event, read_calendar_events, schedule_notification and get_current_date_and_time. An unknown intent returns an ErrorResult and never reaches a handler.

Every intent's parameters are validated (presence, type, email, phone and ISO-8601 shape, time-component ranges) before any handler runs. Outbound actions go through the OS compose or confirm surface: the system mail or SMS composer opens through a mailto: or sms: URI, and add_2_calendar opens the calendar's event editor. The model proposes; the user is the one who presses send or save, so nothing fires silently. For calendar events the executor prefers relative parameters — day_offset, hour, minute and duration_minutes — because small models can't reliably emit ISO-8601, and it also accepts absolute ISO-8601 begin_time and end_time. Handlers are injectable, so apps can provide richer implementations.

## MCP skills over Streamable HTTP

An MCP skill is a tool call against a remote Model Context Protocol server via the run_mcp tool. McpSkillExecutor calls tools on connected MCP servers over Streamable HTTP, mirroring Gallery's runMcpTool. A runMcp call carries only the tool name, so the executor finds the server whose enabled tool list contains it. Every call goes through a permission hook unless the matched tool is flagged alwaysAllow; the hook defaults to deny, so nothing fires without an explicit host decision, and the host wires it to an "Allow once" or "Deny" dialog. You write your own SKILL.md for MCP.

## The SKILL.md file format and skill secrets

A SKILL.md file starts with a YAML frontmatter block fenced by `---` lines, followed by a markdown body. The frontmatter looks like this:

```yaml
name: kebab-case-id
description: One-line summary the model uses to pick the skill.
metadata:
  homepage: https://optional
  require-secret: true
  require-secret-description: how to obtain the key
```

The body is ordinary markdown — for example a Title heading and an Instructions section saying "Call the run_js tool with: data: { field: Type }". The name and description fields are required; everything in metadata is optional and tolerated when missing. The name is a kebab-case identifier such as calculate-hash, and the description is the only skill text put in the discovery prompt. The body, everything after the second `---`, becomes the skill's instructions. parseSkillMd throws SkillMdParseException if there is no frontmatter or if name or description is missing.

A skill whose metadata sets require-secret: true needs an API key or secret supplied at runtime, and require-secret-description holds a human-readable hint on how to obtain it. AgentSession keeps runtime secrets for require-secret skills in a store the loop reads at execution time. Secrets are injected into the executor — as the JS secret argument for JavaScript skills — and are never placed in the model prompt.

## How the skill type is inferred from SKILL.md

There is no type field in the frontmatter. parseSkillMd infers the SkillType from the body's tool mention: run_js gives a JS skill, run_intent an intent skill, run_mcp an MCP skill, and otherwise the skill is text-only. The executor probe chain then uses that type to route the skill to the matching executor.

For JS skills the model does not choose which script runs. run_js always runs the skill's own scripts/index.html, or the scriptName its SKILL.md declares, resolved by JsSkillExecutor. Its tool schema therefore takes only data and no scriptName: a required script-name field only made the model invent a filename, and honouring one would let it name any file under scripts/.

## Two-stage skill discovery and the four built-in agent tools

Two-stage discovery is Gallery's trick for keeping context small. The system prompt lists only each selected skill's name and description, one "- name: description" line per selected skill, built by SkillRegistry.discoveryString(). The full SKILL.md instructions are pulled on demand: the model calls loadSkill(skillName), and the agent loop returns that skill's full instructions from the registry as the tool response. This keeps context small when many skills are selected.

The agent gives the model four built-in tools, exported together as agentTools in the order the model sees them, to pass to createChat(tools: agentTools):

- loadSkill(skillName) loads a skill and returns its full instructions. Its description tells the model to call it first with a skill name from the available skills list, then follow the returned instructions.
- runSkill(skillName, data) runs a JS skill's script in a sandboxed webview; data is a JSON string, or an empty string if the user provided none. It is Gallery's runJs without scriptName.
- runIntent(intent, parameters) runs a native intent to interact with the device; parameters is a JSON string.
- runMcp(toolName, input) calls a tool on a connected MCP server; input is a JSON string. It is Gallery's runMcpTool.

## The default agent system prompt

The default agent system prompt mirrors Gallery's skills-only prompt: route the request to a skill, loadSkill its instructions, then follow them. It tells the model that for every new task it must first find the most relevant skill from the list, then use the loadSkill tool to read its instructions if a relevant skill exists, then follow the skill's instructions exactly and output only the final result when successful, and if no relevant skill is found, answer the user directly. A `__SKILLS__` placeholder in the template is replaced with the registry's two-stage discovery list. AgentSession can build the discovery system prompt for a registry by substituting the selected-skills list into a custom systemPromptTemplate.

## SkillRegistry for adding, selecting and looking up skills

SkillRegistry holds the set of available skills and which are currently selected, meaning enabled for the agent loop. It is per-agent state, with selection toggled by the user in the skill manager, not a global singleton. Skills are keyed by name, and adding a skill with an existing name replaces it. Newly added skills are not selected by default; pass selected: true to add or addAll to select them on add. Selection is tracked separately, so toggling a skill off does not drop it from the catalog. The registry also offers remove, select, unselect, isSelected, getSelected and clear.

Lookup is exact match first, then a normalized fallback: small on-device models often echo a kebab-case skill id back as snake_case or with different casing, for example interactive_map for interactive-map, when they call loadSkill. Rather than fail those, the registry retries on a normalized key (lowercased, underscore replaced by hyphen), so a predictable model distortion doesn't read as "skill not found".

## Skill executors and the probe chain

A SkillExecutor is a pluggable way to run a skill — one executor per skill mechanism. Selection is a probe chain exactly like flutter_edge_ai's inference engine registry: the registered executor with the highest priority whose canExecute is true wins, and the first registered breaks ties. In-package executors use priority 0; a third party raises it to take precedence for a skill both could handle. There is no central type-to-executor map, so a third-party executor self-selects with zero changes elsewhere. An implementation must select on the skill's type alone. Core probes by the kebab-case type ids 'text', 'js', 'intent' and 'mcp'. An executor runs a skill with the model-supplied JSON-string argument and, for require-secret skills, an optional runtime secret that is never placed in the prompt.

## Registering executors per session or globally

There are two equivalent ways to wire executors. You can pass them per session with AgentSession.fromModel(model, registry: registry, executors: [...]). Or you can register them globally once through FlutterEdgeAi.initialize(skillExecutors: [...]), beside inferenceEngines, and then omit executors — fromModel reads the core registry, which mirrors how inference engines are registered. If neither path supplies executors, or a registered provider is not a SkillExecutor, AgentSession throws a StateError, because those are configuration mistakes worth failing loudly on rather than silently doing nothing.

## AgentSession and the agent loop

AgentSession is a thin facade tying a SkillRegistry, the registered executors, an AgentLoop and a model or chat into one ask entry point. AgentSession.fromModel creates the InferenceChat with the agentTools and the discovery system prompt injected; supportsFunctionCalls defaults to true since the agent is meaningless without it. You can also pass an already-created chat to the constructor.

A minimal setup after installing a Gemma 4 model with the `.litertlm` engine from flutter_edge_ai_litertlm:

```dart
final source = AssetSkillSource();
final registry = SkillRegistry()..addAll(await source.load(), selected: true);
final session = await AgentSession.fromModel(model, registry: registry, executors: [
  TextSkillExecutor(),
  JsSkillExecutor(sourceFor: source.jsSkillSourceFor),
  NativeIntentExecutor(),
]);
```

The AgentLoop delegates the tool loop to core's InferenceChat.generateChatResponseWithTools. loadSkill returns the skill's instructions; runSkill, runIntent and runMcp go to the first matching executor. The loop stops on the model's call-free answer, or when maxIterations is reached. maxIterations defaults to 10 and is a hard guard against a runaway tool loop: the loop makes at most this many model generations before bailing.

## Agent events streamed during a turn

ask returns a Stream of AgentEvent so the UI can show a step-by-step panel and render inline results as the turn unfolds. The sealed event types are: SkillLoadEvent, emitted before loadSkill runs so the UI can show "Loading skill X", with whether the skill was found; ToolCallEvent, emitted before an executor runs, with the tool, its arguments and the resolved skill; ToolResultEvent, carrying the SkillResult so the UI can render an inline image, webview or native widget; TextChunkEvent, a streamed token of the model's final answer; DoneEvent, with the full final answer; MaxIterationsEvent, when the iteration cap is reached without a text answer; and AgentErrorEvent, when an executor threw or no executor could handle a call. On an error the loop feeds an error tool response back to the model so it can recover.

## Barge-in and cancelling an agent turn

ask accepts an isCancelled callback. When provided, it is polled between iterations — before the next generation and between dispatched calls — for barge-in. Once it returns true the stream ends promptly, with no DoneEvent and no MaxIterationsEvent. A tool already executing still completes; only the next step is skipped. Tool calls the model already committed to the chat history but that were not executed are answered with a synthetic "status: cancelled" tool response, so the persistent chat is never left with a dangling half-answered call.

## Asking the agent about a photo

ask takes an optional image, so the model can pick a skill from what it saw rather than from the text alone. Build the session with supportImage: true and back it with a multimodal model, then call session.ask('what is this?', imageBytes: photo).

Passing imageBytes to a session built without supportImage: true fails the returned stream with an ArgumentError instead of quietly answering as if there were no image. The flag is what's checked; that the model is actually multimodal stays your responsibility, because it can't be introspected from Dart. The check exists because the engines do not agree on what an unsupported image means: the FFI, mobile MediaPipe and built-in AI sessions drop it silently and answer text-only, while web MediaPipe throws. A silent drop is worse — the model returns a confident answer to a photo it never saw.

## Bundled starter skills

Eight starter skills ship as package assets under assets/skills/, ported verbatim from Gallery under Apache-2.0. They are skills the on-device model runs at inference time — not the package skills flutter_edge_ai bundles for your coding assistant — and the skills CLI does not scan that folder.

| Skill | Type | What it does |
|---|---|---|
| calculate-hash | JS | Hash a piece of text |
| qr-code | JS (image) | Generate a QR code |
| query-wikipedia | JS (data) | Summarize a Wikipedia topic |
| interactive-map | JS (webview) | Show a location on an embedded map |
| send-email | intent | Open the OS mail composer |
| create-calendar-event | intent | Open the calendar event editor |
| get-current-time | intent | Report the current local date and time |
| kitchen-adventure | text-only | A text-adventure dungeon-master persona |

AssetSkillSource loads them: it parses the bundled SKILL.md files and builds the packages/flutter_edge_ai_agent/ asset keys for you, and source.jsSkillSourceFor wires the JS executor to their bundled HTML. load() returns one skill per bundled name or throws a BundledSkillLoadError, a StateError whose failures map names each skill that did not load and why; on web, a deployment that does not serve assets/packages/flutter_edge_ai_agent/ causes it. Example prompts are "Calculate the hash of hello" or "Show Paris on interactive map".

## Platform setup for agent skills

Most skills need no platform setup. On Windows, JS skills require the WebView2 Runtime. On iOS, the create-calendar-event intent opens the calendar editor via add_2_calendar, which needs an NSCalendarsUsageDescription usage description in ios/Runner/Info.plist, and local notifications (schedule_notification) prompt for permission at runtime. On Android, flutter_local_notifications requires core-library desugaring in android/app/build.gradle: set isCoreLibraryDesugaringEnabled = true and add the coreLibraryDesugaring dependency com.android.tools:desugar_jdk_libs:2.1.4. With Android Gradle Plugin 9, flutter_inappwebview_android 1.1.3 still calls getDefaultProguardFile('proguard-android.txt'), which AGP 9 rejects; until a stable release fixes it, add android.r8.proguardAndroidTxt.disallowed=false to android/gradle.properties. AGP deprecates this opt-out and plans to remove it in AGP 10.
