/*
 * Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import org.gradle.api.DefaultTask
import org.gradle.api.file.ConfigurableFileCollection
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.file.RegularFileProperty
import org.gradle.api.tasks.InputFile
import org.gradle.api.tasks.InputFiles
import org.gradle.api.tasks.OutputDirectory
import org.gradle.api.tasks.PathSensitive
import org.gradle.api.tasks.PathSensitivity
import org.gradle.api.tasks.TaskAction

/**
 * Writes the Kotlin code blocks of a skill's Markdown files into one source file, so that the app
 * under test is the skill's own code and cannot drift from it. A file that has ```kotlin blocks
 * gives those; a file with no ``` at all gives its indented code blocks, the form android/skills
 * publishes. A block that starts with `android {`, `dependencies {`, `plugins {` or
 * `repositories {` is the Gradle setup and is left out: the module's build file has the app's
 * dependencies. The `import` lines of every block are hoisted to the top of the file, each once.
 */
abstract class ExtractSkillCode : DefaultTask() {
    /**
     * The Markdown files to read, in the order their code blocks are written out. An `imports.md`
     * among them is the imports reference: a block may import only what it lists. Without one, the
     * imports of all the blocks are the list.
     */
    @get:InputFiles
    @get:PathSensitive(PathSensitivity.NAME_ONLY)
    abstract val sources: ConfigurableFileCollection

    /** The package line of the generated file. */
    @get:InputFile
    @get:PathSensitive(PathSensitivity.NONE)
    abstract val header: RegularFileProperty

    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    @TaskAction
    fun extract() {
        val gradle = Regex("^(android|dependencies|plugins|repositories)\\s*\\{")
        val blocksByFile =
            sources.files.associateWith { file ->
                codeBlocks(file.readText().replace("\r\n", "\n")).filterNot { block ->
                    gradle.containsMatchIn(block.lines().firstOrNull { it.isNotBlank() }.orEmpty())
                }
            }
        val blocks = blocksByFile.values.flatten()
        check(blocks.isNotEmpty()) { "no kotlin block in ${sources.files}" }
        val importLine = Regex("^import\\s+[\\w.]+(\\s+as\\s+\\w+)?$")
        val reference = blocksByFile.filterKeys { it.name == "imports.md" }.values.flatten()
        val listed =
            reference.ifEmpty { blocks }.flatMap { it.lines() }.filter(importLine::matches).toSet()
        for (line in blocks.flatMap { it.lines() }.filter(importLine::matches)) {
            check(line in listed) { "'$line' is used by a block but missing from the imports file" }
        }
        val imports = listed.sorted()
        val code =
            blocks.map { block ->
                block.lines().filterNot(importLine::matches).joinToString("\n").trim()
            }
        val head = header.get().asFile.readText().trimEnd()
        val out = outputDir.get().file("SkillCode.kt").asFile
        out.parentFile.mkdirs()
        val body = code.filter { it.isNotEmpty() }.joinToString("\n\n")
        out.writeText(head + "\n\n" + imports.joinToString("\n") + "\n\n" + body + "\n")
    }

    /** The ```kotlin blocks of [text], or its indented code blocks when it has no ``` at all. */
    private fun codeBlocks(text: String): List<String> {
        if ("```" in text) {
            val kotlinBlock = Regex("```kotlin\\n(.*?)\\n```", RegexOption.DOT_MATCHES_ALL)
            return kotlinBlock.findAll(text).map { it.groupValues[1] }.toList()
        }
        // A block starts with a line indented by four spaces after a blank line and runs over such
        // lines and blank lines up to the next line that is neither.
        val blocks = mutableListOf<MutableList<String>>()
        var open = false
        var previousBlank = true
        for (line in text.lines()) {
            val blank = line.isBlank()
            val indented = !blank && line.startsWith(INDENT)
            if (!open && indented && previousBlank) {
                blocks.add(mutableListOf())
                open = true
            } else if (open && !indented && !blank) {
                open = false
            }
            if (open) {
                blocks.last().add(if (blank) "" else line.removePrefix(INDENT))
            }
            previousBlank = blank
        }
        return blocks.map { it.joinToString("\n").trimEnd() }
    }

    private companion object {
        const val INDENT = "    "
    }
}
