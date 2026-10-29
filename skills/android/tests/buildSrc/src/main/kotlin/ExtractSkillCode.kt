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
 * under test is the skill's own code and cannot drift from it. The code blocks are indented by four
 * spaces, the form android/skills publishes. A block that starts with `android {`,
 * `dependencies {`, `plugins {` or `repositories {` is the Gradle setup and one that starts with
 * `<` is a manifest entry; both are left out, since the module's build file and manifest have
 * them. The `import` lines of every block are hoisted to the top of the file, each once.
 */
abstract class ExtractSkillCode : DefaultTask() {
    /** The Markdown files to read, in the order their code blocks are written out. */
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
        val setup = Regex("^(<|(android|dependencies|plugins|repositories)\\s*\\{)")
        val blocks =
            sources.files.flatMap { file ->
                codeBlocks(file.readText().replace("\r\n", "\n")).filterNot { block ->
                    setup.containsMatchIn(block.lines().firstOrNull { it.isNotBlank() }.orEmpty())
                }
            }
        check(blocks.isNotEmpty()) { "no kotlin block in ${sources.files}" }
        val importLine = Regex("^import\\s+[\\w.]+(\\s+as\\s+\\w+)?$")
        val imports = blocks.flatMap { it.lines() }.filter(importLine::matches).distinct().sorted()
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

    /**
     * The indented code blocks of [text]. A block starts with a line indented by four spaces after
     * a blank line and runs over such lines and blank lines up to the next line that is neither.
     */
    private fun codeBlocks(text: String): List<String> {
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
