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
 * under test is the skill's own code and cannot drift from it.
 */
abstract class ExtractSkillCode : DefaultTask() {
    /** The Markdown files to read, in the order their code blocks are written out. */
    @get:InputFiles
    @get:PathSensitive(PathSensitivity.NONE)
    abstract val sources: ConfigurableFileCollection

    /** The package line and the imports, which the skill leaves to the IDE. */
    @get:InputFile
    @get:PathSensitive(PathSensitivity.NONE)
    abstract val header: RegularFileProperty

    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    @TaskAction
    fun extract() {
        val fence = Regex("```kotlin\\n(.*?)\\n```", RegexOption.DOT_MATCHES_ALL)
        val code =
            sources.files.flatMap { file ->
                fence.findAll(file.readText()).map { it.groupValues[1] }
            }
        check(code.isNotEmpty()) { "no kotlin block in ${sources.files}" }
        val head = header.get().asFile.readText().trimEnd()
        val out = outputDir.get().file("SkillCode.kt").asFile
        out.parentFile.mkdirs()
        out.writeText(head + "\n\n" + code.joinToString("\n\n") + "\n")
    }
}
