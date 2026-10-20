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

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.plugin.compose")
}

// The directory that holds the litert-runtime skill. -PskillsDir=<dir> tests another copy.
val skillsDir = providers.gradleProperty("skillsDir").orElse("..")

// -PlitertVersion=<version> tests another release of the library. From 2.3.0 the GPU accelerator
// is its own artifact, litert-gpu; for 2.2.0 or 2.1.6 add -Pandroid.uniquePackageNames=false
// (step 1 of the skill).
val litertVersion = providers.gradleProperty("litertVersion").orElse("2.3.0")
val litertHasGpuArtifact =
    litertVersion.map { version ->
        val (major, minor) = version.split(".").take(2).map { it.toInt() }
        major > 2 || (major == 2 && minor >= 3)
    }

val extractSkillCode =
    tasks.register<ExtractSkillCode>("extractSkillCode") {
        val skills = rootProject.layout.projectDirectory.dir(skillsDir)
        val skill = skills.map { it.dir("litert-runtime") }
        sources.from(skill.map { it.file("SKILL.md") })
        header.set(layout.projectDirectory.file("skill-header.txt"))
        outputDir.set(layout.buildDirectory.dir("generated/skill"))
    }

// The Kotlin of the skill's migration reference, compiled into the test APK, and the tests in
// src/migrationTest that run it. It uses LiteRT Support, whose com.google.ai.edge.litert.support
// package starts with 2.3.0, so it is built from that version on, and only from a copy of the skill
// that has the reference.
val migrationReference =
    rootProject.layout.projectDirectory.dir(skillsDir).map {
        it.file("litert-runtime/references/migration.md")
    }
val migrationTests = litertHasGpuArtifact.get() && migrationReference.get().asFile.isFile
val extractMigrationCode =
    tasks.register<ExtractSkillCode>("extractMigrationCode") {
        enabled = migrationTests
        sources.from(migrationReference)
        header.set(layout.projectDirectory.file("skill-header.txt"))
        // Not SkillCode.kt: the test APK's SkillCodeKt would shadow the app's.
        fileName.set("MigrationCode.kt")
        outputDir.set(layout.buildDirectory.dir("generated/migration"))
    }

// The app's own model, under the asset name the skill's Classifier opens. The other fixtures are
// test assets.
val copyAppModel =
    tasks.register<Copy>("copyAppModel") {
        from(layout.projectDirectory.file("fixtures/model.tflite"))
        into(layout.buildDirectory.dir("generated/assets"))
    }

android {
    namespace = "com.google.ai.edge.examples.litertruntime"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.google.ai.edge.examples.litertruntime"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        compose = true
    }
    sourceSets.getByName("main").kotlin.directories += "build/generated/skill"
    sourceSets.getByName("main").assets.directories += "build/generated/assets"
    sourceSets.getByName("androidTest").assets.directories += "fixtures"
    if (migrationTests) {
        sourceSets.getByName("androidTest").kotlin.directories +=
            listOf("build/generated/migration", "src/migrationTest/java")
    }
}

tasks.named("preBuild") {
    dependsOn(extractSkillCode, copyAppModel, extractMigrationCode)
}

dependencies {
    implementation("androidx.core:core-ktx:1.17.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.10.0")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.10.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.10.0")
    implementation("androidx.activity:activity-compose:1.12.1")
    implementation(platform("androidx.compose:compose-bom:2025.12.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
    implementation("com.google.ai.edge.litert:litert:${litertVersion.get()}")
    if (litertHasGpuArtifact.get()) {
        implementation("com.google.ai.edge.litert:litert-gpu:${litertVersion.get()}")
    }
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
    androidTestImplementation("androidx.test:runner:1.7.0")
    if (migrationTests) {
        androidTestImplementation("com.google.ai.edge.litert:litert-support:${litertVersion.get()}")
    }
}
