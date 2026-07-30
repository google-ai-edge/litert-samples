plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.google.ai.edge.samples.imagegeneration"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.google.ai.edge.samples.imagegeneration"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters += setOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"))
            // sample convenience: installable without a release keystore
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    lint {
        // lintVitalAnalyzeRelease crashes in AGP 8.7.3 on this project (lint
        // -internal NPE); unit tests + on-device runs are the real gate here
        checkReleaseBuilds = false
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    packaging {
        jniLibs {
            pickFirsts += setOf(
                "**/libc++_shared.so",
                "**/libtensorflowlite_jni.so",
                "**/libtensorflowlite_gpu_jni.so"
            )
        }
    }
}

dependencies {
    // LiteRT — classic Interpreter API (CPU/XNNPACK path, thread control)
    implementation("com.google.ai.edge.litert:litert:2.1.3")

    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.3")

    // Host-JVM tests for the tokenizer / math ports (golden + fixture parity)
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
}
