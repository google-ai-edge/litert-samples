pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}
plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "EfficientDET-Lite"
include(":app")

// LiteRT Qualcomm NPU runtime deployment modules.
include(":litert_npu_runtime_libraries:runtime_strings")
include(":litert_npu_runtime_libraries:qualcomm_runtime_v69")
include(":litert_npu_runtime_libraries:qualcomm_runtime_v73")
include(":litert_npu_runtime_libraries:qualcomm_runtime_v75")
include(":litert_npu_runtime_libraries:qualcomm_runtime_v79")
include(":litert_npu_runtime_libraries:qualcomm_runtime_v81")
 
