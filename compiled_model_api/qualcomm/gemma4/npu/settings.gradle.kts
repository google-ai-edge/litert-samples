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
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        maven {
            url = uri(rootDir.resolve("third_party/android_maven_repository/m2repository"))
            content {
                includeGroup("com.google.ai.edge.litertlm")
            }
        }
        google()
        mavenCentral()
        mavenLocal()
    }
}

rootProject.name = "ModelGarden-QNN-LiteRT"
include(":app")
 
