# ProGuard rules for LiteRT and LiteRT-LM
-keep class com.google.ai.edge.litert.** { *; }
-keep class com.google.ai.edge.litertlm.** { *; }
-keepclassmembers class * {
    native <methods>;
}
