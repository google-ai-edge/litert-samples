# LiteRT 2.2.0's own proguard.txt keeps only @UsedByReflection members; its JNI looks up the Kotlin
# API classes by name. Keep the runtime packages un-renamed and un-stripped.
-keep class com.google.ai.edge.litert.** { *; }
-keepclasseswithmembernames class com.google.ai.edge.litert.** { native <methods>; }

# The classic Interpreter API (org.tensorflow.lite), which the Kitten speaker's three graphs run on:
# its native methods bind to the JNI library by class and method name.
-keep class org.tensorflow.lite.** { *; }
-keepclasseswithmembernames class org.tensorflow.lite.** { native <methods>; }

# litertlm-android ships no proguard.txt, and its JNI reaches Kotlin by name (FindClass / NewObject
# on its input and exception classes, GetMethodID on the streaming callback's onMessage / onDone /
# onError). Keep the whole runtime package un-renamed and un-stripped.
-keep class com.google.ai.edge.litertlm.** { *; }
-keepclasseswithmembernames class com.google.ai.edge.litertlm.** { native <methods>; }

# litert 2.2.0 -> Play AI Delivery -> WorkManager + Room. Under R8 the process dies in
# androidx.startup ("Failed to create an instance of class androidx.work.impl.WorkDatabase")
# because the bundled Room rule keeps the class but not its no-arg constructor.
-keep class * extends androidx.room.RoomDatabase { <init>(); }

-keepclasseswithmembernames,includedescriptorclasses class * {
    native <methods>;
}
