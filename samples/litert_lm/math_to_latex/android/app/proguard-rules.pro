# litertlm-android ships no proguard.txt, and its JNI reaches Kotlin by name (FindClass / NewObject
# on its input and exception classes, GetMethodID on the streaming callback's onMessage / onDone /
# onError). Keep the whole runtime package un-renamed and un-stripped.
-keep class com.google.ai.edge.litertlm.** { *; }
-keepclasseswithmembernames class com.google.ai.edge.litertlm.** { native <methods>; }

-keepclasseswithmembernames,includedescriptorclasses class * {
    native <methods>;
}

# MathView's page calls back into the app through @JavascriptInterface methods, found by name.
-keepclassmembers class * {
    @android.webkit.JavascriptInterface <methods>;
}
