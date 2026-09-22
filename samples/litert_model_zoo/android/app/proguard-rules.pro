# LiteRT native/JNI entry points use class and method names across the Java/native boundary.
-keep class com.google.ai.edge.litert.** { *; }
-keepclasseswithmembernames,includedescriptorclasses class * {
    native <methods>;
}

# Model wrappers are called directly and do not use reflection. Keep reachable entry points and
# permit method optimization.
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.rfdetr.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.zipformer.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.matcha.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.ormbg.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.edsr.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.realesrgan.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.da3.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.pidnet.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.rfdetrseg.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.ppocr.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.ram.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.panns.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.crepe.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.dac.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.cmgan.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.tiger.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.basicpitch.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.movinet.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.twinlite.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.sixdrepnet.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.liveness.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.dehaze.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.modnet.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.nima.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.plantnet.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.vrwkv.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.crowdcount.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.dinov2.** { *; }
-keep,allowoptimization,allowobfuscation class com.google.ai.edge.examples.model_zoo.models.xfeat.** { *; }

# The catalog uses org.json, not kotlinx.serialization; no reflective serialization rules are needed.
