load("@build_bazel_rules_apple//apple:apple.bzl", "apple_dynamic_xcframework_import")
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application")
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

package(default_visibility = ["//visibility:public"])

genrule(
    name = "LiteRtMetalAccelerator_xcframework",
    srcs = ["@litert_archive//litert/swift:LiteRtMetalAccelerator"],
    outs = [
        "LiteRtMetalAccelerator.xcframework/Info.plist",
        "LiteRtMetalAccelerator.xcframework/ios-arm64/LiteRtMetalAccelerator.framework/Info.plist",
        "LiteRtMetalAccelerator.xcframework/ios-arm64/LiteRtMetalAccelerator.framework/LiteRtMetalAccelerator",
        "LiteRtMetalAccelerator.xcframework/ios-arm64-simulator/LiteRtMetalAccelerator.framework/Info.plist",
        "LiteRtMetalAccelerator.xcframework/ios-arm64-simulator/LiteRtMetalAccelerator.framework/LiteRtMetalAccelerator",
    ],
    cmd = "unzip -q $(location @litert_archive//litert/swift:LiteRtMetalAccelerator) -d $(RULEDIR)",
)

apple_dynamic_xcframework_import(
    name = "LiteRtMetalAccelerator",
    xcframework_imports = [":LiteRtMetalAccelerator_xcframework"],
)

swift_library(
    name = "ImageSegmentationSwift",
    srcs = glob(["ImageSegmentation/*.swift"]),
    module_name = "ImageSegmentation",
    deps = [
        "@litert_archive//litert/swift:LiteRT_Swift",
    ],
)

ios_application(
    name = "ImageSegmentationApp",
    bundle_id = "com.google.ai.edge.examples.image_segmentation",
    families = ["iphone", "ipad"],
    frameworks = [
        ":LiteRtMetalAccelerator",
    ],
    infoplists = ["Info.plist"],
    minimum_os_version = "15.0",
    resources = [
        "//samples/litert/image_segmentation/c++_segmentation/build_from_source:model_files",
    ],
    deps = [
        ":ImageSegmentationSwift",
    ],
)
