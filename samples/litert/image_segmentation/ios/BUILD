load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application")
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

package(default_visibility = ["//visibility:public"])

objc_library(
    name = "LiteRTSegmenterObjC",
    srcs = ["ImageSegmentation/LiteRTSegmenter.mm"],
    hdrs = [
        "ImageSegmentation/LiteRTSegmenter.h",
        "ImageSegmentation/ImageSegmentation-Bridging-Header.h",
    ],
    copts = ["-std=c++20", "-fobjc-arc"],
    deps = [
        "@litert_archive//litert/cc:litert_common",
        "@litert_archive//litert/cc:litert_compiled_model",
        "@litert_archive//litert/cc:litert_environment",
        "@litert_archive//litert/cc:litert_model",
        "@litert_archive//litert/cc:litert_opaque_options",
        "@litert_archive//litert/cc:litert_options",
        "@litert_archive//litert/cc:litert_tensor_buffer",
        "@litert_archive//litert/cc:litert_tensor_buffer_requirements",
    ],
)

swift_library(
    name = "ImageSegmentationSwift",
    srcs = glob(["ImageSegmentation/*.swift"]),
    module_name = "ImageSegmentation",
    copts = ["-import-objc-header", "samples/litert/image_segmentation/ios/ImageSegmentation/ImageSegmentation-Bridging-Header.h"],
    deps = [
        ":LiteRTSegmenterObjC",
    ],
)

ios_application(
    name = "ImageSegmentationApp",
    bundle_id = "com.google.ai.edge.examples.image_segmentation",
    families = ["iphone", "ipad"],
    infoplists = ["Info.plist"],
    minimum_os_version = "14.0",
    deps = [
        ":ImageSegmentationSwift",
    ],
    resources = [
        "//samples/litert/image_segmentation/c++_segmentation/build_from_source:model_files",
    ],
)
