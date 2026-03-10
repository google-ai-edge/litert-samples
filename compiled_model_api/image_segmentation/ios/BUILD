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
        "@litert_archive//litert/cc:litert_api_with_dynamic_runtime",
    ],
)

swift_library(
    name = "ImageSegmentationSwift",
    srcs = glob(["ImageSegmentation/*.swift"]),
    module_name = "ImageSegmentation",
    copts = ["-import-objc-header", "ImageSegmentation/ImageSegmentation-Bridging-Header.h"],
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
        "//compiled_model_api/image_segmentation/c++_segmentation/build_from_source:model_files",
    ],
)
