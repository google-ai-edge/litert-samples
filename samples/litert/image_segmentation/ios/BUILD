load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application")
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

package(default_visibility = ["//visibility:public"])

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
    infoplists = ["Info.plist"],
    minimum_os_version = "14.0",
    deps = [
        ":ImageSegmentationSwift",
    ],
    resources = [
        "//samples/litert/image_segmentation/c++_segmentation/build_from_source:model_files",
    ],
)
