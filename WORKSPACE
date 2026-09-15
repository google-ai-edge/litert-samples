# buildifier: disable=load-on-top

workspace(name = "litert")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# LiteRT Archive pointing to the latest commit on main branch
http_archive(
    name = "litert_archive",
    url = "https://github.com/google-ai-edge/LiteRT/archive/refs/heads/main.tar.gz",
    strip_prefix = "LiteRT-main",
    patch_cmds = [
        "sed 's|//litert|@litert_archive//litert|g' litert/build_common/special_rule.bzl > litert/build_common/special_rule.bzl.tmp && mv litert/build_common/special_rule.bzl.tmp litert/build_common/special_rule.bzl",
        # Rewrite "@//" (main-repo) labels so upstream third_party macros resolve here.
        "for f in third_party/*/*.bzl; do if [ -f \"$f\" ]; then sed 's|@//|@litert_archive//|g' \"$f\" > \"$f.tmp\" && mv \"$f.tmp\" \"$f\"; fi; done",
        # Make litert/cc and litert/cc/options targets publicly visible to external workspaces.
        "sed 's|//litert:__subpackages__|//visibility:public|g' litert/cc/BUILD > litert/cc/BUILD.tmp && mv litert/cc/BUILD.tmp litert/cc/BUILD",
        "sed 's|//litert:__subpackages__|//visibility:public|g' litert/cc/options/BUILD > litert/cc/options/BUILD.tmp && mv litert/cc/options/BUILD.tmp litert/cc/options/BUILD",
        # Windows: inject windows_export_all_symbols feature into cc_shared_library for DLL builds.
        "sed 's/cc_shared_library(/cc_shared_library(\\n    features = [\"windows_export_all_symbols\"],/g' litert/c/BUILD > litert/c/BUILD.tmp && mv litert/c/BUILD.tmp litert/c/BUILD",
        # Windows: define the missing static constant kValueNotSet needed by MSVC linker.
        "printf '\\n#if defined(_MSC_VER) && !defined(__clang__)\\nnamespace tflite { namespace profiling { namespace memory { constexpr size_t MemoryUsage::kValueNotSet; } } }\\n#endif\\n' >> tflite/profiling/memory_info.cc",
        # Stub internal-only license package referenced by some upstream BUILD files.
        "mkdir -p third_party/odml",
        "printf 'load(\"@rules_license//rules:license.bzl\", \"license\")\\n\\npackage(default_visibility = [\"//visibility:public\"])\\n\\nlicense(\\n    name = \"license\",\\n    package_name = \"litert\",\\n)\\n' > third_party/odml/BUILD",
        # Drop internal-only hooks dep (code is compiled out in OSS).
        "sed '\\%^ *\"//litert/vendors/google_tensor/hooks\",%d' litert/vendors/google_tensor/dispatch/BUILD > litert/vendors/google_tensor/dispatch/BUILD.tmp && mv litert/vendors/google_tensor/dispatch/BUILD.tmp litert/vendors/google_tensor/dispatch/BUILD",
    ],
)

# Darts Clone. Declare this before TensorFlow's workspace macros so they do not
# install their own incompatible BUILD overlay.
http_archive(
    name = "darts_clone",
    build_file = "@litert_archive//:BUILD.darts_clone",
    sha256 = "4a562824ec2fbb0ef7bd0058d9f73300173d20757b33bb69baa7e50349f65820",
    strip_prefix = "darts-clone-e40ce4627526985a7767444b6ed6893ab6ff8983",
    url = "https://github.com/s-yata/darts-clone/archive/e40ce4627526985a7767444b6ed6893ab6ff8983.tar.gz",
)

http_archive(
    name = "FP16",
    build_file = "@litert_archive//third_party/FP16:FP16.BUILD",
    sha256 = "d973501a40c55126b31accc2d9f08d931ec3cc190c0430309a5e341d3c0ce32a",
    strip_prefix = "FP16-4dfe081cf6bcd15db339cf2680b9281b8451eeb3",
    url = "https://github.com/Maratyszcza/FP16/archive/4dfe081cf6bcd15db339cf2680b9281b8451eeb3.zip",
)

git_repository(
    name = "XNNPACK",
    remote = "https://github.com/google/XNNPACK.git",
    branch = "master",
)

git_repository(
    name = "KleidiAI",
    remote = "https://github.com/ARM-software/kleidiai.git",
    branch = "main",  # Check if main or master. Usually main for modern repos.
)

http_archive(
    name = "rules_shell",
    sha256 = "bc61ef94facc78e20a645726f64756e5e285a045037c7a61f65af2941f4c25e1",
    strip_prefix = "rules_shell-0.4.1",
    url = "https://github.com/bazelbuild/rules_shell/releases/download/v0.4.1/rules_shell-v0.4.1.tar.gz",
)

load("@rules_shell//shell:repositories.bzl", "rules_shell_dependencies", "rules_shell_toolchains")

rules_shell_dependencies()

rules_shell_toolchains()

http_archive(
    name = "rules_platform",
    sha256 = "0aadd1bd350091aa1f9b6f2fbcac8cd98201476289454e475b28801ecf85d3fd",
    urls = [
        "https://github.com/bazelbuild/rules_platform/releases/download/0.1.0/rules_platform-0.1.0.tar.gz",
    ],
)

# Use recent platforms version to support uefi platform.
http_archive(
    name = "platforms",
    sha256 = "3384eb1c30762704fbe38e440204e114154086c8fc8a8c2e3e28441028c019a8",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/1.0.0/platforms-1.0.0.tar.gz",
        "https://github.com/bazelbuild/platforms/releases/download/1.0.0/platforms-1.0.0.tar.gz",
    ],
)

# Use 3.22.0 (from 3.5.1 of tensorflow) to fix binary signing issue on MacOS Tahoe.
http_archive(
    name = "build_bazel_rules_apple",
    sha256 = "a78f26c22ac8d6e3f3fcaad50eace4d9c767688bd7254b75bdf4a6735b299f6a",
    url = "https://github.com/bazelbuild/rules_apple/releases/download/3.22.0/rules_apple.3.22.0.tar.gz",
)

# Must precede apple_rules_dependencies(), which declares an older bazel_skylib.
http_archive(
    name = "bazel_skylib",
    sha256 = "3b5b49006181f5f8ff626ef8ddceaa95e9bb8ad294f7b5d7b11ea9f7ddaf8c59",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/1.9.0/bazel-skylib-1.9.0.tar.gz"],
)

load(
    "@build_bazel_rules_apple//apple:repositories.bzl",
    "apple_rules_dependencies",
)

apple_rules_dependencies()

http_archive(
    name = "build_bazel_rules_swift",
    sha256 = "f7a67197cd8a79debfe70b8cef4dc19d03039af02cc561e31e0718e98cad83ac",
    url = "https://github.com/bazelbuild/rules_swift/releases/download/2.9.0/rules_swift.2.9.0.tar.gz",
)

# Lower the version from 1.24.5 that tensorflow uses to 1.23.1, the highest version which don't have
# issues with missing LC_UUID, DEVELOPER_DIR or SDKROOT on MacOS Tahoe.
http_archive(
    name = "build_bazel_apple_support",
    sha256 = "ee20cc5c0bab47065473c8033d462374dd38d172406ecc8de5c8f08487943f2f",
    url = "https://github.com/bazelbuild/apple_support/releases/download/1.23.1/apple_support.1.23.1.tar.gz",
)

http_archive(
    name = "bazel_features",
    sha256 = "c26b4e69cf02fea24511a108d158188b9d8174426311aac59ce803a78d107648",
    strip_prefix = "bazel_features-1.43.0",
    url = "https://github.com/bazel-contrib/bazel_features/releases/download/v1.43.0/bazel_features-v1.43.0.tar.gz",
)

# Download coremltools of the same version of tensorflow, but with a custom patchcmd until
# tensorflow is updated to do the same patchcmd.
http_archive(
    name = "coremltools",
    build_file = "@litert_archive//third_party/coremltools:coremltools.BUILD",
    patch_cmds = [
        # Append "mlmodel/format/" to the import path of all proto files.
        "sed -i -e 's|import public \"|import public \"mlmodel/format/|g' mlmodel/format/*.proto",
    ],
    sha256 = "37d4d141718c70102f763363a8b018191882a179f4ce5291168d066a84d01c9d",
    strip_prefix = "coremltools-8.0",
    url = "https://github.com/apple/coremltools/archive/8.0.tar.gz",
)

# Load the custom repository rule to select either a local TensorFlow source or a remote http_archive.
load("@litert_archive//:tensorflow_source_rules.bzl", "tensorflow_source_repo")

tensorflow_source_repo(
    name = "org_tensorflow",
    patches = ["@litert_archive//:PATCH.flatbuffers_windows_no_bash"],
    protobuf_patches = ["@litert_archive//:PATCH.protobuf_port_msvc_compat"],
    sha256 = "7bf06cfd5ff9b462b1b25ca4dc3613fa5e3847fd8e291ff0a8de2ca5a812590a",
    strip_prefix = "tensorflow-5c0b7a5946f0f485e3a532b2a00e03f42a6e14c1",
    urls = ["https://github.com/tensorflow/tensorflow/archive/5c0b7a5946f0f485e3a532b2a00e03f42a6e14c1.tar.gz"],
)

# Initialize the TensorFlow repository and all dependencies.
#
# The cascade of load() statements and tf_workspace?() calls works around the
# restriction that load() statements need to be at the top of .bzl files.
# E.g. we can not retrieve a new repository with http_archive and then load()
# a macro from that repository in the same file.

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

# Fetched by tf_workspace3(); must be initialized before tf_workspace2().
load("@bazel_features//:deps.bzl", "bazel_features_deps")

bazel_features_deps()

load("@rules_cc//cc:extensions.bzl", "compatibility_proxy_repo")

compatibility_proxy_repo()

# Initialize hermetic Python
load("@xla//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@rules_ml_toolchain//py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    default_python_version = "system",
    local_wheel_dist_folder = "dist",
    local_wheel_inclusion_list = [
        "tensorflow*",
        "tf_nightly*",
    ],
    local_wheel_workspaces = ["@org_tensorflow//:WORKSPACE"],
    requirements = {
        "3.10": "@org_tensorflow//:requirements_lock_3_10.txt",
        "3.11": "@org_tensorflow//:requirements_lock_3_11.txt",
        "3.12": "@org_tensorflow//:requirements_lock_3_12.txt",
        "3.13": "@org_tensorflow//:requirements_lock_3_13.txt",
        "3.14": "@org_tensorflow//:requirements_lock_3_14.txt",
        "3.14-freethreaded": "@org_tensorflow//:requirements_lock_3_14_freethreaded.txt",
    },
)

load("@rules_ml_toolchain//py:python_register_toolchain.bzl", "python_register_toolchain")

python_register_toolchain()

load("@rules_ml_toolchain//py:python_init_pip.bzl", "python_init_pip")

python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")

install_deps()
# End hermetic Python initialization

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

load(
    "@xla//third_party/py:python_wheel.bzl",
    "python_wheel_version_suffix_repository",
)

python_wheel_version_suffix_repository(name = "tf_wheel_version_suffix")

# Initialize hermetic C++
load(
    "@rules_ml_toolchain//cc/deps:cc_toolchain_deps.bzl",
    "cc_toolchain_deps",
)

cc_toolchain_deps()

register_toolchains("@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64")

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//gpu/cuda:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@litert_archive//third_party/nvidia_sdk:repositories.bzl",
    "local_cuda_repository",
    "local_tensorrt_rtx_repository",
)

local_cuda_repository(name = "local_cuda")

local_tensorrt_rtx_repository(name = "local_tensorrt_rtx")

load(
    "@rules_ml_toolchain//gpu/nccl:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@rules_ml_toolchain//gpu/nccl:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")

load("@litert_archive//third_party/tqdm:workspace.bzl", tqdm = "repo")

tqdm()

load("@litert_archive//third_party/markupsafe:workspace.bzl", markupsafe = "repo")

markupsafe()

load("@litert_archive//third_party/jinja2:workspace.bzl", jinja2 = "repo")

jinja2()

load("@litert_archive//third_party/dawn:workspace.bzl", dawn = "repo")

dawn()

load("@litert_archive//third_party/lark:workspace.bzl", lark = "repo")

lark()

load("@litert_archive//third_party/xdsl:workspace.bzl", xdsl = "repo")

xdsl()

load("@litert_archive//third_party/perfetto:workspace.bzl", perfetto = "repo")

perfetto()

load("@rules_jvm_external//:defs.bzl", "maven_install")

maven_install(
    name = "litert_maven",
    artifacts = [
        "androidx.lifecycle:lifecycle-common:2.8.7",
        "com.google.android.odml:image:aar:1.0.0-beta1",
        "com.google.android.play:ai-delivery:0.1.1-alpha01",
        "com.google.errorprone:error_prone_annotations:2.50.0",
        "com.google.guava:guava:33.4.6-android",
        "org.jetbrains.kotlin:kotlin-stdlib:2.0.21",
        "org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.0",
        "org.jetbrains.kotlinx:kotlinx-coroutines-guava:1.8.0",
        "org.jetbrains.kotlinx:kotlinx-coroutines-play-services:1.8.0",
    ],
    repositories = [
        "https://jcenter.bintray.com",
        "https://maven.google.com",
        "https://dl.google.com/dl/android/maven2",
        "https://repo1.maven.org/maven2",
    ],
    version_conflict_policy = "pinned",
)

# Kotlin rules
http_archive(
    name = "rules_kotlin",
    sha256 = "13d5b767d697473ced9b55547a18a6ab65ab3fae5440555deee8a44c886b50aa",
    url = "https://github.com/bazelbuild/rules_kotlin/releases/download/v2.3.20/rules_kotlin-v2.3.20.tar.gz",
)

# Sentencepiece
http_archive(
    name = "sentencepiece",
    build_file = "@litert_archive//:BUILD.sentencepiece",
    patch_cmds = [
        # Empty config.h seems enough.
        "touch config.h",
        # Replace third_party/absl/ with absl/ in *.h and *.cc files.
        "sed -i -e 's|#include \"third_party/absl/|#include \"absl/|g' *.h *.cc",
        # Replace third_party/darts_clone/ with include/ in *.h and *.cc files.
        "sed -i -e 's|#include \"third_party/darts_clone/|#include \"include/|g' *.h *.cc",
    ],
    patches = ["@litert_archive//:PATCH.sentencepiece"],
    sha256 = "9970f0a0afee1648890293321665e5b2efa04eaec9f1671fcf8048f456f5bb86",
    strip_prefix = "sentencepiece-0.2.0/src",
    url = "https://github.com/google/sentencepiece/archive/refs/tags/v0.2.0.tar.gz",
)

# tomlplusplus
http_archive(
    name = "tomlplusplus",
    build_file = "@litert_archive//:BUILD.tomlplusplus",
    patch_cmds = [
        "echo '#define TOML_IMPLEMENTATION' > toml.cc",
        "echo '#include \"toml.hpp\"' >> toml.cc",
    ],
    sha256 = "8517f65938a4faae9ccf8ebb36631a38c1cadfb5efa85d9a72e15b9e97d25155",
    strip_prefix = "tomlplusplus-3.4.0",
    url = "https://github.com/marzer/tomlplusplus/archive/refs/tags/v3.4.0.tar.gz",
)

# RE2
http_archive(
    name = "com_googlesource_code_re2",
    sha256 = "7b2b3aa8241eac25f674e5b5b2e23d4ac4f0a8891418a2661869f736f03f57f4",
    strip_prefix = "re2-2024-03-01",
    urls = [
        "https://github.com/google/re2/archive/refs/tags/2024-03-01.tar.gz",
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/re2/archive/refs/tags/2024-03-01.tar.gz",
    ],
)

load("@rules_kotlin//kotlin:repositories.bzl", "kotlin_repositories")

kotlin_repositories()

load("@rules_kotlin//kotlin:core.bzl", "kt_register_toolchains")

kt_register_toolchains()

# Direct dependency for the C++ sample application (`image_utils.cc`). Provides `stb_image.h` for image loading and saving.
load("@litert_archive//third_party/stblib:workspace.bzl", stblib = "repo")

stblib()

load("@litert_archive//third_party/models:workspace.bzl", "models")

models()

# Vendor SDKs
load("@litert_archive//third_party/arm:workspace.bzl", "arm_deps")

arm_deps()

load("@litert_archive//third_party/qairt:workspace.bzl", "qairt")

qairt()

# Currently only works with local sdk
load("@litert_archive//third_party/neuro_pilot:workspace.bzl", "neuro_pilot")

neuro_pilot()

load("@litert_archive//third_party/google_tensor:workspace.bzl", "google_tensor")

google_tensor()

# ML Drift ----------------------------------------------------------------------------------
http_archive(
    name = "ml_drift",
    repo_mapping = {
        "@fp16": "@FP16",
    },
    strip_prefix = "ml-drift-main",
)

# LiteRT GPU ----------------------------------------------------------------------------------
load("@litert_archive//third_party/litert_gpu:workspace.bzl", "litert_gpu")

litert_gpu()

# LiteRT Prebuilts ---------------------------------------------------------------------------------
load("@litert_archive//third_party/litert_prebuilts:workspace.bzl", "litert_prebuilts")

litert_prebuilts()

load("@litert_archive//third_party/intel_openvino:openvino.bzl", "openvino_configure")

openvino_configure()

load("@litert_archive//third_party/exynos_ai_litecore:workspace.bzl", "exynos_ai_litecore")

exynos_ai_litecore()

# Android rules. Need latest rules_android_ndk to use NDK 26+.
load("@rules_android_ndk//:rules.bzl", "android_ndk_repository")

android_ndk_repository(
    name = "androidndk",
    api_level = 26,
)

load("@litert_archive//:android_ndk_env.bzl", "check_android_ndk_env")

check_android_ndk_env(name = "android_ndk_env")

load("@android_ndk_env//:current_android_ndk_env.bzl", "ANDROID_NDK_HOME_IS_SET")

register_toolchains("@androidndk//:all" if ANDROID_NDK_HOME_IS_SET else "@android_ndk_env//:all")

# Conditionally declare Android SDK repository at the bottom using built-in maybe.
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

maybe(
    android_sdk_repository,
    name = "androidsdk",
)
