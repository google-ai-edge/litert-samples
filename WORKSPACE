# buildifier: disable=load-on-top

workspace(name = "litert")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

# LiteRT Archive (v2.1.4) with Correct Checksum
http_archive(
    name = "litert_archive",
    url = "https://github.com/google-ai-edge/LiteRT/archive/refs/tags/v2.1.4.tar.gz",
    strip_prefix = "LiteRT-2.1.4",
    sha256 = "be614b050cc6603863acca0c48d0e210c0399ca80974648cb6fb5f70374a5b86",
    patch_cmds = [
        "sed 's|//litert|@litert_archive//litert|g' litert/build_common/special_rule.bzl > litert/build_common/special_rule.bzl.tmp && mv litert/build_common/special_rule.bzl.tmp litert/build_common/special_rule.bzl",
        "sed 's|@//third_party|@litert_archive//third_party|g' third_party/litert_prebuilts/workspace.bzl > third_party/litert_prebuilts/workspace.bzl.tmp && mv third_party/litert_prebuilts/workspace.bzl.tmp third_party/litert_prebuilts/workspace.bzl",
    ],
)

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

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

# Download coremltools of the same version of tensorflow, but with a custom patchcmd until
# tensorflow is updated to do the same patchcmd.
http_archive(
    name = "coremltools",
    build_file = "@//third_party/coremltools:coremltools.BUILD",
    patch_cmds = [
        # Append "mlmodel/format/" to the import path of all proto files.
        "sed -i -e 's|import public \"|import public \"mlmodel/format/|g' mlmodel/format/*.proto",
    ],
    sha256 = "37d4d141718c70102f763363a8b018191882a179f4ce5291168d066a84d01c9d",
    strip_prefix = "coremltools-8.0",
    url = "https://github.com/apple/coremltools/archive/8.0.tar.gz",
)

# Load the custom repository rule to select either a local TensorFlow source or a remote http_archive.
load("@litert_archive//litert:tensorflow_source_rules.bzl", "tensorflow_source_repo")

tensorflow_source_repo(
    name = "org_tensorflow",
    sha256 = "dfdb64ef739ce773f14f703b8a0b98ee5c005595852aa19d9c51aeb53bce0eff",
    strip_prefix = "tensorflow-a9cd5eb2b4536b6c185ed882e0f41d9c10ada5f5",
    urls = ["https://github.com/tensorflow/tensorflow/archive/a9cd5eb2b4536b6c185ed882e0f41d9c10ada5f5.tar.gz"],
)

# Initialize the TensorFlow repository and all dependencies.
#
# The cascade of load() statements and tf_workspace?() calls works around the
# restriction that load() statements need to be at the top of .bzl files.
# E.g. we can not retrieve a new repository with http_archive and then load()
# a macro from that repository in the same file.

# Darts Clone
http_archive(
    name = "darts_clone",
    build_file = "@//:BUILD.darts_clone",
    sha256 = "4a562824ec2fbb0ef7bd0058d9f73300173d20757b33bb69baa7e50349f65820",
    strip_prefix = "darts-clone-e40ce4627526985a7767444b6ed6893ab6ff8983",
    url = "https://github.com/s-yata/darts-clone/archive/e40ce4627526985a7767444b6ed6893ab6ff8983.tar.gz",
)

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

# Initialize hermetic Python
load("@xla//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    default_python_version = "system",
    local_wheel_dist_folder = "dist",
    local_wheel_inclusion_list = [
        "tensorflow*",
        "tf_nightly*",
    ],
    local_wheel_workspaces = ["@org_tensorflow//:WORKSPACE"],
    requirements = {
        "3.9": "@org_tensorflow//:requirements_lock_3_9.txt",
        "3.10": "@org_tensorflow//:requirements_lock_3_10.txt",
        "3.11": "@org_tensorflow//:requirements_lock_3_11.txt",
        "3.12": "@org_tensorflow//:requirements_lock_3_12.txt",
        "3.13": "@org_tensorflow//:requirements_lock_3_13.txt",
    },
)

load("@xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

load("@xla//third_party/py:python_init_pip.bzl", "python_init_pip")

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

# Toolchains for ML projects hermetic builds.
# Details: https://github.com/google-ml-infra/rules_ml_toolchain
http_archive(
    name = "rules_ml_toolchain",
    sha256 = "d67b536f812ba8784d58b1548d0f9cba49237ad280cea694934a6c14da706f30",
    strip_prefix = "rules_ml_toolchain-4a5659fcf7a91d6a25c2abddf3736ab175101a49",
    url = "https://github.com/google-ml-infra/rules_ml_toolchain/archive/4a5659fcf7a91d6a25c2abddf3736ab175101a49.tar.gz",
)

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
    "@rules_ml_toolchain//gpu/nccl:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@rules_ml_toolchain//gpu/nccl:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")

load("@litert_archive//litert/sdk_util:repo.bzl", "configurable_repo")



load("@rules_jvm_external//:defs.bzl", "maven_install")

maven_install(
    name = "litert_maven",
    artifacts = [
        "androidx.lifecycle:lifecycle-common:2.8.7",
        "com.google.android.play:ai-delivery:0.1.1-alpha01",
        "com.google.guava:guava:33.4.6-android",
        "org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.1",
        "org.jetbrains.kotlinx:kotlinx-coroutines-guava:1.10.1",
        "org.jetbrains.kotlinx:kotlinx-coroutines-play-services:1.10.1",
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
    sha256 = "e1448a56b2462407b2688dea86df5c375b36a0991bd478c2ddd94c97168125e2",
    url = "https://github.com/bazelbuild/rules_kotlin/releases/download/v2.1.3/rules_kotlin-v2.1.3.tar.gz",
)

# Sentencepiece
http_archive(
    name = "sentencepiece",
    build_file = "@//:BUILD.sentencepiece",
    patch_cmds = [
        # Empty config.h seems enough.
        "touch config.h",
        # Replace third_party/absl/ with absl/ in *.h and *.cc files.
        "sed -i -e 's|#include \"third_party/absl/|#include \"absl/|g' *.h *.cc",
        # Replace third_party/darts_clone/ with include/ in *.h and *.cc files.
        "sed -i -e 's|#include \"third_party/darts_clone/|#include \"include/|g' *.h *.cc",
    ],
    patches = ["@//:PATCH.sentencepiece"],
    sha256 = "9970f0a0afee1648890293321665e5b2efa04eaec9f1671fcf8048f456f5bb86",
    strip_prefix = "sentencepiece-0.2.0/src",
    url = "https://github.com/google/sentencepiece/archive/refs/tags/v0.2.0.tar.gz",
)

http_archive(
    name = "tqdm",
    build_file = "@litert_archive//third_party/tqdm:tqdm.BUILD",
    add_prefix = "tqdm",
    urls = [
        "https://third-party-mirror.googlesource.com/tqdm/+archive/d593e871a6b3fcc21ca5281aebda0feee0e8732e.tar.gz",
    ],
)

# Manually declared here because Bazel does not inherit repository definitions from external archives (`@litert_archive`), and it is referenced by upstream build rules.
http_archive(
    name = "dawn",
    add_prefix = "dawn",
    urls = [
        "https://github.com/google/dawn/archive/v20250713.025201.tar.gz",
    ],
    build_file = "@litert_archive//third_party/dawn:BUILD",
)

# Direct dependency for the C++ sample application (`image_utils.cc`). Provides `stb_image.h` for image loading and saving.
new_git_repository(
    name = "stblib",
    remote = "https://github.com/nothings/stb",
    commit = "c0c982601f40183e74d84a61237e968dca08380e",
    build_file = "@litert_archive//third_party/stblib:stblib.BUILD",
)

configurable_repo(
    name = "models",
    build_file = "@litert_archive//third_party/models:models.BUILD",
    local_path_env = "LITERT_MODELS",
    url = "https://storage.googleapis.com/litert/models.tar.gz",
)

configurable_repo(
    name = "ats_models",
    build_file = "@litert_archive//third_party/models:ats_models.BUILD",
    local_path_env = "LITERT_ATS_MODELS",
    url = "https://storage.googleapis.com/litert/ats_models.tar.gz",
)

configurable_repo(
    name = "qairt",
    build_file = "@litert_archive//third_party/qairt:qairt.BUILD",
    local_path_env = "LITERT_QAIRT_SDK",
    strip_prefix = "latest",
    url = "https://storage.googleapis.com/litert/litert_qualcomm_sdk_2_37_1_release.tar.gz",
)

# Currently only works with local sdk
configurable_repo(
    name = "neuro_pilot",
    build_file = "@litert_archive//third_party/neuro_pilot:neuro_pilot.BUILD",
    local_path_env = "LITERT_NEURO_PILOT_SDK",
    strip_prefix = "neuro_pilot",
    url = "https://s3.ap-southeast-1.amazonaws.com/mediatek.neuropilot.com/57c17aa0-90b4-4871-a7b6-cdcdc678b3aa.gz",
    symlink_mapping = {
        "v8_latest": "v8_0_8",
        # Just let the compilation pass, we don't expect it to work...
        # TODO: Remove this once we have a working V7 & V9 version.
        "v7_latest": "v8_0_8",
        "v9_latest": "v8_0_8",
    },
)

configurable_repo(
    name = "google_tensor",
    build_file = "@litert_archive//third_party/google_tensor:google_tensor.BUILD",
    local_path_env = "GOOGLE_TENSOR_COMPILER_LIB",
)

# LiteRT Prebuilts ---------------------------------------------------------------------------------
load("@litert_archive//third_party/litert_prebuilts:workspace.bzl", "litert_prebuilts")

litert_prebuilts()

load("@litert_archive//third_party/intel_openvino:openvino.bzl", "openvino_configure")

openvino_configure()

# iOS Build Rules
http_archive(
    name = "build_bazel_apple_support",
    sha256 = "b53f6491e742549f13866628ddffcc75d1f3b2d6987dc4f14a16b242113c890b",
    url = "https://github.com/bazelbuild/apple_support/releases/download/1.17.1/apple_support.1.17.1.tar.gz",
)

http_archive(
    name = "build_bazel_rules_apple",
    sha256 = "34953c6c5666f2bd864a4a2a27599eb6630a42fde18ba57292fa0a7fcb3d851c",
    url = "https://github.com/bazelbuild/rules_apple/releases/download/4.5.0/rules_apple.4.5.0.tar.gz",
)

load("@build_bazel_apple_support//lib:repositories.bzl", "apple_support_dependencies")
apple_support_dependencies()

http_archive(
    name = "build_bazel_rules_swift",
    sha256 = "fbc1843b0d87922d05903b2909a5676ee5f27918840c9497e6e58b901594950a",
    url = "https://github.com/bazelbuild/rules_swift/releases/download/1.18.0/rules_swift.1.18.0.tar.gz",
)

load("@build_bazel_rules_swift//swift:repositories.bzl", "swift_rules_dependencies")
swift_rules_dependencies()

load("@build_bazel_rules_apple//apple:repositories.bzl", "apple_rules_dependencies")
apple_rules_dependencies()
