# buildifier: disable=load-on-top

workspace(name = "litert")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

# LiteRT Archive (v2.1.2) with Correct Checksum
http_archive(
    name = "litert_archive",
    url = "https://github.com/google-ai-edge/LiteRT/archive/refs/tags/v2.1.2.tar.gz",
    strip_prefix = "LiteRT-2.1.2",
    sha256 = "16079585fcd0c7fbb95585db10516d059f35b8860d4a92566e257d67e259473a",
    patch_cmds = [
        "sed 's|\"//ci/tools/python/wheel:__subpackages__\",||g; s|\"//litert:__subpackages__\"|\"//visibility:public\"|g; s|@flatbuffers//:runtime_cc|@flatbuffers//:flatbuffers|g; s|\"@qairt//:qnn_lib_headers\",|\"@qairt//:qnn_lib_headers\", \"@flatbuffers//:flatbuffers\",|g' litert/vendors/qualcomm/compiler/BUILD > litert/vendors/qualcomm/compiler/BUILD.tmp && mv litert/vendors/qualcomm/compiler/BUILD.tmp litert/vendors/qualcomm/compiler/BUILD",
        "sed 's|//litert:litert_public|//visibility:public|g' litert/vendors/qualcomm/dispatch/BUILD > litert/vendors/qualcomm/dispatch/BUILD.tmp && mv litert/vendors/qualcomm/dispatch/BUILD.tmp litert/vendors/qualcomm/dispatch/BUILD",
        "sed 's|\"LiteRtRegisterGpuAccelerator\"|\"LiteRtRegisterAcceleratorGpuOpenCl\"|g' litert/runtime/accelerators/auto_registration.cc > litert/runtime/accelerators/auto_registration.cc.tmp && mv litert/runtime/accelerators/auto_registration.cc.tmp litert/runtime/accelerators/auto_registration.cc",
        "sed 's|//litert|@litert_archive//litert|g' litert/build_common/special_rule.bzl > litert/build_common/special_rule.bzl.tmp && mv litert/build_common/special_rule.bzl.tmp litert/build_common/special_rule.bzl",
        "sed 's|@//third_party|@litert_archive//third_party|g' third_party/litert_prebuilts/workspace.bzl > third_party/litert_prebuilts/workspace.bzl.tmp && mv third_party/litert_prebuilts/workspace.bzl.tmp third_party/litert_prebuilts/workspace.bzl",
        "sed -i '/dsp_backend.h/d' litert/vendors/qualcomm/qnn_manager.cc",
        "sed -i '/case ::qnn::BackendType::kDspBackend:/,+9d' litert/vendors/qualcomm/qnn_manager.cc",
        "sed -i '/dsp_backend/d' litert/vendors/qualcomm/BUILD",
        "sed -i 's|LiteRtRegisterAcceleratorGpuOpenCl|LiteRtRegisterGpuAccelerator|g' litert/runtime/accelerators/auto_registration.cc",
    ],
    repo_mapping = {"@xla": "@local_xla"},
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
    sha256 = "3ec4399033e9691a3375703f418257417191124bcddb754e6ecb53faf68656d2",
    strip_prefix = "tensorflow-bdb78510d0ce35ea98eb298fc770657a16056a2c",
    urls = ["https://github.com/tensorflow/tensorflow/archive/bdb78510d0ce35ea98eb298fc770657a16056a2c.tar.gz"],
)

# Initialize the TensorFlow repository and all dependencies.
#
# The cascade of load() statements and tf_workspace?() calls works around the
# restriction that load() statements need to be at the top of .bzl files.
# E.g. we can not retrieve a new repository with http_archive and then load()
# a macro from that repository in the same file.
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

# Initialize hermetic Python
load("@local_xla//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@local_xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")

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
    },
)

load("@local_xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

load("@local_xla//third_party/py:python_init_pip.bzl", "python_init_pip")

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
    "@local_xla//third_party/py:python_wheel.bzl",
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

http_archive(
    name = "tqdm",
    build_file = "@litert_archive//third_party/tqdm:tqdm.BUILD",
    add_prefix = "tqdm",
    urls = [
        "https://third-party-mirror.googlesource.com/tqdm/+archive/d593e871a6b3fcc21ca5281aebda0feee0e8732e.tar.gz",
    ],
)

http_archive(
    name = "dawn",
    add_prefix = "dawn",
    urls = [
        "https://github.com/google/dawn/archive/v20250713.025201.tar.gz",
    ],
    build_file = "@litert_archive//third_party/dawn:BUILD",
)

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
