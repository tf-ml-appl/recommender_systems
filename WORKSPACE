workspace(name='tf_recommender_systems')

load("//third_party:build_dep.bzl", "tensorflow_http_archive")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "bazel_toolchains",
    sha256 = "294cdd859e57fcaf101d4301978c408c88683fbc46fbc1a3829da92afbea55fb",
    strip_prefix = "bazel-toolchains-8c717f8258cd5f6c7a45b97d974292755852b658",
    urls = [
        "http://mirror.tensorflow.org/github.com/bazelbuild/bazel-toolchains/archive/8c717f8258cd5f6c7a45b97d974292755852b658.tar.gz",
        "https://github.com/bazelbuild/bazel-toolchains/archive/8c717f8258cd5f6c7a45b97d974292755852b658.tar.gz",
    ],
)

tensorflow_http_archive(
    name = "org_tensorflow",
    sha256 = "47edef97c9b23661fd63621d522454f30772ac70a1fb5ff82864e566ef86be78",
    git_commit = "f3cc513887e06150b6f870c522220dabedc58920",
)