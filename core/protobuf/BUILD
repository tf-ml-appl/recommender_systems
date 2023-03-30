load("@org_tensorflow//tensorflow/tsl/platform/default:build_config.bzl", "tf_proto_library") 

tf_proto_library(
    name = "simple_layer_params_proto",
    srcs = ["simple_layer_params.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        "//tensorflow/core/frameowrk:tensor_shape.proto",
    ],
)