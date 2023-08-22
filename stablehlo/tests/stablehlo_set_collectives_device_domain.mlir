// RUN: stablehlo-opt \
// RUN:   '--pass-pipeline=builtin.module(func.func(stablehlo-set-collectives-device-domain{device_domain=super}))' \
// RUN:   --split-input-file "%s" \
// RUN: | FileCheck "%s"

//        CHECK-LABEL: func.func @op_without_device_domain
func.func @op_without_device_domain(%arg0: tensor<2xf32>) -> tensor<4xf32> {
//              CHECK: stablehlo.all_gather
//         CHECK-SAME: device_domain = "super"
  %0 = "stablehlo.all_gather"(%arg0) {
    all_gather_dim = 0 : i64,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    use_global_device_ids} : (tensor<2xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

//        CHECK-LABEL: func.func @op_with_device_domain
func.func @op_with_device_domain(%arg0: tensor<2xf32>) -> tensor<4xf32> {
//              CHECK: stablehlo.all_gather
//         CHECK-SAME: device_domain = "already_set"
  %0 = "stablehlo.all_gather"(%arg0) {
    device_domain = "already_set",
    all_gather_dim = 0 : i64,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    use_global_device_ids} : (tensor<2xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
