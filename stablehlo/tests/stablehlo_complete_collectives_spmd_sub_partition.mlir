// RUN: stablehlo-opt \
// RUN:   --super-sub-device-map-file="%test_source_root/super_sub_device_map.yaml" \
// RUN:   --pass-pipeline="builtin.module(func.func(stablehlo-complete-collectives-spmd-sub-partition))" \
// RUN:   --split-input-file "%s" \
// RUN: | FileCheck "%s"

//       CHECK: func.func @all_reduce
func.func @all_reduce(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "stablehlo.all_reduce"(%arg0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }) {
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
//   CHECK-DAG{LITERAL}: replica_groups = dense<[[0, 1, 2], [3, 4, 5]]>
    replica_groups = dense<[[0, 1, 2]]> : tensor<1x3xi64>,
    use_global_device_ids,
//   CHECK-DAG{LITERAL}: device_domain = "complete"
    device_domain = "sub"
  } : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
