// RUN: stablehlo-opt \
// RUN:   --super-sub-device-map-file="super_sub_device_map.yaml" \
// RUN:   --pass-pipeline="builtin.module(func.func(stablehlo-complete-collectives-spmd-sub-partition))" \
// RUN:   --split-input-file %s

module {
  func.func public @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) {
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
      replica_groups = dense<[[0, 1, 2]]> : tensor<1x3xi64>,
      use_global_device_ids,
      sub_partition
    } : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
