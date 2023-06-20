// RUN: stablehlo-opt \
// RUN:   --stablehlo-xla-cc-lib-path="%xla_cc_lib_path" \
// RUN:   --pass-pipeline="builtin.module(stablehlo-xla-collectives-optimization)" \
// RUN:   --split-input-file %s \
// RUN: | FileCheck %s

//       CHECK: module @all_reduce_reassociate
module @all_reduce_reassociate {
//       CHECK: func.func @main
  func.func public @main(
//  CHECK-SAME: [[ARG0:%[A-Za-z0-9_]+]]: tensor<2x3xf32>
    %arg0: tensor<2x3xf32>,
//  CHECK-SAME: [[ARG1:%[A-Za-z0-9_]+]]: tensor<2x3xf32>
    %arg1: tensor<2x3xf32>
  ) -> tensor<2x3xf32> {
//       CHECK: [[ADD:%.+]] = stablehlo.add [[ARG0]], [[ARG1]] : tensor<2x3xf32>
//       CHECK: [[AR:%.+]] = "stablehlo.all_reduce"([[ADD]])
    %ar0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%a: tensor<f32>, %b: tensor<f32>):
      %res = stablehlo.add %a, %b : tensor<f32>
      stablehlo.return %res : tensor<f32>
    }) {
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>,
      use_global_device_ids
      } : (tensor<2x3xf32>) -> tensor<2x3xf32>
    %ar1 = "stablehlo.all_reduce"(%arg1) ({
    ^bb0(%a: tensor<f32>, %b: tensor<f32>):
      %res = stablehlo.add %a, %b : tensor<f32>
      stablehlo.return %res : tensor<f32>
    }) {
      channel_handle = #stablehlo.channel_handle<handle = 2, type = 0>,
      replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>,
      use_global_device_ids
      } : (tensor<2x3xf32>) -> tensor<2x3xf32>
    %add = stablehlo.add %ar0, %ar1 : tensor<2x3xf32>
//       CHECK: return [[AR]] : tensor<2x3xf32>
    return %add : tensor<2x3xf32>
  }
}
