// RUN: stablehlo-opt \
// RUN:   --pass-pipeline="builtin.module(func.func(stablehlo-distributed-legalize))" \
// RUN:   --split-input-file %s | FileCheck %s

// CHECK-LABEL: @sharding_custom_op
func.func @sharding_custom_op(
//       CHECK: %[[ARG0:[A-Za-z0-9]+]]: tensor<4xf32>
  %arg0: tensor<4xf32>
) -> tensor<4xf32> {
//       CHECK: %[[RES:[A-Za-z0-9]+]] = stablehlo.custom_call @Sharding(%[[ARG0]]) {mhlo.sharding = "{devices=[2]0,1}"} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = stablehlo.custom_call @sharding.tensor_annotate(%arg0)
    {mhlo.sharding = "{devices=[2]0,1}"}
    : (tensor<4xf32>) -> tensor<4xf32>
//       CHECK: return %[[RES]] : tensor<4xf32>
  return %0 : tensor<4xf32>
}
