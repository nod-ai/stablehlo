// RUN: stablehlo-opt \
// RUN:   --stablehlo-xla-cc-lib-path="%xla_cc_lib_path" \
// RUN:   --stablehlo-distributed-pass-pipeline \
// RUN:   --split-input-file %s | FileCheck %s

// CHECK-LABEL: @distributed_pass_pipeline
module @distributed_pass_pipeline attributes {
  mhlo.num_partitions = 2 : i32,
  mhlo.num_replicas = 1 : i32
} {
  func.func public @main(
//       CHECK: %arg0: tensor<2x3xi32> {mhlo.sharding = "{devices=[1,2]0,1}"}
    %arg0: tensor<2x6xi32>,
//       CHECK: %arg1: tensor<3x2xi32> {mhlo.sharding = "{devices=[2,1]0,1}"}
    %arg1: tensor<6x2xi32>
//       CHECK: -> tensor<2x2xi32> {
  ) -> tensor<2x2xi32> {
    %0 = stablehlo.custom_call @sharding.tensor_annotate(%arg0)
      {mhlo.sharding = "{devices=[1,2]0,1}"}
    : (tensor<2x6xi32>) -> tensor<2x6xi32>
    %1 = stablehlo.custom_call @sharding.tensor_annotate(%arg1)
      {mhlo.sharding = "{devices=[2,1]0,1}"}
    : (tensor<6x2xi32>) -> tensor<6x2xi32>
    %2 = stablehlo.dot_general %0, %1,
      contracting_dims = [1] x [0],
      precision = [DEFAULT, DEFAULT]
      : (tensor<2x6xi32>, tensor<6x2xi32>) -> tensor<2x2xi32>
    return %2 : tensor<2x2xi32>
  }
}
