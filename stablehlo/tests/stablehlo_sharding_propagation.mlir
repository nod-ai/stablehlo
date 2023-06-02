// RUN: stablehlo-opt \
// RUN:   --stablehlo-xla-cc-lib-path="%xla_cc_lib_path" \
// RUN:   --pass-pipeline="builtin.module(stablehlo-xla-sharding-propagation{is_spmd=0 propagate_metadata=0 allow_spmd_sharding_propagation_to_output=1 allow_spmd_sharding_propagation_to_parameters=1 cse_prevention_only=false})" \
// RUN:   --split-input-file %s \
// RUN: | FileCheck %s

//       CHECK: module @pjit_f
//   CHECK-NOT: mhlo.num_partitions
//   CHECK-NOT: mhlo.num_replicas
module @pjit_f attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
//       CHECK: func.func @main
//  CHECK-SAME: [[ARG0:%[A-Za-z0-9_]+]]: tensor<16x16xi32> {mhlo.sharding = "{devices=[1,2,4]0,2,4,6,1,3,5,7 last_tile_dim_replicate}"}
//  CHECK-SAME: [[ARG1:%[A-Za-z0-9_]+]]: tensor<16x16xi32> {mhlo.sharding = "{devices=[1,2,4]0,2,4,6,1,3,5,7 last_tile_dim_replicate}"}
//  CHECK-SAME: -> (tensor<16x16xi32> {mhlo.sharding = "{devices=[1,4,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}"})
  func.func public @main(
    %arg0: tensor<16x16xi32> {mhlo.sharding = "{devices=[1,2,4]0,2,4,6,1,3,5,7 last_tile_dim_replicate}"},
    %arg1: tensor<16x16xi32> {mhlo.sharding = "{devices=[1,2,4]0,2,4,6,1,3,5,7 last_tile_dim_replicate}"}
  ) -> (tensor<16x16xi32> {mhlo.sharding = "{devices=[1,4,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}"}) {
//       CHECK: [[DOT:%.+]] = stablehlo.dot [[ARG0]], [[ARG1]], precision = [DEFAULT, DEFAULT]
//  CHECK-SAME: {mhlo.sharding = "{devices=[1,2,4]0,2,4,6,1,3,5,7 last_tile_dim_replicate}"}
//  CHECK-SAME: : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi32>
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi32>
//       CHECK: [[ADD:%.+]] = stablehlo.add [[ARG0]], [[DOT]]
//  CHECK-SAME: {mhlo.sharding = "{devices=[1,2,4]0,2,4,6,1,3,5,7 last_tile_dim_replicate}"}
//  CHECK-SAME: : tensor<16x16xi32>
    %1 = stablehlo.add %arg0, %0 : tensor<16x16xi32>
//       CHECK: return [[ADD]] : tensor<16x16xi32>
    return %1 : tensor<16x16xi32>
  }
}
