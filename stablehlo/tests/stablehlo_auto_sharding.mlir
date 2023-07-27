// RUN: stablehlo-opt \
// RUN:   --stablehlo-xla-cc-lib-path="%xla_cc_lib_path" \
// RUN:   --pass-pipeline="builtin.module(stablehlo-xla-auto-sharding{device_mesh_shape=2,4 device_mesh_ids=0,1,2,3,4,5,6,7 device_mesh_alpha=1,2 device_mesh_beta=3,4})" \
// RUN:   --split-input-file %s \
// RUN: | FileCheck %s

// CHECK{LITERAL}: module @pjit_f attributes {mhlo.cross_program_prefetches = [], mhlo.dynamic_parameter_bindings = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
// CHECK{LITERAL}:   func.func @main(%arg0: tensor<16x16xi32> {mhlo.sharding = "{devices=[2,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}"}, %arg1: tensor<16x16xi32> {mhlo.sharding = "{devices=[1,4,2]0,4,1,5,2,6,3,7 last_tile_dim_replicate}"}) -> (tensor<16x16xi32> {mhlo.sharding = "{devices=[2,4]0,1,2,3,4,5,6,7}"}) {
// CHECK{LITERAL}:     %0 = stablehlo.dot %arg0, %arg1, precision = [DEFAULT, DEFAULT] {mhlo.sharding = "{devices=[2,4]0,1,2,3,4,5,6,7}"} : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi32>
// CHECK{LITERAL}:     %1 = stablehlo.add %arg0, %0 {mhlo.sharding = "{devices=[2,4]0,1,2,3,4,5,6,7}"} : tensor<16x16xi32>
// CHECK{LITERAL}:     return %1 : tensor<16x16xi32>
// CHECK{LITERAL}:   }
// CHECK{LITERAL}: }
module @pjit_f {
  func.func public @main(
    %arg0: tensor<16x16xi32>,
    %arg1: tensor<16x16xi32>
  ) -> (tensor<16x16xi32>) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi32>
    %1 = stablehlo.add %arg0, %0 : tensor<16x16xi32>
    return %1 : tensor<16x16xi32>
  }
}
