// RUN: stablehlo-opt \
// RUN:   --stablehlo-xla-cc-lib-path="%xla_cc_lib_path" \
// RUN:   --pass-pipeline="builtin.module(stablehlo-xla-spmd-partitioner)" \
// RUN:   --split-input-file %s \
// RUN: | FileCheck %s


// CHECK{LITERAL}: module @spmd_partitioner attributes {mhlo.cross_program_prefetches = [], mhlo.is_dynamic = false, mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32, mhlo.spmd_output_sharding = "{devices=[1,4,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}", mhlo.spmd_parameters_shardings = ["{devices=[1,2,4]0,2,4,6,1,3,5,7 last_tile_dim_replicate}", "{devices=[1,2,4]0,2,4,6,1,3,5,7 last_tile_dim_replicate}"], mhlo.use_auto_spmd_partitioning = false} {
// CHECK{LITERAL}:   func.func @main(%arg0: tensor<16x8xi32> {mhlo.sharding = "{devices=[1,2,4]0,2,4,6,1,3,5,7 last_tile_dim_replicate}"}, %arg1: tensor<16x8xi32> {mhlo.sharding = "{devices=[1,2,4]0,2,4,6,1,3,5,7 last_tile_dim_replicate}"}) -> tensor<16x4xi32> {
// CHECK{LITERAL}:     %0 = "stablehlo.collective_permute"(%arg0) {channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 0], [2, 1], [4, 2], [6, 3], [1, 4], [3, 5], [5, 6], [7, 7]]> : tensor<8x2xi64>} : (tensor<16x8xi32>) -> tensor<16x8xi32>
// CHECK{LITERAL}:     %1 = "stablehlo.all_gather"(%0) {all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 2, type = 0>, replica_groups = dense<[[0, 4], [2, 6], [1, 5], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids} : (tensor<16x8xi32>) -> tensor<16x16xi32>
// CHECK{LITERAL}:     %2 = stablehlo.dot %1, %arg1, precision = [DEFAULT, DEFAULT] : (tensor<16x16xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
// CHECK{LITERAL}:     %3 = stablehlo.add %arg0, %2 : tensor<16x8xi32>
// CHECK{LITERAL}:     %4 = "stablehlo.all_gather"(%3) {all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 3, type = 0>, replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>, use_global_device_ids} : (tensor<16x8xi32>) -> tensor<16x16xi32>
// CHECK{LITERAL}:     %5 = stablehlo.constant dense<0> : tensor<i32>
// CHECK{LITERAL}:     %6 = stablehlo.constant dense<[0, 4, 8, 12]> : tensor<4xi32>
// CHECK{LITERAL}:     %7 = stablehlo.constant dense<[0, 0, 1, 1, 2, 2, 3, 3]> : tensor<8xui32>
// CHECK{LITERAL}:     %8 = stablehlo.partition_id : tensor<ui32>
// CHECK{LITERAL}:     %9 = stablehlo.dynamic_slice %7, %8, sizes = [1] : (tensor<8xui32>, tensor<ui32>) -> tensor<1xui32>
// CHECK{LITERAL}:     %10 = stablehlo.reshape %9 : (tensor<1xui32>) -> tensor<ui32>
// CHECK{LITERAL}:     %11 = stablehlo.dynamic_slice %6, %10, sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
// CHECK{LITERAL}:     %12 = stablehlo.reshape %11 : (tensor<1xi32>) -> tensor<i32>
// CHECK{LITERAL}:     %13 = stablehlo.dynamic_slice %4, %5, %12, sizes = [16, 4] : (tensor<16x16xi32>, tensor<i32>, tensor<i32>) -> tensor<16x4xi32>
// CHECK{LITERAL}:     return %13 : tensor<16x4xi32>
// CHECK{LITERAL}:   }
// CHECK{LITERAL}: }
module @spmd_partitioner attributes {
    mhlo.num_partitions = 8,
    mhlo.num_replicas = 1,
    mhlo.cross_program_prefetches = [],
    mhlo.dynamic_parameter_bindings = [],
    mhlo.is_dynamic = false,
    mhlo.use_auto_spmd_partitioning = false} {
  func.func public @main(
    %arg0: tensor<16x16xi32> {mhlo.sharding = "{devices=[1,2,4]0,2,4,6,1,3,5,7 last_tile_dim_replicate}"},
    %arg1: tensor<16x16xi32> {mhlo.sharding = "{devices=[1,2,4]0,2,4,6,1,3,5,7 last_tile_dim_replicate}"}
  ) -> (tensor<16x16xi32> {mhlo.sharding = "{devices=[1,4,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      {mhlo.sharding = "{devices=[1,2,4]0,2,4,6,1,3,5,7 last_tile_dim_replicate}"}
      : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi32>
    %1 = stablehlo.add %arg0, %0
      {mhlo.sharding = "{devices=[1,2,4]0,2,4,6,1,3,5,7 last_tile_dim_replicate}"}
      : tensor<16x16xi32>
    return %1 : tensor<16x16xi32>
  }
}

// -----

//        CHECK-LABEL: module @all_gather_cross_replica
module @all_gather_cross_replica attributes {
  mhlo.num_partitions = 3 : i32,
  mhlo.num_replicas = 2 : i32
} {
  func.func @main(
    %arg0: tensor<9x2xf32> {mhlo.sharding = "{devices=[3,1]0,1,2}"}
    ) -> (tensor<9x4xf32> {mhlo.sharding = "{devices=[3,1]0,1,2}"}) {
    %0 = "stablehlo.all_gather"(%arg0) {
      all_gather_dim = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      mhlo.sharding = "{devices=[3,1]0,1,2}"
    } : (tensor<9x2xf32>) -> tensor<9x4xf32>
    return %0 : tensor<9x4xf32>
  }
}

// -----

//        CHECK-LABEL: module @all_reduce_cross_replica
module @all_reduce_cross_replica attributes {
  mhlo.num_partitions = 3 : i32,
  mhlo.num_replicas = 2 : i32
} {
  func.func @main(
    %arg0: tensor<9x2xf32> {mhlo.sharding = "{devices=[3,1]0,1,2}"}
    ) -> (tensor<9x2xf32> {mhlo.sharding = "{devices=[3,1]0,1,2}"}) {
    %0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) {
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      mhlo.sharding = "{devices=[3,1]0,1,2}"
    } : (tensor<9x2xf32>) -> tensor<9x2xf32>
    return %0 : tensor<9x2xf32>
  }
}
