// RUN: stablehlo-opt \
// RUN:   --stablehlo-xla-cc-lib-path="%xla_cc_lib_path" \
// RUN:   --pass-pipeline="builtin.module(stablehlo-xla-sharding-propagation{is_spmd=1 propagate_metadata=0 allow_spmd_sharding_propagation_to_output=1 allow_spmd_sharding_propagation_to_parameters=1})" \
// RUN:   --split-input-file %s \
// RUN: | FileCheck %s

// CHECK-LABEL: module @short_func
//  CHECK-SAME: mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32
module @short_func attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
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

// -----

//        CHECK-LABEL: module @all_gather_global_device_ids
module @all_gather_global_device_ids {
  func.func @main(
    %arg0: tensor<9x2xf32> {mhlo.sharding = "{devices=[1,2]0,1}"}
    ) -> tensor<9x4xf32> {
    %0 = "stablehlo.all_gather"(%arg0) {
      all_gather_dim = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>,
      use_global_device_ids
    } : (tensor<9x2xf32>) -> tensor<9x4xf32>
    return %0 : tensor<9x4xf32>
  }
}

// -----

//        CHECK-LABEL: module @all_gather_cross_replica
module @all_gather_cross_replica {
  func.func @main(
    %arg0: tensor<9x2xf32> {mhlo.sharding = "{devices=[1,2]0,1}"}
    ) -> tensor<9x4xf32> {
    %0 = "stablehlo.all_gather"(%arg0) {
      all_gather_dim = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      use_global_device_ids
    } : (tensor<9x2xf32>) -> tensor<9x4xf32>
    return %0 : tensor<9x4xf32>
  }
}

// -----

//        CHECK-LABEL: module @all_gather_cross_replica_and_partition
module @all_gather_cross_replica_and_partition {
  func.func @main(
    %arg0: tensor<9x2xf32> {mhlo.sharding = "{devices=[1,2]0,1}"}
    ) -> tensor<9x4xf32> {
    %0 = "stablehlo.all_gather"(%arg0) {
      all_gather_dim = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<9x2xf32>) -> tensor<9x4xf32>
    return %0 : tensor<9x4xf32>
  }
}

// -----

//        CHECK-LABEL: module @all_gather_propagate_from_results
module @all_gather_propagate_from_results {
  func.func @main(
    %arg0: tensor<9x2xf32>
    ) -> (tensor<9x4xf32> {mhlo.sharding = "{devices=[1,2]0,1}"}) {
    %0 = "stablehlo.all_gather"(%arg0) {
      all_gather_dim = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<9x2xf32>) -> tensor<9x4xf32>
    return %0 : tensor<9x4xf32>
  }
}

// -----

//        CHECK-LABEL: module @all_reduce_global_device_ids
module @all_reduce_global_device_ids {
  func.func @main(
    %arg0: tensor<9x2xf32> {mhlo.sharding = "{devices=[1,2]0,1}"}
    ) -> tensor<9x2xf32> {
    %0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) {
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>,
      use_global_device_ids
    } : (tensor<9x2xf32>) -> tensor<9x2xf32>
    return %0 : tensor<9x2xf32>
  }
}

// -----

//        CHECK-LABEL: module @all_reduce_cross_replica
module @all_reduce_cross_replica {
  func.func @main(
    %arg0: tensor<9x2xf32> {mhlo.sharding = "{devices=[1,2]0,1}"}
    ) -> tensor<9x2xf32> {
    %0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) {
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<9x2xf32>) -> tensor<9x2xf32>
    return %0 : tensor<9x2xf32>
  }
}

// -----

//        CHECK-LABEL: module @all_reduce_cross_replica_and_partition
module @all_reduce_cross_replica_and_partition {
  func.func @main(
    %arg0: tensor<9x2xf32> {mhlo.sharding = "{devices=[1,2]0,1}"}
    ) -> tensor<9x2xf32> {
    %0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) {
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<9x2xf32>) -> tensor<9x2xf32>
    return %0 : tensor<9x2xf32>
  }
}

// -----

//        CHECK-LABEL: module @all_reduce_cross_without_channel_id
module @all_reduce_cross_without_channel_id {
  func.func @main(
    %arg0: tensor<9x2xf32> {mhlo.sharding = "{devices=[1,2]0,1}"}
    ) -> tensor<9x2xf32> {
    %0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<9x2xf32>) -> tensor<9x2xf32>
    return %0 : tensor<9x2xf32>
  }
}

// -----

//        CHECK-LABEL: module @resharding
module @resharding {
  func.func @main(
    %arg0: tensor<9x2xf32> {mhlo.sharding = "{devices=[1,2]0,1}"},
    %arg1: tensor<9x2xf32> {mhlo.sharding = "{devices=[2,1]0,1}"}
    ) -> tensor<9x2xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<9x2xf32>
    %1 = stablehlo.add %0, %0 {mhlo.sharding = "{devices=[2,1]0,1}"} : tensor<9x2xf32>
    %2 = stablehlo.custom_call @Sharding(%1) {mhlo.sharding = "{replicated}"} : (tensor<9x2xf32>) -> tensor<9x2xf32>
    return %2 : tensor<9x2xf32>
  }
}

// -----

//        CHECK-LABEL: module @propagate_from_results_to_parameters
module @propagate_from_results_to_parameters {
  func.func @main(
    %arg0: tensor<9x2xf32>,
    %arg1: tensor<9x2xf32>
    ) -> (tensor<9x2xf32> {mhlo.sharding = "{devices=[2,1]0,1}"}) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<9x2xf32>
    return %0 : tensor<9x2xf32>
  }
}

// -----

//        CHECK-LABEL: module @conv
module @conv attributes {
  mhlo.num_partitions = 2 : i32,
  mhlo.num_replicas = 1 : i32
} {
  func.func public @main(
    %arg0: tensor<8x480x640x3xf32> {mhlo.sharding = "{devices=[1,1,2,1]0,1}"},
    %arg1: tensor<4x5x3x16xf32> {mhlo.sharding = "{replicated}"}
  ) -> (tensor<8x477x636x16xf32>) {
    %0 = stablehlo.convolution(%arg0, %arg1)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {
        stride = [1, 1],
        pad = [[0, 0], [0, 0]],
        lhs_dilate = [1, 1],
        rhs_dilate = [1, 1],
        reverse = [0, 0]
      } {
        batch_group_count = 1 : i64,
        feature_group_count = 1 : i64,
        precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
      } : (tensor<8x480x640x3xf32>, tensor<4x5x3x16xf32>) -> tensor<8x477x636x16xf32>
    return %0 : tensor<8x477x636x16xf32>
  }
}