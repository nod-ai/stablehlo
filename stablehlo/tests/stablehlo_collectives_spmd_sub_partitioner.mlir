// RUN: stablehlo-opt \
// RUN:   --stablehlo-xla-cc-lib-path="%xla_cc_lib_path" \
// RUN:   --super-sub-device-map-file="%test_source_root/super_sub_device_map_4x3.yaml" \
// RUN:   --pass-pipeline="builtin.module(stablehlo-collectives-spmd-sub-partitioner)" \
// RUN:   --split-input-file "%s" \
// RUN: | FileCheck "%s"

//        CHECK-LABEL: module @all_gather
module @all_gather attributes {
  mhlo.num_partitions = 2 : i32,
  mhlo.num_replicas = 2 : i32,
  mhlo.spmd_parameters_shardings = ["{replicated}"],
  mhlo.spmd_output_sharding = "{replicated}"
} {
  func.func @main(
//              CHECK: %[[ARG0:[A-Za-z0-9]+]]: tensor<9x2xf32>
    %arg0: tensor<9x2xf32>
    ) -> tensor<9x4xf32> {
//              CHECK: %[[FULL_TO_SHARD_SHAPE_RES:[A-Za-z0-9]+]] = stablehlo.custom_call @SPMDFullToShardShape(%[[ARG0]]) {mhlo.sharding = "{manual}"} : (tensor<9x2xf32>) -> tensor<3x2xf32>
//              CHECK: %[[ALL_GATHER_RES:[A-Za-z0-9]+]] = "stablehlo.all_gather"(%[[FULL_TO_SHARD_SHAPE_RES]]) {
    %0 = "stablehlo.all_gather"(%arg0) {
      all_gather_dim = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
// CHECK-DAG{LITERAL}: replica_groups = dense<[[0, 3], [6, 9], [1, 4], [7, 10], [2, 5], [8, 11]]>
      replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>,
      use_global_device_ids,
// CHECK-DAG{LITERAL}: mhlo.sharding = "{manual}"
      mhlo.sharding = "{devices=[3,1]0,1,2}",
// CHECK-DAG{LITERAL}: device_domain = "complete"
      device_domain = "super"
//     CHECK{LITERAL}: } : (tensor<3x2xf32>) -> tensor<3x4xf32>
    } : (tensor<9x2xf32>) -> tensor<9x4xf32>
//              CHECK: %[[SHARD_TO_FULL_SHAPE_RES:[A-Za-z0-9]+]] = stablehlo.custom_call @SPMDShardToFullShape(%[[ALL_GATHER_RES]]) {mhlo.sharding = "{devices=[3,1]0,1,2}"} : (tensor<3x4xf32>) -> tensor<9x4xf32>
//              CHECK: return %[[SHARD_TO_FULL_SHAPE_RES]] : tensor<9x4xf32>
    return %0 : tensor<9x4xf32>
  }
}

// -----

//        CHECK-LABEL: module @all_reduce
module @all_reduce attributes {
  mhlo.num_partitions = 2 : i32,
  mhlo.num_replicas = 2 : i32,
  mhlo.spmd_parameters_shardings = ["{replicated}"],
  mhlo.spmd_output_sharding = "{replicated}"
} {
  func.func @main(
//              CHECK: %[[ARG0:[A-Za-z0-9]+]]: tensor<9x2xf32>
    %arg0: tensor<9x2xf32>
    ) -> tensor<9x2xf32> {
//              CHECK: %[[FULL_TO_SHARD_SHAPE_RES:[A-Za-z0-9]+]] = stablehlo.custom_call @SPMDFullToShardShape(%[[ARG0]]) {mhlo.sharding = "{manual}"} : (tensor<9x2xf32>) -> tensor<3x2xf32>
//              CHECK: %[[ALL_REDUCE_RES:[A-Za-z0-9]+]] = "stablehlo.all_reduce"(%[[FULL_TO_SHARD_SHAPE_RES]]) ({
    %0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) {
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
// CHECK-DAG{LITERAL}: replica_groups = dense<[[0, 3], [6, 9], [1, 4], [7, 10], [2, 5], [8, 11]]>
      replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>,
      use_global_device_ids,
// CHECK-DAG{LITERAL}: mhlo.sharding = "{manual}"
      mhlo.sharding = "{devices=[3,1]0,1,2}",
// CHECK-DAG{LITERAL}: device_domain = "complete"
      device_domain = "super"
//     CHECK{LITERAL}: } : (tensor<3x2xf32>) -> tensor<3x2xf32>
    } : (tensor<9x2xf32>) -> tensor<9x2xf32>
//              CHECK: %[[SHARD_TO_FULL_SHAPE_RES:[A-Za-z0-9]+]] = stablehlo.custom_call @SPMDShardToFullShape(%[[ALL_REDUCE_RES]]) {mhlo.sharding = "{devices=[3,1]0,1,2}"} : (tensor<3x2xf32>) -> tensor<9x2xf32>
//              CHECK: return %[[SHARD_TO_FULL_SHAPE_RES]] : tensor<9x2xf32>
    return %0 : tensor<9x2xf32>
  }
}

// -----

//        CHECK-LABEL: module @all_to_all
module @all_to_all attributes {
  mhlo.num_partitions = 2 : i32,
  mhlo.num_replicas = 2 : i32,
  mhlo.spmd_parameters_shardings = ["{replicated}"],
  mhlo.spmd_output_sharding = "{replicated}"
} {
  func.func @main(
//              CHECK: %[[ARG0:[A-Za-z0-9]+]]: tensor<9x4x2xf32>
    %arg0: tensor<9x4x2xf32>
    ) -> tensor<9x2x4xf32> {
//              CHECK: %[[FULL_TO_SHARD_SHAPE_RES:[A-Za-z0-9]+]] = stablehlo.custom_call @SPMDFullToShardShape(%[[ARG0]]) {mhlo.sharding = "{manual}"} : (tensor<9x4x2xf32>) -> tensor<3x4x2xf32>
//              CHECK: %[[ALL_TO_ALL_RES:[A-Za-z0-9]+]] = "stablehlo.all_to_all"(%[[FULL_TO_SHARD_SHAPE_RES]]) {
    %0 = "stablehlo.all_to_all"(%arg0) {
      split_dimension = 1 : i64,
      concat_dimension = 2 : i64,
      split_count = 2 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
// CHECK-DAG{LITERAL}: replica_groups = dense<[[0, 3], [6, 9], [1, 4], [7, 10], [2, 5], [8, 11]]>
      replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>,
      use_global_device_ids,
// CHECK-DAG{LITERAL}: mhlo.sharding = "{manual}"
      mhlo.sharding = "{devices=[3,1,1]0,1,2}",
// CHECK-DAG{LITERAL}: device_domain = "complete"
      device_domain = "super"
//     CHECK{LITERAL}: } : (tensor<3x4x2xf32>) -> tensor<3x2x4xf32>
    } : (tensor<9x4x2xf32>) -> tensor<9x2x4xf32>
//              CHECK: %[[SHARD_TO_FULL_SHAPE_RES:[A-Za-z0-9]+]] = stablehlo.custom_call @SPMDShardToFullShape(%[[ALL_TO_ALL_RES]]) {mhlo.sharding = "{devices=[3,1,1]0,1,2}"} : (tensor<3x2x4xf32>) -> tensor<9x2x4xf32>
//              CHECK: return %[[SHARD_TO_FULL_SHAPE_RES]] : tensor<9x2x4xf32>
    return %0 : tensor<9x2x4xf32>
  }
}

// -----

//        CHECK-LABEL: module @reduce_scatter
module @reduce_scatter attributes {
  mhlo.num_partitions = 2 : i32,
  mhlo.num_replicas = 2 : i32,
  mhlo.spmd_parameters_shardings = ["{replicated}"],
  mhlo.spmd_output_sharding = "{replicated}"
} {
  func.func @main(
//              CHECK: %[[ARG0:[A-Za-z0-9]+]]: tensor<9x4xf32>
    %arg0: tensor<9x4xf32>
    ) -> tensor<9x2xf32> {
//              CHECK: %[[FULL_TO_SHARD_SHAPE_RES:[A-Za-z0-9]+]] = stablehlo.custom_call @SPMDFullToShardShape(%[[ARG0]]) {mhlo.sharding = "{manual}"} : (tensor<9x4xf32>) -> tensor<3x4xf32>
//              CHECK: %[[REDUCE_SCATTER_RES:[A-Za-z0-9]+]] = "stablehlo.reduce_scatter"(%[[FULL_TO_SHARD_SHAPE_RES]]) ({
    %0 = "stablehlo.reduce_scatter"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) {
      scatter_dimension = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
// CHECK-DAG{LITERAL}: replica_groups = dense<[[0, 3], [6, 9], [1, 4], [7, 10], [2, 5], [8, 11]]>
      replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>,
      use_global_device_ids,
// CHECK-DAG{LITERAL}: mhlo.sharding = "{manual}"
      mhlo.sharding = "{devices=[3,1]0,1,2}",
// CHECK-DAG{LITERAL}: device_domain = "complete"
      device_domain = "super"
//     CHECK{LITERAL}: } : (tensor<3x4xf32>) -> tensor<3x2xf32>
    } : (tensor<9x4xf32>) -> tensor<9x2xf32>
//              CHECK: %[[SHARD_TO_FULL_SHAPE_RES:[A-Za-z0-9]+]] = stablehlo.custom_call @SPMDShardToFullShape(%[[REDUCE_SCATTER_RES]]) {mhlo.sharding = "{devices=[3,1]0,1,2}"} : (tensor<3x2xf32>) -> tensor<9x2xf32>
//              CHECK: return %[[SHARD_TO_FULL_SHAPE_RES]] : tensor<9x2xf32>
    return %0 : tensor<9x2xf32>
  }
}

// -----

//        CHECK-LABEL: module @sharding_module_attributes
module @sharding_module_attributes attributes {
  mhlo.num_partitions = 2 : i32,
  mhlo.num_replicas = 2 : i32,
//     CHECK{LITERAL}: mhlo.frontend_attributes = {
//     CHECK{LITERAL}: super_partition_spmd_output_sharding = "{replicated}"
//     CHECK{LITERAL}: super_partition_spmd_parameters_sharding = "{{replicated}}"
// CHECK-NOT{LITERAL}: mhlo.spmd_parameters_shardings
  mhlo.spmd_parameters_shardings = ["{replicated}"],
// CHECK-NOT{LITERAL}: mhlo.spmd_output_sharding
  mhlo.spmd_output_sharding = "{replicated}"
} {
  func.func @main(%arg0: tensor<1xf32>) -> tensor<1xf32> {
    return %arg0 : tensor<1xf32>
  }
}

// -----

//        CHECK-LABEL: module @num_partitions_and_replicas_attributes
module @num_partitions_and_replicas_attributes attributes {
//     CHECK{LITERAL}: mhlo.frontend_attributes
  mhlo.frontend_attributes = {
// CHECK-NOT{LITERAL}: sub_partition_num_partitions
    sub_partition_num_partitions = "3",
// CHECK-NOT{LITERAL}: sub_partition_num_replicas
    sub_partition_num_replicas = "1"
//     CHECK{LITERAL}: super_partition_num_partitions = "1"
//     CHECK{LITERAL}: super_partition_num_replicas = "4"
  },
//     CHECK{LITERAL}: mhlo.num_partitions = 3 : i32
  mhlo.num_partitions = 1 : i32,
//     CHECK{LITERAL}: mhlo.num_replicas = 1 : i32
  mhlo.num_replicas = 4 : i32,
  mhlo.spmd_parameters_shardings = ["{replicated}"],
  mhlo.spmd_output_sharding = "{replicated}"
} {
  func.func @main(%arg0: tensor<1xf32>) -> tensor<1xf32> {
    return %arg0 : tensor<1xf32>
  }
}

// -----

//        CHECK-LABEL: module @sub_partition_num_partitions_and_replicas_from_super_sub_device_map
module @sub_partition_num_partitions_and_replicas_from_super_sub_device_map attributes {
//     CHECK{LITERAL}: mhlo.frontend_attributes
  mhlo.frontend_attributes = {
//     CHECK{LITERAL}: super_partition_num_partitions = "1"
//     CHECK{LITERAL}: super_partition_num_replicas = "4"
  },
//     CHECK{LITERAL}: mhlo.num_partitions = 1 : i32
  mhlo.num_partitions = 1 : i32,
//     CHECK{LITERAL}: mhlo.num_replicas = 3 : i32
  mhlo.num_replicas = 4 : i32,
  mhlo.spmd_parameters_shardings = ["{replicated}"],
  mhlo.spmd_output_sharding = "{replicated}"
} {
  func.func @main(%arg0: tensor<1xf32>) -> tensor<1xf32> {
    return %arg0 : tensor<1xf32>
  }
}
