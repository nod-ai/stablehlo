// RUN: stablehlo-opt \
// RUN:   --stablehlo-xla-cc-lib-path="%xla_cc_lib_path" \
// RUN:   --super-sub-device-map-file="%test_source_root/super_sub_device_map_4x3.yaml" \
// RUN:   --pass-pipeline="builtin.module(func.func(stablehlo-collectives-spmd-sub-partitioner))" \
// RUN:   --split-input-file "%s" \
// RUN: | FileCheck "%s"

//        CHECK-LABEL: func.func @all_gather
func.func @all_gather(
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

// -----

//        CHECK-LABEL: func.func @all_reduce
func.func @all_reduce(
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

//        CHECK-LABEL: func.func @all_to_all
func.func @all_to_all(
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

// -----

//        CHECK-LABEL: func.func @reduce_scatter
func.func @reduce_scatter(
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
