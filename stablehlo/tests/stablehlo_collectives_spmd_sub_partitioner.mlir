// RUN: stablehlo-opt \
// RUN:   --super-sub-device-map-file="%test_source_root/super_sub_device_map_4x3.yaml" \
// RUN:   --pass-pipeline="builtin.module(func.func(stablehlo-collectives-spmd-sub-partitioner))" \
// RUN:   --split-input-file "%s" \
// RUN: | FileCheck "%s"

//        CHECK-LABEL: func.func @all_gather
func.func @all_gather(
//              CHECK: %[[ARG0:[A-Za-z0-9]+]]: tensor<9x2xf32>
  %arg0: tensor<9x2xf32>
  ) -> tensor<9x4xf32> {
//              CHECK: %[[PRE_SHARDING_RES:[A-Za-z0-9]+]] = stablehlo.custom_call @Sharding(%[[ARG0]]) {mhlo.sharding = "{devices=[3,1]0,1,2}"} : (tensor<9x2xf32>) -> tensor<9x2xf32>
//              CHECK: %[[FULL_TO_SHARD_SHAPE_RES:[A-Za-z0-9]+]] = stablehlo.custom_call @SPMDFullToShardShape(%[[PRE_SHARDING_RES]]) {mhlo.sharding = "{manual}"} : (tensor<9x2xf32>) -> tensor<3x2xf32> 
//              CHECK: %[[ALL_GATHER_RES:[A-Za-z0-9]+]] = "stablehlo.all_gather"(%[[FULL_TO_SHARD_SHAPE_RES]]) {
  %1 = "stablehlo.all_gather"(%0) {
    all_gather_dim = 1 : i64,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
// CHECK-DAG{LITERAL}: replica_groups = dense<[[0, 3], [1, 4], [2, 5], [6, 9], [7, 10], [8, 11]]>
    replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>,
    use_global_device_ids,
    mhlo.sharding = "{devices=[3,1]0,1,2}",
// CHECK-DAG{LITERAL}: device_domain = "complete"
    device_domain = "super"
//     CHECK{LITERAL}: } : (tensor<3x2xf32>) -> tensor<3x4xf32>
  } : (tensor<9x2xf32>) -> tensor<9x4xf32>
//              CHECK: %[[POST_SHARDING_RES:[A-Za-z0-9]+]] = stablehlo.custom_call @Sharding(%[[ALL_GATHER_RES]]) {mhlo.sharding = "{manual}"} : (tensor<3x4xf32>) -> tensor<3x4xf32>
//              CHECK: %[[SHARD_TO_FULL_SHAPE_RES:[A-Za-z0-9]+]] = stablehlo.custom_call @SPMDShardToFullShape(%[[POST_SHARDING_RES]]) {mhlo.sharding = "{devices=[3,1]0,1,2}"} : (tensor<3x4xf32>) -> tensor<9x4xf32>
//              CHECK: return %[[SHARD_TO_FULL_SHAPE_RES]] : tensor<9x4xf32>
  return %1 : tensor<9x4xf32>
}
