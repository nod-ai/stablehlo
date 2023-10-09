// RUN: stablehlo-opt \
// RUN:   --stablehlo-xla-cc-lib-path="%xla_cc_lib_path" \
// RUN:   --pass-pipeline="builtin.module(stablehlo-xla-sharding-propagation-and-spmd-partitioner{is_spmd=1 propagate_metadata=0 allow_spmd_sharding_propagation_to_output=1 allow_spmd_sharding_propagation_to_parameters=1})" \
// RUN:   --split-input-file %s \
// RUN: | FileCheck %s

// CHECK{LITERAL}: module @m attributes {mhlo.cross_program_prefetches = [], mhlo.is_dynamic = false, mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32, mhlo.spmd_output_sharding = "{devices=[2,1]0,1}", mhlo.spmd_parameters_shardings = ["{devices=[2,1]0,1}", "{devices=[2,1]0,1}"], mhlo.use_auto_spmd_partitioning = false} {
// CHECK{LITERAL}:   func.func @main(%arg0: tensor<8x16xi32> {mhlo.sharding = "{devices=[2,1]0,1}"}, %arg1: tensor<8x16xi32> {mhlo.sharding = "{devices=[2,1]0,1}"}) -> tensor<8x16xi32> {
// CHECK{LITERAL}:     %0 = stablehlo.add %arg0, %arg1 : tensor<8x16xi32>
// CHECK{LITERAL}:     %1 = "stablehlo.all_reduce"(%0) ({
// CHECK{LITERAL}:     ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
// CHECK{LITERAL}:       %3 = stablehlo.add %arg2, %arg3 : tensor<i32>
// CHECK{LITERAL}:       stablehlo.return %3 : tensor<i32>
// CHECK{LITERAL}:     }) {channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids} : (tensor<8x16xi32>) -> tensor<8x16xi32>
// CHECK{LITERAL}:     %2 = stablehlo.add %0, %1 : tensor<8x16xi32>
// CHECK{LITERAL}:     return %2 : tensor<8x16xi32>
// CHECK{LITERAL}:   }
// CHECK{LITERAL}: }
module @m attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<16x16xi32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<16x16xi32> {mhlo.sharding = "{replicated}"}) -> (tensor<16x16xi32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<16x16xi32>
    %1 = stablehlo.custom_call @Sharding(%0) {mhlo.sharding = "{devices=[2,1]0,1}"} : (tensor<16x16xi32>) -> tensor<16x16xi32>
    %2 = stablehlo.custom_call @SPMDFullToShardShape(%1) {mhlo.sharding = "{manual}"} : (tensor<16x16xi32>) -> tensor<8x16xi32>
    %3 = call @shmap_body(%2) : (tensor<8x16xi32>) -> tensor<8x16xi32>
    %4 = stablehlo.custom_call @Sharding(%3) {mhlo.sharding = "{manual}"} : (tensor<8x16xi32>) -> tensor<8x16xi32>
    %5 = stablehlo.custom_call @SPMDShardToFullShape(%4) {mhlo.sharding = "{devices=[2,1]0,1}"} : (tensor<8x16xi32>) -> tensor<16x16xi32>
    %6 = stablehlo.add %0, %5 : tensor<16x16xi32>
    return %6 : tensor<16x16xi32>
  }
  func.func private @shmap_body(%arg0: tensor<8x16xi32>) -> (tensor<8x16xi32>) {
    %0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
    }) {channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids} : (tensor<8x16xi32>) -> tensor<8x16xi32>
    return %0 : tensor<8x16xi32>
  }
}

// -----

//        CHECK-LABEL: module @all_reduce_cross_replica
module @all_reduce_cross_replica attributes {
  mhlo.num_partitions = 2 : i32,
  mhlo.num_replicas = 1 : i32
} {
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
