// RUN: stablehlo-opt \
// RUN:   --stablehlo-xla-cc-lib-path="%xla_cc_lib_path" \
// RUN:   --pass-pipeline="builtin.module(stablehlo-xla-sharding-propagation-and-spmd-partitioner{is_spmd=0 propagate_metadata=0 allow_spmd_sharding_propagation_to_output=1 allow_spmd_sharding_propagation_to_parameters=1 cse_prevention_only=false num_partitions=2 num_replicas=1})" \
// RUN:   --split-input-file %s \
// RUN: | FileCheck %s

// CHECK{LITERAL}: module @m attributes {mhlo.cross_program_prefetches = [], mhlo.dynamic_parameter_bindings = [], mhlo.is_dynamic = false, mhlo.spmd_output_sharding = "{devices=[2,1]0,1}", mhlo.spmd_parameters_shardings = ["{devices=[2,1]0,1}", "{devices=[2,1]0,1}"], mhlo.use_auto_spmd_partitioning = false} {
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
