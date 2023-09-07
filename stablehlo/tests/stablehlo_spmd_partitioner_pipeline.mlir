// RUN: stablehlo-opt \
// RUN:   --mlir-print-ir-before-all \
// RUN:   --mlir-print-ir-after-all \
// RUN:   --stablehlo-xla-cc-lib-path="%xla_cc_lib_path" \
// RUN:   --super-sub-device-map-file="%test_source_root/super_sub_device_map_2x2.yaml" \
// RUN:   --pass-pipeline="builtin.module( \
// RUN:     func.func( \
// RUN:       stablehlo-set-collectives-device-domain{device_domain=super} \
// RUN:     ), \
// RUN:     stablehlo-collectives-spmd-sub-partitioner, \
// RUN:     func.func( \
// RUN:       stablehlo-move-device-domain-to-frontend-attributes \
// RUN:     ), \
// RUN:     stablehlo-xla-spmd-partitioner{num_partitions=2 num_replicas=1}, \
// RUN:     func.func( \
// RUN:       stablehlo-move-device-domain-from-frontend-attributes, \
// RUN:       stablehlo-set-collectives-device-domain{device_domain=sub} \
// RUN:     ), \
// RUN:     stablehlo-complete-collectives-spmd-sub-partition \
// RUN:   )" \
// RUN:   --split-input-file %s

module @spmd_pipeline attributes {
  mhlo.spmd_parameters_shardings = ["{replicated}", "{replicated}"],
  mhlo.spmd_output_sharding = "{replicated}"
} {
  func.func public @main(
    %arg0: tensor<2x6xi32> {mhlo.sharding = "{devices=[1,2]0,1}"},
    %arg1: tensor<6x2xi32> {mhlo.sharding = "{devices=[2,1]0,1}"}
  ) -> (tensor<4x4xi32> {mhlo.sharding = "{replicated}"}) {
    %0 = "stablehlo.all_gather" (%arg0) {
      all_gather_dim = 0 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      use_global_device_ids,
      mhlo.sharding = "{devices=[1,2]0,1}"
    } : (tensor<2x6xi32>) -> tensor<4x6xi32>
    %1 = "stablehlo.all_gather" (%arg1) {
      all_gather_dim = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      use_global_device_ids,
      mhlo.sharding = "{devices=[2,1]0,1}"
    } : (tensor<6x2xi32>) -> tensor<6x4xi32>
    %2 = stablehlo.dot_general %0, %1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      {mhlo.sharding = "{replicated}"}
      : (tensor<4x6xi32>, tensor<6x4xi32>) -> tensor<4x4xi32>
    return %2 : tensor<4x4xi32>
  }
}
