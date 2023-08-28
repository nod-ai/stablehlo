// RUN: stablehlo-opt \
// RUN:   --stablehlo-xla-cc-lib-path="%xla_cc_lib_path" \
// RUN:   --super-sub-device-map-file="%test_source_root/super_sub_device_map_8x8.yaml" \
// RUN:   --pass-pipeline="builtin.module(stablehlo-complete-collectives-spmd-sub-partition)" \
// RUN:   --split-input-file "%s" \
// RUN: | FileCheck "%s"

//        CHECK-LABEL: module @module_sharding_partial_replication
module @module_sharding_partial_replication attributes {
  mhlo.num_partitions = 2 : i32,
  mhlo.num_replicas = 2 : i32,
// CHECK-DAG{LITERAL}: mhlo.spmd_parameters_shardings = "{devices=[8,1,1,8]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63 last_tile_dim_replicate}"
  mhlo.spmd_parameters_shardings = "{replicated}",
// CHECK-DAG{LITERAL}: mhlo.spmd_output_sharding = "{devices=[2,1,8,4]54,55,62,63,52,53,60,61,50,51,58,59,48,49,56,57,38,39,46,47,36,37,44,45,34,35,42,43,32,33,40,41,22,23,30,31,20,21,28,29,18,19,26,27,16,17,24,25,6,7,14,15,4,5,12,13,2,3,10,11,0,1,8,9 last_tile_dim_replicate}"
  mhlo.spmd_output_sharding = "{devices=[1,1,4,2]7,6,5,4,3,2,1,0 last_tile_dim_replicate}",
  mhlo.frontend_attributes = {
    super_partition_spmd_parameters_sharding = "{devices=[8,1,1]0,1,2,3,4,5,6,7}",
    super_partition_spmd_output_sharding = "{devices=[2,1,2,2]7,6,5,4,3,2,1,0 last_tile_dim_replicate}"
  }
} {
  func.func @main(%arg0: tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
    return %arg0 : tensor<1x2x3xf32>
  }
}
