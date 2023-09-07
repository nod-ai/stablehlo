// RUN: stablehlo-opt \
// RUN:   --stablehlo-xla-cc-lib-path="%xla_cc_lib_path" \
// RUN:   --super-sub-device-map-file="%test_source_root/super_sub_device_map_2x4.yaml" \
// RUN:   --pass-pipeline="builtin.module(stablehlo-complete-collectives-spmd-sub-partition)" \
// RUN:   --split-input-file "%s" \
// RUN: | FileCheck "%s"

//        CHECK-LABEL: module @all_gather
module @all_gather attributes {
  mhlo.num_partitions = 4 : i32,
  mhlo.num_replicas = 1 : i32,
  mhlo.spmd_parameters_shardings = ["{replicated}"],
  mhlo.spmd_output_sharding = "{replicated}",
  mhlo.frontend_attributes = {
    super_partition_spmd_parameters_sharding = "{{replicated}}",
    super_partition_spmd_output_sharding = "{replicated}"
  }
} {
  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x6xf32> {
    %0 = "stablehlo.all_gather"(%arg0) {
      all_gather_dim = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
// CHECK-DAG{LITERAL}: replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]>
      replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>,
      use_global_device_ids,
// CHECK-DAG{LITERAL}: device_domain = "complete"
      device_domain = "sub"
    } : (tensor<2x3xf32>) -> tensor<2x6xf32>
    return %0 : tensor<2x6xf32>
  }
}

// -----

//        CHECK-LABEL: module @all_reduce
module @all_reduce attributes {
  mhlo.num_partitions = 4 : i32,
  mhlo.num_replicas = 1 : i32,
  mhlo.spmd_parameters_shardings = ["{replicated}"],
  mhlo.spmd_output_sharding = "{replicated}",
  mhlo.frontend_attributes = {
    super_partition_spmd_parameters_sharding = "{{replicated}}",
    super_partition_spmd_output_sharding = "{replicated}"
  }
} {
  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) {
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
// CHECK-DAG{LITERAL}: replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]>
      replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>,
      use_global_device_ids,
// CHECK-DAG{LITERAL}: device_domain = "complete"
      device_domain = "sub"
    } : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}

// -----

//        CHECK-LABEL: module @all_to_all
module @all_to_all attributes {
  mhlo.num_partitions = 4 : i32,
  mhlo.num_replicas = 1 : i32,
  mhlo.spmd_parameters_shardings = ["{replicated}"],
  mhlo.spmd_output_sharding = "{replicated}",
  mhlo.frontend_attributes = {
    super_partition_spmd_parameters_sharding = "{{replicated}}",
    super_partition_spmd_output_sharding = "{replicated}"
  }
} {
  func.func @main(%arg0: tensor<2x6xf32>) -> tensor<4x3xf32> {
    %0 = "stablehlo.all_to_all"(%arg0) {
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      split_dimension = 1,
      concat_dimension = 0,
      split_count = 2,
// CHECK-DAG{LITERAL}: replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]>
      replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>,
// CHECK-DAG{LITERAL}: use_global_device_ids
// CHECK-DAG{LITERAL}: device_domain = "complete"
      device_domain = "sub"
    } : (tensor<2x6xf32>) -> tensor<4x3xf32>
    return %0 : tensor<4x3xf32>
  }
}

// -----

//        CHECK-LABEL: module @reduce_scatter
module @reduce_scatter attributes {
  mhlo.num_partitions = 4 : i32,
  mhlo.num_replicas = 1 : i32,
  mhlo.spmd_parameters_shardings = ["{replicated}"],
  mhlo.spmd_output_sharding = "{replicated}",
  mhlo.frontend_attributes = {
    super_partition_spmd_parameters_sharding = "{{replicated}}",
    super_partition_spmd_output_sharding = "{replicated}"
  }
} {
  func.func @main(%arg0: tensor<2x6xf32>) -> tensor<2x3xf32> {
    %0 = "stablehlo.reduce_scatter"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) {
      scatter_dimension = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
// CHECK-DAG{LITERAL}: replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]>
      replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>,
      use_global_device_ids,
// CHECK-DAG{LITERAL}: device_domain = "complete"
      device_domain = "sub"
    } : (tensor<2x6xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}

// -----

//        CHECK-LABEL: module @cross_replica
module @cross_replica attributes {
  mhlo.num_partitions = 2 : i32,
  mhlo.num_replicas = 2 : i32,
  mhlo.spmd_parameters_shardings = ["{replicated}"],
  mhlo.spmd_output_sharding = "{replicated}",
  mhlo.frontend_attributes = {
    super_partition_spmd_parameters_sharding = "{{replicated}}",
    super_partition_spmd_output_sharding = "{replicated}"
  }
} {
  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x6xf32> {
    %0 = "stablehlo.all_gather"(%arg0) {
      all_gather_dim = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
// CHECK-DAG{LITERAL}: replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]>
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
// CHECK-DAG{LITERAL}: use_global_device_ids
// CHECK-DAG{LITERAL}: device_domain = "complete"
      device_domain = "sub"
    } : (tensor<2x3xf32>) -> tensor<2x6xf32>
    return %0 : tensor<2x6xf32>
  }
}

// -----

//        CHECK-LABEL: module @cross_replica_and_partition
module @cross_replica_and_partition attributes {
  mhlo.num_partitions = 2 : i32,
  mhlo.num_replicas = 2 : i32,
  mhlo.spmd_parameters_shardings = ["{replicated}"],
  mhlo.spmd_output_sharding = "{replicated}",
  mhlo.frontend_attributes = {
    super_partition_spmd_parameters_sharding = "{{replicated}}",
    super_partition_spmd_output_sharding = "{replicated}"
  }
} {
  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x6xf32> {
    %0 = "stablehlo.all_gather"(%arg0) {
      all_gather_dim = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
// CHECK-DAG{LITERAL}: replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]>
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
// CHECK-DAG{LITERAL}: use_global_device_ids
// CHECK-DAG{LITERAL}: device_domain = "complete"
      device_domain = "sub"
    } : (tensor<2x3xf32>) -> tensor<2x6xf32>
    return %0 : tensor<2x6xf32>
  }
}

// -----

//        CHECK-LABEL: module @cross_partition
module @cross_partition attributes {
  mhlo.num_partitions = 2 : i32,
  mhlo.num_replicas = 2 : i32,
  mhlo.spmd_parameters_shardings = ["{replicated}"],
  mhlo.spmd_output_sharding = "{replicated}",
  mhlo.frontend_attributes = {
    super_partition_spmd_parameters_sharding = "{{replicated}}",
    super_partition_spmd_output_sharding = "{replicated}"
  }
} {
  func.func @main(%arg0: tensor<2x6xf32>) -> tensor<4x3xf32> {
    %0 = "stablehlo.all_to_all"(%arg0) {
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      split_dimension = 1,
      concat_dimension = 0,
      split_count = 2,
// CHECK-DAG{LITERAL}: replica_groups = dense<[[0, 2], [1, 3], [4, 6], [5, 7]]>
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
// CHECK-DAG{LITERAL}: device_domain = "complete"
      device_domain = "sub"
    } : (tensor<2x6xf32>) -> tensor<4x3xf32>
    return %0 : tensor<4x3xf32>
  }
}

// -----

//        CHECK-LABEL: module @module_sharding_replicated
module @module_sharding_replicated attributes {
  mhlo.num_partitions = 2 : i32,
  mhlo.num_replicas = 2 : i32,
// CHECK-DAG{LITERAL}: mhlo.spmd_parameters_shardings = ["{replicated}"]
  mhlo.spmd_parameters_shardings = ["{replicated}"],
// CHECK-DAG{LITERAL}: mhlo.spmd_output_sharding = "{replicated}"
  mhlo.spmd_output_sharding = "{devices=[1,1,4]0,1,2,3 last_tile_dim_replicate}",
// CHECK-NOT{LITERAL}: mhlo.frontend_attributes
  mhlo.frontend_attributes = {
// CHECK-NOT{LITERAL}: super_partition_spmd_parameters_sharding
    super_partition_spmd_parameters_sharding = "{{replicated}}",
// CHECK-NOT{LITERAL}: super_partition_spmd_parameters_sharding
    super_partition_spmd_output_sharding = "{devices=[1,1,2]0,1 last_tile_dim_replicate}"
  }
} {
  func.func @main(%arg0: tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
    return %arg0 : tensor<1x2x3xf32>
  }
}

// -----

//        CHECK-LABEL: module @module_sharding_tiled
module @module_sharding_tiled attributes {
  mhlo.num_partitions = 2 : i32,
  mhlo.num_replicas = 2 : i32,
// CHECK-DAG{LITERAL}: mhlo.spmd_parameters_shardings = ["{devices=[1,8]0,1,2,3,4,5,6,7}"]
  mhlo.spmd_parameters_shardings = ["{devices=[1,4]0,1,2,3}"],
// CHECK-DAG{LITERAL}: mhlo.spmd_output_sharding = "{devices=[1,8]4,5,6,7,0,1,2,3}"
  mhlo.spmd_output_sharding = "{devices=[1,4]0,1,2,3}",
  mhlo.frontend_attributes = {
    super_partition_spmd_parameters_sharding = "{{devices=[1,2]0,1}}",
    super_partition_spmd_output_sharding = "{devices=[1,2]1,0}"
  }
} {
  func.func @main(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }
}

// -----

//        CHECK-LABEL: module @module_sharding_partial_replication
module @module_sharding_partial_replication attributes {
  mhlo.num_partitions = 2 : i32,
  mhlo.num_replicas = 2 : i32,
// CHECK-DAG{LITERAL}: mhlo.spmd_parameters_shardings = ["{devices=[2,2,2]0,2,4,6,1,3,5,7 last_tile_dim_replicate}"]
  mhlo.spmd_parameters_shardings = ["{devices=[2,1,2]0,2,1,3 last_tile_dim_replicate}"],
// CHECK-DAG{LITERAL}: mhlo.spmd_output_sharding = "{devices=[1,2,4]0,1,4,5,2,3,6,7 last_tile_dim_replicate}"
  mhlo.spmd_output_sharding = "{devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}",
  mhlo.frontend_attributes = {
    super_partition_spmd_parameters_sharding = "{{devices=[1,2]0,1}}",
    super_partition_spmd_output_sharding = "{devices=[1,1,2]0,1 last_tile_dim_replicate}"
  }
} {
  func.func @main(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }
}

// -----

//        CHECK-LABEL: module @module_sharding_tuple
module @module_sharding_tuple attributes {
  mhlo.num_partitions = 2 : i32,
  mhlo.num_replicas = 2 : i32,
// CHECK-DAG{LITERAL}: mhlo.spmd_output_sharding = "{{devices=[2,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}, {devices=[2,1,4]4,5,6,7,0,1,2,3 last_tile_dim_replicate}}"
  mhlo.spmd_parameters_shardings = ["{replicated}", "{replicated}"],
// CHECK-DAG{LITERAL}: mhlo.spmd_parameters_shardings = ["{devices=[1,2,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}", "{devices=[1,2,4]4,5,6,7,0,1,2,3 last_tile_dim_replicate}"]
  mhlo.spmd_output_sharding = "{{replicated}, {replicated}}",
  mhlo.frontend_attributes = {
    super_partition_spmd_parameters_sharding = "{{devices=[1,2]0,1}, {devices=[1,2]1,0}}",
    super_partition_spmd_output_sharding = "{{devices=[2,1]0,1}, {devices=[2,1]1,0}}"
  }
} {
  func.func @main(%arg0: tensor<1x2xf32>, %arg1: tensor<3x4xf32>) -> (tensor<1x2xf32>, tensor<3x4xf32>) {
    return %arg0, %arg1 : tensor<1x2xf32>, tensor<3x4xf32>
  }
}
