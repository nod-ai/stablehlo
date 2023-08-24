// RUN: stablehlo-opt \
// RUN:   --super-sub-device-map-file="%test_source_root/super_sub_device_map_2x4.yaml" \
// RUN:   --pass-pipeline="builtin.module(func.func(stablehlo-complete-collectives-spmd-sub-partition))" \
// RUN:   --split-input-file "%s" \
// RUN: | FileCheck "%s"

module attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
//        CHECK-LABEL: func.func @all_gather
  func.func @all_gather(%arg0: tensor<2x3xf32>) -> tensor<2x6xf32> {
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

module attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
//        CHECK-LABEL: func.func @all_reduce
  func.func @all_reduce(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
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

module attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
//        CHECK-LABEL: func.func @all_to_all
  func.func @all_to_all(%arg0: tensor<2x6xf32>) -> tensor<4x3xf32> {
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

module attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
//        CHECK-LABEL: func.func @reduce_scatter
  func.func @reduce_scatter(%arg0: tensor<2x6xf32>) -> tensor<2x3xf32> {
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

module attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 2 : i32} {
//        CHECK-LABEL: func.func @cross_replica
  func.func @cross_replica(%arg0: tensor<2x3xf32>) -> tensor<2x6xf32> {
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

module attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 2 : i32} {
//        CHECK-LABEL: func.func @cross_replica_and_partition
  func.func @cross_replica_and_partition(%arg0: tensor<2x3xf32>) -> tensor<2x6xf32> {
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

module attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 2 : i32} {
//        CHECK-LABEL: func.func @cross_partition
  func.func @cross_partition(%arg0: tensor<2x6xf32>) -> tensor<4x3xf32> {
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
