/* Copyright 2022 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

include "mlir/Pass/PassBase.td"

def StablehloCanonicalizeDynamismPass : Pass<"stablehlo-canonicalize-dynamism", "func::FuncOp"> {
  let summary = "Canonicalizes dynamic StableHLO ops into static ops.";
  let description = [{
    Replaces dynamic StableHLO ops like DynamicReshapeOp with the corresponding
    static counterparts like ReshapeOp if all the dynamic elements of these ops
    are actually constant.

    For example, if the output_shape operand of DynamicReshapeOp is a constant
    value, then the operation can be transformed to ReshapeOp.
  }];
}

def StablehloLegalizeToVhloPass : Pass<"stablehlo-legalize-to-vhlo", "ModuleOp"> {
  let summary = "Legalize StableHLO to VHLO.";
  let dependentDialects = ["mlir::vhlo::VhloDialect"];
}

def StablehloRefineShapesPass : Pass<"stablehlo-refine-shapes", "ModuleOp"> {
  let summary = "Refines shapes across a StableHLO program.";
  let description = [{
    Walks through a StableHLO program refining shapes within ops.

    The flagship use case for this pass is specializing dynamically-shaped
    programs to static shapes. If a dynamically-shaped StableHLO program has the
    right structure, then updating its argument types from dynamic shapes to
    static shapes and running this pass will propagate static shapes across
    the program.
  }];
}

def VhloLegalizeToStablehloPass : Pass<"vhlo-legalize-to-stablehlo", "ModuleOp"> {
  let summary = "Legalize VHLO to StableHLO.";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::stablehlo::StablehloDialect",
                           "mlir::shape::ShapeDialect", "mlir::quant::QuantizationDialect"];
}

def VhloToVersionPass : Pass<"vhlo-to-version"> {
  let summary = "Convert between versions of VHLO.";
  let options = [
    Option<"targetVersionOption", "target", "std::string", "",
           "The target version. Must be a version of the form #.#.# or 'current'.">,
  ];
}

def CollectivesSpmdSubPartitioner : InterfacePass<"stablehlo-collectives-spmd-sub-partitioner", "mlir::FunctionOpInterface"> {
  let summary = "Partition only collective operations that have sharding annotations and operate on the super devices.";
  let description = [{
    Each super-pratition collective is further subdivided into a set of devices with new device IDs.
    This induces a new finer grained complete-partition.
    The collectives and their operands must have sharding in a sub-parititon scheme,
    while the collectives' replica_groups are still in the original super-parition scheme.

    For example lets have a device set {0, 1}.
    These devices are further subdivided into a list of devices
    ```
    0 -> [10, 11, 12]
    1 -> [13, 14, 15]
    ```

    All-reduce example
    Input:
    ```mlir
    %result = "stablehlo.all_reduce"(%operand) ({
      ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
        %0 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %0 : tensor<f32>
    }) {
      # The communication is described in the original device scheme.
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
      use_global_device_ids = true,
      # Sharding in the new subdivided scheme.
      # The devices mean the indexes in the subdivision, not the new IDs.
      mhlo.sharding = "{devices=[1,3]0,1,2}",
      super_partition
    } : (tensor<2x9xf32>) -> tensor<2x9xf32>
    ```

    Output:
    ```mlir
    %0 = stablehlo.custom_call @Sharding(%operand) {mhlo.sharding = "{devices=[1,3]0,1,2}"}
      : (tensor<2x9xf32>) -> tensor<2x9xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {mhlo.sharding = "{manual}"}
      : (tensor<2x9xf32>) -> tensor<2x3xf32>
    %2 = "stablehlo.all_reduce"(%1) ({
      ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
        %0 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %0 : tensor<f32>
    }) {
      # Communication between devices described in the new scheme.
      replica_groups = dense<[[10, 13], [11, 14], [12, 15]]> : tensor<3x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
      use_global_device_ids = true,
      complete_partition
    } : (tensor<2x3xf32>) -> tensor<2x3xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {mhlo.sharding = "{manual}"}
      : (tensor<2x3xf32>) -> tensor<2x3xf32>
    %result = stablehlo.custom_call @SPMDShardToFullShape(%3) {mhlo.sharding = "{devices=[1,3]0,1,2}"} 
      : (tensor<2x3xf32>) -> tensor<2x9xf32>
    ```

    All-gather example
    Input:
    ```mlir
    %result = "stablehlo.all_gather"(%operand) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
      use_global_device_ids = true,
      mhlo.sharding = "{devices=[3,1]0,1,2}",
      super_partition
    } : (tensor<9x2xf32>) -> tensor<9x4xf32>
    ```

    Output:
    ```mlir
    %0 = stablehlo.custom_call @Sharding(%operand) {mhlo.sharding = "{devices=[3,1]0,1,2}"}
      : (tensor<9x2xf32>) -> tensor<9x2xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {mhlo.sharding = "{manual}"}
      : (tensor<9x2xf32>) -> tensor<3x2xf32>
    %result = "stablehlo.all_gather"(%1) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[10, 13], [11, 14], [12, 15]]> : tensor<3x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
      use_global_device_ids = true,
      complete_partition
    } : (tensor<3x2xf32>) -> tensor<3x4xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {mhlo.sharding = "{manual}"}
      : (tensor<3x4xf32>) -> tensor<3x4xf32>
    %result = stablehlo.custom_call @SPMDShardToFullShape(%3) {mhlo.sharding = "{devices=[3,1]0,1,2}"} 
      : (tensor<3x4xf32>) -> tensor<9x4xf32>
    ```

    Reduce-scatter example
    ```
    0 -> [100, 101, 102]
    1 -> [103, 104, 105]
    2 -> [106, 107, 108]
    3 -> [109, 110, 111]
    ```

    Input:
    ```mlir
    %result = "stablehlo.reduce_scatter"(%operand) ({
      ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%0) : (tensor<f32>) -> ()
    }) {
      scatter_dimension = 1 : i64,
      replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
      use_global_device_ids = true,
      mhlo.sharding = "{devices=[3,1]0,1,2}",
      super_partition
    } : (tensor<9x4xf32>) -> tensor<9x2xf32>
    ```

    Output:
    ```mlir
    %0 = stablehlo.custom_call @Sharding(%operand) {mhlo.sharding = "{devices=[3,1]0,1,2}"}
      : (tensor<9x4xf32>) -> tensor<9x4xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {mhlo.sharding = "{manual}"}
      : (tensor<9x4xf32>) -> tensor<3x4xf32>
    %result = "stablehlo.reduce_scatter"(%operand) {
      scatter_dimension = 1 : i64,
      replica_groups = dense<[[100, 103], [101, 104], [102, 105], [106, 109], [107, 110], [108, 111]]> : tensor<6x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
      use_global_device_ids = true,
      complete_partition
    } : (tensor<3x4xf32>) -> tensor<3x2xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {mhlo.sharding = "{manual}"}
      : (tensor<3x2xf32>) -> tensor<3x2xf32>
    %result = stablehlo.custom_call @SPMDShardToFullShape(%3) {mhlo.sharding = "{devices=[3,1]0,1,2}"} 
      : (tensor<3x2xf32>) -> tensor<9x2xf32>
    ```
  }];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def CompleteCollectivesSpmdSubPartition : InterfacePass<"stablehlo-complete-collectives-spmd-sub-partition", "mlir::FunctionOpInterface"> {
  let summary = "Complete a partition of collectives that operate on a sub-partition.";
  let description = [{
    Lets have a mapping of a super-device to a sub-device.
    For example:
    ```
           0,  1,  2,  3   # sub-partition indexes.
    0 -> [10, 11, 12, 13],
    1 -> [14, 15, 16, 17]
    ```
    A sub-partition collective operation is a collective operation that operates on the indexes of a sub-partition.
    ```
    %result = "stablehlo.all_gather"(%operand) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
      use_global_device_ids = true,
      sub_partition
    } : (tensor<2x3xf32>) -> tensor<2x12xf32>
    ```

    A super-partition collective is a collective operation that operates on the super-device ids.
    ```
    %result = "stablehlo.all_gather"(%operand) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
      use_global_device_ids = true,
      super_partition
    } : (tensor<2x3xf32>) -> tensor<2x6xf32>
    ```

    A complete-partition collective is a collective operation that describes both the super and sub device levels.
    ```
    %result = "stablehlo.all_gather"(%operand) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[10, 14], [11, 15], [12, 16], [13, 17]]> : tensor<4x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
      use_global_device_ids = true
    } : (tensor<2x3xf32>) -> tensor<2x6xf32>
    ```

    All-gather example
    Input:
    ```
    %result = "stablehlo.all_gather"(%operand) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
      use_global_device_ids = true,
      sub_partition
    } : (tensor<2x3xf32>) -> tensor<2x6xf32>
    ```

    Result:
    ```
    %result = "stablehlo.all_gather"(%operand) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[10, 11], [14, 15], [12, 13], [16, 17]]> : tensor<4x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
      use_global_device_ids = true,
      complete_partition
    } : (tensor<2x3xf32>) -> tensor<2x6xf32>
    ```
  }];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def MarkUnmarkedCollectivesAsSuperPartition : InterfacePass<"stablehlo-mark-unmarked-collectives-as-super-partition", "mlir::FunctionOpInterface"> {
  let summary = "Add the super_partition attribute to StableHLO collectives.";
  let description = [{
    Example

    Input:
    ```
    %result = "stablehlo.all_gather"(%operand) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
      use_global_device_ids = true
    } : (tensor<2x3xf32>) -> tensor<2x6xf32>
    ```

    Result:
    ```
    %result = "stablehlo.all_gather"(%operand) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
      use_global_device_ids = true,
      super_partition
    } : (tensor<2x3xf32>) -> tensor<2x6xf32>
    ```
  }];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

// def FenceOffSuperPartitionCollectives : InterfacePass<"stablehlo-mark-collectives-as-super-partition", "mlir::FunctionOpInterface"> {
//   let summary = "Add the super_partition attribute to StableHLO collectives.";
//   let description = [{
//     Example
//     Super-device to sub-device map:
//     ```
//     0 -> [100, 101, 102]
//     1 -> [110, 111, 112]
//     ``` 

//     Input:
//     ```
//     %result = "stablehlo.all_gather"(%operand) {
//       all_gather_dim = 1 : i64,
//       replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
//       channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
//       use_global_device_ids = true,
//       # Sub-device sharding.
//       mhlo.sharding = "{devices=[3,1]0,1,2}",
//       super_partition
//     } : (tensor<6x4xf32>) -> tensor<6x8xf32>
//     ```

//     Result:
//     ```
//     %1 = stablehlo.custom_call @Sharding(%0) {mhlo.sharding = "{devices=[1,2]0,1}"} : (tensor<6x4xf32>) -> tensor<6x4xf32>
//     %2 = stablehlo.custom_call @SPMDFullToShardShape(%1) {mhlo.sharding = "{manual}"} : (tensor<6x4xf32>) -> tensor<6x4xf32>
//     %result = "stablehlo.all_gather"(%operand) {
//       all_gather_dim = 1 : i64,
//       replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
//       channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
//       use_global_device_ids = true,
//       mhlo.sharding = "{devices=[3,1]0,1,2}",
//       super_partition
//     } : (tensor<6x4xf32>) -> tensor<6x8xf32>
//     %4 = stablehlo.custom_call @Sharding(%3) {mhlo.sharding = "{manual}"} : (tensor<8x4xi32>) -> tensor<8x4xi32>
//     %5 = stablehlo.custom_call @SPMDShardToFullShape(%4) {mhlo.sharding = "{devices=[2,4]0,1,2,3,4,5,6,7}"} : (tensor<8x4xi32>) -> tensor<16x16xi32>
//     ```
//   }];
//   let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
// }
