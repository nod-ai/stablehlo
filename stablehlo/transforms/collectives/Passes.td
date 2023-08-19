include "mlir/Pass/PassBase.td"

def CollectivesSpmdSubPartitioner
  : InterfacePass<"stablehlo-collectives-spmd-sub-partitioner", "mlir::FunctionOpInterface"> {
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
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids,
      # Sharding in the new subdivided scheme.
      # The devices mean the indexes in the subdivision, not the new IDs.
      mhlo.sharding = "{devices=[1,3]0,1,2}",
      device_domain = "super"
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
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids,
      device_domain = "complete"
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
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids,
      mhlo.sharding = "{devices=[3,1]0,1,2}",
      device_domain = "super"
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
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids,
      device_domain = "complete"
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
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids,
      mhlo.sharding = "{devices=[3,1]0,1,2}",
      device_domain = "super"
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
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids,
      device_domain = "complete"
    } : (tensor<3x4xf32>) -> tensor<3x2xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {mhlo.sharding = "{manual}"}
      : (tensor<3x2xf32>) -> tensor<3x2xf32>
    %result = stablehlo.custom_call @SPMDShardToFullShape(%3) {mhlo.sharding = "{devices=[3,1]0,1,2}"} 
      : (tensor<3x2xf32>) -> tensor<9x2xf32>
    ```
  }];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def CompleteCollectivesSpmdSubPartition
  : InterfacePass<"stablehlo-complete-collectives-spmd-sub-partition", "mlir::FunctionOpInterface"> {
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
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids,
      device_domain = "sub"
    } : (tensor<2x3xf32>) -> tensor<2x12xf32>
    ```

    A super-partition collective is a collective operation that operates on the super-device ids.
    ```
    %result = "stablehlo.all_gather"(%operand) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids,
      device_domain = "super"
    } : (tensor<2x3xf32>) -> tensor<2x6xf32>
    ```

    A complete-partition collective is a collective operation that describes both the super and sub device levels.
    ```
    %result = "stablehlo.all_gather"(%operand) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[10, 14], [11, 15], [12, 16], [13, 17]]> : tensor<4x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids
    } : (tensor<2x3xf32>) -> tensor<2x6xf32>
    ```

    All-gather example
    Input:
    ```
    %result = "stablehlo.all_gather"(%operand) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids,
      device_domain = "sub"
    } : (tensor<2x3xf32>) -> tensor<2x6xf32>
    ```

    Result:
    ```
    %result = "stablehlo.all_gather"(%operand) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[10, 11], [14, 15], [12, 13], [16, 17]]> : tensor<4x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids,
      device_domain = "complete"
    } : (tensor<2x3xf32>) -> tensor<2x6xf32>
    ```
  }];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def SetCollectivesDeviceDomain : Pass<"stablehlo-set-collectives-device-domain"> {
  let summary = "Collectives without device_domain have theirs set.";
  let description = [{
    Set the device_domain attribute in all collectives that don't have already a device domain
    to what is provided in the device-domain pass option.

    Example

    Input:
    ```
    %result = "stablehlo.all_gather"(%operand) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids
    } : (tensor<2x3xf32>) -> tensor<2x6xf32>
    ```

    Result:
    ```
    %result = "stablehlo.all_gather"(%operand) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids,
      device_domain = "super"
    } : (tensor<2x3xf32>) -> tensor<2x6xf32>
    ```
  }];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
  let options = [
    Option<"deviceDomain", "device-domain", "std::string", /*default=*/"",
      /*decription=*/"">,
  ];
}

def MoveDeviceDomainToFrontendAttributes
  : Pass<"stablehlo-move-device-domain-to-frontend-attributes"> {
  let summary = "Moves the device_domain attribute inside stablehlo.frontend_attributes.";
  let description = [{
    Example

    Input:
    ```
    %result = "stablehlo.all_gather"(%operand) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids,
      device_domain = "sub"
    } : (tensor<2x3xf32>) -> tensor<2x6xf32>
    ```

    Result:
    ```
    %result = "stablehlo.all_gather"(%operand) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids,
      stablehlo.frontend_attributes = { device_domain = "sub" }
    } : (tensor<2x3xf32>) -> tensor<2x6xf32>
    ```
  }];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}
