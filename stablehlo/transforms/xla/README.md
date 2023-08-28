# How To

The MLIR passes here use passes from a [fork](https://github.com/nod-ai/xla/tree/xla_cc) of the XLA compiler.
This fork exposes some XLA passes through a C interface inside a shared library `xla_cc`.
You need to build it first. Then build this project by following the [build instructions](/README.md#build-instructions).

The passes expose the command line argument `--stablehlo-xla-cc-lib-path` that specifies the path to `xla_cc` to be dynamically loaded.
You can also relay on the runtime libraray search path to find the library.

## Example

```mlir
# input.mlir
module attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(
    %arg0: tensor<16x16xi32> {mhlo.sharding = "{devices=[1,2]0,1}"},
    %arg1: tensor<16x16xi32> {mhlo.sharding = "{devices=[2,1]0,1}"}
  ) -> tensor<16x16xi32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi32>
    return %0 : tensor<16x16xi32>
  }
}
```

```
export LD_LIBRARY_PATH=/path/to/my/xla_cc/build/dir
stablehlo-opt \
    --pass-pipeline="builtin.module(stablehlo-xla-sharding-propagation-and-spmd-partitioner{is_spmd=1 propagate_metadata=1 allow_spmd_sharding_propagation_to_output=1 allow_spmd_sharding_propagation_to_parameters=1 cse_prevention_only=false})" \
    input.mlir
```

This results in an SPMD program sharded across 2 devices. 
```mlir
module @main attributes {
  mhlo.cross_program_prefetches = [],
  mhlo.dynamic_parameter_bindings = [],
  mhlo.is_dynamic = false,
  mhlo.num_partitions = 2 : i32,
  mhlo.num_replicas = 1 : i32,
  mhlo.spmd_output_sharding = "{replicated}",
  mhlo.spmd_parameters_shardings = ["{devices=[1,2]0,1}", "{devices=[2,1]0,1}"],
  mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(
    %arg0: tensor<16x8xi32> {mhlo.sharding = "{devices=[1,2]0,1}"},
    %arg1: tensor<8x16xi32> {mhlo.sharding = "{devices=[2,1]0,1}"}) -> tensor<16x16xi32> {
    %0 = stablehlo.dot %arg0, %arg1, precision = [DEFAULT, DEFAULT] : (tensor<16x8xi32>, tensor<8x16xi32>) -> tensor<16x16xi32>
    %1 = "stablehlo.all_reduce"(%0) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %2 = stablehlo.add %arg2, %arg3 : tensor<i32>
      stablehlo.return %2 : tensor<i32>
    }) {channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<0> : tensor<1x1xi64>} : (tensor<16x16xi32>) -> tensor<16x16xi32>
    return %1 : tensor<16x16xi32>
  }
}
```
