# MHLO sharding format

The sharding attribute `mhlo.sharding` defines how the results of an operation should be sharded.
```mlir
%2 = stablehlo.add %0, %1 {mhlo.sharding = "{devices=[2,1]0,1}"} : tensor<4x3xi32>
```

Each device is described by its ID as an integer.
The `devices` field is a tensor that describes the tiling of the output tensor
and the tile assignment to devices.
In the above example there are 2 devices { 0, 1 }. They are arrange in a tensor
```
[
    [0],
    [1]
]
```
The output tensor is sharded along its 0-th axis.
Each shard would have a shape `2x3`.
If we have an unsharded output tensor
```
[
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
]
```
Device `0` would get the shard
```
[
    [1, 2, 3],
    [4, 5, 6],
]
```
and device `1` would get
```
[
    [7, 8, 9],
    [10, 11, 12]
]
```

---

```
%2 = stablehlo.add %0, %1 {mhlo.sharding = "{devices=[1,2,4]0,1,2,3,4,5,6,7}"}
    : tensor<3x4x4xi32>
```
The sharding above would result in tile assignment
```
[
    [
        [0, 1, 2, 3],
        [4, 5, 6, 7]
    ]
]
```
The resulting shape of each shard would be 3x2x1.

## Fully replicated

```
{mhlo.sharding = "{replicated}"}
```
This means that there is no sharding of the output tensor,
but instead each device gets a replica of the full tensor.

## Partial replication

In this mode the output tensor is both sharded and replicated. Each shard is replicated on some of the devices.

```
%2 = stablehlo.add %0, %1
    {mhlo.sharding = "{devices=[1,2,4]0,1,2,3,4,5,6,7} last_tile_dim_replicate"}
    : tensor<4x3xi32>
```
In the above example the last dimension does not describe a normal tile dimension,
but instead lists multiple devices that are assigned to each tile.
The tiling is described by the dimensions except the last.

If we have an unsharded output 4x3 tensor
```
[
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
]
```

Devices [0, 1, 2, 3] would all get the same 2x3 tensor
```
[
    [1, 2, 3],
    [4, 5, 6],
]
```

Devices [4, 5, 6, 7] would all get the same 2x3 tensor
```
[
    [7, 8, 9],
    [10, 11, 12]
]
```
