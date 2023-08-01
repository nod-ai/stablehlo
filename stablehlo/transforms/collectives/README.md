# SPMD partitioning IR that contains collectives

The order of passes should be as follows.
 * Assign sub-device sharding to all operations.
 This can be manual annotation and then sharding propagation.
 Some other auto-sharding algorithm.
 * Mark all existing collectives with the `stablehlo-mark-collectives-as-super-partition` pass.
 * Partition the marked super-partition collectives into a complete-partition
 with the `stablehlo-collectives-spmd-sub-partitioner` pass.
 * Run the `stablehlo-xla-spmd-partitioner` pass.
 This will partition everything else besides the complete-partition collectives.
 * Designate all generated collectives from the previous pass as sub-partition.
 * Run `stablehlo-complete-collectives-spmd-sub-partition` to convert sub-partition collectives
 to complete-partition.

 There may be a need to convert the partition type designation attributes to HLO metadata to survive the conversion to/form HLO.
