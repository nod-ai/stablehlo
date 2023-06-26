

#ifndef STABLEHLO_TRANSFORMS_XLA_XLAPASSES
#define STABLEHLO_TRANSFORMS_XLA_XLAPASSES

include "mlir/Pass/PassBase.td"

def ShadingPropagation : Pass<"stablehlo-xla-sharding-propagation", "mlir::ModuleOp"> {
  let summary = "Propagate StableHLO tensor sharding annotations.";
  let description = [{
    Uses the XLO compiler. Translates first from StableHLO to XLA HLO.
    Then runs the XLA pass `xla::ShardingPropagation`.
    The result is converted back to StableHLO.
  }];
  let constructor = "mlir::stablehlo::createShadingPropagationPass()";
  let options = [
    Option<"is_spmd", "is_spmd", "bool",
      /*default=*/"false", "">,
    Option<"propagate_metadata", "propagate_metadata", "bool",
      /*default=*/"false", "">,
    ListOption<"allow_spmd_sharding_propagation_to_output",
      "allow_spmd_sharding_propagation_to_output", "char", "">,
    ListOption<"allow_spmd_sharding_propagation_to_parameters",
      "allow_spmd_sharding_propagation_to_parameters", "char", "">,
    Option<"cse_prevention_only", "cse_prevention_only", "bool",
           /*default=*/"false", "">,
  ];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def SpmdPartitioner : Pass<"stablehlo-xla-spmd-partitioner", "mlir::ModuleOp"> {
  let summary = "SPMDize a StableHLO program that has tensor sharding annotations.";
  let description = [{
    Uses the XLO compiler. Translates first from StableHLO to XLA HLO.
    Then runs the XLA pass `xla::spmd::StatefulRngSpmdPartitioner`.
    The result is converted back to StableHLO.
  }];
  let constructor = "mlir::stablehlo::createSpmdPartitionerPass()";
  let options = [
    Option<"num_partitions", "num_partitions", "int64_t",
      /*default=*/"-1", "">,
    Option<"num_replicas", "num_replicas", "int64_t",
      /*default=*/"-1", "">,
  ];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def ShadingPropagationAndSpmdPartitioner : Pass<"stablehlo-xla-sharding-propagation-and-spmd-partitioner", "mlir::ModuleOp"> {
  let summary = "Sharding propagation and SPMD partitioner.";
  let description = [{}];
  let constructor = "mlir::stablehlo::createShadingPropagationAndSpmdPartitionerPass()";
  let options = [
    Option<"is_spmd", "is_spmd", "bool",
      /*default=*/"false", "">,
    Option<"propagate_metadata", "propagate_metadata", "bool",
      /*default=*/"false", "">,
    ListOption<"allow_spmd_sharding_propagation_to_output",
      "allow_spmd_sharding_propagation_to_output", "char", "">,
    ListOption<"allow_spmd_sharding_propagation_to_parameters",
      "allow_spmd_sharding_propagation_to_parameters", "char", "">,
    Option<"cse_prevention_only", "cse_prevention_only", "bool",
           /*default=*/"false", "">,
    Option<"num_partitions", "num_partitions", "int64_t",
      /*default=*/"-1", "">,
    Option<"num_replicas", "num_replicas", "int64_t",
      /*default=*/"-1", "">,
  ];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def CollectivesOptimization : Pass<"stablehlo-xla-collectives-optimization", "mlir::ModuleOp"> {
  let summary = "Optimize collectives in a StableHLO program.";
  let description = [{
    Uses the XLO compiler. Translates first from StableHLO to XLA HLO.
    Then runs an XLA pipeline.
    The result is converted back to StableHLO.
  }];
  let constructor = "mlir::stablehlo::createCollectivesOptimizationPass()";
  let options = [
    Option<"reassociate_converted_all_reduce", "reassociate_converted_all_reduce",
      "bool", /*default=*/"true", "">,
    Option<"enable_while_loop_reduce_scatter_code_motion",
      "enable_while_loop_reduce_scatter_code_motion", "bool", /*default=*/"false", "">,
    Option<"enable_data_parallel_collective_optimizer",
      "enable_data_parallel_collective_optimizer", "bool", /*default=*/"false", "">,
  ];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def AutoSharding : Pass<"stablehlo-xla-auto-sharding", "mlir::ModuleOp"> {
  let summary = "Alpa auto sharding.";
  let description = [{
    Uses the XLO compiler. Translates first from StableHLO to XLA HLO.
    Then runs the XLA pass `xla::AutoSharding`.
    The result is converted back to StableHLO.
  }];
  let constructor = "mlir::stablehlo::createAutoShardingPass()";
  let options = [
    Option<"preserve_shardings", "preserve_shardings", "std::string",
      /*default=*/"\"kKeepInputOutputShardings\"", "">,
    Option<"simplify_graph", "simplify_graph", "bool",
      /*default=*/"true", "">,
    Option<"memory_budget_per_device", "memory_budget_per_device", "int64_t",
      /*default=*/"-1", "">,
    Option<"memory_budget_ratio", "memory_budget_ratio", "float",
      /*default=*/"1.1", "">,
    Option<"force_all_gather_cost", "force_all_gather_cost", "bool",
      /*default=*/"false", "">,
    Option<"all_gather_cost", "all_gather_cost", "double",
      /*default=*/"0", "">,
    Option<"force_all_to_all_cost", "force_all_to_all_cost", "bool",
      /*default=*/"false", "">,
    Option<"all_to_all_cost", "all_to_all_cost", "double",
      /*default=*/"0", "">,
    Option<"force_batch_dim_to_mesh_dim", "force_batch_dim_to_mesh_dim", "int",
      /*default=*/"-1", "">,
    Option<"allow_replicated_parameters", "allow_replicated_parameters", "bool",
      /*default=*/"true", "">,
    Option<"prefer_reduce_scatter", "prefer_reduce_scatter", "bool",
      /*default=*/"false", "">,
    Option<"reduce_scatter_grad_acc_friendly", "reduce_scatter_grad_acc_friendly", "bool",
      /*default=*/"false", "">,
    Option<"reduce_scatter_aggressive_partition", "reduce_scatter_aggressive_partition", "bool",
      /*default=*/"false", "">,
    Option<"batch_matmul_always_split_batch", "batch_matmul_always_split_batch", "bool",
      /*default=*/"false", "">,
    Option<"allow_recompute_heavy_op", "allow_recompute_heavy_op", "bool",
      /*default=*/"false", "">,
    Option<"allow_mixed_mesh_shape", "allow_mixed_mesh_shape", "bool",
      /*default=*/"false", "">,
    Option<"grad_acc_num_micro_batches", "grad_acc_num_micro_batches", "int",
      /*default=*/"1", "">,
    Option<"load_solution_vector", "load_solution_vector", "bool",
      /*default=*/"false", "">,
    Option<"solve_nd_sharding_iteratively", "solve_nd_sharding_iteratively", "bool",
      /*default=*/"true", "">,
    Option<"force_simple_heuristic", "force_simple_heuristic", "std::string",
      /*default=*/"", "">,
    Option<"force_strategy", "force_strategy", "bool",
      /*default=*/"false", "">,
    ListOption<"force_strategy_inst_indices", "force_strategy_inst_indices", "int64_t", "">,
    ListOption<"force_strategy_stra_names", "force_strategy_stra_names", "std::string", "">,
    ListOption<"device_mesh_shape", "device_mesh_shape", "int64_t", "">,
    ListOption<"device_mesh_ids", "device_mesh_ids", "int64_t", "">,
    ListOption<"device_mesh_alpha", "device_mesh_alpha", "double", "">,
    ListOption<"device_mesh_beta", "device_mesh_beta", "double", "">,
    Option<"load_strategy", "load_strategy", "bool",
      /*default=*/"false", "">,
    Option<"try_multiple_mesh_shapes", "try_multiple_mesh_shapes", "bool",
      /*default=*/"false", "">,
    Option<"solver_timeout_in_seconds", "solver_timeout_in_seconds", "int64_t",
      /*default=*/"3600", "">,
    ListOption<"strategy_vector", "strategy_vector", "int64_t", "">,
  ];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

#endif // STABLEHLO_TRANSFORMS_XLA_XLAPASSES
