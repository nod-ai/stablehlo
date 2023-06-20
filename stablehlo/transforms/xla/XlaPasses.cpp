#include "XlaPasses.h"

#include <dlfcn.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "xla/xla_cc.h"
#include "xla/xla_cc_loader.h"
#include "xla/xla_cc_loader_impl.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_SHADINGPROPAGATION
#define GEN_PASS_DEF_SPMDPARTITIONER
#define GEN_PASS_DEF_COLLECTIVESOPTIMIZATION
#define GEN_PASS_DEF_AUTOSHARDING
#include "stablehlo/transforms/xla/XlaPasses.h.inc"

namespace {

llvm::cl::opt<std::string> xlaCcLibPath = llvm::cl::opt<std::string>(
    "stablehlo-xla-cc-lib-path",
    llvm::cl::desc("Path to the shared library libxla_cc_shared.so."
                   " This path should be absolue or a filename to be searched "
                   "in the OS runtime library search paths.\n"
                   "On Unix-like systems the path is passed to dlopen verbatim."
                   " You can use the environment variable LD_LIBRARY_PATH to "
                   "add search locations."),
    llvm::cl::init("libxla_cc_shared.so"));

template <typename Fn>
struct Destroyer {
  Destroyer(Fn fn) : fn(fn) {}
  ~Destroyer() { fn(); }

  Fn fn;
};

template <typename Fn>
Destroyer<Fn> makeDestroyer(Fn fn) {
  return Destroyer<Fn>(fn);
}

FailureOr<OwningOpRef<ModuleOp>> loadMlir(const char* buffer, size_t size,
                                          MLIRContext& context) {
  std::unique_ptr<llvm::MemoryBuffer> memoryBuffer =
      llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(buffer, size),
                                       /*BufferName=*/"",
                                       /*RequiresNullTerminator=*/false);
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memoryBuffer), llvm::SMLoc());
  return parseSourceFile<ModuleOp>(sourceMgr, &context);
}

void move(ModuleOp src, ModuleOp dst) {
  dst.getBodyRegion().takeBody(src.getBodyRegion());
  dst->setAttrs(src->getAttrDictionary());
  src->setAttrs(SmallVector<NamedAttribute, 1>());
}

FailureOr<std::unique_ptr<xla::HloModule, xla::api::DestroyHloModule>>
convertToHloModule(ModuleOp moduleOp) {
  std::string rawBytecodeBuffer;
  llvm::raw_string_ostream os(rawBytecodeBuffer);
  if (failed(writeBytecodeToFile(moduleOp.getOperation(), os))) {
    emitError(moduleOp->getLoc(), "Failed serializing MLIR to bytecode.");
    return failure();
  }

  xla::HloModule* hloModule = nullptr;
  if (xla::api::stableHloBufferToXlaHlo(rawBytecodeBuffer.data(),
                                        rawBytecodeBuffer.size(),
                                        &hloModule) != XlaStatus::OK) {
    emitError(moduleOp->getLoc(),
              "Failed converting MLIR bytecode to XLA HLO.");
    return failure();
  }

  return std::unique_ptr<xla::HloModule, xla::api::DestroyHloModule>(
      hloModule, xla::api::destroyHloModule);
}

FailureOr<OwningOpRef<ModuleOp>> convertFromHloModule(
    const xla::HloModule& hloModule, MLIRContext& context,
    const Location& location) {
  char* mlirBytecodeBuffer = nullptr;
  size_t mlirBytecodeBufferSize = 0;
  if (xla::api::xlaHloToStableHloBuffer(hloModule, &mlirBytecodeBuffer,
                                        &mlirBytecodeBufferSize) !=
      XlaStatus::OK) {
    emitError(location, "Failed converting XLA HLO to MLIR.");
    return failure();
  }
  auto mlirByteCodeBufferDestroyer = makeDestroyer([mlirBytecodeBuffer]() {
    xla::api::destroyCharBuffer(mlirBytecodeBuffer);
  });

  FailureOr<OwningOpRef<ModuleOp>> moduleOpOrFailure =
      loadMlir(mlirBytecodeBuffer, mlirBytecodeBufferSize, context);
  if (failed(moduleOpOrFailure)) {
    emitError(location, "Failed loading MLIR coming from XLA HLO.");
    return failure();
  }

  return moduleOpOrFailure;
}

std::unordered_map<std::string, XlaAutoShardingOptionPreserveShardingsType>
    strToAutoShardingOptionPreserveShardingsTypeMap = {
        {"kKeepAllShardings", kKeepAllShardings},
        {"kKeepInputOutputShardings", kKeepInputOutputShardings},
        {"kRemoveAllShardings", kRemoveAllShardings},
};

FailureOr<XlaAutoShardingOptionPreserveShardingsType>
strToAutoShardingOptionPreserveShardingsType(const std::string& x) {
  auto preserveShardingsIt =
      strToAutoShardingOptionPreserveShardingsTypeMap.find(x);
  if (preserveShardingsIt ==
      strToAutoShardingOptionPreserveShardingsTypeMap.end()) {
    std::cerr << "Failed parsing \"" << x
              << "\" as XlaAutoShardingOptionPreserveShardingsType"
              << std::endl;
    return failure();
  }
  return preserveShardingsIt->second;
}

template <typename ShapeIt>
int64_t shapeElementsCount(ShapeIt begin, ShapeIt end) {
  if (begin == end) {
    return 0;
  }
  return std::accumulate(begin, end, 1,
                         [](int64_t prod, int64_t x) { return prod * x; });
}

struct ShadingPropagationPass
    : public impl::ShadingPropagationBase<ShadingPropagationPass> {
  using ShadingPropagationBase::ShadingPropagationBase;

  LogicalResult initialize(MLIRContext* context) override {
    xlaCcLibHandle = xla::api::loadLibrary(xlaCcLibPath.getValue().c_str());
    if (!xlaCcLibHandle) {
      return LogicalResult::failure();
    }

    std::transform(
        allow_spmd_sharding_propagation_to_output.begin(),
        allow_spmd_sharding_propagation_to_output.end(),
        std::back_inserter(allow_spmd_sharding_propagation_to_output_vec),
        [](char x) { return x != '0'; });
    std::transform(
        allow_spmd_sharding_propagation_to_parameters.begin(),
        allow_spmd_sharding_propagation_to_parameters.end(),
        std::back_inserter(allow_spmd_sharding_propagation_to_parameters_vec),
        [](char x) { return x != '0'; });

    return LogicalResult::success();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    auto hloModule = convertToHloModule(moduleOp);
    if (failed(hloModule)) {
      return signalPassFailure();
    }

    XlaShardingPropagationOption option = getOptionFromPassArgs();
    if (xla::api::runShardingPropagationPass(hloModule.value().get(),
                                             &option) != XlaStatus::OK) {
      emitError(moduleOp->getLoc(), "Sharding propagation pass failed.");
      return signalPassFailure();
    }

    FailureOr<OwningOpRef<ModuleOp>> newModuleOpOrFailure =
        convertFromHloModule(*hloModule.value(), getContext(),
                             moduleOp.getLoc());

    move(newModuleOpOrFailure.value().get(), moduleOp);
  }

 private:
  XlaShardingPropagationOption getOptionFromPassArgs() {
    XlaShardingPropagationOption res;
    res.is_spmd = is_spmd;
    res.propagate_metadata = propagate_metadata;

    res.allow_spmd_sharding_propagation_to_output = reinterpret_cast<bool*>(
        &allow_spmd_sharding_propagation_to_output_vec[0]);
    res.allow_spmd_sharding_propagation_to_output_size =
        allow_spmd_sharding_propagation_to_output_vec.size();

    res.allow_spmd_sharding_propagation_to_parameters = reinterpret_cast<bool*>(
        &allow_spmd_sharding_propagation_to_parameters_vec[0]);
    res.allow_spmd_sharding_propagation_to_parameters_size =
        allow_spmd_sharding_propagation_to_parameters_vec.size();

    res.cse_prevention_only = cse_prevention_only;
    return res;
  }

  std::shared_ptr<void> xlaCcLibHandle;
  std::vector<char> allow_spmd_sharding_propagation_to_output_vec;
  std::vector<char> allow_spmd_sharding_propagation_to_parameters_vec;
};

struct SpmdPartitionerPass
    : public impl::SpmdPartitionerBase<SpmdPartitionerPass> {
  using SpmdPartitionerBase::SpmdPartitionerBase;

  LogicalResult initialize(MLIRContext* context) override {
    xlaCcLibHandle = xla::api::loadLibrary(xlaCcLibPath.getValue().c_str());
    if (!xlaCcLibHandle) {
      return LogicalResult::failure();
    }
    return LogicalResult::success();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    auto hloModule = convertToHloModule(moduleOp);
    if (failed(hloModule)) {
      return signalPassFailure();
    }

    XlaSpmdPartitionerOption option = getOptionFromPassArgs();
    if (xla::api::runSpmdPartitionerPass(hloModule.value().get(), &option) !=
        XlaStatus::OK) {
      emitError(moduleOp->getLoc(), "SPMD partitioner pass failed.");
      return signalPassFailure();
    }

    FailureOr<OwningOpRef<ModuleOp>> newModuleOpOrFailure =
        convertFromHloModule(*hloModule.value(), getContext(),
                             moduleOp.getLoc());

    move(newModuleOpOrFailure.value().get(), moduleOp);
  }

 private:
  XlaSpmdPartitionerOption getOptionFromPassArgs() {
    XlaSpmdPartitionerOption res;
    res.num_partitions = num_partitions;
    res.num_replicas = num_replicas;
    return res;
  }

  std::shared_ptr<void> xlaCcLibHandle;
};

struct CollectivesOptimizationPass
    : public impl::CollectivesOptimizationBase<CollectivesOptimizationPass> {
  using CollectivesOptimizationBase::CollectivesOptimizationBase;

  LogicalResult initialize(MLIRContext* context) override {
    xlaCcLibHandle = xla::api::loadLibrary(xlaCcLibPath.getValue().c_str());
    if (!xlaCcLibHandle) {
      return LogicalResult::failure();
    }
    return LogicalResult::success();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    auto hloModule = convertToHloModule(moduleOp);
    if (failed(hloModule)) {
      return signalPassFailure();
    }

    XlaCollectivesOptimizationOption option = getOptionFromPassArgs();
    if (xla::api::runCollectivesOptimizationPipeline(
            hloModule.value().get(), &option) != XlaStatus::OK) {
      emitError(moduleOp->getLoc(), "Collectives optimization pass failed.");
      return signalPassFailure();
    }

    FailureOr<OwningOpRef<ModuleOp>> newModuleOpOrFailure =
        convertFromHloModule(*hloModule.value(), getContext(),
                             moduleOp.getLoc());

    move(newModuleOpOrFailure.value().get(), moduleOp);
  }

 private:
  XlaCollectivesOptimizationOption getOptionFromPassArgs() {
    XlaCollectivesOptimizationOption res;
    res.reassociate_converted_all_reduce = reassociate_converted_all_reduce;
    res.enable_while_loop_reduce_scatter_code_motion =
        enable_while_loop_reduce_scatter_code_motion;
    res.enable_data_parallel_collective_optimizer =
        enable_data_parallel_collective_optimizer;
    return res;
  }

  std::shared_ptr<void> xlaCcLibHandle;
};

struct AutoShardingPass : public impl::AutoShardingBase<AutoShardingPass> {
  using AutoShardingBase::AutoShardingBase;

  LogicalResult initialize(MLIRContext* context) override {
    xlaCcLibHandle = xla::api::loadLibrary(xlaCcLibPath.getValue().c_str());
    if (!xlaCcLibHandle) {
      return LogicalResult::failure();
    }

    FailureOr<XlaAutoShardingOption> opt = getOptionFromPassArgs();
    if (failed(opt)) {
      std::cerr << "Failed parsing AutoShardingPass arguments." << std::endl;
      return LogicalResult::failure();
    }
    option = opt.value();

    return LogicalResult::success();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    auto hloModule = convertToHloModule(moduleOp);
    if (failed(hloModule)) {
      return signalPassFailure();
    }

    if (xla::api::runAutoShardingPass(hloModule.value().get(), &option) !=
        XlaStatus::OK) {
      emitError(moduleOp->getLoc(), "Auto sharding pass failed.");
      return signalPassFailure();
    }

    FailureOr<OwningOpRef<ModuleOp>> newModuleOpOrFailure =
        convertFromHloModule(*hloModule.value(), getContext(),
                             moduleOp.getLoc());

    move(newModuleOpOrFailure.value().get(), moduleOp);
  }

 private:
  void processArgs() {
    std::transform(force_strategy_stra_names.begin(),
                   force_strategy_stra_names.end(),
                   std::back_inserter(force_strategy_stra_names_vec),
                   [](std::string& x) { return x.data(); });
  }

  FailureOr<XlaAutoShardingOption> getOptionFromPassArgs() {
    XlaAutoShardingOption res;
    res.enable = true;

    FailureOr<XlaAutoShardingOptionPreserveShardingsType> preserveShardings =
        strToAutoShardingOptionPreserveShardingsType(preserve_shardings);
    if (failed(preserveShardings)) {
      return failure();
    }

    res.preserve_shardings = preserveShardings.value();
    res.simplify_graph = simplify_graph;
    res.memory_budget_per_device = memory_budget_per_device;
    res.memory_budget_ratio = memory_budget_ratio;
    res.force_all_gather_cost = force_all_gather_cost;
    res.all_gather_cost = all_gather_cost;
    res.force_all_to_all_cost = force_all_to_all_cost;
    res.all_to_all_cost = all_to_all_cost;
    res.force_batch_dim_to_mesh_dim = force_batch_dim_to_mesh_dim;
    res.allow_replicated_parameters = allow_replicated_parameters;
    res.prefer_reduce_scatter = prefer_reduce_scatter;
    res.reduce_scatter_grad_acc_friendly = reduce_scatter_grad_acc_friendly;
    res.reduce_scatter_aggressive_partition =
        reduce_scatter_aggressive_partition;
    res.batch_matmul_always_split_batch = batch_matmul_always_split_batch;
    res.allow_recompute_heavy_op = allow_recompute_heavy_op;
    res.allow_mixed_mesh_shape = allow_mixed_mesh_shape;
    res.grad_acc_num_micro_batches = grad_acc_num_micro_batches;
    res.load_solution_vector = load_solution_vector;
    res.solve_nd_sharding_iteratively = solve_nd_sharding_iteratively;
    res.force_simple_heuristic = force_simple_heuristic.data();
    res.force_strategy = force_strategy;
    res.force_strategy_size = force_strategy_inst_indices.size();

    res.force_strategy_inst_indices = &*force_strategy_inst_indices.begin();
    if (force_strategy_inst_indices.size() !=
        force_strategy_stra_names.size()) {
      std::cerr << "Error: force_strategy_inst_indices size "
                << force_strategy_inst_indices.size()
                << " is diffrent from force_strategy_stra_names size "
                << force_strategy_stra_names.size() << "." << std::endl;
      return failure();
    }
    force_strategy_stra_names_vec.clear();
    std::transform(force_strategy_stra_names.begin(),
                   force_strategy_stra_names.end(),
                   std::back_inserter(force_strategy_stra_names_vec),
                   [](std::string& x) { return x.data(); });
    res.force_strategy_stra_names = force_strategy_stra_names_vec.data();

    res.device_mesh_shape = &*device_mesh_shape.begin();
    res.device_mesh_shape_size = device_mesh_shape.size();
    int64_t expectedDeviceCount =
        shapeElementsCount(device_mesh_shape.begin(), device_mesh_shape.end());
    if (expectedDeviceCount != static_cast<int64_t>(device_mesh_ids.size())) {
      std::cerr << "Error: mismatch between size of device_mesh_ids of and "
                   "device_mesh_shape."
                << std::endl;
      return failure();
    }
    res.device_mesh_ids = &*device_mesh_ids.begin();

    if (device_mesh_shape.size() != device_mesh_alpha.size()) {
      std::cerr << "Error: divece_mesh_shape size " << device_mesh_shape.size()
                << " must be equal to device_mesh_alpha size "
                << device_mesh_alpha.size() << "." << std::endl;
      return failure();
    }
    res.device_mesh_alpha = &*device_mesh_alpha.begin();

    if (device_mesh_shape.size() != device_mesh_beta.size()) {
      std::cerr << "Error: divece_mesh_shape size " << device_mesh_shape.size()
                << " must be equal to device_mesh_beta size "
                << device_mesh_beta.size() << "." << std::endl;
      return failure();
    }
    res.device_mesh_beta = &*device_mesh_beta.begin();

    res.load_strategy = load_strategy;
    res.try_multiple_mesh_shapes = try_multiple_mesh_shapes;
    res.solver_timeout_in_seconds = solver_timeout_in_seconds;
    res.strategy_vector = &*strategy_vector.begin();
    res.strategy_vector_size = strategy_vector.size();
    return res;
  }

  XlaAutoShardingOption option;
  std::vector<char*> force_strategy_stra_names_vec;
  std::shared_ptr<void> xlaCcLibHandle;
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createShadingPropagationPass() {
  return std::make_unique<ShadingPropagationPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createSpmdPartitionerPass() {
  return std::make_unique<SpmdPartitionerPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createCollectivesOptimizationPass() {
  return std::make_unique<CollectivesOptimizationPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createAutoShardingPass() {
  return std::make_unique<AutoShardingPass>();
}

}  // namespace stablehlo
}  // namespace mlir
