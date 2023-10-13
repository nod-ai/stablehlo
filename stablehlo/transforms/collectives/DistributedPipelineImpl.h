#include <memory>

#include "Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassRegistry.h"
#include "stablehlo/conversions/distributed/Passes.h"
#include "stablehlo/transforms/xla/XlaPasses.h"

namespace mlir {
namespace stablehlo {

void populateDistributedPassPipeline(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createLegalizeDistributed());
  pm.addPass(createRenameEntryToMain());
  pm.addPass(createShadingPropagationAndSpmdPartitionerPass(
      ShadingPropagationAndSpmdPartitionerOptions{
          .is_spmd = true,
          .allow_spmd_sharding_propagation_to_output = ArrayRef<char>({1}),
          .allow_spmd_sharding_propagation_to_parameters = ArrayRef<char>({1}),
      }));
  pm.addPass(createCollectivesOptimizationPass());
  pm.addPass(createRenameMainToEntry());
}

void registerDistributedPassPipeline() {
  PassPipelineRegistration<>(
      "stablehlo-distributed-pass-pipeline",
      "Run sharding propagation and SPMD partitioner pipeline.",
      populateDistributedPassPipeline);
}

}  // namespace stablehlo
}  // namespace mlir
