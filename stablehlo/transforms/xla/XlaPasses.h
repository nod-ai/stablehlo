#ifndef STABLEHLO_TRANSFORMS_XLA_XLAPASSES_H
#define STABLEHLO_TRANSFORMS_XLA_XLAPASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {

class ModuleOp;

namespace stablehlo {

#define GEN_PASS_DECL
#include "stablehlo/transforms/xla/XlaPasses.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createShadingPropagationPass();
std::unique_ptr<OperationPass<ModuleOp>> createSpmdPartitionerPass();
std::unique_ptr<OperationPass<ModuleOp>> createCollectivesOptimizationPass();
std::unique_ptr<OperationPass<ModuleOp>> createAutoShardingPass();

#define GEN_PASS_REGISTRATION
#include "stablehlo/transforms/xla/XlaPasses.h.inc"

inline void registerAllXlaPasses() { registerXlaPassesPasses(); }

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_TRANSFORMS_XLA_XLAPASSES_H
