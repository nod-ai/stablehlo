#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/collectives/Passes.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_COLLECTIVESSPMDSUBPARTITIONER
#include "stablehlo/transforms/collectives/Passes.h.inc"

namespace {

struct CollectivesSpmdSubPartitionerPass
    : public impl::CollectivesSpmdSubPartitionerBase<
          CollectivesSpmdSubPartitionerPass> {
  using CollectivesSpmdSubPartitionerBase::CollectivesSpmdSubPartitionerBase;

  LogicalResult initialize(MLIRContext* context) override {
    return LogicalResult::failure();
  }

  void runOnOperation() override {}
};

}  // namespace

}  // namespace stablehlo
}  // namespace mlir
