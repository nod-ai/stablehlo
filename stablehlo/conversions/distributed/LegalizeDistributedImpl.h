#include "Passes.h"
#include "RewritePatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_LEGALIZEDISTRIBUTED
#include "stablehlo/conversions/distributed/Passes.h.inc"

struct LegalizeDistributedPass
    : public impl::LegalizeDistributedBase<LegalizeDistributedPass> {
  using LegalizeDistributedBase::LegalizeDistributedBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateShardingAnnotationOpTargetNameRewritePattern(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation().getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  std::shared_ptr<void> xlaCcLibHandle;
};

}  // namespace stablehlo
}  // namespace mlir
