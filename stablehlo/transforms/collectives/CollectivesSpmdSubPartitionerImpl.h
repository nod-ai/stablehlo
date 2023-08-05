#include "CollectivesPassesCli.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/collectives/Passes.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_COLLECTIVESSPMDSUBPARTITIONER
#include "stablehlo/transforms/collectives/Passes.h.inc"

namespace {

struct AllGatherPattern : public OpRewritePattern<AllGatherOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AllGatherOp op,
                                PatternRewriter& rewriter) const final {
    return failure();
  }
};

void populateCollectivesSpmdSubPartitionerRewritePatterns(
    RewritePatternSet& patterns) {
  patterns.add<AllGatherPattern>(patterns.getContext());
}

struct CollectivesSpmdSubPartitionerPass
    : public impl::CollectivesSpmdSubPartitionerBase<
          CollectivesSpmdSubPartitionerPass> {
  CollectivesSpmdSubPartitionerPass() { registerCollectiveCliOptions(); }

  CollectivesSpmdSubPartitionerPass(
      const CollectivesSpmdSubPartitionerPass& other)
      : CollectivesSpmdSubPartitionerBase(other) {
    registerCollectiveCliOptions();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateCollectivesSpmdSubPartitionerRewritePatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace stablehlo
}  // namespace mlir
