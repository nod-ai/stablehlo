#include <algorithm>

#include "CollectivesPassesCli.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/collectives/Passes.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_COMPLETECOLLECTIVESSPMDSUBPARTITION
#include "stablehlo/transforms/collectives/Passes.h.inc"

namespace {

// Complete sub-partition replica groups to a complete-partition replica groups.
DenseIntElementsAttr completeSubReplicaGroups(
    const DenseIntElementsAttr& subReplicaGroups,
    const SuperSubDeviceIdMap& superSubDeviceMap) {
  assert(!superSubDeviceMap.empty());
  auto shape = subReplicaGroups.getShapedType().getShape();
  std::vector<int64_t> resultShape(shape.begin(), shape.end());
  resultShape[0] *= superSubDeviceMap.size();
  SmallVector<APInt, 16> resultArray(resultShape[0] * resultShape[1]);
  int64_t i = 0;
  for (auto& superSubDeviceMapPair : superSubDeviceMap) {
    auto& completeDevices = superSubDeviceMapPair.second;
    std::transform(subReplicaGroups.begin(), subReplicaGroups.end(),
                   resultArray.begin() + i * shape[0] * shape[1],
                   [&completeDevices](const APInt& subDevice) {
                     return llvm::APInt(
                         64, completeDevices[subDevice.getSExtValue()],
                         /*isSigned=*/true);
                   });
    ++i;
  }
  return DenseIntElementsAttr::get(
      subReplicaGroups.getShapedType().clone(resultShape), resultArray);
}

template <typename Op>
struct CompleteSpmdSubPartitionPattern : public OpRewritePattern<Op> {
 public:
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op, PatternRewriter& rewriter) const final {
    StringAttr deviceDomain =
        op->template getAttrOfType<StringAttr>("device_domain");
    if (!deviceDomain || deviceDomain.getValue() != "sub") {
      return failure();
    }

    DenseIntElementsAttr replicaGroupsAttr =
        op->template getAttrOfType<DenseIntElementsAttr>("replica_groups");
    if (!replicaGroupsAttr) {
      emitError(op.getLoc(),
                "Expected operation attribute replica_groups not found.");
      return failure();
    }

    if (!op->hasAttr("use_global_device_ids")) {
      emitError(op.getLoc(),
                "Expected attribute use_global_device_ids. "
                "No other modes are supported.");
      return failure();
    }

    DenseIntElementsAttr newReplicaGroupsAttr = completeSubReplicaGroups(
        replicaGroupsAttr, getCollectiveOptions().superSubDeviceMap);
    op.setReplicaGroupsAttr(newReplicaGroupsAttr);
    op->setAttr("device_domain",
                StringAttr::get(this->getContext(), "complete"));

    return success();
  }
};

void populateSpmdSubPartitionCompletionRewritePatterns(
    RewritePatternSet& patterns) {
  patterns.add<CompleteSpmdSubPartitionPattern<AllGatherOp>,
               CompleteSpmdSubPartitionPattern<AllReduceOp>,
               CompleteSpmdSubPartitionPattern<ReduceScatterOp>>(
      patterns.getContext());
}

struct CompleteCollectivesSpmdSubPartitionPass
    : public impl::CompleteCollectivesSpmdSubPartitionBase<
          CompleteCollectivesSpmdSubPartitionPass> {
  CompleteCollectivesSpmdSubPartitionPass() { registerCollectiveCliOptions(); }

  CompleteCollectivesSpmdSubPartitionPass(
      const CompleteCollectivesSpmdSubPartitionPass& other)
      : CompleteCollectivesSpmdSubPartitionBase(other) {
    registerCollectiveCliOptions();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateSpmdSubPartitionCompletionRewritePatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace stablehlo
}  // namespace mlir
