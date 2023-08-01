#include <algorithm>

#include "CollectivesPassesCli.h"
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
  int64_t strideDim0 = shape[1];
  SmallVector<APInt, 16> resultArray(resultShape[0] * resultShape[1]);
  int64_t i = 0;
  for (auto& superSubDeviceMapPair : superSubDeviceMap) {
    auto& subDevices = superSubDeviceMapPair.second;
    std::transform(subReplicaGroups.begin(), subReplicaGroups.end(),
                   resultArray.begin() + i * strideDim0,
                   [&subDevices](const APInt& subDeviceIndex) {
                     return llvm::APInt(
                         64, subDevices[subDeviceIndex.getSExtValue()],
                         /*isSigned=*/true);
                   });
    ++i;
  }
  return DenseIntElementsAttr::get(
      subReplicaGroups.getShapedType().clone(resultShape), resultArray);
}

struct CompleteAllReduceSpmdSubPartition
    : public OpRewritePattern<AllReduceOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AllReduceOp op,
                                PatternRewriter& rewriter) const final {
    if (!op->hasAttr("sub_partition")) {
      return failure();
    }

    DenseIntElementsAttr replicaGroupsAttr =
        op->getAttrOfType<DenseIntElementsAttr>("replica_groups");
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
    op->removeAttr("sub_partition");
    op->setAttr("complete_partition", UnitAttr::get(getContext()));

    return success();
  }
};

void populateSpmdSubPartitionCompletionRewritePatterns(
    RewritePatternSet& patterns) {
  patterns.add<CompleteAllReduceSpmdSubPartition>(patterns.getContext());
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

  // CompleteCollectivesSpmdSubPartitionPass(const
  // CompleteCollectivesSpmdSubPartitionOptions &options) :
  // CompleteCollectivesSpmdSubPartitionBase(options) {
  //   registerCollectiveCliOptions();
  // }

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
