#include <algorithm>
#include <iterator>

#include "CollectivesPassesCli.h"
#include "Utils.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
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
template <typename Op>
FailureOr<DenseIntElementsAttr> completeSubReplicaGroups(
    Op op, const SuperSubDeviceIdMap& superSubDeviceMap) {
  assert(!superSubDeviceMap.empty());
  SmallVector<int64_t, 64> deviceGroups;
  SmallVector<int64_t, 2> deviceGroupsShape;
  ModuleOp moduleOp = op->template getParentOfType<ModuleOp>();
  IntegerAttr numPartitionsAttr =
      moduleOp->getAttrOfType<IntegerAttr>("mhlo.num_partitions");
  if (!numPartitionsAttr) {
    emitError(moduleOp->getLoc())
        << "mhlo.num_partitions is a required attribute.";
    return failure();
  }
  auto numPartitions = numPartitionsAttr.getInt();
  IntegerAttr numReplicasAttr =
      moduleOp->getAttrOfType<IntegerAttr>("mhlo.num_replicas");
  if (!numReplicasAttr) {
    emitError(moduleOp->getLoc())
        << "mhlo.num_replicas is a required attribute.";
    return failure();
  }
  auto numReplicas = numReplicasAttr.getInt();
  getReplicaGroupsAsGlobalDeviceIds(op, numPartitions, numReplicas,
                                    std::back_inserter(deviceGroups),
                                    std::back_inserter(deviceGroupsShape));

  std::vector<int64_t> resultShape(deviceGroupsShape.begin(),
                                   deviceGroupsShape.end());
  resultShape[0] *= superSubDeviceMap.size();
  SmallVector<APInt, 16> resultArray(resultShape[0] * resultShape[1]);
  int64_t i = 0;
  for (auto& superSubDeviceMapPair : superSubDeviceMap) {
    auto& completeDevices = superSubDeviceMapPair.second;
    std::transform(
        deviceGroups.begin(), deviceGroups.end(),
        resultArray.begin() + i * deviceGroupsShape[0] * deviceGroupsShape[1],
        [&completeDevices](int64_t subDevice) {
          return llvm::APInt(64, completeDevices[subDevice],
                             /*isSigned=*/true);
        });
    ++i;
  }
  return DenseIntElementsAttr::get(
      op.getReplicaGroups().getShapedType().clone(resultShape), resultArray);
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

    FailureOr<DenseIntElementsAttr> newReplicaGroupsAttr =
        completeSubReplicaGroups(op, getCollectiveOptions().superSubDeviceMap);
    if (failed(newReplicaGroupsAttr)) {
      return failure();
    }
    op.setReplicaGroupsAttr(newReplicaGroupsAttr.value());
    op->setAttr("use_global_device_ids", UnitAttr::get(op->getContext()));
    op->setAttr("device_domain",
                StringAttr::get(this->getContext(), "complete"));

    return success();
  }
};

struct CollectivePermuteCompleteSpmdSubPartitionPattern
    : public OpRewritePattern<CollectivePermuteOp> {
 public:
  using OpRewritePattern<CollectivePermuteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CollectivePermuteOp op,
                                PatternRewriter& rewriter) const final {
    emitError(
        op.getLoc(),
        "Sub-partition completion for collective_permute is not implemented.");
    return failure();
  }
};

void populateSpmdSubPartitionCompletionRewritePatterns(
    RewritePatternSet& patterns) {
  patterns.add<CompleteSpmdSubPartitionPattern<AllGatherOp>,
               CompleteSpmdSubPartitionPattern<AllReduceOp>,
               CompleteSpmdSubPartitionPattern<AllToAllOp>,
               CompleteSpmdSubPartitionPattern<ReduceScatterOp>,
               CollectivePermuteCompleteSpmdSubPartitionPattern>(
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
