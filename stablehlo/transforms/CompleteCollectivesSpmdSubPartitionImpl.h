#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include <algorithm>
#include "CollectivesPassesCli.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_COMPLETECOLLECTIVESSPMDSUBPARTITION
#include "stablehlo/transforms/Passes.h.inc"

namespace {

DenseIntElementsAttr getCompleteSubReplicaGroups(const DenseIntElementsAttr& subReplicaGroups,
  const SuperSubDeviceIdMap& superSubDeviceMap) {
  assert(!superSubDeviceMap.empty());
  auto shape = subReplicaGroups.getShapedType().getShape();
  std::vector<int64_t> resultShape(shape.begin(), shape.end());
  resultShape[0] *= superSubDeviceMap.begin()->second.size();
  int64_t strideDim0 = shape[1];
  int64_t resultStrideDim0 = resultShape[1];
  SmallVector<int64_t, 16> resultArray(resultShape[0] * resultShape[1]);
  int64_t i = 0;
  for (auto& superSubDeviceMapPair : superSubDeviceMap) {

    ++i;
  }
  // for (auto subReplicaGroupsIt = subReplicaGroups.value_begin<int64_t>();
  //   subReplicaGroupsIt != subReplicaGroups.value_end<int64_t>();
  //   subReplicaGroupsIt += strideDim0) {
  //   auto groupEnd = subReplicaGroupsIt + strideDim0;

  //   for (groupIt = subReplicaGroupsIt
  // }
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

    DenseIntElementsAttr replicaGroupsAttr = op->getAttrOfType<DenseIntElementsAttr>("replica_groups");
    if (!replicaGroupsAttr) {
      emitError(op.getLoc(), "Expected operation attribute replica_groups not found.");
      return failure();
    }

    BoolAttr useGlobalDeviceIds = op->getAttrOfType<BoolAttr>("use_global_device_ids");
    if (!useGlobalDeviceIds || !useGlobalDeviceIds.getValue()) {
      emitError(op.getLoc(), "Expected attribute use_global_device_ids to be true.");
      return failure();
    }

    DenseIntElementsAttr newReplicaGroupsAttr = getCompleteSubReplicaGroups(replicaGroupsAttr);
    op.setReplicaGroupsAttr(newReplicaGroupsAttr);

    return success();
  }
};

struct CompleteCollectivesSpmdSubPartitionPass
    : public impl::CompleteCollectivesSpmdSubPartitionBase<
          CompleteCollectivesSpmdSubPartitionPass> {
  // using CompleteCollectivesSpmdSubPartitionBase::
  //     CompleteCollectivesSpmdSubPartitionBase;

  CompleteCollectivesSpmdSubPartitionPass() {
    registerCollectiveCliOptions();
  }

  CompleteCollectivesSpmdSubPartitionPass(const CompleteCollectivesSpmdSubPartitionPass &other) : CompleteCollectivesSpmdSubPartitionBase(other) {
    registerCollectiveCliOptions();
  }

  CompleteCollectivesSpmdSubPartitionPass(const CompleteCollectivesSpmdSubPartitionOptions &options) : CompleteCollectivesSpmdSubPartitionBase(options) {
    registerCollectiveCliOptions();
  }

  void runOnOperation() override {}
};

}  // namespace

}  // namespace stablehlo
}  // namespace mlir
