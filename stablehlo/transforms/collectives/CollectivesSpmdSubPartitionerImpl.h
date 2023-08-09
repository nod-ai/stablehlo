#include <memory>
#include <utility>

#include "CollectivesPassesCli.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/collectives/Passes.h"
#include "stablehlo/transforms/xla/XlaCcLibLoader.h"
#include "xla/xla_cc.h"
#include "xla/xla_cc_loader.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_COLLECTIVESSPMDSUBPARTITIONER
#include "stablehlo/transforms/collectives/Passes.h.inc"

namespace {

SmallVector<int64_t, 2> superReplicaGroupsCompletionShape(
    const ArrayRef<int64_t> replicaGroupsShape,
    const SuperSubDeviceIdMap& superSubDeviceMap) {
  assert(replicaGroupsShape.size() == 2);
  assert(superSubDeviceMap.size() > 1);
  return {
      replicaGroupsShape[0] * int64_t(superSubDeviceMap.begin()->second.size()),
      replicaGroupsShape[1]};
}

void completeSuperReplicaGroups(const ArrayRef<int64_t> replicaGroups,
                                const ArrayRef<int64_t> replicaGroupsShape,
                                const SuperSubDeviceIdMap& superSubDeviceMap,
                                MutableArrayRef<int64_t> outReplicaGroups) {
  SmallVector<DeviceId, 2> outShape =
      superReplicaGroupsCompletionShape(replicaGroupsShape, superSubDeviceMap);
  assert(outShape[0] * outShape[1] == int64_t(outReplicaGroups.size()));
  for (size_t completeDevicesIndex = 0;
       completeDevicesIndex < superSubDeviceMap.begin()->second.size();
       ++completeDevicesIndex) {
    std::transform(
        replicaGroups.begin(), replicaGroups.end(),
        outReplicaGroups.begin() + completeDevicesIndex *
                                       replicaGroupsShape[0] *
                                       replicaGroupsShape[1],
        [completeDevicesIndex, &superSubDeviceMap](DeviceId superDevice) {
          auto it = superSubDeviceMap.find(superDevice);
          assert(it != superSubDeviceMap.end());
          return it->second[completeDevicesIndex];
        });
  }
}

DenseIntElementsAttr completeSuperReplicaGroups(
    const DenseIntElementsAttr& superReplicaGroups,
    const SuperSubDeviceIdMap& superSubDeviceMap) {
  // auto superReplicaGroupsShape =
  // superReplicaGroups.getShapedType().getShape();
  SmallVector<int64_t, 32> superReplicaGroupsArr(superReplicaGroups.size());
  std::transform(superReplicaGroups.begin(), superReplicaGroups.end(),
                 superReplicaGroupsArr.begin(),
                 [](const APInt& deviceId) { return deviceId.getSExtValue(); });
  auto resultReplicaGroupsShape = superReplicaGroupsCompletionShape(
      superReplicaGroups.getShapedType().getShape(), superSubDeviceMap);
  SmallVector<int64_t, 32> resultReplicaGroupsArr(resultReplicaGroupsShape[0] *
                                                  resultReplicaGroupsShape[1]);

  completeSuperReplicaGroups(superReplicaGroupsArr,
                             superReplicaGroups.getShapedType().getShape(),
                             superSubDeviceMap, resultReplicaGroupsArr);
  return DenseIntElementsAttr::get(
      superReplicaGroups.getShapedType().clone(resultReplicaGroupsShape),
      resultReplicaGroupsArr);
}

Operation* insertShardingProlog(ImplicitLocOpBuilder& builder, Value operand,
                                StringAttr sharding, Type resultType) {
  CustomCallOp shardingOp = builder.create<CustomCallOp>(
      TypeRange(operand.getType()), ValueRange(operand));
  shardingOp.setCallTargetName(builder.getAttr<StringAttr>("Sharding"));
  shardingOp->setAttr("mhlo.sharding", sharding);
  CustomCallOp spmdFullToShardShapeOp = builder.create<CustomCallOp>(
      TypeRange(resultType), shardingOp->getResults());
  spmdFullToShardShapeOp.setCallTargetName(
      builder.getAttr<StringAttr>("SPMDFullToShardShape"));
  spmdFullToShardShapeOp->setAttr("mhlo.sharding",
                                  builder.getAttr<StringAttr>("{manual}"));
  return spmdFullToShardShapeOp.getOperation();
}

Operation* insertShardingEpilog(ImplicitLocOpBuilder& builder, Value operand,
                                StringAttr sharding, Type resultType) {
  CustomCallOp shardingOp = builder.create<CustomCallOp>(
      TypeRange(operand.getType()), ValueRange(operand));
  shardingOp.setCallTargetName(builder.getAttr<StringAttr>("Sharding"));
  shardingOp->setAttr("mhlo.sharding", builder.getAttr<StringAttr>("{manual}"));
  CustomCallOp spmdShardToFullShape = builder.create<CustomCallOp>(
      TypeRange(resultType), shardingOp->getResults());
  spmdShardToFullShape.setCallTargetName(
      builder.getAttr<StringAttr>("SPMDShardToFullShape"));
  spmdShardToFullShape->setAttr("mhlo.sharding", sharding);
  return spmdShardToFullShape.getOperation();
}

ShapedType getShardingTileShape(xla::HloSharding* hloSharding,
                                ShapedType shape) {
  ArrayRef<int64_t> shapeArr = shape.getShape();
  size_t resultShapeSize;
  xla::api::hloShardingTileShape(hloSharding, shapeArr.data(), shapeArr.size(),
                                 nullptr, &resultShapeSize);
  llvm::SmallVector<int64_t, 10> resultShapeArr(resultShapeSize);
  xla::api::hloShardingTileShape(hloSharding, shapeArr.data(), shapeArr.size(),
                                 resultShapeArr.data(), &resultShapeSize);
  return shape.cloneWith(resultShapeArr, shape.getElementType());
}

template <typename Op>
FailureOr<std::pair<ShapedType, ShapedType>> getOperandAndResultShardedShapes(
    Op op) {
  StringAttr shardingAttr =
      op->template getAttrOfType<StringAttr>("mhlo.sharding");
  if (!shardingAttr) {
    emitError(op.getLoc(),
              "Expected operation attribute mhlo.sharding not found.");
    return failure();
  }

  xla::HloSharding* hloSharding = nullptr;
  XlaStatus xlaStatus = xla::api::parseHloSharding(
      shardingAttr.data(), shardingAttr.size(), &hloSharding);
  if (xlaStatus != XlaStatus::OK) {
    emitError(op.getLoc()) << "Failed to parse sharding \""
                           << shardingAttr.str() << "\".";
    return failure();
  }
  std::unique_ptr<xla::HloSharding, xla::api::DestroyHloSharding>
      hloShardingDeleter(hloSharding, xla::api::destroyHloSharding);

  if (xla::api::hloShardingIsTuple(hloSharding)) {
    emitError(op.getLoc()) << "Tuple sharding \"" << shardingAttr.str()
                           << "\" not supported.";
    return failure();
  }

  if (op->getResultTypes().size() > 1) {
    emitError(op.getLoc()) << "Expected 1 result got "
                           << op->getResultTypes().size() << ".";
    return failure();
  }
  if (op->getOperandTypes().size() > 1) {
    emitError(op.getLoc()) << "Expected 1 operand got "
                           << op->getResultTypes().size() << ".";
    return failure();
  }
  return std::make_pair(
      getShardingTileShape(
          hloSharding, op->getOperand(0).getType().template cast<ShapedType>()),
      getShardingTileShape(
          hloSharding, op->getResult(0).getType().template cast<ShapedType>()));
}

bool checkInputCollectiveOp(Operation* op) {
  StringAttr deviceDomain =
      op->template getAttrOfType<StringAttr>("device_domain");
  if (!deviceDomain || deviceDomain.str() != "super") {
    return false;
  }
  if (!op->hasAttr("use_global_device_ids")) {
    emitError(op->getLoc()) << "Only use_global_device_ids mode is supported.";
    return false;
  }
  return true;
}

template <typename Op>
FailureOr<Op> createReplacementCollectiveOperation(
    Type resultType, Value operand, Op originalOp,
    const SuperSubDeviceIdMap& superSubDeviceMap);

template <>
FailureOr<AllGatherOp> createReplacementCollectiveOperation<AllGatherOp>(
    Type resultType, Value operand, AllGatherOp originalOp,
    const SuperSubDeviceIdMap& superSubDeviceMap) {
  ImplicitLocOpBuilder builder(originalOp->getLoc(), originalOp.getOperation());
  if (operand.getType()
              .cast<ShapedType>()
              .getShape()[originalOp.getAllGatherDim()] !=
          originalOp.getOperand()
              .getType()
              .getShape()[originalOp.getAllGatherDim()] ||
      resultType.cast<ShapedType>().getShape()[originalOp.getAllGatherDim()] !=
          originalOp.getResult()
              .getType()
              .getShape()[originalOp.getAllGatherDim()]) {
    emitError(originalOp.getLoc())
        << "Sharding along all_gather_dim is not supported.";
    return failure();
  }
  AllGatherOp newOp = builder.create<AllGatherOp>(
      resultType, operand, originalOp.getAllGatherDim(),
      completeSuperReplicaGroups(originalOp.getReplicaGroups(),
                                 superSubDeviceMap),
      originalOp.getChannelHandleAttr(), true);
  newOp->setAttr("device_domain",
                 StringAttr::get(originalOp->getContext(), "complete"));
  return newOp;
}

template <>
FailureOr<AllReduceOp> createReplacementCollectiveOperation<AllReduceOp>(
    Type resultType, Value operand, AllReduceOp originalOp,
    const SuperSubDeviceIdMap& superSubDeviceMap) {
  ImplicitLocOpBuilder builder(originalOp->getLoc(), originalOp.getOperation());
  AllReduceOp newOp = builder.create<AllReduceOp>(
      resultType, operand,
      completeSuperReplicaGroups(originalOp.getReplicaGroups(),
                                 superSubDeviceMap),
      originalOp.getChannelHandleAttr(), true);
  newOp.getRegion().takeBody(originalOp.getComputation());
  newOp->setAttr("device_domain",
                 StringAttr::get(originalOp->getContext(), "complete"));
  return newOp;
}

// sub-partition -> complete-partition
template <typename Op>
FailureOr<Operation*> spmdPartitionCollective(
    PatternRewriter& rewriter, Op op,
    const SuperSubDeviceIdMap& superSubDeviceMap) {
  if (!checkInputCollectiveOp(op.getOperation())) {
    return failure();
  }
  FailureOr<std::pair<ShapedType, ShapedType>> shardedArgAndResShapes =
      getOperandAndResultShardedShapes(op);
  if (failed(shardedArgAndResShapes)) {
    return failure();
  }

  ImplicitLocOpBuilder builder(op->getLoc(), op.getOperation());
  Operation* shardingProlog = insertShardingProlog(
      builder, op.getOperand(),
      op->template getAttrOfType<StringAttr>("mhlo.sharding"),
      shardedArgAndResShapes->first);
  assert(shardingProlog->getResults().size() == 1);

  FailureOr<Op> newOp = createReplacementCollectiveOperation<Op>(
      shardedArgAndResShapes->second, shardingProlog->getResult(0), op,
      superSubDeviceMap);
  if (failed(newOp)) {
    return failure();
  }

  FailureOr<Operation*> newResultOp = insertShardingEpilog(
      builder, newOp->getResult(),
      op->template getAttrOfType<StringAttr>("mhlo.sharding"),
      op.getResult().getType());
  if (failed(newResultOp)) {
    return failure();
  }
  rewriter.replaceOp(op.getOperation(), newResultOp.value()->getResults());
  return newResultOp;
}

template <typename Op>
struct CollectiveOpPatternRewriter : public OpRewritePattern<Op> {
 public:
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op, PatternRewriter& rewriter) const final {
    return spmdPartitionCollective(rewriter, op,
                                   getCollectiveOptions().superSubDeviceMap);
  }
};

void populateCollectivesSpmdSubPartitionerRewritePatterns(
    RewritePatternSet& patterns) {
  patterns.add<CollectiveOpPatternRewriter<AllGatherOp>,
               CollectiveOpPatternRewriter<AllReduceOp>>(patterns.getContext());
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

  LogicalResult initialize(MLIRContext* context) override {
    xlaCcLibHandle = xla::api::loadLibrary(xlaCcLibPath().c_str());
    if (!xlaCcLibHandle) {
      return failure();
    }
    return success();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateCollectivesSpmdSubPartitionerRewritePatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  std::shared_ptr<void> xlaCcLibHandle;
};

}  // namespace

}  // namespace stablehlo
}  // namespace mlir
