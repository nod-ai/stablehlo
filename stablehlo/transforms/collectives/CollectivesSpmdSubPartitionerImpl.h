#include <cstddef>
#include <iterator>
#include <memory>
#include <string>
#include <utility>

#include "CollectivesPassesCli.h"
#include "Utils.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
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

LogicalResult handleNumPartitionsAndReplicasAttributes(
    ModuleOp op, const SuperSubDeviceIdMap& superSubDeviceMap) {
  Builder builder(op->getContext());

  DictionaryAttr frontendAttributes =
      op->getAttrOfType<DictionaryAttr>("mhlo.frontend_attributes");
  if (!frontendAttributes) {
    frontendAttributes = DictionaryAttr::get(op->getContext());
  }

  Attribute subPartitionNumPartitionsAttr =
      frontendAttributes.get("sub_partition_num_partitions");
  Attribute subPartitionNumReplicasAttr =
      frontendAttributes.get("sub_partition_num_replicas");
  if (bool(subPartitionNumPartitionsAttr) !=
      bool(subPartitionNumReplicasAttr)) {
    MLIR_EMIT_ERROR(op.getLoc())
        << "sub_partition_num_partitions and sub_partition_num_replicas "
           "must both be present or not at the same time in "
           "mhlo.frontend_attributes";
    return failure();
  }
  int32_t subPartitionNumPartitions;
  int32_t subPartitionNumReplicas;
  if (subPartitionNumPartitionsAttr) {
    FAILURE_OR_ASSIGN_OR_RETURN(
        subPartitionNumPartitions,
        toInteger<int32_t>(
            subPartitionNumPartitionsAttr.cast<StringAttr>().strref(),
            op->getLoc()));
    FAILURE_OR_ASSIGN_OR_RETURN(
        subPartitionNumReplicas,
        toInteger<int32_t>(
            subPartitionNumReplicasAttr.cast<StringAttr>().strref(),
            op->getLoc()));
  } else {
    subPartitionNumPartitions = 1;
    assert(!superSubDeviceMap.empty());
    subPartitionNumReplicas = superSubDeviceMap.begin()->second.size();
  }

  IntegerAttr superPartitionNumPartitionsAttr =
      op->getAttrOfType<IntegerAttr>("mhlo.num_partitions");
  if (!superPartitionNumPartitionsAttr) {
    MLIR_EMIT_ERROR(op->getLoc())
        << "Required integer attribute mhlo.num_partitions not found.";
  }
  int32_t superPartitionNumPartitions =
      superPartitionNumPartitionsAttr.getValue().getSExtValue();

  IntegerAttr superPartitionNumReplicasAttr =
      op->getAttrOfType<IntegerAttr>("mhlo.num_replicas");
  if (!superPartitionNumReplicasAttr) {
    MLIR_EMIT_ERROR(op->getLoc())
        << "Required integer attribute mhlo.num_replicas not found.";
  }
  int32_t superPartitionNumReplicas =
      superPartitionNumReplicasAttr.getValue().getSExtValue();

  frontendAttributes = setAttributes(
      frontendAttributes,
      {NamedAttribute(
           builder.getStringAttr("super_partition_num_partitions"),
           builder.getStringAttr(std::to_string(superPartitionNumPartitions))),
       NamedAttribute(
           builder.getStringAttr("super_partition_num_replicas"),
           builder.getStringAttr(std::to_string(superPartitionNumReplicas)))});

  frontendAttributes = removeAttributes(
      frontendAttributes,
      {"sub_partition_num_partitions", "sub_partition_num_replicas"});

  op->setAttr("mhlo.frontend_attributes", frontendAttributes);
  op->setAttr("mhlo.num_partitions",
              builder.getI32IntegerAttr(subPartitionNumPartitions));
  op->setAttr("mhlo.num_replicas",
              builder.getI32IntegerAttr(subPartitionNumReplicas));
  return success();
}

LogicalResult handleModuleIoShardingAttributes(ModuleOp op) {
  DictionaryAttr frontendAttributes =
      op->getAttrOfType<DictionaryAttr>("mhlo.frontend_attributes");
  if (!frontendAttributes) {
    frontendAttributes = DictionaryAttr::get(op->getContext());
  }

  ArrayAttr argsShardingAttr =
      op->getAttrOfType<ArrayAttr>("mhlo.spmd_parameters_shardings");
  if (!argsShardingAttr) {
    MLIR_EMIT_ERROR(op->getLoc())
        << "Array attribute \"mhlo.spmd_parameters_shardings\" not found in "
           "module "
           "op.";
    return failure();
  }
  HloShardingPtr argsHloSharding = makeHloShardingPtr(nullptr);
  FAILURE_OR_ASSIGN_OR_RETURN(
      argsHloSharding, makeHloShardingTuple(argsShardingAttr, op.getLoc()));
  Attribute argsShardingStrAttr;
  FAILURE_OR_ASSIGN_OR_RETURN(
      argsShardingStrAttr,
      makeHloShardingAttr(*argsHloSharding.get(), op.getLoc()));

  StringAttr resultsShardingAttr =
      op->getAttrOfType<StringAttr>("mhlo.spmd_output_sharding");
  if (!resultsShardingAttr) {
    MLIR_EMIT_ERROR(op->getLoc())
        << "Attribute \"mhlo.spmd_output_sharding\" not found in module op.";
    return failure();
  }

  DictionaryAttr newFrontendAttributes = setAttributes(
      frontendAttributes,
      {
          NamedAttribute(
              StringAttr::get(op->getContext(),
                              "super_partition_spmd_parameters_sharding"),
              argsShardingStrAttr),
          NamedAttribute(
              StringAttr::get(op->getContext(),
                              "super_partition_spmd_output_sharding"),
              resultsShardingAttr),
      });
  op->setAttr("mhlo.frontend_attributes", newFrontendAttributes);

  op->removeAttr("mhlo.spmd_parameters_shardings");
  op->removeAttr("mhlo.spmd_output_sharding");

  return success();
}

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
  CustomCallOp spmdFullToShardShapeOp =
      builder.create<CustomCallOp>(TypeRange(resultType), ValueRange(operand));
  spmdFullToShardShapeOp.setCallTargetName(
      builder.getAttr<StringAttr>("SPMDFullToShardShape"));
  spmdFullToShardShapeOp->setAttr("mhlo.sharding",
                                  builder.getAttr<StringAttr>("{manual}"));
  return spmdFullToShardShapeOp.getOperation();
}

Operation* insertShardingEpilog(ImplicitLocOpBuilder& builder, Value operand,
                                StringAttr sharding, Type resultType) {
  CustomCallOp spmdShardToFullShape =
      builder.create<CustomCallOp>(TypeRange(resultType), ValueRange(operand));
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

template <>
FailureOr<AllToAllOp> createReplacementCollectiveOperation<AllToAllOp>(
    Type resultType, Value operand, AllToAllOp originalOp,
    const SuperSubDeviceIdMap& superSubDeviceMap) {
  ImplicitLocOpBuilder builder(originalOp->getLoc(), originalOp.getOperation());
  if (operand.getType()
              .cast<ShapedType>()
              .getShape()[originalOp.getSplitDimension()] !=
          originalOp.getOperand()
              .getType()
              .getShape()[originalOp.getSplitDimension()] ||
      resultType.cast<ShapedType>()
              .getShape()[originalOp.getSplitDimension()] !=
          originalOp.getResult()
              .getType()
              .getShape()[originalOp.getSplitDimension()]) {
    emitError(originalOp.getLoc())
        << "Sharding along split_dimension is not supported.";
    return failure();
  }
  if (operand.getType()
              .cast<ShapedType>()
              .getShape()[originalOp.getConcatDimension()] !=
          originalOp.getOperand()
              .getType()
              .getShape()[originalOp.getConcatDimension()] ||
      resultType.cast<ShapedType>()
              .getShape()[originalOp.getConcatDimension()] !=
          originalOp.getResult()
              .getType()
              .getShape()[originalOp.getConcatDimension()]) {
    emitError(originalOp.getLoc())
        << "Sharding along concat_dimension is not supported.";
    return failure();
  }
  AllToAllOp newOp = builder.create<AllToAllOp>(
      resultType, operand, originalOp.getSplitDimension(),
      originalOp.getConcatDimension(), originalOp.getSplitCount(),
      completeSuperReplicaGroups(originalOp.getReplicaGroups(),
                                 superSubDeviceMap),
      originalOp.getChannelHandleAttr());
  newOp->setAttr("use_global_device_ids",
                 UnitAttr::get(originalOp->getContext()));
  newOp->setAttr("device_domain",
                 StringAttr::get(originalOp->getContext(), "complete"));
  return newOp;
}

template <>
FailureOr<ReduceScatterOp>
createReplacementCollectiveOperation<ReduceScatterOp>(
    Type resultType, Value operand, ReduceScatterOp originalOp,
    const SuperSubDeviceIdMap& superSubDeviceMap) {
  if (operand.getType()
              .cast<ShapedType>()
              .getShape()[originalOp.getScatterDimension()] !=
          originalOp.getOperand()
              .getType()
              .getShape()[originalOp.getScatterDimension()] ||
      resultType.cast<ShapedType>()
              .getShape()[originalOp.getScatterDimension()] !=
          originalOp.getResult()
              .getType()
              .getShape()[originalOp.getScatterDimension()]) {
    emitError(originalOp.getLoc())
        << "Sharding along scatter_dimension is not supported.";
    return failure();
  }
  ImplicitLocOpBuilder builder(originalOp->getLoc(), originalOp.getOperation());
  ReduceScatterOp newOp = builder.create<ReduceScatterOp>(
      resultType, operand, originalOp.getScatterDimension(),
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
  newOp.value()->setAttr("mhlo.sharding",
                         StringAttr::get(op->getContext(), "{manual}"));

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
               CollectiveOpPatternRewriter<AllReduceOp>,
               CollectiveOpPatternRewriter<AllToAllOp>,
               CollectiveOpPatternRewriter<ReduceScatterOp>>(
      patterns.getContext());
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
    if (failed(handleModuleIoShardingAttributes(getOperation()))) {
      return signalPassFailure();
    }
    if (failed(handleNumPartitionsAndReplicasAttributes(
            getOperation(), getCollectiveOptions().superSubDeviceMap))) {
      return signalPassFailure();
    }

    FailureOr<func::FuncOp> mainFunc = getMainFunc(getOperation());
    if (failed(mainFunc)) {
      emitError(getOperation().getLoc())
          << "Module operation does not have \"main\" function.";
      return signalPassFailure();
    }
    RewritePatternSet patterns(&getContext());
    populateCollectivesSpmdSubPartitionerRewritePatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(mainFunc->getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  std::shared_ptr<void> xlaCcLibHandle;
};

}  // namespace

}  // namespace stablehlo
}  // namespace mlir
