#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>

#include "CollectivesPassesCli.h"
#include "Utils.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
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

#define GEN_PASS_DEF_COMPLETECOLLECTIVESSPMDSUBPARTITION
#include "stablehlo/transforms/collectives/Passes.h.inc"

namespace {

LogicalResult handleNumPartitionsAndReplicasAttributes(ModuleOp op) {
  Builder builder(op->getContext());

  DictionaryAttr frontendAttributes =
      op->getAttrOfType<DictionaryAttr>("mhlo.frontend_attributes");
  if (!frontendAttributes) {
    frontendAttributes = DictionaryAttr::get(op->getContext());
  }

  Attribute superPartitionNumPartitionsAttr =
      frontendAttributes.get("super_partition_num_partitions");
  if (!superPartitionNumPartitionsAttr) {
    MLIR_EMIT_ERROR(op.getLoc()) << "Attribute super_partition_num_partitions "
                                    "not found in mhlo.frontend_attributes.";
    return failure();
  }
  int32_t superPartitionNumPartitions;
  FAILURE_OR_ASSIGN_OR_RETURN(
      superPartitionNumPartitions,
      toInteger<int32_t>(
          superPartitionNumPartitionsAttr.cast<StringAttr>().strref(),
          op->getLoc()));

  Attribute superPartitionNumReplicasAttr =
      frontendAttributes.get("super_partition_num_replicas");
  if (!superPartitionNumReplicasAttr) {
    MLIR_EMIT_ERROR(op.getLoc()) << "Attribute super_partition_num_replicas "
                                    "not found in mhlo.frontend_attributes.";
    return failure();
  }
  int32_t superPartitionNumReplicas;
  FAILURE_OR_ASSIGN_OR_RETURN(
      superPartitionNumReplicas,
      toInteger<int32_t>(
          superPartitionNumReplicasAttr.cast<StringAttr>().strref(),
          op->getLoc()));

  IntegerAttr subPartitionNumPartitionsAttr =
      op->getAttrOfType<IntegerAttr>("mhlo.num_partitions");
  if (!subPartitionNumPartitionsAttr) {
    MLIR_EMIT_ERROR(op->getLoc())
        << "Required integer attribute mhlo.num_partitions not found.";
  }
  int32_t subPartitionNumPartitions =
      subPartitionNumPartitionsAttr.getValue().getSExtValue();

  IntegerAttr subPartitionNumReplicasAttr =
      op->getAttrOfType<IntegerAttr>("mhlo.num_replicas");
  if (!subPartitionNumReplicasAttr) {
    MLIR_EMIT_ERROR(op->getLoc())
        << "Required integer attribute mhlo.num_replicas not found.";
  }
  int32_t subPartitionNumReplicas =
      subPartitionNumReplicasAttr.getValue().getSExtValue();

  int32_t completePartitionNumPartitions =
      superPartitionNumPartitions * subPartitionNumPartitions;
  op->setAttr("mhlo.num_partitions",
              builder.getI32IntegerAttr(completePartitionNumPartitions));

  int32_t completePartitionNumReplicas =
      superPartitionNumReplicas * subPartitionNumReplicas;
  op->setAttr("mhlo.num_replicas",
              builder.getI32IntegerAttr(completePartitionNumReplicas));

  frontendAttributes = removeAttributes(
      frontendAttributes,
      {"super_partition_num_partitions", "super_partition_num_replicas"});
  if (frontendAttributes.empty()) {
    op->removeAttr("mhlo.frontend_attributes");
  } else {
    op->setAttr("mhlo.frontend_attributes", frontendAttributes);
  }

  return success();
}

template <typename ShapeIt>
auto shapeElementsCount(ShapeIt begin, ShapeIt end) {
  return std::reduce(begin, end, 1, std::multiplies<>{});
}

template <typename SuperShardingDevicesIt, typename SuperShardingShapeIt,
          typename SubShardingDevicesIt, typename SubShardingShapeIt,
          typename OutShardingDevicesIt, typename OutShardingShapeIt>
LogicalResult concatenateShardings(
    SuperShardingDevicesIt superShardingDevicesBegin,
    SuperShardingShapeIt superShardingShapeBegin,
    SuperShardingShapeIt superShardingShapeEnd,
    SubShardingDevicesIt subShardingDevicesBegin,
    SubShardingShapeIt subShardingShapeBegin,
    SubShardingShapeIt subShardingShapeEnd,
    const SuperSubDeviceIdMap& superSubDeviceMap, Location loc,
    OutShardingDevicesIt outShardingDevicesBegin,
    OutShardingShapeIt outShardingShapeItBegin) {
  OutShardingShapeIt outShardingShapeIt = std::copy(
      superShardingShapeBegin, superShardingShapeEnd, outShardingShapeItBegin);

  std::copy(subShardingShapeBegin, subShardingShapeEnd, outShardingShapeIt);

  auto superShardingDevicesSize =
      shapeElementsCount(superShardingShapeBegin, superShardingShapeEnd);
  auto superShardingDevicesEnd = superShardingDevicesBegin;
  std::advance(superShardingDevicesEnd, superShardingDevicesSize);

  auto subShardingDevicesSize =
      shapeElementsCount(subShardingShapeBegin, subShardingShapeEnd);
  auto subShardingDevicesEnd = subShardingDevicesBegin;
  std::advance(subShardingDevicesEnd, subShardingDevicesSize);

  auto outShardingDevicesIt = outShardingDevicesBegin;
  for (SuperShardingDevicesIt superShardingDevicesIt =
           superShardingDevicesBegin;
       superShardingDevicesIt != superShardingDevicesEnd;
       ++superShardingDevicesIt) {
    auto superSubDeviceMapIt = superSubDeviceMap.find(*superShardingDevicesIt);
    if (superSubDeviceMapIt == superSubDeviceMap.end()) {
      MLIR_EMIT_ERROR(loc) << "Super device " << *superShardingDevicesIt
                           << " not found in super-sub device map.";
      return failure();
    }
    outShardingDevicesIt = std::transform(
        subShardingDevicesBegin, subShardingDevicesEnd, outShardingDevicesIt,
        [&superSubDeviceMapIt](auto subDeviceId) {
          return superSubDeviceMapIt->second[subDeviceId];
        });
  }

  return success();
}

// Always replicate on last time dimension even if it is with size 1.
template <typename AllDevicesIt, typename OutDevicesIt, typename OutShapeIt>
LogicalResult canonicalTileAssignment(const xla::HloSharding& sharding,
                                      size_t tensorRanks,
                                      AllDevicesIt allDevicesBegin,
                                      AllDevicesIt allDevicesEnd, Location loc,
                                      OutDevicesIt outDevicesBegin,
                                      OutShapeIt outShapeBegin) {
  if (xla::api::hloShardingIsReplicated(&sharding)) {
    std::copy(allDevicesBegin, allDevicesEnd, outDevicesBegin);
    OutShapeIt outShapeIt = std::fill_n(outShapeBegin, tensorRanks, 1);
    *outShapeIt = std::distance(allDevicesBegin, allDevicesEnd);
  } else if (xla::api::HloShardingIsTiled(&sharding)) {
    const int64_t* shardingDevices;
    const int64_t* shardingShape;
    size_t shardingShapeSize;
    xla::api::hloShardingTileAssignmentDevices(
        &sharding, &shardingDevices, &shardingShape, &shardingShapeSize);
    auto shardingDevicesSize =
        shapeElementsCount(shardingShape, shardingShape + shardingShapeSize);
    std::copy(shardingDevices, shardingDevices + shardingDevicesSize,
              outDevicesBegin);
    OutShapeIt outShapeIt = std::copy(
        shardingShape, shardingShape + shardingShapeSize, outShapeBegin);
    if (!xla::api::hloShardingReplicateOnLastTileDim(&sharding)) {
      *outShapeIt = 1;
    }
  } else {
    MLIR_EMIT_ERROR(loc) << "Sharding type unsupported.";
  }
  return success();
}

template <typename TypesIt, typename OutIt>
LogicalResult getTypesRank(TypesIt typesBegin, TypesIt typesEnd, OutIt outBegin,
                           Location loc) {
  for (; typesBegin != typesEnd; ++typesBegin, ++outBegin) {
    ShapedType shapedType = typesBegin->template cast<ShapedType>();
    if (!shapedType || !shapedType.hasRank()) {
      MLIR_EMIT_ERROR(loc) << "Only shaped types expected.";
      return failure();
    }
    *outBegin = shapedType.getRank();
  }
  return success();
}

template <typename OutIt>
LogicalResult funcOpArgumentsRank(func::FuncOp op, OutIt outBegin) {
  LogicalResult res =
      getTypesRank(op.getArgumentTypes().begin(), op.getArgumentTypes().end(),
                   outBegin, op->getLoc());
  if (failed(res)) {
    MLIR_EMIT_ERROR(op->getLoc())
        << "Failed getting rank of function arguments.";
  }
  return res;
}

template <typename OutIt>
LogicalResult funcOpResultsRank(func::FuncOp op, OutIt outBegin) {
  LogicalResult res =
      getTypesRank(op.getResultTypes().begin(), op.getResultTypes().end(),
                   outBegin, op->getLoc());
  if (failed(res)) {
    MLIR_EMIT_ERROR(op->getLoc()) << "Failed getting rank of function results.";
  }
  return res;
}

template <typename OutDevicesIt>
void getSuperDevices(const SuperSubDeviceIdMap& superSubDeviceMap,
                     OutDevicesIt outDevicesBegin) {
  std::transform(superSubDeviceMap.begin(), superSubDeviceMap.end(),
                 outDevicesBegin, [](const auto& pair) { return pair.first; });
}

template <typename OutDevicesIt>
void getSubDevices(const SuperSubDeviceIdMap& superSubDeviceMap,
                   OutDevicesIt outDevicesBegin) {
  assert(!superSubDeviceMap.empty());
  for (DeviceId i = 0; i < DeviceId(superSubDeviceMap.begin()->second.size());
       ++i, ++outDevicesBegin) {
    *outDevicesBegin = i;
  }
}

template <typename AllSuperDevicesIt, typename AllSubDevicesIt,
          typename TensorRanksIt>
FailureOr<HloShardingPtr> completeSharding(
    const xla::HloSharding& superSharding,
    AllSuperDevicesIt allSuperDevicesBegin,
    AllSuperDevicesIt allSuperDevicesEnd, const xla::HloSharding& subSharding,
    AllSubDevicesIt allSubDevicesBegin, AllSubDevicesIt allSubDevicesEnd,
    Location loc, TensorRanksIt tensorRanksIt,
    const SuperSubDeviceIdMap& superSubDeviceMap) {
  bool isSuperShardingTuple = xla::api::hloShardingIsTuple(&superSharding);
  bool isSubShardingTuple = xla::api::hloShardingIsTuple(&subSharding);
  if (isSuperShardingTuple != isSubShardingTuple) {
    MLIR_EMIT_ERROR(loc)
        << "super-partition and sub-partition shardings must both be tuple or "
           "not simultaneously.";
    return failure();
  }
  if (isSuperShardingTuple) {
    // Recurse on the tuple.

    size_t superShardingElementsSize;
    xla::api::hloShardingTupleElements(&superSharding, nullptr,
                                       &superShardingElementsSize);
    llvm::SmallVector<xla::HloSharding*, 64> superShardingElements(
        superShardingElementsSize);
    xla::api::hloShardingTupleElements(
        &superSharding,
        const_cast<const xla::HloSharding**>(superShardingElements.data()),
        &superShardingElementsSize);

    size_t subShardingElementsSize;
    xla::api::hloShardingTupleElements(&subSharding, nullptr,
                                       &subShardingElementsSize);
    llvm::SmallVector<xla::HloSharding*, 64> subShardingElements(
        subShardingElementsSize);
    xla::api::hloShardingTupleElements(
        &subSharding,
        const_cast<const xla::HloSharding**>(subShardingElements.data()),
        &subShardingElementsSize);

    if (superShardingElementsSize != subShardingElementsSize) {
      MLIR_EMIT_ERROR(loc)
          << "Super-sharding and sub-sharding tuples must have the same number "
             "of elements,"
          << superShardingElementsSize << " != " << subShardingElementsSize
          << ".";
      return failure();
    }

    llvm::SmallVector<HloShardingPtr, 64> completeShardingElements;
    completeShardingElements.reserve(subShardingElementsSize);
    auto completeShardingElementsInserterIt =
        std::back_inserter(completeShardingElements);
    for (size_t i = 0; i < subShardingElementsSize;
         ++i, ++completeShardingElementsInserterIt) {
      size_t tensorRanksElement = tensorRanksIt[i];
      FailureOr<HloShardingPtr> completeShardingPtr = completeSharding(
          *superShardingElements[i], allSuperDevicesBegin, allSuperDevicesEnd,
          *subShardingElements[i], allSubDevicesBegin, allSubDevicesEnd, loc,
          &tensorRanksElement, superSubDeviceMap);
      if (failed(completeShardingPtr)) {
        return failure();
      }
      *completeShardingElementsInserterIt =
          std::move(completeShardingPtr.value());
    }
    llvm::SmallVector<const xla::HloSharding*, 64>
        completeShardingElementsRawPtr(completeShardingElements.size());
    std::transform(completeShardingElements.begin(),
                   completeShardingElements.end(),
                   completeShardingElementsRawPtr.begin(),
                   [](const HloShardingPtr& p) { return p.get(); });
    xla::HloSharding* completeSharding;
    if (xla::api::makeHloShardingTuple(completeShardingElementsRawPtr.data(),
                                       completeShardingElementsRawPtr.size(),
                                       &completeSharding) != XlaStatus::OK) {
      MLIR_EMIT_ERROR(loc) << "Failed creating tuple sharding.";
    }
    return makeHloShardingPtr(completeSharding);
  }

  // Tupleless case.

  using Devices = SmallVector<int64_t, 64>;
  using Shape = SmallVector<int64_t, 8>;
  Devices canonicalSuperShardingDevices;
  Shape canonicalSuperShardingShape;
  if (failed(canonicalTileAssignment(
          superSharding, *tensorRanksIt, allSuperDevicesBegin,
          allSuperDevicesEnd, loc,
          std::back_inserter(canonicalSuperShardingDevices),
          std::back_inserter(canonicalSuperShardingShape)))) {
    return failure();
  }

  Devices canonicalSubShardingDevices;
  Shape canonicalSubShardingShape;
  if (failed(canonicalTileAssignment(
          subSharding, *tensorRanksIt, allSubDevicesBegin, allSubDevicesEnd,
          loc, std::back_inserter(canonicalSubShardingDevices),
          std::back_inserter(canonicalSubShardingShape)))) {
    return failure();
  }

  if (canonicalSuperShardingShape.size() != canonicalSubShardingShape.size()) {
    MLIR_EMIT_ERROR(loc) << "Super and sub sharding don't have the same rank.";
    return failure();
  }

  Devices concatenatedShardingsDevices;
  Shape concatenatedShardingsShape;
  if (failed(concatenateShardings(
          canonicalSuperShardingDevices.begin(),
          canonicalSuperShardingShape.begin(),
          canonicalSuperShardingShape.end(),
          canonicalSubShardingDevices.begin(),
          canonicalSubShardingShape.begin(), canonicalSubShardingShape.end(),
          superSubDeviceMap, loc,
          std::back_inserter(concatenatedShardingsDevices),
          std::back_inserter(concatenatedShardingsShape)))) {
    return failure();
  }

  Devices completeShardingDevices(concatenatedShardingsDevices.size());
  Shape completeShardingShape(concatenatedShardingsShape.size());
  Shape permutation(completeShardingShape);
  for (size_t i = 0; i < canonicalSuperShardingShape.size(); ++i) {
    permutation[2 * i] = i;
    permutation[2 * i + 1] = i + canonicalSuperShardingShape.size();
  }
  transpose(concatenatedShardingsDevices.begin(),
            concatenatedShardingsDevices.end(),
            concatenatedShardingsShape.begin(),
            concatenatedShardingsShape.end(), permutation.begin(),
            completeShardingDevices.begin(), completeShardingShape.begin());

  completeShardingShape.resize(canonicalSuperShardingShape.size());
  for (size_t i = 0; i < completeShardingShape.size(); ++i) {
    completeShardingShape[i] =
        canonicalSuperShardingShape[i] * canonicalSubShardingShape[i];
  }

  xla::HloSharding* completeSharding;
  if (xla::api::makeTiledHloSharding(
          completeShardingDevices.data(), completeShardingShape.data(),
          completeShardingShape.size(), /*replicateOnLastTileDim=*/true,
          &completeSharding) != XlaStatus::OK) {
    MLIR_EMIT_ERROR(loc)
        << "Failed creating complete sharding from super-sub sharding pair.";
    return failure();
  }
  return makeHloShardingPtr(completeSharding);
}

template <typename AllSuperDevicesIt, typename AllSubDevicesIt,
          typename TensorRanksIt>
FailureOr<Attribute> completeSharding(
    StringAttr superShardingAttr, AllSuperDevicesIt allSuperDevicesBegin,
    AllSuperDevicesIt allSuperDevicesEnd, Attribute subShardingAttr,
    AllSubDevicesIt allSubDevicesBegin, AllSubDevicesIt allSubDevicesEnd,
    Location loc, TensorRanksIt tensorRanksIt,
    const SuperSubDeviceIdMap& superSubDeviceMap) {
  FailureOr<HloShardingPtr> superShardingPtr =
      makeHloSharding(superShardingAttr, loc);
  if (failed(superShardingPtr)) {
    return failure();
  }

  HloShardingPtr subShardingPtr = makeHloShardingPtr(nullptr);
  if (subShardingAttr.isa<StringAttr>()) {
    StringAttr subShardingStrAttr = subShardingAttr.cast<StringAttr>();
    FAILURE_OR_ASSIGN_OR_RETURN(subShardingPtr,
                                makeHloSharding(subShardingStrAttr, loc));
  } else if (subShardingAttr.isa<ArrayAttr>()) {
    FAILURE_OR_ASSIGN_OR_RETURN(
        subShardingPtr,
        makeHloShardingTuple(subShardingAttr.cast<ArrayAttr>(), loc));
  } else {
    MLIR_EMIT_ERROR(loc) << "Sub-sharding attribute has unexpected type.";
  }

  FailureOr<HloShardingPtr> completeShardingPtr = completeSharding(
      *superShardingPtr.value().get(), allSuperDevicesBegin, allSuperDevicesEnd,
      *subShardingPtr.get(), allSubDevicesBegin, allSubDevicesEnd, loc,
      tensorRanksIt, superSubDeviceMap);
  if (failed(completeShardingPtr)) {
    return failure();
  }

  return makeHloShardingAttr(*completeShardingPtr.value().get(), loc,
                             bool(subShardingAttr.isa<ArrayAttr>()));
}

LogicalResult completeSharding(ModuleOp op,
                               const SuperSubDeviceIdMap& superSubDeviceMap) {
  ArrayAttr subPartitionArgumentsShardingAttr =
      op->getAttrOfType<ArrayAttr>("mhlo.spmd_parameters_shardings");
  if (!subPartitionArgumentsShardingAttr) {
    MLIR_EMIT_ERROR(op->getLoc())
        << "mhlo.spmd_parameters_shardings is a required array attribute.";
    return failure();
  }

  StringAttr subPartitionResultsShardingAttr =
      op->getAttrOfType<StringAttr>("mhlo.spmd_output_sharding");
  if (!subPartitionResultsShardingAttr) {
    MLIR_EMIT_ERROR(op->getLoc())
        << "mhlo.spmd_output_sharding is a required string attribute.";
    return failure();
  }

  DictionaryAttr frontendAttributes =
      op->getAttrOfType<DictionaryAttr>("mhlo.frontend_attributes");
  if (!frontendAttributes) {
    MLIR_EMIT_ERROR(op->getLoc())
        << "mhlo.frontend_attributes is a required string attribute.";
    return failure();
  }

  StringAttr superPartitionArgumentsShardingAttr =
      frontendAttributes.getAs<StringAttr>(
          "super_partition_spmd_parameters_sharding");
  if (!superPartitionArgumentsShardingAttr) {
    MLIR_EMIT_ERROR(op->getLoc())
        << "string attribute super_partition_spmd_parameters_sharding "
           "not found in mhlo.frontend_attributes.";
    return failure();
  }

  StringAttr superPartitionResultsShardingAttr =
      frontendAttributes.getAs<StringAttr>(
          "super_partition_spmd_output_sharding");
  if (!superPartitionResultsShardingAttr) {
    MLIR_EMIT_ERROR(op->getLoc())
        << "string attribute super_partition_spmd_output_sharding not "
           "found in mhlo.frontend_attributes.";
    return failure();
  }

  FailureOr<func::FuncOp> mainFunc = getMainFunc(op);
  if (failed(mainFunc)) {
    MLIR_EMIT_ERROR(op->getLoc()) << "Function \"main\" not found.";
    return failure();
  }

  SmallVector<int64_t, 64> superDevices;
  getSuperDevices(superSubDeviceMap, std::back_inserter(superDevices));
  std::sort(superDevices.begin(), superDevices.end());
  SmallVector<int64_t, 64> subDevices;
  getSubDevices(superSubDeviceMap, std::back_inserter(subDevices));

  SmallVector<int64_t, 64> argumentsRank;
  if (failed(funcOpArgumentsRank(mainFunc.value(),
                                 std::back_inserter(argumentsRank)))) {
    return failure();
  }
  FailureOr<Attribute> argumentsCompleteShardingAttr = completeSharding(
      superPartitionArgumentsShardingAttr, superDevices.begin(),
      superDevices.end(), subPartitionArgumentsShardingAttr, subDevices.begin(),
      subDevices.end(), op->getLoc(), argumentsRank.begin(), superSubDeviceMap);
  if (failed(argumentsCompleteShardingAttr)) {
    MLIR_EMIT_ERROR(op->getLoc())
        << "Failed completing sub-super arguments sharding.";
    return failure();
  }
  op->setAttr("mhlo.spmd_parameters_shardings",
              argumentsCompleteShardingAttr.value());

  SmallVector<int64_t, 64> resultsRank;
  if (failed(funcOpResultsRank(mainFunc.value(),
                               std::back_inserter(resultsRank)))) {
    return failure();
  }
  FailureOr<Attribute> resultsCompleteShardingAttr = completeSharding(
      superPartitionResultsShardingAttr, superDevices.begin(),
      superDevices.end(), subPartitionResultsShardingAttr, subDevices.begin(),
      subDevices.end(), op->getLoc(), argumentsRank.begin(), superSubDeviceMap);
  if (failed(resultsCompleteShardingAttr)) {
    MLIR_EMIT_ERROR(op->getLoc())
        << "Failed completing super-sub results sharding.";
    return failure();
  }
  op->setAttr("mhlo.spmd_output_sharding", resultsCompleteShardingAttr.value());

  DictionaryAttr newFrontendAttributes = removeAttributes(
      frontendAttributes, {"super_partition_spmd_parameters_sharding",
                           "super_partition_spmd_output_sharding"});
  if (newFrontendAttributes.empty()) {
    op->removeAttr("mhlo.frontend_attributes");
  } else {
    op->setAttr("mhlo.frontend_attributes", newFrontendAttributes);
  }

  return success();
}

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
    MLIR_EMIT_ERROR(moduleOp->getLoc())
        << "mhlo.num_partitions is a required attribute.";
    return failure();
  }
  auto numPartitions = numPartitionsAttr.getInt();
  IntegerAttr numReplicasAttr =
      moduleOp->getAttrOfType<IntegerAttr>("mhlo.num_replicas");
  if (!numReplicasAttr) {
    MLIR_EMIT_ERROR(moduleOp->getLoc())
        << "mhlo.num_replicas is a required attribute.";
    return failure();
  }
  auto numReplicas = numReplicasAttr.getInt();

  if (failed(getReplicaGroupsAsGlobalDeviceIds(
          op, numPartitions, numReplicas, std::back_inserter(deviceGroups),
          std::back_inserter(deviceGroupsShape)))) {
    return failure();
  }

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
    MLIR_EMIT_ERROR(op.getLoc()) << "Sub-partition completion for "
                                    "collective_permute is not implemented.";
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

  LogicalResult initialize(MLIRContext* context) override {
    xlaCcLibHandle = xla::api::loadLibrary(xlaCcLibPath().c_str());
    if (!xlaCcLibHandle) {
      return LogicalResult::failure();
    }
    return LogicalResult::success();
  }

  void runOnOperation() override {
    if (failed(completeSharding(getOperation(),
                                getCollectiveOptions().superSubDeviceMap))) {
      return signalPassFailure();
    }

    FailureOr<func::FuncOp> mainFunc = getMainFunc(getOperation());
    assert(!failed(mainFunc));
    RewritePatternSet patterns(&getContext());
    populateSpmdSubPartitionCompletionRewritePatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(mainFunc.value(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }

    if (failed(handleNumPartitionsAndReplicasAttributes(getOperation()))) {
      return signalPassFailure();
    }
  }

  std::shared_ptr<void> xlaCcLibHandle;
};

}  // namespace

}  // namespace stablehlo
}  // namespace mlir
