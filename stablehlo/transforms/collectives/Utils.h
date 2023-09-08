#ifndef STABLEHLO_TRANSFORMS_COLLECTIVES_UTILS_H
#define STABLEHLO_TRANSFORMS_COLLECTIVES_UTILS_H

#include <algorithm>
#include <charconv>
#include <initializer_list>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/xla_cc.h"
#include "xla/xla_cc_loader.h"

#define MLIR_EMIT_ERROR(loc) \
  mlir::emitError(loc) << __FILE__ << ":" << __LINE__ << ": "

#define FAILURE_OR_ASSIGN_OR_RETURN(name, expr)               \
  auto FAILURE_OR_ASSIGN_OR_RETURN_FailureOr_##name = expr;   \
  if (failed(FAILURE_OR_ASSIGN_OR_RETURN_FailureOr_##name)) { \
    return failure();                                         \
  }                                                           \
  name = std::move(FAILURE_OR_ASSIGN_OR_RETURN_FailureOr_##name.value())

namespace mlir {
namespace stablehlo {

using HloShardingPtr =
    std::unique_ptr<xla::HloSharding, xla::api::DestroyHloSharding>;
using XlaCharBuffer = std::unique_ptr<char[], xla::api::DestroyCharBuffer>;

template <typename Int>
inline FailureOr<Int> toInteger(StringRef s, Location loc) {
  Int res;
  std::from_chars_result ret =
      std::from_chars(s.data(), s.data() + s.size(), res);
  if (ret.ec != std::errc()) {
    MLIR_EMIT_ERROR(loc) << "Failed converting \"" << s << "\" to integer.";
  }
  return res;
}

inline HloShardingPtr makeHloShardingPtr(xla::HloSharding* sharding) {
  return HloShardingPtr(sharding, xla::api::destroyHloSharding);
}

inline FailureOr<std::string> hloShardingToStr(const xla::HloSharding& sharding,
                                               Location loc) {
  char* shardingStr;
  size_t shardingStrSize;
  if (xla::api::hloShardingToString(&sharding, &shardingStr,
                                    &shardingStrSize) != XlaStatus::OK) {
    MLIR_EMIT_ERROR(loc) << "Failed converting HLO sharding to string.";
    return failure();
  }
  auto charBuff = XlaCharBuffer(shardingStr, xla::api::destroyCharBuffer);
  return std::string(shardingStr, shardingStrSize);
}

inline FailureOr<StringAttr> makeHloShardingStrAttr(
    const xla::HloSharding& sharding, Location loc) {
  char* shardingStr;
  size_t shardingStrSize;
  if (xla::api::hloShardingToString(&sharding, &shardingStr,
                                    &shardingStrSize) != XlaStatus::OK) {
    MLIR_EMIT_ERROR(loc) << "Failed converting HLO sharding to string.";
    return failure();
  }
  auto shardingStrPtr = XlaCharBuffer(shardingStr, xla::api::destroyCharBuffer);
  return StringAttr::get(loc->getContext(),
                         StringRef(shardingStr, shardingStrSize));
}

inline FailureOr<ArrayAttr> makeHloShardingArrayAttr(
    const xla::HloSharding& sharding, Location loc) {
  if (!xla::api::hloShardingIsTuple(&sharding)) {
    MLIR_EMIT_ERROR(loc) << "Expected HLO sharding tuple.";
    return failure();
  }

  size_t shardingElementsSize;
  xla::api::hloShardingTupleElements(&sharding, nullptr, &shardingElementsSize);
  llvm::SmallVector<xla::HloSharding*, 64> shardingElements(
      shardingElementsSize);
  xla::api::hloShardingTupleElements(
      &sharding, const_cast<const xla::HloSharding**>(shardingElements.data()),
      &shardingElementsSize);
  llvm::SmallVector<Attribute, 64> shardingAttrs;
  shardingAttrs.reserve(shardingElementsSize);
  for (xla::HloSharding* s : shardingElements) {
    FailureOr<StringAttr> strAttr = makeHloShardingStrAttr(*s, loc);
    if (failed(strAttr)) {
      return failure();
    }
    shardingAttrs.push_back(strAttr.value());
  }
  return ArrayAttr::get(loc->getContext(), shardingAttrs);
}

inline FailureOr<Attribute> makeHloShardingAttr(
    const xla::HloSharding& sharding, Location loc, bool expandTuple = false) {
  if (expandTuple && xla::api::hloShardingIsTuple(&sharding)) {
    ArrayAttr arrayAttr;
    FAILURE_OR_ASSIGN_OR_RETURN(arrayAttr,
                                makeHloShardingArrayAttr(sharding, loc));
    return arrayAttr;
  }

  StringAttr strAttr;
  FAILURE_OR_ASSIGN_OR_RETURN(strAttr, makeHloShardingStrAttr(sharding, loc));
  return strAttr;
}

inline FailureOr<HloShardingPtr> makeHloSharding(StringAttr attr,
                                                 Location loc) {
  xla::HloSharding* subSharding;
  if (xla::api::parseHloSharding(attr.data(), attr.size(), &subSharding) !=
      XlaStatus::OK) {
    MLIR_EMIT_ERROR(loc) << "Failed parsing sharding attribute.";
    return failure();
  }
  return makeHloShardingPtr(subSharding);
}

inline FailureOr<HloShardingPtr> makeHloShardingTuple(ArrayAttr attr,
                                                      Location loc) {
  SmallVector<HloShardingPtr, 64> hloShardings;
  auto attrItRange = attr.getAsRange<StringAttr>();
  hloShardings.reserve(std::distance(attrItRange.begin(), attrItRange.end()));
  for (StringAttr strAttr : attrItRange) {
    FailureOr<HloShardingPtr> hloSharding = makeHloSharding(strAttr, loc);
    if (failed(hloSharding)) {
      return failure();
    }
    hloShardings.emplace_back(std::move(hloSharding.value()));
  }

  SmallVector<xla::HloSharding*, 64> hloShardingsRaw(hloShardings.size());
  std::transform(hloShardings.begin(), hloShardings.end(),
                 hloShardingsRaw.begin(),
                 [](const HloShardingPtr& p) { return p.get(); });
  xla::HloSharding* resShardingRaw;
  if (xla::api::makeHloShardingTuple(
          const_cast<const xla::HloSharding**>(hloShardingsRaw.data()),
          hloShardingsRaw.size(), &resShardingRaw) != XlaStatus::OK) {
    MLIR_EMIT_ERROR(loc) << "Failed creating HLO sharding tuple.";
  }
  return makeHloShardingPtr(resShardingRaw);
}

inline int64_t getChannelId(const std::optional<ChannelHandleAttr>& attr) {
  if (!attr) {
    return 0;
  }
  return attr->getHandle();
}

inline int64_t processId(int64_t replicaId, int64_t partitionId,
                         uint64_t numReplicasPerPartition) {
  return partitionId * numReplicasPerPartition + replicaId;
}

template <typename Op, typename OutputDeviceIdsIt,
          typename OutputDeviceIdsShapeIt>
struct GetReplicaGroupsAsGlobalDeviceIds {
  LogicalResult operator()(Op op, uint64_t numPartitions,
                           uint64_t numReplicasPerPartition,
                           OutputDeviceIdsIt outputGlobalDeviceIdsIt,
                           OutputDeviceIdsShapeIt outputGlobalDeviceIdsShapeIt);
};

template <typename Op, typename OutputDeviceIdsIt,
          typename OutputDeviceIdsShapeIt>
LogicalResult getReplicaGroupsAsGlobalDeviceIds(
    Op op, uint64_t numPartitions, uint64_t numReplicasPerPartition,
    OutputDeviceIdsIt outputGlobalDeviceIdsIt,
    OutputDeviceIdsShapeIt outputGlobalDeviceIdsShapeIt) {
  return GetReplicaGroupsAsGlobalDeviceIds<Op, OutputDeviceIdsIt,
                                           OutputDeviceIdsShapeIt>()(
      op, numPartitions, numReplicasPerPartition, outputGlobalDeviceIdsIt,
      outputGlobalDeviceIdsShapeIt);
}

template <typename InIdsIt, typename InIdsShapeIt, typename OutIdsIt,
          typename OutIdsShapeIt>
void crossReplicaGroupsToGlobalDeviceIds(uint64_t numPartitions,
                                         uint64_t numReplicasPerPartition,
                                         InIdsIt inIdsBegin,
                                         InIdsShapeIt inIdsShapeBegin,
                                         InIdsShapeIt inIdsShapeEnd,
                                         OutIdsIt outIdsBegin,
                                         OutIdsShapeIt outIdsShapeBegin) {
  assert(std::distance(inIdsShapeBegin, inIdsShapeEnd) == 2);
  auto inIdsSize = inIdsShapeBegin[0] * inIdsShapeBegin[1];
  *outIdsShapeBegin = *inIdsShapeBegin * numPartitions;
  std::copy(++inIdsShapeBegin, inIdsShapeEnd, ++outIdsShapeBegin);
  for (uint64_t partitionId = 0; partitionId < numPartitions; ++partitionId) {
    outIdsBegin = std::transform(
        inIdsBegin, inIdsBegin + inIdsSize, outIdsBegin,
        [numReplicasPerPartition, partitionId](int64_t replicaId) {
          return processId(replicaId, partitionId, numReplicasPerPartition);
        });
  }
}

template <typename InIdsIt, typename InIdsShapeIt, typename OutIdsIt,
          typename OutIdsShapeIt>
void crossPartitionGroupsToGlobalDeviceIds(uint64_t numPartitions,
                                           uint64_t numReplicasPerPartition,
                                           InIdsIt inIdsBegin,
                                           InIdsShapeIt inIdsShapeBegin,
                                           InIdsShapeIt inIdsShapeEnd,
                                           OutIdsIt outIdsBegin,
                                           OutIdsShapeIt outIdsShapeBegin) {
  assert(std::distance(inIdsShapeBegin, inIdsShapeEnd) == 2);
  auto inIdsSize = inIdsShapeBegin[0] * inIdsShapeBegin[1];
  *outIdsShapeBegin = *inIdsShapeBegin * numReplicasPerPartition;
  std::copy(++inIdsShapeBegin, inIdsShapeEnd, ++outIdsShapeBegin);
  for (uint64_t replicaId = 0; replicaId < numReplicasPerPartition;
       ++replicaId) {
    outIdsBegin = std::transform(
        inIdsBegin, inIdsBegin + inIdsSize, outIdsBegin,
        [numReplicasPerPartition, replicaId](int64_t partitionId) {
          return processId(replicaId, partitionId, numReplicasPerPartition);
        });
  }
}

template <typename InIdsIt, typename InIdsShapeIt, typename OutIdsIt,
          typename OutIdsShapeIt>
void crossReplicaAndPartitionGroupsToGlobalDeviceIds(
    uint64_t numPartitions, uint64_t numReplicasPerPartition,
    InIdsIt inIdsBegin, InIdsShapeIt inIdsShapeBegin,
    InIdsShapeIt inIdsShapeEnd, OutIdsIt outIdsBegin,
    OutIdsShapeIt outIdsShapeBegin) {
  assert(std::distance(inIdsShapeBegin, inIdsShapeEnd) == 2);
  auto inIdsSize = inIdsShapeBegin[0] * inIdsShapeBegin[1];
  *outIdsShapeBegin = *inIdsShapeBegin;
  ++outIdsShapeBegin;
  ++inIdsShapeBegin;
  *outIdsShapeBegin = *inIdsShapeBegin * numPartitions;
  std::copy(++inIdsShapeBegin, inIdsShapeEnd, ++outIdsShapeBegin);
  for (uint64_t partitionId = 0; partitionId < numPartitions; ++partitionId) {
    outIdsBegin = std::transform(
        inIdsBegin, inIdsBegin + inIdsSize, outIdsBegin,
        [numReplicasPerPartition, partitionId](int64_t replicaId) {
          return processId(replicaId, partitionId, numReplicasPerPartition);
        });
  }
}

template <typename Op, typename OutIdsIt, typename OutIdsShapeIt>
LogicalResult getReplicaGroupsAsGlobalDeviceIds1(
    Op op, uint64_t numPartitions, uint64_t numReplicasPerPartition,
    OutIdsIt outIdsBegin, OutIdsShapeIt outIdsShapeBegin) {
  int64_t channelId = getChannelId(op.getChannelHandle());
  bool useGlobalDeviceIds = op.getUseGlobalDeviceIds();
  ArrayRef<int64_t> replicaGroupsShape =
      op.getReplicaGroups().getShapedType().getShape();
  auto replicaGroups = op.getReplicaGroups().template getValues<int64_t>();
  assert(std::distance(replicaGroupsShape.begin(), replicaGroupsShape.end()) ==
         2);
  if (channelId <= 0 && !useGlobalDeviceIds) {
    if (replicaGroups.size() != numReplicasPerPartition) {
      MLIR_EMIT_ERROR(op->getLoc())
          << "Mismatch between replica_groups size and "
             "number of replicas per partition.";
      return failure();
    }
    crossReplicaGroupsToGlobalDeviceIds(
        numPartitions, numReplicasPerPartition, replicaGroups.begin(),
        replicaGroupsShape.begin(), replicaGroupsShape.end(), outIdsBegin,
        outIdsShapeBegin);
  } else if (channelId > 0 && !useGlobalDeviceIds) {
    if (replicaGroups.size() != numReplicasPerPartition) {
      MLIR_EMIT_ERROR(op->getLoc())
          << "Mismatch between replica_groups size and "
             "number of replicas per partition.";
      return failure();
    }
    crossReplicaAndPartitionGroupsToGlobalDeviceIds(
        numPartitions, numReplicasPerPartition, replicaGroups.begin(),
        replicaGroupsShape.begin(), replicaGroupsShape.end(), outIdsBegin,
        outIdsShapeBegin);
  } else if (channelId > 0 && useGlobalDeviceIds) {
    if (replicaGroups.size() != numPartitions * numReplicasPerPartition) {
      MLIR_EMIT_ERROR(op->getLoc())
          << "Mismatch between replica_groups size and "
             "number of partitions and replicas.";
      return failure();
    }
    std::copy(replicaGroupsShape.begin(), replicaGroupsShape.end(),
              outIdsShapeBegin);
    std::copy(replicaGroups.begin(),
              replicaGroups.begin() +
                  replicaGroupsShape.begin()[0] * replicaGroupsShape.begin()[1],
              outIdsBegin);
  } else {
    MLIR_EMIT_ERROR(op->getLoc())
        << "Unsupported mode: channel_id=" << channelId
        << " and use_global_device_ids=" << useGlobalDeviceIds << ".";
    return failure();
  }

  return success();
}

template <typename Op, typename OutIdsIt, typename OutIdsShapeIt>
LogicalResult getReplicaGroupsAsGlobalDeviceIds2(
    Op op, uint64_t numPartitions, uint64_t numReplicasPerPartition,
    OutIdsIt outIdsBegin, OutIdsShapeIt outIdsShapeBegin) {
  int64_t channelId = getChannelId(op.getChannelHandle());
  ArrayRef<int64_t> replicaGroupsShape =
      op.getReplicaGroups().getShapedType().getShape();
  auto replicaGroups = op.getReplicaGroups().template getValues<int64_t>();
  assert(std::distance(replicaGroupsShape.begin(), replicaGroupsShape.end()) ==
         2);
  if (channelId <= 0) {
    if (replicaGroups.size() != numReplicasPerPartition) {
      MLIR_EMIT_ERROR(op->getLoc())
          << "Mismatch between replica_groups size and "
             "number of replicas per partition.";
      return failure();
    }
    crossReplicaGroupsToGlobalDeviceIds(
        numPartitions, numReplicasPerPartition, replicaGroups.begin(),
        replicaGroupsShape.begin(), replicaGroupsShape.end(), outIdsBegin,
        outIdsShapeBegin);
  } else if (channelId > 0) {
    if (replicaGroups.size() != numPartitions) {
      MLIR_EMIT_ERROR(op->getLoc())
          << "Mismatch between replica_groups size and number of partitions.";
      return failure();
    }
    crossPartitionGroupsToGlobalDeviceIds(
        numPartitions, numReplicasPerPartition, replicaGroups.begin(),
        replicaGroupsShape.begin(), replicaGroupsShape.end(), outIdsBegin,
        outIdsShapeBegin);
  }

  return success();
}

template <typename OutputDeviceIdsIt, typename OutputDeviceIdsShapeIt>
struct GetReplicaGroupsAsGlobalDeviceIds<AllGatherOp, OutputDeviceIdsIt,
                                         OutputDeviceIdsShapeIt> {
  LogicalResult operator()(
      AllGatherOp op, uint64_t numPartitions, uint64_t numReplicasPerPartition,
      OutputDeviceIdsIt outputGlobalDeviceIdsIt,
      OutputDeviceIdsShapeIt outputGlobalDeviceIdsShapeIt) {
    return getReplicaGroupsAsGlobalDeviceIds1(
        op, numPartitions, numReplicasPerPartition, outputGlobalDeviceIdsIt,
        outputGlobalDeviceIdsShapeIt);
  }
};

template <typename OutputDeviceIdsIt, typename OutputDeviceIdsShapeIt>
struct GetReplicaGroupsAsGlobalDeviceIds<AllReduceOp, OutputDeviceIdsIt,
                                         OutputDeviceIdsShapeIt> {
  LogicalResult operator()(
      AllReduceOp op, uint64_t numPartitions, uint64_t numReplicasPerPartition,
      OutputDeviceIdsIt outputGlobalDeviceIdsIt,
      OutputDeviceIdsShapeIt outputGlobalDeviceIdsShapeIt) {
    return getReplicaGroupsAsGlobalDeviceIds1(
        op, numPartitions, numReplicasPerPartition, outputGlobalDeviceIdsIt,
        outputGlobalDeviceIdsShapeIt);
  }
};

template <typename OutputDeviceIdsIt, typename OutputDeviceIdsShapeIt>
struct GetReplicaGroupsAsGlobalDeviceIds<ReduceScatterOp, OutputDeviceIdsIt,
                                         OutputDeviceIdsShapeIt> {
  LogicalResult operator()(
      ReduceScatterOp op, uint64_t numPartitions,
      uint64_t numReplicasPerPartition,
      OutputDeviceIdsIt outputGlobalDeviceIdsIt,
      OutputDeviceIdsShapeIt outputGlobalDeviceIdsShapeIt) {
    return getReplicaGroupsAsGlobalDeviceIds1(
        op, numPartitions, numReplicasPerPartition, outputGlobalDeviceIdsIt,
        outputGlobalDeviceIdsShapeIt);
  }
};

template <typename OutputDeviceIdsIt, typename OutputDeviceIdsShapeIt>
struct GetReplicaGroupsAsGlobalDeviceIds<AllToAllOp, OutputDeviceIdsIt,
                                         OutputDeviceIdsShapeIt> {
  LogicalResult operator()(
      AllToAllOp op, uint64_t numPartitions, uint64_t numReplicasPerPartition,
      OutputDeviceIdsIt outputGlobalDeviceIdsIt,
      OutputDeviceIdsShapeIt outputGlobalDeviceIdsShapeIt) {
    return getReplicaGroupsAsGlobalDeviceIds2(
        op, numPartitions, numReplicasPerPartition, outputGlobalDeviceIdsIt,
        outputGlobalDeviceIdsShapeIt);
  }
};

template <typename AttrNamesIt>
DictionaryAttr removeAttributes(DictionaryAttr dict, AttrNamesIt attrNamesBegin,
                                AttrNamesIt attrNamesEnd) {
  // TODO: make this work with llvm::SmallDenseMap<StringRef, Attribute> map;
  std::map<StringRef, Attribute> map;
  auto mapInserterIt = std::inserter(map, map.end());
  std::transform(dict.begin(), dict.end(), mapInserterIt,
                 [](const NamedAttribute& attr) {
                   return std::make_pair(attr.getName(), attr.getValue());
                 });
  std::for_each(attrNamesBegin, attrNamesEnd,
                [&map](const auto& attrName) { map.erase(attrName); });
  SmallVector<NamedAttribute, 64> namedAttrs;
  std::transform(map.begin(), map.end(), std::back_inserter(namedAttrs),
                 [&dict](const auto& pair) {
                   return NamedAttribute(
                       StringAttr::get(dict.getContext(), pair.first),
                       pair.second);
                 });
  return DictionaryAttr::get(dict.getContext(), namedAttrs);
}

template <typename AttrName>
DictionaryAttr removeAttributes(DictionaryAttr dict,
                                std::initializer_list<AttrName> list) {
  return removeAttributes(dict, list.begin(), list.end());
}

template <typename NamedAttrsIt>
DictionaryAttr setAttributes(DictionaryAttr dict, NamedAttrsIt attrsBegin,
                             NamedAttrsIt attrsEnd) {
  // TODO: make this work with llvm::SmallDenseMap<StringRef, Attribute> map;
  std::map<StringRef, Attribute> map;
  auto mapInserterIt = std::inserter(map, map.end());
  std::transform(dict.begin(), dict.end(), mapInserterIt,
                 [](const NamedAttribute& attr) {
                   return std::make_pair(attr.getName(), attr.getValue());
                 });
  std::for_each(attrsBegin, attrsEnd, [&map](const auto& namedAttr) {
    map.insert({namedAttr.getName(), namedAttr.getValue()});
  });
  SmallVector<NamedAttribute, 64> namedAttrs;
  std::transform(map.begin(), map.end(), std::back_inserter(namedAttrs),
                 [&dict](const auto& pair) {
                   return NamedAttribute(
                       StringAttr::get(dict.getContext(), pair.first),
                       pair.second);
                 });
  return DictionaryAttr::get(dict.getContext(), namedAttrs);
}

template <typename NamedAttr>
DictionaryAttr setAttributes(DictionaryAttr dict,
                             std::initializer_list<NamedAttr> list) {
  return setAttributes(dict, list.begin(), list.end());
}

template <typename ShapeIt, typename IndexIt>
auto flattenIndex(ShapeIt shapeBegin, ShapeIt shapeEnd, IndexIt indexBegin) {
  assert(shapeBegin != shapeEnd);
  using Index = std::decay_t<decltype(*indexBegin)>;
  Index res = 0;
  ShapeIt shapeIt = shapeBegin;
  ++shapeIt;
  IndexIt indexIt = indexBegin;
  for (; shapeIt != shapeEnd; ++shapeIt, ++indexIt) {
    res = *shapeIt * (*indexIt + res);
  }
  res += *indexIt;
  return res;
}

template <typename It, typename ShapeIt, typename IndexIt>
void tensorAdvance(It& begin, ShapeIt shapeBegin, ShapeIt shapeEnd,
                   IndexIt indexBegin) {
  auto flatIdx = flattenIndex(shapeBegin, shapeEnd, indexBegin);
  std::advance(begin, flatIdx);
}

template <typename BasesIt, typename DigitsIt>
void successor(BasesIt basesBegin, BasesIt basesEnd, DigitsIt digitsBegin) {
  DigitsIt digitsIt = digitsBegin;
  BasesIt basesIt = basesBegin;
  for (; basesIt != basesEnd; ++basesIt, ++digitsIt) {
    ++(*digitsIt);
    if (*digitsIt == *basesIt) {
      *digitsIt = 0;
    } else {
      break;
    }
  }
}

template <typename ShapeIt, typename IndexIt>
void nextTensorIndex(ShapeIt shapeBegin, ShapeIt shapeEnd, IndexIt indexBegin) {
  IndexIt indexEnd = indexBegin;
  std::advance(indexEnd, std::distance(shapeBegin, shapeEnd));
  successor(std::make_reverse_iterator(shapeEnd),
            std::make_reverse_iterator(shapeBegin),
            std::make_reverse_iterator(indexEnd));
}

template <typename It, typename PermutationIt, typename OutIt>
void permute(It begin, It end, PermutationIt permutationBegin, OutIt outBegin) {
  OutIt outIt = outBegin;
  PermutationIt permutationIt = permutationBegin;
  PermutationIt permutationEnd = permutationBegin;
  std::advance(permutationEnd, std::distance(begin, end));
  for (; permutationIt != permutationEnd; ++permutationIt, ++outIt) {
    *outIt = *(begin + *permutationIt);
  }
}

template <typename It, typename ShapeIt, typename PermutationIt, typename OutIt,
          typename OutShapeIt>
void transpose(It begin, It end, ShapeIt shapeBegin, ShapeIt shapeEnd,
               PermutationIt permutationBegin, OutIt outBegin,
               OutShapeIt outShapeBegin) {
  // Not efficient as some dimensions ranges can be squashed.

  ptrdiff_t shapeSize = std::distance(shapeBegin, shapeEnd);

  permute(shapeBegin, shapeEnd, permutationBegin, outShapeBegin);
  OutShapeIt outShapeEnd = outShapeBegin;
  std::advance(outShapeEnd, shapeSize);

  using Index = std::decay_t<decltype(*shapeBegin)>;
  SmallVector<Index, 16> srcIndex(shapeSize, 0);
  SmallVector<Index, 16> dstIndex(shapeSize);

  for (It it = begin; it != end; ++it) {
    permute(srcIndex.begin(), srcIndex.end(), permutationBegin,
            dstIndex.begin());
    OutIt outIt = outBegin;
    tensorAdvance(outIt, outShapeBegin, outShapeEnd, dstIndex.begin());
    *outIt = *it;
    nextTensorIndex(shapeBegin, shapeEnd, srcIndex.begin());
  }
}

template <typename It, typename ShapeIt, typename OutIt, typename OutShapeIt>
void swapAxes(It begin, It end, ShapeIt shapeBegin, ShapeIt shapeEnd,
              size_t axis1Index, size_t axis2Index, OutIt outBegin,
              OutShapeIt outShapeBegin) {
  auto nDims = std::distance(shapeBegin, shapeEnd);
  SmallVector<size_t, 16> permutation(nDims);
  std::iota(permutation.begin(), permutation.end(), size_t(0));
  std::swap(permutation[axis1Index], permutation[axis2Index]);
  transpose(begin, end, shapeBegin, shapeEnd, permutation.begin(), outBegin,
            outShapeBegin);
}

template <typename It, typename ShapeIt, typename OutIt, typename OutShapeIt>
void moveAxis(It begin, It end, ShapeIt shapeBegin, ShapeIt shapeEnd,
              size_t source, size_t destination, OutIt outBegin,
              OutShapeIt outShapeBegin) {
  // Reshape by adding an extra 1-sized dimension and then swap axes.

  using ShapeT = std::decay_t<decltype(*shapeBegin)>;
  auto shapeSize = std::distance(shapeBegin, shapeEnd);
  SmallVector<ShapeT, 16> swapShape(shapeSize + 1);
  auto swapShapeIt =
      std::copy(shapeBegin, shapeBegin + destination, swapShape.begin());
  *swapShapeIt = 1;
  swapShapeIt++;
  std::copy(shapeBegin + destination, shapeEnd, swapShapeIt);
  SmallVector<ShapeT, 16> outSwapShape(shapeSize + 1);
  swapAxes(begin, end, swapShape.begin(), swapShape.end(), source, destination,
           outBegin, outSwapShape.begin());
  OutShapeIt outShapeIt = std::copy(
      outSwapShape.begin(), outSwapShape.begin() + source, outShapeBegin);
  std::copy(outSwapShape.begin() + source + 1, outSwapShape.end(), outShapeIt);
}

inline FailureOr<func::FuncOp> getMainFunc(Operation* op) {
  func::FuncOp mainFunc;
  op->walk([&mainFunc](func::FuncOp func) {
    if (func.getSymName() == "main") {
      mainFunc = func;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!mainFunc) {
    return failure();
  }
  return mainFunc;
}

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_TRANSFORMS_COLLECTIVES_UTILS_H
