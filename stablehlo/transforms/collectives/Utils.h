#ifndef STABLEHLO_TRANSFORMS_COLLECTIVES_UTILS_H
#define STABLEHLO_TRANSFORMS_COLLECTIVES_UTILS_H

#include <algorithm>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace stablehlo {

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
      emitError(op->getLoc()) << "Mismatch between replica_groups size and "
                                 "number of replicas per partition.";
      return failure();
    }
    crossReplicaGroupsToGlobalDeviceIds(
        numPartitions, numReplicasPerPartition, replicaGroups.begin(),
        replicaGroupsShape.begin(), replicaGroupsShape.end(), outIdsBegin,
        outIdsShapeBegin);
  } else if (channelId > 0 && !useGlobalDeviceIds) {
    if (replicaGroups.size() != numReplicasPerPartition) {
      emitError(op->getLoc()) << "Mismatch between replica_groups size and "
                                 "number of replicas per partition.";
      return failure();
    }
    crossReplicaAndPartitionGroupsToGlobalDeviceIds(
        numPartitions, numReplicasPerPartition, replicaGroups.begin(),
        replicaGroupsShape.begin(), replicaGroupsShape.end(), outIdsBegin,
        outIdsShapeBegin);
  } else if (channelId > 0 && useGlobalDeviceIds) {
    if (replicaGroups.size() != numPartitions * numReplicasPerPartition) {
      emitError(op->getLoc()) << "Mismatch between replica_groups size and "
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
    emitError(op->getLoc())
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
      emitError(op->getLoc()) << "Mismatch between replica_groups size and "
                                 "number of replicas per partition.";
      return failure();
    }
    crossReplicaGroupsToGlobalDeviceIds(
        numPartitions, numReplicasPerPartition, replicaGroups.begin(),
        replicaGroupsShape.begin(), replicaGroupsShape.end(), outIdsBegin,
        outIdsShapeBegin);
  } else if (channelId > 0) {
    if (replicaGroups.size() != numPartitions) {
      emitError(op->getLoc())
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

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_TRANSFORMS_COLLECTIVES_UTILS_H
