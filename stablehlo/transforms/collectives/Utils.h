#ifndef STABLEHLO_TRANSFORMS_COLLECTIVES_UTILS_H
#define STABLEHLO_TRANSFORMS_COLLECTIVES_UTILS_H

#include <algorithm>

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
  void operator()(Op op, uint64_t numPartitions,
                  uint64_t numReplicasPerPartition,
                  OutputDeviceIdsIt outputGlobalDeviceIdsIt,
                  OutputDeviceIdsShapeIt outputGlobalDeviceIdsShapeIt);
};

template <typename Op, typename OutputDeviceIdsIt,
          typename OutputDeviceIdsShapeIt>
void getReplicaGroupsAsGlobalDeviceIds(
    Op op, uint64_t numPartitions, uint64_t numReplicasPerPartition,
    OutputDeviceIdsIt outputGlobalDeviceIdsIt,
    OutputDeviceIdsShapeIt outputGlobalDeviceIdsShapeIt) {
  GetReplicaGroupsAsGlobalDeviceIds<Op, OutputDeviceIdsIt,
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
void getReplicaGroupsAsGlobalDeviceIds1(Op op, uint64_t numPartitions,
                                        uint64_t numReplicasPerPartition,
                                        OutIdsIt outIdsBegin,
                                        OutIdsShapeIt outIdsShapeBegin) {
  int64_t channelId = getChannelId(op.getChannelHandle());
  bool useGlobalDeviceIds = op.getUseGlobalDeviceIds();
  ArrayRef<int64_t> replicaGroupsShape =
      op.getReplicaGroups().getShapedType().getShape();
  auto replicaGroups = op.getReplicaGroups().template getValues<int64_t>();
  assert(std::distance(replicaGroupsShape.begin(), replicaGroupsShape.end()) ==
         2);
  if (channelId <= 0 && !useGlobalDeviceIds) {
    crossReplicaGroupsToGlobalDeviceIds(
        numPartitions, numReplicasPerPartition, replicaGroups.begin(),
        replicaGroupsShape.begin(), replicaGroupsShape.end(), outIdsBegin,
        outIdsShapeBegin);
  } else if (channelId > 0 && !useGlobalDeviceIds) {
    crossReplicaAndPartitionGroupsToGlobalDeviceIds(
        numPartitions, numReplicasPerPartition, replicaGroups.begin(),
        replicaGroupsShape.begin(), replicaGroupsShape.end(), outIdsBegin,
        outIdsShapeBegin);
  } else if (channelId > 0 && useGlobalDeviceIds) {
    std::copy(replicaGroupsShape.begin(), replicaGroupsShape.end(),
              outIdsShapeBegin);
    std::copy(replicaGroups.begin(),
              replicaGroups.begin() +
                  replicaGroupsShape.begin()[0] * replicaGroupsShape.begin()[1],
              outIdsBegin);
  } else {
    assert(false && "Not a valid case.");
  }
}

template <typename Op, typename OutIdsIt, typename OutIdsShapeIt>
void getReplicaGroupsAsGlobalDeviceIds2(Op op, uint64_t numPartitions,
                                        uint64_t numReplicasPerPartition,
                                        OutIdsIt outIdsBegin,
                                        OutIdsShapeIt outIdsShapeBegin) {
  int64_t channelId = getChannelId(op.getChannelHandle());
  ArrayRef<int64_t> replicaGroupsShape =
      op.getReplicaGroups().getShapedType().getShape();
  auto replicaGroups = op.getReplicaGroups().template getValues<int64_t>();
  assert(std::distance(replicaGroupsShape.begin(), replicaGroupsShape.end()) ==
         2);
  if (channelId <= 0) {
    crossReplicaGroupsToGlobalDeviceIds(
        numPartitions, numReplicasPerPartition, replicaGroups.begin(),
        replicaGroupsShape.begin(), replicaGroupsShape.end(), outIdsBegin,
        outIdsShapeBegin);
  } else if (channelId > 0) {
    crossPartitionGroupsToGlobalDeviceIds(
        numPartitions, numReplicasPerPartition, replicaGroups.begin(),
        replicaGroupsShape.begin(), replicaGroupsShape.end(), outIdsBegin,
        outIdsShapeBegin);
  } else {
    assert(false && "Not a valid case.");
  }
}

template <typename OutputDeviceIdsIt, typename OutputDeviceIdsShapeIt>
struct GetReplicaGroupsAsGlobalDeviceIds<AllGatherOp, OutputDeviceIdsIt,
                                         OutputDeviceIdsShapeIt> {
  void operator()(AllGatherOp op, uint64_t numPartitions,
                  uint64_t numReplicasPerPartition,
                  OutputDeviceIdsIt outputGlobalDeviceIdsIt,
                  OutputDeviceIdsShapeIt outputGlobalDeviceIdsShapeIt) {
    getReplicaGroupsAsGlobalDeviceIds1(
        op, numPartitions, numReplicasPerPartition, outputGlobalDeviceIdsIt,
        outputGlobalDeviceIdsShapeIt);
  }
};

template <typename OutputDeviceIdsIt, typename OutputDeviceIdsShapeIt>
struct GetReplicaGroupsAsGlobalDeviceIds<AllReduceOp, OutputDeviceIdsIt,
                                         OutputDeviceIdsShapeIt> {
  void operator()(AllReduceOp op, uint64_t numPartitions,
                  uint64_t numReplicasPerPartition,
                  OutputDeviceIdsIt outputGlobalDeviceIdsIt,
                  OutputDeviceIdsShapeIt outputGlobalDeviceIdsShapeIt) {
    getReplicaGroupsAsGlobalDeviceIds1(
        op, numPartitions, numReplicasPerPartition, outputGlobalDeviceIdsIt,
        outputGlobalDeviceIdsShapeIt);
  }
};

template <typename OutputDeviceIdsIt, typename OutputDeviceIdsShapeIt>
struct GetReplicaGroupsAsGlobalDeviceIds<ReduceScatterOp, OutputDeviceIdsIt,
                                         OutputDeviceIdsShapeIt> {
  void operator()(ReduceScatterOp op, uint64_t numPartitions,
                  uint64_t numReplicasPerPartition,
                  OutputDeviceIdsIt outputGlobalDeviceIdsIt,
                  OutputDeviceIdsShapeIt outputGlobalDeviceIdsShapeIt) {
    getReplicaGroupsAsGlobalDeviceIds1(
        op, numPartitions, numReplicasPerPartition, outputGlobalDeviceIdsIt,
        outputGlobalDeviceIdsShapeIt);
  }
};

template <typename OutputDeviceIdsIt, typename OutputDeviceIdsShapeIt>
struct GetReplicaGroupsAsGlobalDeviceIds<AllToAllOp, OutputDeviceIdsIt,
                                         OutputDeviceIdsShapeIt> {
  void operator()(AllToAllOp op, uint64_t numPartitions,
                  uint64_t numReplicasPerPartition,
                  OutputDeviceIdsIt outputGlobalDeviceIdsIt,
                  OutputDeviceIdsShapeIt outputGlobalDeviceIdsShapeIt) {
    getReplicaGroupsAsGlobalDeviceIds2(
        op, numPartitions, numReplicasPerPartition, outputGlobalDeviceIdsIt,
        outputGlobalDeviceIdsShapeIt);
  }
};

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_TRANSFORMS_COLLECTIVES_UTILS_H
