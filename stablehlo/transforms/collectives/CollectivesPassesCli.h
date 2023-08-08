#ifndef STABLEHLO_TRANSFORMS_COLLECTIVESPASSESCLI_H
#define STABLEHLO_TRANSFORMS_COLLECTIVESPASSESCLI_H

#include "llvm/ADT/DenseMap.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace stablehlo {

using DeviceId = int64_t;
using SuperSubDeviceIdMap = llvm::DenseMap<DeviceId, SmallVector<DeviceId, 8>>;

struct CollectiveOptions {
  SuperSubDeviceIdMap superSubDeviceMap;
};

void registerCollectiveCliOptions();
CollectiveOptions& getCollectiveOptions();

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_TRANSFORMS_COLLECTIVESPASSESCLI_H
