#include "llvm/ADT/DenseMap.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace stablehlo {

using DeviceId = int;
using SuperSubDeviceIdMap = llvm::DenseMap<DeviceId, SmallVector<DeviceId, 8>>;

void registerCollectiveCliOptions();

}  // namespace stablehlo
}  // namespace mlir
