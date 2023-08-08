#ifndef STABLEHLO_TRANSFORMS_XLA_XLACCLIBLOADER_H
#define STABLEHLO_TRANSFORMS_XLA_XLACCLIBLOADER_H

#include <string>

#include "xla/xla_cc_loader.h"

namespace mlir {
namespace stablehlo {

std::string& xlaCcLibPath();

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_TRANSFORMS_XLA_XLACCLIBLOADER_H
