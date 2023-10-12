/* Copyright 2022 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef STABLEHLO_TRANSFORMS_COLLECTIVES_CUSTOMCALLOPTARGETNAMEREWRITEPATTERN_H
#define STABLEHLO_TRANSFORMS_COLLECTIVES_CUSTOMCALLOPTARGETNAMEREWRITEPATTERN_H

#include "mlir/IR/PatternMatch.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace stablehlo {

struct CustomCallOpTargetNameRewritePattern
    : public OpRewritePattern<CustomCallOp> {
 public:
  CustomCallOpTargetNameRewritePattern(MLIRContext* context, StringRef srcName,
                                       StringRef dstName)
      : OpRewritePattern<CustomCallOp>(context),
        srcName(srcName),
        dstName(dstName) {}

  LogicalResult matchAndRewrite(CustomCallOp op,
                                PatternRewriter& rewriter) const final {
    if (op.getCallTargetName() != srcName) {
      return failure();
    }

    op.setCallTargetName(dstName);

    return success();
  }

 private:
  std::string srcName;
  std::string dstName;
};

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_TRANSFORMS_COLLECTIVES_CUSTOMCALLOPTARGETNAMEREWRITEPATTERN_H
