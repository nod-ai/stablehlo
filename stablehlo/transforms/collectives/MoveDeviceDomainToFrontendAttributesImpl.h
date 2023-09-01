#include <algorithm>
#include <iostream>
#include <iterator>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/collectives/Passes.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_MOVEDEVICEDOMAINTOFRONTENDATTRIBUTES
#include "stablehlo/transforms/collectives/Passes.h.inc"

namespace {

LogicalResult moveDeviceDomainToFrontendAttributes(Operation* op) {
  StringAttr deviceDomainAttr = op->getAttrOfType<StringAttr>("device_domain");
  if (!deviceDomainAttr) {
    return failure();
  }

  DictionaryAttr frontendAttrs =
      op->getAttrOfType<DictionaryAttr>("mhlo.frontend_attributes");
  SmallVector<NamedAttribute, 16> frontendAttrValue;
  if (frontendAttrs) {
    std::optional<NamedAttribute> deviceDomainAttrInFrontendAttrs =
        frontendAttrs.getNamed("device_domain");
    if (deviceDomainAttrInFrontendAttrs) {
      StringAttr deviceDomainAttrInFrontendAttrsValue =
          deviceDomainAttrInFrontendAttrs->getValue().cast<StringAttr>();
      if (!deviceDomainAttrInFrontendAttrsValue ||
          deviceDomainAttrInFrontendAttrsValue != deviceDomainAttr) {
        emitError(op->getLoc())
            << "Can't move device_domain Attributed. It is already present in "
               "mhlo.frontend_attributes with different value.";
        return failure();
      }
    }

    ArrayRef<NamedAttribute> existingFrontendAttrsValue =
        frontendAttrs.getValue();
    std::copy(existingFrontendAttrsValue.begin(),
              existingFrontendAttrsValue.end(),
              std::back_inserter(frontendAttrValue));
  }

  frontendAttrValue.emplace_back(
      StringAttr::get(op->getContext(), "device_domain"), deviceDomainAttr);
  op->setAttr("mhlo.frontend_attributes",
              DictionaryAttr::get(op->getContext(), frontendAttrValue));
  op->removeAttr("device_domain");
  return success();
}

struct MoveDeviceDomainToFrontendAttributesRewritePattern
    : public RewritePattern {
 public:
  MoveDeviceDomainToFrontendAttributesRewritePattern(MLIRContext* context)
      : RewritePattern(MatchAnyOpTypeTag(), PatternBenefit(1), context) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const final {
    return moveDeviceDomainToFrontendAttributes(op);
  }
};

struct MoveDeviceDomainToFrontendAttributesPass
    : public impl::MoveDeviceDomainToFrontendAttributesBase<
          MoveDeviceDomainToFrontendAttributesPass> {
  using MoveDeviceDomainToFrontendAttributesBase::
      MoveDeviceDomainToFrontendAttributesBase;

  void runOnOperation() override {
    std::cout << "Running pass on "
              << getOperation()->getName().getStringRef().str() << std::endl;
    RewritePatternSet patterns(&getContext());
    patterns.add<MoveDeviceDomainToFrontendAttributesRewritePattern>(
        &getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace stablehlo
}  // namespace mlir
