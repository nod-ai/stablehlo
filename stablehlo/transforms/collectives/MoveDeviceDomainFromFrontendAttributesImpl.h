#include <algorithm>
#include <iostream>
#include <iterator>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/collectives/Passes.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_MOVEDEVICEDOMAINFROMFRONTENDATTRIBUTES
#include "stablehlo/transforms/collectives/Passes.h.inc"

namespace {

LogicalResult moveDeviceDomainFromFrontendAttributes(Operation* op) {
  DictionaryAttr frontendAttrs =
      op->getAttrOfType<DictionaryAttr>("stablehlo.frontend_attributes");
  if (!frontendAttrs) {
    return failure();
  }

  std::optional<NamedAttribute> deviceDomainAttrInFrontendAttrs =
      frontendAttrs.getNamed("device_domain");
  if (!deviceDomainAttrInFrontendAttrs) {
    return failure();
  }

  Attribute deviceDomainAttr = op->getAttr("device_domain");
  if (deviceDomainAttr) {
    emitError(op->getLoc()) << "Can't move device_domain attribute out of "
                               "stablehlo.frontend_attributes."
                               " It is already present in. ";
    return failure();
  }

  ArrayRef<NamedAttribute> existingFrontendAttrsValue =
      frontendAttrs.getValue();
  SmallVector<NamedAttribute, 32> frontendAttrsValueWithoutDeviceDomain;
  std::copy_if(existingFrontendAttrsValue.begin(),
               existingFrontendAttrsValue.end(),
               std::back_inserter(frontendAttrsValueWithoutDeviceDomain),
               [](NamedAttribute attr) {
                 return attr.getName().getValue() != "device_domain";
               });
  op->setAttr("stablehlo.frontend_attributes",
              DictionaryAttr::get(op->getContext(),
                                  frontendAttrsValueWithoutDeviceDomain));

  op->setAttr(deviceDomainAttrInFrontendAttrs->getName(),
              deviceDomainAttrInFrontendAttrs->getValue());

  return success();
}

struct MoveDeviceDomainFromFrontendAttributesRewritePattern
    : public RewritePattern {
 public:
  MoveDeviceDomainFromFrontendAttributesRewritePattern(MLIRContext* context)
      : RewritePattern(MatchAnyOpTypeTag(), PatternBenefit(1), context) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const final {
    return moveDeviceDomainFromFrontendAttributes(op);
  }
};

struct MoveDeviceDomainFromFrontendAttributesPass
    : public impl::MoveDeviceDomainFromFrontendAttributesBase<
          MoveDeviceDomainFromFrontendAttributesPass> {
  using MoveDeviceDomainFromFrontendAttributesBase::
      MoveDeviceDomainFromFrontendAttributesBase;

  void runOnOperation() override {
    std::cout << "Running pass on "
              << getOperation()->getName().getStringRef().str() << std::endl;
    RewritePatternSet patterns(&getContext());
    patterns.add<MoveDeviceDomainFromFrontendAttributesRewritePattern>(
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
