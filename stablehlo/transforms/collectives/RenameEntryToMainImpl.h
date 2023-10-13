#include <memory>

#include "Passes.h"
#include "Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_RENAMEENTRYTOMAIN
#include "stablehlo/transforms/collectives/Passes.h.inc"

struct RenameEntryToMainPass
    : public impl::RenameEntryToMainBase<RenameEntryToMainPass> {
  using RenameEntryToMainBase::RenameEntryToMainBase;

  void runOnOperation() override {
    getOperation()->walk([this](func::FuncOp funcOp) {
      if (funcOp.getSymName() == entry_func) {
        DictionaryAttr frontendAttributes =
            getOperation()->getAttrOfType<DictionaryAttr>(
                "mhlo.frontend_attributes");
        if (!frontendAttributes) {
          frontendAttributes = DictionaryAttr::get(funcOp->getContext());
        }
        frontendAttributes = setAttributes(
            frontendAttributes,
            {NamedAttribute(StringAttr::get(funcOp->getContext(), "entry_func"),
                            funcOp.getSymNameAttr())});
        getOperation()->setAttr("mhlo.frontend_attributes", frontendAttributes);
        funcOp.setSymName("main");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }
};

}  // namespace stablehlo
}  // namespace mlir
