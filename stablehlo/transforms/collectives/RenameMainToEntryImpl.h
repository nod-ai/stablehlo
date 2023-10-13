#include <algorithm>
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

#define GEN_PASS_DEF_RENAMEMAINTOENTRY
#include "stablehlo/transforms/collectives/Passes.h.inc"

struct RenameMainToEntryPass
    : public impl::RenameMainToEntryBase<RenameMainToEntryPass> {
  using RenameMainToEntryBase::RenameMainToEntryBase;

  void runOnOperation() override {
    getOperation()->walk([this](func::FuncOp funcOp) {
      if (funcOp.getSymName() == "main") {
        DictionaryAttr frontendAttributes =
            getOperation()->getAttrOfType<DictionaryAttr>(
                "mhlo.frontend_attributes");
        if (!frontendAttributes) {
          return WalkResult::interrupt();
        }
        auto entryAttrIt =
            std::find_if(frontendAttributes.begin(), frontendAttributes.end(),
                         [](NamedAttribute namedAttr) {
                           return namedAttr.getName() == "entry_func";
                         });
        if (entryAttrIt == frontendAttributes.end()) {
          return WalkResult::interrupt();
        }
        funcOp.setSymName(entryAttrIt->getValue().cast<StringAttr>());
        frontendAttributes =
            removeAttributes(frontendAttributes, {"entry_func"});
        if (frontendAttributes.empty()) {
          getOperation()->removeAttr("mhlo.frontend_attributes");
        } else {
          getOperation()->setAttr("mhlo.frontend_attributes",
                                  frontendAttributes);
        }
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }
};

}  // namespace stablehlo
}  // namespace mlir
