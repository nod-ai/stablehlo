#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/collectives/Passes.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_SETCOLLECTIVESDEVICEDOMAIN
#include "stablehlo/transforms/collectives/Passes.h.inc"

namespace {

LogicalResult addCollectiveToTypeSet(StringRef operationName,
                                     llvm::SmallDenseSet<TypeID>& set,
                                     MLIRContext* ctx) {
  auto opName = RegisteredOperationName::lookup(operationName, ctx);
  if (!opName) {
    llvm::errs() << "Operation " << operationName << " not registered.\n";
    return failure();
  }
  set.insert(opName->getTypeID());
  return success();
}

FailureOr<llvm::SmallDenseSet<TypeID>> makeCollectivesTypeIdSet(
    MLIRContext* ctx) {
  llvm::SmallDenseSet<TypeID> set;
  SmallVector<std::string, 10> collectiveOpNames(
      {"stablehlo.all_gather", "stablehlo.all_reduce", "stablehlo.all_to_all",
       "stablehlo.reduce_scatter"});
  for (auto& opNameStr : collectiveOpNames) {
    auto status = addCollectiveToTypeSet(opNameStr, set, ctx);
    if (failed(status)) {
      return status;
    }
  }
  return set;
}

bool isCollectiveOperation(Operation* op,
                           llvm::SmallDenseSet<TypeID>& collectivesTypeIdSet) {
  return collectivesTypeIdSet.contains(op->getName().getTypeID());
}

struct SetCollectivesDeviceDomainPass
    : public impl::SetCollectivesDeviceDomainBase<
          SetCollectivesDeviceDomainPass> {
  using SetCollectivesDeviceDomainBase::SetCollectivesDeviceDomainBase;

  LogicalResult initialize(MLIRContext* context) override {
    FailureOr<llvm::SmallDenseSet<TypeID>> collectivesTypeIdSetStatus =
        makeCollectivesTypeIdSet(context);
    if (failed(collectivesTypeIdSetStatus)) {
      return failure();
    }
    collectivesTypeIdSet = collectivesTypeIdSetStatus.value();
    return success();
  }

  void runOnOperation() override {
    Operation* op = getOperation();
    if (!isCollectiveOperation(op, collectivesTypeIdSet)) {
      return;
    }
    // if (op->getAttrOfType<StringAttr>())
  }

 private:
  llvm::SmallDenseSet<TypeID> collectivesTypeIdSet;
};

}  // namespace

}  // namespace stablehlo
}  // namespace mlir
