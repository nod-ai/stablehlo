#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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

LogicalResult setDeviceDomainIfUnset(Operation* op, StringRef deviceDomain) {
  Attribute deviceDomainAttr = op->getAttr("device_domain");
  if (deviceDomainAttr) {
    return failure();
  }

  op->setAttr("device_domain", StringAttr::get(op->getContext(), deviceDomain));
  return success();
}

bool isCollectiveOperation(
    Operation* op, const llvm::SmallDenseSet<TypeID>& collectivesTypeIdSet) {
  return collectivesTypeIdSet.contains(op->getName().getTypeID());
}

struct SetCollectivesDeviceDomainRewritePattern : public RewritePattern {
 public:
  SetCollectivesDeviceDomainRewritePattern(
      MLIRContext* context,
      const llvm::SmallDenseSet<TypeID>& collectivesTypeIdSet,
      StringRef deviceDomain)
      : RewritePattern(MatchAnyOpTypeTag(), PatternBenefit(1), context),
        collectivesTypeIdSet(collectivesTypeIdSet),
        deviceDomain(deviceDomain) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const final {
    if (!isCollectiveOperation(op, collectivesTypeIdSet)) {
      return failure();
    }
    return setDeviceDomainIfUnset(op, deviceDomain);
  }

 private:
  const llvm::SmallDenseSet<TypeID>& collectivesTypeIdSet;
  StringRef deviceDomain;
};

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
    RewritePatternSet patterns(&getContext());
    patterns.add<SetCollectivesDeviceDomainRewritePattern>(
        &getContext(), collectivesTypeIdSet, deviceDomain);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }

 private:
  llvm::SmallDenseSet<TypeID> collectivesTypeIdSet;
};

}  // namespace

}  // namespace stablehlo
}  // namespace mlir
