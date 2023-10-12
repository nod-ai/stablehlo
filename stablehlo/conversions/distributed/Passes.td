include "mlir/Pass/PassBase.td"

def LegalizeDistributed
  : InterfacePass<"stablehlo-distributed-legalize", "FunctionOpInterface"> {
  let summary = "Legalize input IR distributed related operations to StableHLO.";
  let description = [{
    Currently just converts StableHLO custom call operation
    `sharding.tensor_annotate` to `Sharding`.
  }];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}
