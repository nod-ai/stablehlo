set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name StablehloCollectivesTransforms)
add_public_tablegen_target(StablehloCollectivesPassesIncGen)

add_mlir_dialect_library(
  StablehloCollectivesPasses
  PARTIAL_SOURCES_INTENDED
  CollectivesPasses.cpp

  DEPENDS
  StablehloCollectivesPassesIncGen

  LINK_LIBS
  PUBLIC
  MLIRFunctionInterfaces
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  StablehloBase
  StablehloTypeInference
  StablehloOps
  StablehloConversionsDistributedPasses
  StablehloXlaPasses
  PRIVATE
  StablehloXlaCcLibLoader
)
