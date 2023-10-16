if (NOT STABLEHLO_ENABLE_DISTRIBUTED)
  return()
endif()

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name StablehloConversionsDistributed)
add_public_tablegen_target(StablehloConversionsDistributedPassesIncGen)

add_mlir_dialect_library(
  StablehloConversionsDistributedPasses
  PARTIAL_SOURCES_INTENDED
  Passes.cpp

  DEPENDS
  StablehloConversionsDistributedPassesIncGen

  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRSupport
  StablehloBase
  StablehloTypeInference
  StablehloOps
)
