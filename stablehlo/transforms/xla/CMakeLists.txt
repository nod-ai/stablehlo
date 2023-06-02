set(LLVM_TARGET_DEFINITIONS XlaPasses.td)
mlir_tablegen(XlaPasses.h.inc -gen-pass-decls -name XlaPasses)
add_public_tablegen_target(StablehloXlaPassesIncGen)
add_dependencies(mlir-headers StablehloXlaPassesIncGen)

add_mlir_doc(Passes XlaPasses ./ -gen-pass-doc)

add_mlir_dialect_library(StablehloXlaPasses
  XlaPasses.cpp

  DEPENDS
  StablehloXlaPassesIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  LLVMSupport
)
target_include_directories(StablehloXlaPasses PRIVATE ${XLA_DIR})
