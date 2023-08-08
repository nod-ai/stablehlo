set(LLVM_OPTIONAL_SOURCES
  XlaCcLibLoader.cpp
  XlaPasses.cpp
)

add_mlir_library(StablehloXlaCcLibLoader
  XlaCcLibLoader.cpp

  ADDITIONAL_HEADER_DIRS
  ${XLA_DIR}
)
target_include_directories(StablehloXlaCcLibLoader PUBLIC ${XLA_DIR})

set(LLVM_TARGET_DEFINITIONS XlaPasses.td)
mlir_tablegen(XlaPasses.h.inc -gen-pass-decls -name XlaPasses)
add_public_tablegen_target(StablehloXlaPassesIncGen)
add_dependencies(mlir-headers StablehloXlaPassesIncGen)

add_mlir_doc(Passes XlaPasses ./ -gen-pass-doc)

add_mlir_dialect_library(StablehloXlaPasses
  XlaPasses.cpp

  DEPENDS
  StablehloOps
  StablehloXlaPassesIncGen

  LINK_LIBS
  PUBLIC
  MLIRPass
  LLVMSupport
  PRIVATE
  StablehloXlaCcLibLoader
)
