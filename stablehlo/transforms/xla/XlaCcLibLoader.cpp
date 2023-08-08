#include "XlaCcLibLoader.h"

#include "llvm/Support/CommandLine.h"
#include "xla/xla_cc_loader_impl.h"

namespace mlir {
namespace stablehlo {

namespace {

llvm::cl::opt<std::string> xlaCcLibPathOpt = llvm::cl::opt<std::string>(
    "stablehlo-xla-cc-lib-path",
    llvm::cl::desc("Path to the shared library libxla_cc_shared.so."
                   " This path should be absolue or a filename to be searched "
                   "in the OS runtime library search paths.\n"
                   "On Unix-like systems the path is passed to dlopen verbatim."
                   " You can use the environment variable LD_LIBRARY_PATH to "
                   "add search locations."),
    llvm::cl::init("libxla_cc_shared.so"));

}  // namespace

std::string& xlaCcLibPath() { return xlaCcLibPathOpt.getValue(); }

}  // namespace stablehlo
}  // namespace mlir
