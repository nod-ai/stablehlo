#include <charconv>
#include <fstream>
#include <memory>
#include <utility>

#include "CollectivesPassesCli.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace stablehlo {

struct CollectiveCliOptions {
  llvm::cl::opt<std::string> superSubDeviceMapFile{
      "super-sub-device-map-file",
      llvm::cl::desc(
          "Path to a YAML file that describes the map between original global "
          "super-device IDs to a list of sub-device IDs. "
          "This will be used to sub-partition a program that already contains "
          "collective operations. "
          R"(Example:
        0: [0, 1, 2] # Map super-device 0 to sub-devices 0, 1 and 2.
        1: [3, 4, 5]
        )")};
};

}  // namespace stablehlo
}  // namespace mlir

namespace llvm {
namespace yaml {
template <>
struct CustomMappingTraits<mlir::stablehlo::SuperSubDeviceIdMap> {
  static void inputOne(IO &io, StringRef key,
                       mlir::stablehlo::SuperSubDeviceIdMap &v) {
    int val;
    std::from_chars_result fromCharsRes =
        std::from_chars(key.data(), key.data() + key.size(), val);
    if (fromCharsRes.ec != std::errc()) {
      io.setError(Twine("Failed parsing into SuperSubDeviceIdMap, key \"") +
                  key + "\" not an integer.");
    }
    io.mapRequired(key.str().c_str(), v[val]);
  }

  static void output(IO &io, mlir::stablehlo::SuperSubDeviceIdMap &v) {
    // TODO
  }
};
}  // namespace yaml
}  // namespace llvm

namespace mlir {
namespace stablehlo {

llvm::ErrorOr<SuperSubDeviceIdMap> parseSuperSubDeviceMap(
    const Twine &filepath) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileMemBuff =
      llvm::MemoryBuffer::getFile(filepath, /*IsText=*/true);
  if (!fileMemBuff) {
    return fileMemBuff.getError();
  }

  llvm::yaml::Input yamlInput(fileMemBuff->get()->getBuffer());
  SuperSubDeviceIdMap res;
  yamlInput >> res;
  if (yamlInput.error()) {
    return yamlInput.error();
  }

  return res;
}

llvm::ErrorOr<CollectiveOptions> parseCollectiveCliOptions(
    const CollectiveCliOptions &collectiveCliOptions) {
  CollectiveOptions res;
  llvm::ErrorOr<SuperSubDeviceIdMap> superSubDeviceMap = parseSuperSubDeviceMap(
      collectiveCliOptions.superSubDeviceMapFile.getValue());
  if (!superSubDeviceMap) {
    return superSubDeviceMap.getError();
  }

  res.superSubDeviceMap = std::move(superSubDeviceMap.get());
  return res;
}

llvm::ManagedStatic<CollectiveCliOptions> collectiveCliOptions;

struct CollectiveOptionsCreatorFromCli {
  static void *call() {
    std::unique_ptr<CollectiveOptions> res =
        std::make_unique<CollectiveOptions>();
    llvm::ErrorOr<CollectiveOptions> opts =
        parseCollectiveCliOptions(*collectiveCliOptions);
    if (!opts) {
      llvm::errs() << "Failed parsing collective CLI options: "
                   << opts.getError().message();
    }
    *res = std::move(opts.get());
    return res.release();
  }
};

llvm::ManagedStatic<CollectiveOptions, CollectiveOptionsCreatorFromCli>
    collectiveOptions;

void registerCollectiveCliOptions() {
  // Make sure that the options struct has been constructed.
  *collectiveCliOptions;
}

CollectiveOptions &getCollectiveOptions() { return *collectiveOptions; }

}  // namespace stablehlo
}  // namespace mlir
