/* Copyright 2022 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef STABLEHLO_TRANSFORMS_COLLECTIVES_PASSES_H
#define STABLEHLO_TRANSFORMS_COLLECTIVES_PASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace stablehlo {
#define GEN_PASS_DECL_RENAMEENTRYTOMAIN
#define GEN_PASS_DECL_RENAMEMAINTOENTRY
#define GEN_PASS_DECL_COLLECTIVESSPMDSUBPARTITIONER
#define GEN_PASS_DECL_COMPLETECOLLECTIVESSPMDSUBPARTITION
#define GEN_PASS_DECL_SETCOLLECTIVESDEVICEDOMAIN
#define GEN_PASS_DECL_MOVEDEVICEDOMAINTOFRONTENDATTRIBUTES
#define GEN_PASS_DECL_MOVEDEVICEDOMAINFROMFRONTENDATTRIBUTES
#define GEN_PASS_REGISTRATION
#include "stablehlo/transforms/collectives/Passes.h.inc"

void populateDistributedPassPipeline(OpPassManager &pm);
void registerDistributedPassPipeline();

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_TRANSFORMS_COLLECTIVES_PASSES_H
