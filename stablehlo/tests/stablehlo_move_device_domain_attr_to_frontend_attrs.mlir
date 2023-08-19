// RUN: stablehlo-opt \
// RUN:   --pass-pipeline="builtin.module(func.func(stablehlo-move-device-domain-to-frontend-attributes))" \
// RUN:   --split-input-file "%s" \
// RUN: | FileCheck "%s"

//        CHECK-LABEL: func.func @empty_frontend_attributes
func.func @empty_frontend_attributes(%arg0: tensor<2xf32>) -> tensor<2xf32> {
//              CHECK: { stablehlo.frontend_attributes = { device_domain = "super" } }
  %0 = "stablehlo.abs" (%arg0) { device_domain="super" }: (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
