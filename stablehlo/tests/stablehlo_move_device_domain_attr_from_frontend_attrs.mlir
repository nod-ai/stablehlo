// RUN: stablehlo-opt \
// RUN:   --pass-pipeline="builtin.module(func.func(stablehlo-move-device-domain-from-frontend-attributes))" \
// RUN:   --split-input-file "%s" \
// RUN: | FileCheck "%s"

//        CHECK-LABEL: func.func @without_device_domain_attribute
func.func @without_device_domain_attribute(%arg0: tensor<2xf32>) -> tensor<2xf32> {
//              CHECK: {stablehlo.frontend_attributes = {some_other_attr = "val"}}
  %0 = "stablehlo.abs" (%arg0) { stablehlo.frontend_attributes = {some_other_attr = "val"} } : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

//        CHECK-LABEL: func.func @with_device_domain_attribute
func.func @with_device_domain_attribute(%arg0: tensor<2xf32>) -> tensor<2xf32> {
//              CHECK: {device_domain = "super", stablehlo.frontend_attributes = {some_other_attr = "val"}}
  %0 = "stablehlo.abs" (%arg0)
    { stablehlo.frontend_attributes = {some_other_attr = "val", device_domain="super"} }
    : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
