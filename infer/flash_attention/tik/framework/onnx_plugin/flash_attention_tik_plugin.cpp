/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "register/register.h"

namespace domi {
// Onnx ParseParams
Status ParseParamFlashAttentionTik(const ge::Operator& opSrc, ge::Operator& opDest)
{
    // To do: Implement the operator plugin by referring to the Onnx Operator Development Guide.
    return SUCCESS;
}

// register FlashAttentionTik op info to GE
REGISTER_CUSTOM_OP("FlashAttentionTik")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::11::FlashAttentionTik",
        "ai.onnx::12::FlashAttentionTik",
        "ai.onnx::13::FlashAttentionTik",
        "ai.onnx::14::FlashAttentionTik",
        "ai.onnx::15::FlashAttentionTik",
        "ai.onnx::16::FlashAttentionTik",
        "ai.onnx::17::FlashAttentionTik",
        "ai.onnx::18::FlashAttentionTik"})
    .ParseParamsByOperatorFn(ParseParamFlashAttentionTik)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
