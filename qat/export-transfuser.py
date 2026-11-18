# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys

from torchpack.utils.config import configs
from mmcv import Config
from mmcv.runner.fp16_utils import auto_fp16
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval
from mmcv.runner import wrap_fp16_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
try:
    from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
    import lean.quantize as quantize
except ImportError as _e:
    print(f"[warn] 跳过 pytorch_quantization（{_e}）")
    TensorQuantizer = None
    quantize = None


class SubclassHeadBBox(nn.Module):
    # 导出包装：直接调用 TransFusionHead.forward_single（与训练/推理同一份代码），
    # 只把"打分组合"和"解码头取值"留在外面，避免导出脚本与训练逻辑漂移。
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.classes_eye = nn.Parameter(torch.eye(parent.heads.object.num_classes).float())

    def forward(self, inputs):
        # classes_eye 是模块参数，不作为图输入（否则 ONNX 会多一个输入）
        return self._forward(inputs, self.classes_eye)

    @auto_fp16(apply_to=("inputs", "classes_eye"))
    def _forward(self, inputs, classes_eye):
        head = self.parent.heads.object
        # forward_single 不使用 img_inputs / metas，传 None 即可
        pred = head.forward_single(inputs, None, None)[0]
        # 每个 query 的种子类别 one_hot（等价训练里的 one_hot(query_labels)）
        one_hot = classes_eye.index_select(0, head.query_labels.reshape(-1))[None].permute(0, 2, 1)
        # 与 get_bboxes 完全一致的分数组合：cls_prob * 种子热图值 * one_hot(种子类)
        score = pred["heatmap"].sigmoid() * pred["query_heatmap_score"] * one_hot
        vel = pred.get("vel")
        if vel is None:
            # 没有 vel 头时给一个零张量，保证 ONNX 输出个数固定
            vel = score.new_zeros((score.shape[0], 2, score.shape[-1]))
        return score, pred["rot"], pred["dim"], pred["center"], pred["height"], vel


class SubclassFuser(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    @auto_fp16(apply_to=("features",))
    def forward(self, features):
        if self.parent.fuser is not None:
            x = self.parent.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.parent.decoder["backbone"](x)
        x = self.parent.decoder["neck"](x)
        return x[0]

class CustomLayerNormImpl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, weight, bias, eps, x_shape):
        return F.layer_norm(input, normalized_shape, weight, bias, eps)

    @staticmethod
    def symbolic(g, input, normalized_shape, weight, bias, eps, x_shape):
        y = g.op("nv::CustomLayerNormalization", input, weight, bias, axis_i=-1, epsilon_f=eps)
        y.setType(input.type().with_sizes(x_shape))
        return y

class CustomLayerNorm(nn.LayerNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return CustomLayerNormImpl.apply(
            input, self.normalized_shape, self.weight, self.bias, self.eps, input.size())

    @staticmethod
    def convert(old: nn.LayerNorm):
        Custom_layernorm = CustomLayerNorm(old.normalized_shape, old.eps, old.elementwise_affine)
        if Custom_layernorm.weight is not None:
            Custom_layernorm.weight.data = old.weight.data
            Custom_layernorm.bias.data   = old.bias.data
        return Custom_layernorm

def replace_layernorm(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            parent, child = name.rsplit(".", 1)
            parent = model.get_submodule(parent)
            setattr(parent, child, CustomLayerNorm.convert(module))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export transfusion to onnx file")
    parser.add_argument("--ckpt", type=str, default="qat/ckpt/bevfusion_ptq.pth", help="Pretrain model")
    parser.add_argument('--fp16', action= 'store_true')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    # 覆盖按类配额（逗号分隔，例如 "20,28,32,20,48,24,36,20,36"）；
    # 不传则用 pickle 里模型自带的值（由 to_deploy_ckpt.py 从 yaml 烘进去）
    parser.add_argument('--per-class-topk', type=str, default=None)
    # 用标准 LayerNorm（torch 原生分解）导出，供 onnxruntime 做 int8 PTQ 用；
    # 默认仍用 nv::CustomLayerNormalization（TensorRT 友好，但 ORT 不认）
    parser.add_argument('--std-layernorm', action='store_true')
    args = parser.parse_args()
    model = __ckpt = torch.load(args.ckpt, map_location="cpu")
    model = getattr(__ckpt, "module", __ckpt)

    # 打印 pickle 里记录的配置来源（to_deploy_ckpt.py 写入）
    print(f"[info] deploy cfg: {getattr(model, '_deploy_cfg_path', '<未记录>')}")
    if args.per_class_topk:
        q = [int(x) for x in args.per_class_topk.split(',') if x.strip()]
        head = model.heads['object']
        head.per_class_topk = q
        head._per_class_adaptive = True
        head.num_proposals = int(sum(q))
        print(f"[info] 覆盖 per_class_topk={q} (sum={sum(q)})")
    
    suffix = "int8"
    if args.fp16:
        suffix = "fp16"
        if quantize is not None:
            quantize.disable_quantization(model).apply()
    
    save_root = f"qat/onnx_{suffix}"
    os.makedirs(save_root, exist_ok=True)

    dev = torch.device(args.device)
    model.eval()
    fuser    = SubclassFuser(model).to(dev)
    headbbox = SubclassHeadBBox(model).to(dev)
    if not args.std_layernorm:
        replace_layernorm(headbbox)

    # 通道数按模型实际值取（nuScenes 是 80/256，本数据集是 64/128）
    cam_ch = model.encoders.camera.vtransform.C
    lidar_ch = model.fuser.in_channels[1]
    bev_h = int((model.heads['object'].test_cfg['grid_size'][0]) // model.heads['object'].test_cfg['out_size_factor'])
    bev_w = int((model.heads['object'].test_cfg['grid_size'][1]) // model.heads['object'].test_cfg['out_size_factor'])
    head_in = model.heads['object'].in_channels
    print(f"[info] cam {cam_ch}ch | lidar {lidar_ch}ch | BEV {bev_h}x{bev_w} | head in {head_in}")

    if TensorQuantizer is not None:
        TensorQuantizer.use_fb_fake_quant = True
    with torch.no_grad():
        camera_features = torch.randn(1, cam_ch, bev_h, bev_w, device=dev)
        lidar_features  = torch.randn(1, lidar_ch, bev_h, bev_w, device=dev)

        fuser_onnx_path = f"{save_root}/fuser.onnx"
        torch.onnx.export(fuser, [camera_features, lidar_features], fuser_onnx_path, opset_version=13, 
            input_names=["camera", "lidar"],
            output_names=["middle"],
        )
        print(f"🚀 The export is completed. ONNX save as {fuser_onnx_path} 🤗, Have a nice day~")

        boxhead_onnx_path = f"{save_root}/head.bbox.onnx"
        head_input = torch.randn(1, head_in, bev_h, bev_w, device=dev)
        torch.onnx.export(headbbox, head_input, f"{save_root}/head.bbox.onnx", opset_version=13, 
            input_names=["middle"],
            output_names=["score", "rot", "dim", "reg", "height", "vel"],
        )
        print(f"🚀 The export is completed. ONNX save as {boxhead_onnx_path} 🤗, Have a nice day~")
