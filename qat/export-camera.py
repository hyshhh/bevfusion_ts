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


import warnings
warnings.filterwarnings("ignore")

import argparse
import os

import onnx
import torch
from onnxsim import simplify
from torchpack.utils.config import configs
from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval

from torch import nn

# pytorch_quantization 的 cuda_ext 是按别的 torch 版本编译的（ABI 不匹配），
# 这里做成可选：fp16 导出不需要量化，缺这个包也能跑
try:
    from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
    import lean.quantize as quantize
except ImportError as _e:
    print(f"[warn] 跳过 pytorch_quantization（{_e}），按非量化模型导出")
    TensorQuantizer = None
    quantize = None

def parse_args():
    parser = argparse.ArgumentParser(description="Export bevfusion model")
    parser.add_argument('--ckpt', type=str, default='qat/ckpt/bevfusion_ptq.pth')
    parser.add_argument('--fp16', action= 'store_true')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    return args

class SubclassCameraModule(nn.Module):
    def __init__(self, model):
        super(SubclassCameraModule, self).__init__()
        self.model = model

    def forward(self, img, depth):
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

        feat = self.model.encoders.camera.backbone(img)
        feat = self.model.encoders.camera.neck(feat)
        if not isinstance(feat, torch.Tensor):
            feat = feat[0]

        BN, C, H, W = map(int, feat.size())
        feat = feat.view(B, int(BN / B), C, H, W)

        def get_cam_feats(vt, x, d):
            B, N, C, fH, fW = map(int, x.shape)
            x = x.view(B * N, C, fH, fW)
            if hasattr(vt, "dtransform"):          # DepthLSSTransform
                d = d.view(B * N, *d.shape[2:])
                x = torch.cat([vt.dtransform(d), x], dim=1)
            x = vt.depthnet(x)                     # LSSTransform 直接用特征图

            depth = x[:, : vt.D].softmax(dim=1)
            feat = x[:, vt.D : (vt.D + vt.C)].permute(0, 2, 3, 1)
            return feat, depth
        
        return get_cam_feats(self.model.encoders.camera.vtransform, feat, depth)

def main():
    args = parse_args()

    __ckpt = torch.load(args.ckpt, map_location="cpu")
    model = getattr(__ckpt, "module", __ckpt)
    suffix = "int8"
    if args.fp16:
        suffix = "fp16"
        if quantize is not None:
            quantize.disable_quantization(model).apply()
        
    data = torch.load("example-data/example-data.pth", map_location="cpu")
    dev = torch.device(args.device)
    img = data["img"].data[0].to(dev)
    points = [i.to(dev) for i in data["points"].data[0]]

    camera_model = SubclassCameraModule(model)
    camera_model.to(dev).eval()
    depth = torch.zeros(len(points), img.shape[1], 1, img.shape[-2], img.shape[-1]).to(dev)

    downsample_model = model.encoders.camera.vtransform.downsample

    save_root = f"qat/onnx_{suffix}"
    os.makedirs(save_root, exist_ok=True)

    with torch.no_grad():
        camera_backbone_onnx = f"{save_root}/camera.backbone.onnx"
        camera_vtransform_onnx = f"{save_root}/camera.vtransform.onnx"
        if TensorQuantizer is not None:
            TensorQuantizer.use_fb_fake_quant = True
        torch.onnx.export(
            camera_model,
            (img, depth),
            camera_backbone_onnx,
            input_names=["img", "depth"],
            output_names=["camera_feature", "camera_depth_weights"],
            opset_version=13,
            do_constant_folding=True,
        )

        onnx_orig = onnx.load(camera_backbone_onnx)
        onnx_simp, check = simplify(onnx_orig)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(onnx_simp, camera_backbone_onnx)
        print(f"🚀 The export is completed. ONNX save as {camera_backbone_onnx} 🤗, Have a nice day~")

        if downsample_model is not None:
            downsample_model.to(dev).eval()
            downsample_in = torch.zeros(1, 80, 360, 360, device=dev)
            torch.onnx.export(
                downsample_model,
                downsample_in,
                camera_vtransform_onnx,
                input_names=["feat_in"],
                output_names=["feat_out"],
                opset_version=13,
                do_constant_folding=True,
            )
            print(f"🚀 The export is completed. ONNX save as {camera_vtransform_onnx} 🤗, Have a nice day~")
        else:
            print("downsample 为 None，跳过 camera.vtransform.onnx")

if __name__ == "__main__":
    main()
