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
from mmcv.cnn import fuse_conv_bn
from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval
from mmcv.runner import wrap_fp16_model
import torch
import argparse
import os

# custom functional package
# pytorch_quantization 的 cuda_ext 在本环境 ABI 不匹配；fp16 导出不需要真正的量化，
# 装不上时退回到 qat/_pq_stub 的恒等实现
try:
    import pytorch_quantization  # noqa: F401
    import pytorch_quantization.tensor_quant as _tq  # noqa: F401
except Exception as _e:  # noqa: BLE001
    import sys as _sys
    # 失败的 import 可能残留半成品子模块，先清干净再指向 stub
    for _m in [m for m in list(_sys.modules) if m.startswith("pytorch_quantization")]:
        del _sys.modules[_m]
    print(f"[warn] pytorch_quantization 不可用（{str(_e)[:80]}），改用本地 stub")
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pq_stub"))

import lean.funcs as funcs
import lean.exptool as exptool
import lean.quantize as quantize 



def fuse_sparse_conv_bn(model):
    """把 spconv 卷积后面紧跟的 BatchNorm 折进卷积。

    官方 tracer 只注册了 SparseConvolution / ReLU / QuantAdd / __add__ / dense /
    reshape / permute，没有 BatchNorm——他们的模型是 BN 已折叠的（所以导出后
    conv 节点带 bias）。不折叠的话：conv 没有 bias，残差 Add 的两个操作数里
    有一个是 BN 输出（未被追踪）→ get_tensor_id 断言失败。
    """
    from mmdet3d.ops.spconv.conv import SparseConvolution

    def _fuse_pair(conv, bn):
        w = conv.weight.data
        gamma, beta = bn.weight.data, bn.bias.data
        mean, var, eps = bn.running_mean, bn.running_var, bn.eps
        scale = gamma / torch.sqrt(var + eps)
        w.mul_(scale.view(*([1] * (w.dim() - 1)), -1))
        b_conv = conv.bias.data.clone() if conv.bias is not None else 0.0
        bias = (b_conv - mean) * scale + beta
        if conv.bias is None:
            conv.bias = torch.nn.Parameter(bias.clone(), requires_grad=False)
        else:
            conv.bias.data.copy_(bias)
        return torch.nn.Identity()

    def _walk(parent):
        # 先递归所有子模块（原来的写法漏掉了最后一个子模块）
        for _, mod in list(parent.named_children()):
            _walk(mod)
        # 再在本层把 conv 后面紧跟的 BN 折进去
        children = list(parent.named_children())
        for i in range(len(children) - 1):
            mod = children[i][1]
            nxt_name, nxt = children[i + 1]
            if isinstance(mod, SparseConvolution) and isinstance(nxt, torch.nn.BatchNorm1d):
                setattr(parent, nxt_name, _fuse_pair(mod, nxt))

    _walk(model)
    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Export scn to onnx file")
    parser.add_argument("--in-channel", type=int, default=0,
                        help="SCN 输入通道数；0=从模型首层卷积自动推导")
    parser.add_argument("--ckpt", type=str, default="qat/ckpt/bevfusion_ptq.pth", help="SCN Checkpoint (scn backbone checkpoint)")
    parser.add_argument("--save", type=str, default="qat/onnx/lidar.backbone.onnx", help="output onnx")
    parser.add_argument("--inverse", action="store_true", help="Transfer the coordinate order of the index from xyz to zyx")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="cpu 仅用于调试；fp16 只在 cuda 上启用")
   
    args = parser.parse_args()
    inverse_indices = args.inverse
    if inverse_indices:
        args.save = os.path.splitext(args.save)[0] + ".zyx.onnx"
    else:
        args.save = os.path.splitext(args.save)[0] + ".xyz.onnx"
    
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    model = __ckpt = torch.load(args.ckpt, map_location="cpu")
    model = getattr(__ckpt, "module", __ckpt)
    dev = torch.device(args.device)
    model.eval().to(dev)
    if dev.type == "cuda":
        model.half()
    model = model.encoders.lidar.backbone
    fuse_sparse_conv_bn(model)          # BN 折进卷积，否则残差 Add 追不到
    # ReLU 必须改成非原地：tracer 用张量对象做 key，inplace 会让输入输出同名，
    # 导出的图节点顺序错乱 + 大量 dangling input
    n_inplace = 0
    for mod in model.modules():
        if isinstance(mod, torch.nn.ReLU) and mod.inplace:
            mod.inplace = False
            n_inplace += 1
    print(f"[info] 关闭 {n_inplace} 个原地 ReLU")

    # 自动推导输入通道：spconv 权重是 (kd, kh, kw, Cin, Cout)
    if not args.in_channel:
        args.in_channel = int(model.conv_input[0].weight.shape[3])
    print(f"[info] SCN 输入通道 = {args.in_channel}")

    if quantize is not None:
        quantize.disable_quantization(model).apply()

    # Set layer attributes
    for name, module in model.named_modules():
        module.precision = "int8"
        module.output_precision = "int8"
    
    model.conv_input.precision = "fp16"
    model.conv_out.output_precision = "fp16"

    dtype = torch.half if dev.type == "cuda" else torch.float
    # 只用 1 个体素会让 rule book 退化，indice_conv 形状推不出来；
    # 用一批随机体素（同一 batch，坐标在 sparse_shape 内）
    n_vox = 3500
    shape = model.sparse_shape
    voxels = torch.randn(n_vox, args.in_channel, device=dev, dtype=dtype)
    coors = torch.stack([
        torch.zeros(n_vox, dtype=torch.int32, device=dev),
        torch.randint(0, int(shape[2]), (n_vox,), device=dev).int(),   # z
        torch.randint(0, int(shape[1]), (n_vox,), device=dev).int(),   # y
        torch.randint(0, int(shape[0]), (n_vox,), device=dev).int(),   # x
    ], 1)
    batch_size = 1
    
    exptool.export_onnx(model, voxels, coors, batch_size, inverse_indices, args.save)

    # ---- 后处理：tracer 漏掉 conv_out 末尾的 ReLU（conv 的 in/out tensor id 同名导致）----
    import onnx
    from onnx import helper
    om = onnx.load(args.save)
    g = om.graph
    has_tail_relu = any(isinstance(c, torch.nn.ReLU) for c in model.conv_out.children())
    scatter = next((n for n in g.node if n.op_type == 'ScatterDense'), None)
    if scatter is not None and has_tail_relu:
        src = scatter.input[0]
        conv_idx = None
        for i in range(len(g.node) - 1, -1, -1):
            if src in g.node[i].output and g.node[i].op_type in ('SparseConvolution', 'Relu'):
                conv_idx = i
                break
        if conv_idx is not None and g.node[conv_idx].op_type == 'SparseConvolution':
            conv_out_name = src + '_convout'
            g.node[conv_idx].output[0] = conv_out_name
            relu = helper.make_node('Relu', [conv_out_name], [src + '_reluout'], name='relu_out_fix')
            g.node.insert(conv_idx + 1, relu)
            scatter.input[0] = src + '_reluout'
            onnx.save(om, args.save)
            print('[fix] conv_out 后补 ReLU: %s -> %s -> %s' % (conv_out_name, src + '_reluout', scatter.name))
        else:
            print('[fix] 末尾已是 ReLU，跳过')
