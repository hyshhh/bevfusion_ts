"""按部署需要拆分子模块导出 ONNX。

整模型无法导出：LiDAR 稀疏卷积（spconv）与 bev_pool 是自定义 CUDA 算子。

可导出的模块：

    camera       img[1,3,256,704]                       -> feat[1,256,32,88]
    depthnet     feat[1,256,32,88]                      -> depth_context[1,D+C,32,88]
    bev_encoder  cam_bev[1,64,180,180] + lidar_bev[1,128,180,180] -> [1,512,180,180]

不可导出（由部署端用 CUDA 实现）：LiDAR 分支、bev_pool。

用法:
    python tools/export_modules.py configs/YZdata/camera+lidar_180x180_spconv.yaml \
        runs/yz_spconv_z21/epoch_100.pth output/

    # 只导某几个
    python tools/export_modules.py <cfg> <ckpt> output/ --modules depthnet
"""
import argparse
import os

import numpy as np
import onnx
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from torchpack.utils.config import configs as tp_configs

from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


class CameraEncoder(torch.nn.Module):
    """相机 backbone + neck。"""

    def __init__(self, model):
        super().__init__()
        self.backbone = model.encoders["camera"]["backbone"]
        self.neck = model.encoders["camera"]["neck"]

    def forward(self, img):
        x = self.backbone(img)
        x = self.neck(x)
        return x if isinstance(x, torch.Tensor) else x[0]


class DepthNet(torch.nn.Module):
    """LSS 的深度/上下文分支，输出 [B, D+C, H, W]（softmax 留给部署端）。"""

    def __init__(self, model):
        super().__init__()
        self.depthnet = model.encoders["camera"]["vtransform"].depthnet

    def forward(self, feat):
        return self.depthnet(feat)


class BevEncoder(torch.nn.Module):
    """融合 + BEV 主干 + FPN，输出检测头输入。"""

    def __init__(self, model):
        super().__init__()
        self.fuser = model.fuser
        self.backbone = model.decoder["backbone"]
        self.neck = model.decoder["neck"]

    def forward(self, cam_bev, lidar_bev):
        x = self.fuser([cam_bev, lidar_bev])
        x = self.backbone(x)
        x = self.neck(x)
        return x if torch.is_tensor(x) else x[0]   # SECONDFPN 返回 [tensor]


def export_onnx(net, dummy, out_path, input_names, output_names, opset, dynamic_batch, simplify=True):
    torch.onnx.export(
        net,
        dummy,
        out_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={n: {0: "batch"} for n in input_names} if dynamic_batch else None,
    )
    m = onnx.load(out_path)
    try:
        if not simplify:
            raise ImportError
        from onnxsim import simplify

        simplified, ok = simplify(m)
        if ok:
            m = simplified
            onnx.save(m, out_path)
    except Exception as e:  # noqa: BLE001
        print(f"  onnxsim 跳过: {type(e).__name__}")
    onnx.checker.check_model(m)
    print(f"  算子: {sorted({n.op_type for n in m.graph.node})}")
    return m


def check_numeric(net, dummy, out_path):
    import onnxruntime as ort

    net.eval()  # torch.onnx.export 可能把模块留在 train 模式，BN 会用 batch 统计量
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    feeds = {}
    for inp, d in zip(sess.get_inputs(), dummy):
        feeds[inp.name] = d.detach().cpu().numpy()
    with torch.no_grad():
        ref = net(*dummy).detach().cpu().numpy()
    got = sess.run(None, feeds)[0]
    err = float(np.abs(ref - got).max())
    rel = err / (np.abs(ref).max() + 1e-9)
    print(f"  形状 {ref.shape} | PyTorch vs ONNX 最大差 {err:.3e}（相对 {rel:.2e}）")


def main():
    parser = argparse.ArgumentParser(description="export deployable BEVFusion sub-modules")
    parser.add_argument("config")
    parser.add_argument("checkpoint", nargs="?", default=None)
    parser.add_argument("outdir", nargs="?", default="output")
    parser.add_argument("--modules", nargs="+", default=["camera", "depthnet", "bev_encoder"])
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="导出设备；GPU 被训练占用时用 cpu")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    tp_configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(tp_configs), filename=args.config)
    # 有 checkpoint 时跳过 torchvision 预训练（权重会被完全覆盖）
    cfg.model.encoders.camera.backbone.init_cfg = None
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location="cpu")
    dev = torch.device(args.device)
    model.eval().to(dev)

    h, w = cfg.image_size
    fh, fw = h // 8, w // 8
    vt = model.encoders["camera"]["vtransform"]
    D, C = vt.D, vt.C
    bev = (
        int((cfg.point_cloud_range[3] - cfg.point_cloud_range[0]) / cfg.bev_stride),
        int((cfg.point_cloud_range[4] - cfg.point_cloud_range[1]) / cfg.bev_stride),
    )
    cam_ch = vt.C          # LSSTransform 里 BEV 通道名是 C
    # depthnet 的输入是 neck 的输出（相��特征图），不是 BEV
    neck_ch = model.encoders["camera"]["neck"].out_channels
    lidar_ch = model.fuser.in_channels[1]

    for name in args.modules:
        print(f"[{name}]")
        out_path = os.path.join(args.outdir, f"{name}.onnx")
        with torch.no_grad():
            if name == "camera":
                net = CameraEncoder(model)
                dummy = (torch.randn(1, 3, h, w, device=args.device),)
                export_onnx(net, dummy, out_path, ["img"], ["feat"], args.opset, args.dynamic_batch,
                            simplify=not args.no_simplify)
            elif name == "depthnet":
                net = DepthNet(model)
                dummy = (torch.randn(1, neck_ch, fh, fw, device=args.device),)
                export_onnx(net, dummy, out_path, ["feat"], ["depth_context"], args.opset,
                            args.dynamic_batch, simplify=not args.no_simplify)
                print(f"  D={D} C={C} 总通道 {D + C}（前 {D} 深度 logits，后 {C} 上下文）")
            elif name == "bev_encoder":
                net = BevEncoder(model)
                dummy = (
                    torch.randn(1, cam_ch, bev[1], bev[0], device=args.device),
                    torch.randn(1, lidar_ch, bev[1], bev[0], device=args.device),
                )
                export_onnx(net, dummy, out_path, ["cam_bev", "lidar_bev"], ["bev_feat"],
                            args.opset, args.dynamic_batch, simplify=not args.no_simplify)
            else:
                raise ValueError(f"unknown module: {name}")
            check_numeric(net, dummy, out_path)
        print(f"  -> {out_path}\n")


if __name__ == "__main__":
    main()
