"""一条命令导出全部部署产物到一个文件夹。

    python tools/export_all.py \
        configs/YZdata/camera+lidar_180x180_spconv.yaml \
        runs/yz_spconv_z21/epoch_100.pth deploy/

产出：

    deploy/
    ├── camera.onnx            img      -> 相机特征
    ├── depthnet.onnx          特征     -> 深度 logits + 上下文
    ├── bev_encoder.onnx       两路 BEV -> 检测头输入
    ├── head.onnx              BEV 特征 -> 6 组预测张量
    ├── lidar/
    │   ├── lidar_weights.npz  LiDAR 稀疏卷积权重（原生 CUDA 实现用）
    │   └── layers.json        层清单（名字/形状/顺序）
    ├── manifest.json          全部尺寸常量
    └── README.md              部署说明

LiDAR 分支无法由 PyTorch 直接导 ONNX（spconv 是自定义 CUDA 算子、形状随数据变化），
需要部署框架自己的 export-scn 脚本或原生 CUDA 实现，本脚本导出其权重。
"""
import argparse
import json
import os
import shutil

import numpy as np
import onnx
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from torchpack.utils.config import configs as tp_configs

from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


# ---------------------------------------------------------------- 模块包装
class CameraEncoder(torch.nn.Module):
    def __init__(self, model, out_shape):
        super().__init__()
        self.backbone = model.encoders["camera"]["backbone"]
        self.neck = model.encoders["camera"]["neck"]
        self.out_shape = tuple(out_shape)

    def forward(self, img):
        x = self.neck(self.backbone(img))
        x = x if torch.is_tensor(x) else x[0]
        # 显式 reshape：FPN 里的 Resize 会让导出器把输出形状标成动态
        return x.reshape(*self.out_shape)


class DepthNet(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.depthnet = model.encoders["camera"]["vtransform"].depthnet

    def forward(self, feat):
        return self.depthnet(feat)


class BevEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fuser = model.fuser
        self.backbone = model.decoder["backbone"]
        self.neck = model.decoder["neck"]

    def forward(self, cam_bev, lidar_bev):
        x = self.neck(self.backbone(self.fuser([cam_bev, lidar_bev])))
        return x if torch.is_tensor(x) else x[0]


class Head(torch.nn.Module):
    HEAD_KEYS = ["heatmap", "center", "height", "dim", "rot", "vel"]

    def __init__(self, model):
        super().__init__()
        self.head = model.heads["object"]

    def forward(self, bev_feat):
        out = self.head(bev_feat, None)[0][0]
        # 显式 reshape：直接输出会让导出器把 batch 维推断成 num_proposals，
        # 导致 ONNX 里声明的形状变成 [200, ...]
        P = self.head.num_proposals
        dims = [self.head.num_classes, 2, 1, 3, 2, 2]
        return tuple(out[k].reshape(1, d, P) for k, d in zip(self.HEAD_KEYS, dims))


# ---------------------------------------------------------------- 导出工具
def export_onnx(net, dummy, path, input_names, output_names, opset, simplify=True):
    torch.onnx.export(
        net, dummy, path, input_names=input_names, output_names=output_names,
        opset_version=opset, do_constant_folding=True,
    )
    m = onnx.load(path)
    if simplify:
        try:
            from onnxsim import simplify as _simplify

            sm, ok = _simplify(m)
            if ok:
                onnx.save(sm, path)
                m = sm
        except Exception:
            pass
    onnx.checker.check_model(m)
    return m


def verify(net, dummy, path, tag):
    import onnxruntime as ort

    net.eval()  # 导出后模块可能被留在 train 模式，BN 会用 batch 统计量
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feeds = {i.name: d.detach().cpu().numpy() for i, d in zip(sess.get_inputs(), dummy)}
    with torch.no_grad():
        ref = net(*dummy)
    ref = ref if isinstance(ref, (tuple, list)) else (ref,)
    got = sess.run(None, feeds)
    err = max(float(np.abs(r.detach().cpu().numpy() - g).max()) for r, g in zip(ref, got))
    print(f"    {tag:12s} 数值最大差 {err:.3e}")


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="导出完整部署包")
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("outdir", nargs="?", default="deploy")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    ap.add_argument("--skip-lidar", action="store_true", help="不导出 lidar 权重")
    ap.add_argument("--simplify", action="store_true",
                    help="用 onnxsim 简化；注意它可能把 batch 维改错，默认关闭")
    args = ap.parse_args()

    outdir = args.outdir
    lidar_dir = os.path.join(outdir, "lidar")
    os.makedirs(lidar_dir, exist_ok=True)

    tp_configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(tp_configs), filename=args.config)
    cfg.model.encoders.camera.backbone.init_cfg = None  # checkpoint 会覆盖
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval()
    dev = torch.device(args.device)
    model.to(dev)

    h, w = cfg.image_size
    fh, fw = h // 8, w // 8
    vt = model.encoders["camera"]["vtransform"]
    D, C = vt.D, vt.C
    neck_ch = model.encoders["camera"]["neck"].out_channels
    bev_w = int((cfg.point_cloud_range[3] - cfg.point_cloud_range[0]) / cfg.bev_stride)
    bev_h = int((cfg.point_cloud_range[4] - cfg.point_cloud_range[1]) / cfg.bev_stride)
    lidar_ch = model.fuser.in_channels[1]
    head_ch = model.decoder["neck"].out_channels
    head_ch = sum(head_ch) if isinstance(head_ch, (list, tuple)) else head_ch

    jobs = [
        ("camera.onnx", CameraEncoder(model, (1, neck_ch, fh, fw)),
         (torch.randn(1, 3, h, w, device=dev),),
         ["img"], ["feat"]),
        ("depthnet.onnx", DepthNet(model), (torch.randn(1, neck_ch, fh, fw, device=dev),),
         ["feat"], ["depth_context"]),
        ("bev_encoder.onnx", BevEncoder(model),
         (torch.randn(1, C, bev_h, bev_w, device=dev),
          torch.randn(1, lidar_ch, bev_h, bev_w, device=dev)),
         ["cam_bev", "lidar_bev"], ["bev_feat"]),
        ("head.onnx", Head(model), (torch.randn(1, head_ch, bev_h, bev_w, device=dev),),
         ["bev_feat"], Head.HEAD_KEYS),
    ]

    manifest = {
        "config": args.config,
        "checkpoint": os.path.abspath(args.checkpoint),
        "image_size": [h, w],
        "feature_size": [fh, fw],
        "point_cloud_range": list(cfg.point_cloud_range),
        "voxel_size": list(cfg.voxel_size),
        "voxel_grid": list(cfg.voxel_grid),
        "bev_size": [bev_h, bev_w],
        "zbound": list(cfg.model.encoders.camera.vtransform.zbound),
        "dbound": list(cfg.model.encoders.camera.vtransform.dbound),
        "bev_stride": float(cfg.bev_stride),
        "camera_bev_channels": C,
        "lidar_bev_channels": lidar_ch,
        "depth_bins": D,
        "depth_context_channels": D + C,
        "head_in_channels": head_ch,
        "head_num_proposals": getattr(model.heads["object"], "num_proposals", None),
        "num_classes": int(model.heads["object"].num_classes),
        "classes": list(cfg.object_classes),
        "fp16": False,
        "modules": {},
    }

    for name, net, dummy, in_names, out_names in jobs:
        print(f"[{name}]")
        path = os.path.join(outdir, name)
        net = net.to(dev).eval()
        with torch.no_grad():
            m = export_onnx(net, dummy, path, in_names, out_names, args.opset,
                            simplify=args.simplify)
            verify(net, dummy, path, "onnx")
        ios = {
            "inputs": [{"name": i.name,
                        "shape": [d.dim_value for d in i.type.tensor_type.shape.dim]}
                       for i in m.graph.input],
            "outputs": [{"name": o.name,
                         "shape": [d.dim_value for d in o.type.tensor_type.shape.dim]}
                        for o in m.graph.output],
            "nodes": len(m.graph.node),
        }
        manifest["modules"][name] = ios
        print(f"    输入 {[i['shape'] for i in ios['inputs']]} -> "
              f"输出 {[o['shape'] for o in ios['outputs']]} | 节点 {ios['nodes']}")

    # ---------------- LiDAR 权重 ----------------
    if not args.skip_lidar:
        print("[lidar]")
        sd = model.state_dict()
        prefix = "encoders.lidar.backbone."
        layers = []
        arrays = {}
        for k, v in sd.items():
            if not k.startswith(prefix):
                continue
            short = k[len(prefix):]
            arrays[short] = v.detach().cpu().numpy()
            layers.append({"name": short, "shape": list(v.shape)})
        np.savez_compressed(os.path.join(lidar_dir, "lidar_weights.npz"), **arrays)
        with open(os.path.join(lidar_dir, "layers.json"), "w") as f:
            json.dump(
                {"sparse_shape": list(cfg.model.encoders.lidar.backbone.sparse_shape),
                 "encoder_channels": cfg.model.encoders.lidar.backbone.encoder_channels,
                 "encoder_paddings": cfg.model.encoders.lidar.backbone.encoder_paddings,
                 "layers": layers},
                f, indent=2,
            )
        mb = sum(a.nbytes for a in arrays.values()) / 2 ** 20
        print(f"    {len(layers)} 个张量 ({mb:.1f} MB) -> lidar/lidar_weights.npz")
        print("    注意：LiDAR 稀疏卷积不能由 PyTorch 直接导 ONNX，需部署框架的 "
              "export-scn（如 qat/export-scn.py）或原生 CUDA 实现")
        manifest["modules"]["lidar"] = {
            "format": "npz + layers.json",
            "num_tensors": len(layers),
            "note": "spconv 不自带 ONNX 支持，需部署框架的 export-scn 或原生实现",
        }

    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "..", "configs", "YZdata", "DEPLOY_README.md"),
        os.path.join(outdir, "README.md"),
    ) if os.path.exists(
        os.path.join(os.path.dirname(__file__), "..", "configs", "YZdata", "DEPLOY_README.md")
    ) else None

    print(f"\n完成 -> {os.path.abspath(outdir)}")
    total = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, fs in os.walk(outdir) for f in fs
    )
    print(f"总计 {total / 2 ** 20:.1f} MB")


if __name__ == "__main__":
    main()
