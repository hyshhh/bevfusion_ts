"""在真实 val 帧上 dump 各 ONNX 模块的输入张量，作为 int8 PTQ 的校准数据。

产出 npz（键名 = ONNX 输入名，第 0 维是样本数）：
    img      (N, 1, 3, 256, 704)   相机预处理后的图像（camera.backbone 的输入）
    depth    (N, 1, 1, 256, 704)   全零占位（导出时的 depth 输入）
    camera   (N, 1, 64, 180, 180)  相机 BEV 特征（fuser 的输入之一）
    lidar    (N, 1, 128, 180, 180) LiDAR BEV 特征（fuser 的另一个输入）
    middle   (N, 1, 512, 180, 180) fuser 输出（head.bbox 的输入）

用法:
    python tools/dump_calib.py <config> <ckpt> \
        --ann data/data_pkl/custom_3000/infos_val.pkl \
        --num 16 --out calib/yzz8_calib16.npz
"""
import argparse
import os

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from torchpack.utils.config import configs as tp_configs

import mmdet3d  # noqa: F401  确保注册
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--ann", default="data/data_pkl/custom_3000/infos_val.pkl")
    ap.add_argument("--num", type=int, default=16, help="校准帧数")
    ap.add_argument("--out", default="calib/calib.npz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp16-store", action="store_true", default=True,
                    help="中间张量以 fp16 存（省一半空间，校准时再转回 fp32）")
    args = ap.parse_args()

    tp_configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(tp_configs), filename=args.config)

    ds_cfg = dict(cfg.data.test)
    ds_cfg["ann_file"] = args.ann
    ds = build_dataset(ds_cfg)

    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model.to(args.device), device_ids=[0]).eval()

    # 抓 fuser 的输入（camera/lidar BEV）和输出（middle）
    grabbed = {}

    def fuser_pre(mod, inputs):
        feats = inputs[0]                      # list[Tensor]
        for t in feats:
            if t.shape[1] == 64:
                grabbed["camera"] = t.detach()
            elif t.shape[1] == 128:
                grabbed["lidar"] = t.detach()

    def head_pre(mod, inputs):
        # head.bbox 的输入 = decoder neck 的输出（512 通道），不是 fuser 的直接输出
        x = inputs[0]
        if isinstance(x, (list, tuple)):
            x = x[0]
        grabbed["middle"] = x.detach()

    model.module.fuser.register_forward_pre_hook(fuser_pre)
    model.module.heads["object"].register_forward_pre_hook(head_pre)

    store = np.float16 if args.fp16_store else np.float32
    buf = {"img": [], "depth": [], "camera": [], "lidar": [], "middle": []}

    n = 0
    with torch.no_grad():
        for i in range(len(ds)):
            if n >= args.num:
                break
            data = collate([ds[i]], samples_per_gpu=1)
            try:
                model(return_loss=False, **data)
            except Exception as e:
                print(f"[skip] 帧 {i}: {type(e).__name__}: {str(e)[:80]}")
                continue
            if "middle" not in grabbed:
                continue
            img = data["img"].data[0]                     # (1, Ncam, 3, H, W)
            if img.dim() == 4:
                img = img.unsqueeze(0)
            buf["img"].append(img.detach().float().cpu().numpy().astype(np.float32))
            buf["depth"].append(np.zeros((1, img.shape[1], 1, img.shape[-2], img.shape[-1]),
                                         dtype=np.float32))
            for k in ("camera", "lidar", "middle"):
                buf[k].append(grabbed[k].float().cpu().numpy().astype(store))
            n += 1
            if n % 4 == 0:
                print(f"  已采集 {n}/{args.num} 帧")
    if n == 0:
        raise RuntimeError("没有采到任何样本")

    out = {k: np.concatenate(v, axis=0) for k, v in buf.items()}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"\n已保存 {args.out}（{n} 帧，{os.path.getsize(args.out)/2**20:.0f} MB）")
    for k, v in out.items():
        print(f"  {k:8} {v.shape} {v.dtype}")


if __name__ == "__main__":
    main()
