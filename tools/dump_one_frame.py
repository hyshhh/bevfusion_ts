"""用训练代码跑单帧推理，打印 scores_3d.max() 并 dump head 输入特征。

用法:
    python tools/dump_one_frame.py configs/YZdata/camera+lidar_180x180_spconv.yaml \
        runs/yz_spconv_z21/epoch_100.pth 000004 [--device cuda|cpu]

产出:
    middle_pt.bin     head 输入特征 (1,512,180,180) float16，C 连续排布
"""
import argparse
import os

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import DataContainer, scatter
from mmcv.runner import load_checkpoint
from torchpack.utils.config import configs as tp_configs

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


CAM_MATS = ("camera_intrinsics", "camera2ego", "lidar2camera",
           "camera2lidar", "lidar2image", "img_aug_matrix")
LIDAR_MATS = ("lidar2ego", "lidar_aug_matrix")


def make_batch(sample, dev):
    """把单帧样本组装成 batch（补上 batch 维），相当于 dataloader+scatter 的结果。"""
    out = {}
    for k, v in sample.items():
        d = v.data if isinstance(v, DataContainer) else v
        if k == "img":                       # (N,C,H,W) -> (1,N,C,H,W)
            out[k] = d.unsqueeze(0).to(dev)
        elif k == "points":                  # 单个张量 -> [张量]
            out[k] = [d.to(dev)]
        elif k in CAM_MATS:                  # (N,4,4) -> (1,N,4,4)
            out[k] = d.unsqueeze(0).to(dev)
        elif k in LIDAR_MATS:                # (4,4) -> (1,4,4)
            out[k] = d.unsqueeze(0).to(dev)
        elif k in ("gt_bboxes_3d", "gt_labels_3d", "metas"):
            out[k] = [d]
        elif torch.is_tensor(d):
            out[k] = d.unsqueeze(0).to(dev)
        else:
            out[k] = [d]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("frame", help="帧号，如 000004")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out", default="middle_pt.bin")
    ap.add_argument("--fp16", action="store_true", help="半精度推理，显存紧张时用")
    args = ap.parse_args()

    tp_configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(tp_configs), filename=args.config)
    cfg.model.encoders.camera.backbone.init_cfg = None

    ds = build_dataset(cfg.data.val, dict(test_mode=True))
    idx = next(i for i, x in enumerate(ds.data_infos) if args.frame in x["lidar_path"])
    print(f"[info] 帧 {args.frame} 在 val 集索引 {idx}: {ds.data_infos[idx]['lidar_path']}")
    sample = ds[idx]

    dev = torch.device(args.device)
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval().to(dev)
    if args.fp16:
        from mmcv.runner import wrap_fp16_model
        wrap_fp16_model(model)
        print("[info] fp16 推理")
    print(f"[info] 权重 {os.path.basename(args.checkpoint)} | 设备 {args.device}")

    batch = make_batch(sample, dev)
    if args.fp16:
        batch = {k: ([x.half() if torch.is_tensor(x) else x for x in v] if isinstance(v, list)
                     else (v.half() if torch.is_tensor(v) and v.is_floating_point() else v))
                 for k, v in batch.items()}
    img_metas = batch["metas"]

    # 直接走模型自己的推理路径，同时用 hook 抓 head 的输入特征
    captured = {}

    def _grab_head_input(module, inputs):
        x = inputs[0]
        if isinstance(x, (list, tuple)):     # SECONDFPN 返回 [tensor]
            x = x[0]
        captured["middle"] = x.detach()

    for k in ("img", "points", "camera2ego", "lidar2image"):
        v = batch[k]
        if torch.is_tensor(v):
            print(f"[dbg] {k}: tensor {tuple(v.shape)}")
        else:
            print(f"[dbg] {k}: {type(v).__name__} len={len(v)} "
                  f"{[tuple(x.shape) if torch.is_tensor(x) else type(x).__name__ for x in v][:3]}")

    h = model.heads["object"].register_forward_pre_hook(_grab_head_input)
    with torch.no_grad():
        results = model(return_loss=False, rescale=False, **batch)
    h.remove()

    middle = captured["middle"]
    print(f"[info] head 输入 {tuple(middle.shape)} | min {float(middle.min()):.4f} "
          f"max {float(middle.max()):.4f} mean {float(middle.mean()):.4f}")

    arr = middle.cpu().numpy().astype("float16")
    arr.tofile(args.out)
    print(f"[save] {args.out}  {arr.shape} {arr.dtype}  {arr.nbytes/2**20:.1f} MB")

    # results: list[list[dict]]，取 scores_3d
    all_scores = []
    for per_img in results:
        # results 可能是 [[dict], ...] 或 [dict, ...]
        items = [per_img] if isinstance(per_img, dict) else per_img
        for ret in items:
            s3 = ret.get("scores_3d", None)
            if s3 is not None and len(s3):
                all_scores.append(torch.as_tensor(s3).float().cpu())
            print(f"[info] 该帧框数 {len(s3) if s3 is not None else 0} | "
                  f"labels {torch.as_tensor(ret['labels_3d']).cpu().tolist()[:10]} "
                  f"| bbox_3d 形状 {tuple(torch.as_tensor(ret['boxes_3d'].tensor).shape)}")

    if all_scores:
        s = torch.cat(all_scores)
        print()
        print(f"=== scores_3d.max() = {float(s.max()):.6f} ===")
        print(f"=== scores_3d 前 10 = {[round(float(x), 4) for x in s.flatten()[:10]]} ===")
        print(f"[info] 框数 {len(s)} | 分数 >0.3 的 {int((s > 0.3).sum())} | "
              f">0.1 的 {int((s > 0.1).sum())} | 中位 {float(s.median()):.4f}")
    else:
        print("=== 没有任何框输出 ===")


if __name__ == "__main__":
    main()
