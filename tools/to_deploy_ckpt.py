"""把训练产出的 checkpoint 转成 qat/ 导出脚本能吃的格式。

qat/export-*.py 期望的输入是：

    torch.load(ckpt)         ->  带 .module 的对象（DataParallel 包装的整模型）
    example-data/*.pth       ->  一个 batch 的字典（img / points 是 DataContainer）

而训练产出的是 {state_dict, meta, epoch} 字典，两者不兼容。
本脚本生成这两样东西：

    qat/ckpt/deploy_<name>.pth
    example-data/example-data.pth

用法:
    python tools/to_deploy_ckpt.py \
        configs/YZdata/camera+lidar_180x180_spconv.yaml \
        runs/yz_spconv_z21/epoch_100.pth
"""
import argparse
import os

import torch
from mmcv import Config
from mmcv.parallel import DataContainer
from mmcv.runner import load_checkpoint
from torchpack.utils.config import configs as tp_configs

import mmdet3d  # noqa: F401  确保注册
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


CAM_KEYS = ("camera_intrinsics", "camera2ego", "lidar2camera",
            "camera2lidar", "lidar2image", "img_aug_matrix")


def expand_cams(batch, n):
    """把单相机 batch 复制成 n 相机（仅用于导出 dummy 数据）。"""
    out = dict(batch)
    for k in ("img",) + CAM_KEYS:
        if k not in out:
            continue
        c = out[k]
        t = c.data[0] if isinstance(c.data, (list, tuple)) else c.data
        rep = t.repeat(1, n, *([1] * (t.dim() - 2)))
        out[k] = DataContainer([rep])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default="qat/ckpt")
    ap.add_argument("--example", default="example-data/example-data.pth")
    ap.add_argument("--num-cams", type=int, default=0,
                    help="把 example-data 扩成 N 相机（0=保持原样）")
    args = ap.parse_args()

    tp_configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(tp_configs), filename=args.config)

    # 1) 整模型对象
    cfg.model.encoders.camera.backbone.init_cfg = None
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval()
    os.makedirs(args.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    ckpt_path = os.path.join(args.out, f"deploy_{name}.pth")
    # 把配置来源烘进模型对象：以后导出只要这个 pth 就够了，
    # 也能在导出时核对"权重是用哪份 yaml 训的"
    model._deploy_cfg_path = os.path.abspath(args.config)
    model._deploy_cfg = cfg
    model._deploy_src_ckpt = os.path.abspath(args.checkpoint)
    head = model.heads["object"] if "object" in model.heads else None
    if head is not None:
        model._deploy_per_class_topk = list(getattr(head, "per_class_topk", 0)) \
            if isinstance(getattr(head, "per_class_topk", 0), (list, tuple)) \
            else int(getattr(head, "per_class_topk", 0))
        model._deploy_num_proposals = int(getattr(head, "num_proposals", 0))

    # 裸模型保存；qat 脚本用 getattr(obj, "module", obj) 取回，两种格式都兼容
    torch.save(model, ckpt_path)
    print(f"模型 checkpoint -> {ckpt_path}（{os.path.getsize(ckpt_path) / 2 ** 20:.0f} MB）")
    print(f"     cfg        : {model._deploy_cfg_path}")
    print(f"     per_class_topk: {getattr(model, '_deploy_per_class_topk', None)} | "
          f"num_proposals: {getattr(model, '_deploy_num_proposals', None)}")

    # 2) 一个真实 batch 当 example-data（qat 脚本用它做 dummy 输入）
    ds = build_dataset(cfg.data.train)
    loader = build_dataloader(ds, 1, 0, dist=False, shuffle=False)
    # 直接保存 dataloader 的 batch：img.data 已经是 [tensor(1,1,3,H,W)]，
    # points.data 是 [[tensor(N,4)]]，正好是 qat 脚本期望的结构
    batch = next(iter(loader))
    os.makedirs(os.path.dirname(args.example), exist_ok=True)
    if args.num_cams and args.num_cams > 1:
        batch = expand_cams(batch, args.num_cams)
    torch.save(batch, args.example)
    print(f"example-data  -> {args.example}")
    for k, v in batch.items():
        if isinstance(v, DataContainer):
            d = v.data[0] if isinstance(v.data, (list, tuple)) else v.data
            shape = tuple(d.shape) if torch.is_tensor(d) else type(d).__name__
            print(f"    {k:18s} {shape}")
        elif torch.is_tensor(v):
            print(f"    {k:18s} {tuple(v.shape)}")


if __name__ == "__main__":
    main()
