"""QAT 模型的校准/导出准备：把 pytorch_quantization 的 quantizer 插进模型，
跑若干真实帧统计每层 amax，然后把整模型 pickle 出来给 qat/export-*.py 用。

两种用途：
  1) 只想快速拿一个 int8 模型（不做 QAT 训练）→ 本脚本即可；
  2) QAT 训练之后想重新校准/导出 → 把 --load-from 指向 QAT 的 ckpt。

用法:
    python tools/qat_calibrate.py <config> <ckpt> \
        --num 32 --out qat/ckpt/deploy_qat.pth
"""
import argparse
import os
import sys

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from torchpack.utils.config import configs as tp_configs

import mmdet3d  # noqa: F401
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--ann", default="data/data_pkl/custom_3000/infos_train.pkl")
    ap.add_argument("--num", type=int, default=32, help="校准帧数")
    ap.add_argument("--out", default="qat/ckpt/deploy_qat.pth")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp16", action="store_true", help="导出 fp16（int8 不要开）")
    args = ap.parse_args()

    tp_configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(tp_configs), filename=args.config)

    # 必须在 build_model 之前 patch（Conv1d/2d/3d、Linear → 带 fake-quant 的版本）
    from tools.train_qat import enable_qat
    quant_nn = enable_qat()

    ds = build_dataset(dict(cfg.data.test, ann_file=args.ann))
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model.to(args.device), device_ids=[0]).eval()
    raw = model.module

    # 打开全部 quantizer 的校准模式，前向若干真实帧统计 amax
    n_q = 0
    for m in raw.modules():
        if "TensorQuantizer" in type(m).__name__:
            m.enable_calib()
            m.use_fb_fake_quant = True
            n_q += 1
    print(f"[QAT] quantizer {n_q} 个，开始校准 {args.num} 帧")

    ok = 0
    with torch.no_grad():
        for i in range(args.num):
            data = collate([ds[i]], samples_per_gpu=1)
            try:
                # MMDataParallel 负责把 DataContainer 搬到 GPU，
                # 且 BEVFusion.forward 需要 camera2ego 等参数，必须传整份 data
                model(return_loss=False, **data)
                ok += 1
            except Exception as e:
                print(f"[skip] 帧 {i}: {type(e).__name__}: {str(e)[:80]}")
            if (i + 1) % 8 == 0:
                print(f"  校准 {i+1}/{args.num}（成功 {ok}）")
    if ok == 0:
        raise RuntimeError("没有任何帧校准成功")

    # 收起校准模式，导出用 fake-quant（ONNX 里会变成 Q/DQ 节点）
    for m in raw.modules():
        if "TensorQuantizer" in type(m).__name__:
            m.disable_calib()
            m.use_fb_fake_quant = True
    if args.fp16:
        from mmcv.runner import wrap_fp16_model
        wrap_fp16_model(raw)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    raw = raw.cpu()
    model = raw
    model._deploy_cfg_path = os.path.abspath(args.config)
    model._deploy_cfg = cfg
    model._deploy_src_ckpt = os.path.abspath(args.checkpoint)
    head = model.heads["object"] if "object" in model.heads else None
    if head is not None:
        model._deploy_per_class_topk = list(getattr(head, "per_class_topk", 0)) \
            if isinstance(getattr(head, "per_class_topk", 0), (list, tuple)) else 0
        model._deploy_num_proposals = int(getattr(head, "num_proposals", 0))
    torch.save(model, args.out)
    print(f"[QAT] 已保存 {args.out}（{os.path.getsize(args.out)/2**20:.0f} MB）")
    print("      下一步: python qat/export-transfuser.py --ckpt=%s" % args.out)


if __name__ == "__main__":
    main()
