"""评测一个 QAT（带 fake-quant）的 checkpoint：模拟 int8 的 mAP。

与 tools/test.py 的区别：必须在 build_model 之前 enable_qat()，否则加载进来的是
无 quantizer 的普通模型，评出来的就不是量化后的精度。

用法:
    python tools/eval_qat.py <config> <ckpt> [--ann ...] [--iou 0.3] [--score-thr 0.05]
"""
import argparse
import os
import pickle
import time

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from torchpack.utils.config import configs as tp_configs

import mmdet3d  # noqa: F401
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.evaluation.bev_ap import compute_bev_ap, format_ap_table
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


def evaluate_model(model, cfg, ann=None, iou=0.3, score_thr=0.05,
                   ranges=("0-30", "30-50", "50-80", "80+", "0-80"), tag=""):
    """对已构建/已校准好的模型跑一遍 val + BEV-AP，打印逐类表格并返回结果。"""
    from mmcv.parallel import MMDataParallel as _MMDP
    from mmdet3d.datasets import build_dataloader, build_dataset

    ann = ann or cfg.ann_val
    ds = build_dataset(dict(cfg.data.test, ann_file=ann))
    loader = build_dataloader(ds, 1, 2, dist=False, shuffle=False)
    wrapper = model if isinstance(model, _MMDP) else _MMDP(model, device_ids=[0])
    wrapper.eval()

    t0 = time.time()
    results = single_gpu_test(wrapper, loader)
    infer_time = time.time() - t0

    with open(ann, "rb") as f:
        infos = pickle.load(f)["infos"]
    classes = list(cfg.object_classes)
    if len(results) != len(infos):
        raise RuntimeError(f"预测 {len(results)} 条，验证集 {len(infos)} 帧，不一致")

    t1 = time.time()
    res = compute_bev_ap(infos, classes, results, iou=iou,
                         ranges=list(ranges), score_thr=score_thr)
    print(f"[eval] 推理 {infer_time:.1f}s，AP 计算 {time.time()-t1:.1f}s" + (f" | {tag}" if tag else ""))
    print(format_ap_table(classes, res["ranges"], res["ap"], res["mAP"]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--ann", default=None, help="默认用 cfg.ann_val")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--score-thr", type=float, default=0.05)
    ap.add_argument("--ranges", nargs="+", default=["0-30", "30-50", "50-80", "80+", "0-80"])
    ap.add_argument("--qat", action="store_true", default=True,
                    help="按 QAT 方式建模型（默认开）；--no-qat 关掉")
    ap.add_argument("--no-qat", dest="qat", action="store_false")
    ap.add_argument("--calib-method", default="histogram", choices=["max", "histogram"])
    ap.add_argument("--calib-percentile", type=float, default=99.99)
    ap.add_argument("--calibrate", type=int, default=0,
                    help="评测前先用 N 帧校准 amax（>0 时才做；ckpt 里没存 amax 必须给）")
    args = ap.parse_args()

    tp_configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(tp_configs), filename=args.config)

    if args.qat:
        from tools.train_qat import enable_qat
        # 必须把校准方式传进去：quantizer 在构造时就绑定 calibrator，
        # 构造之后再改 --calib-method 是无效的
        enable_qat(calib_method=args.calib_method)
        print("[eval] QAT 模式：模型含 fake-quant（等价模拟 int8）")
    else:
        print("[eval] 普通模式：fp32/fp16 精度")

    ann = args.ann or cfg.ann_val
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model.cuda(), device_ids=[0]).eval()

    t0 = time.time()
    if args.calibrate > 0:
        from mmcv.parallel import MMDataParallel as _MMDP
        raw = model.module
        _ds = build_dataset(dict(cfg.data.test, ann_file=ann))
        loader = build_dataloader(_ds, 1, 2, dist=False, shuffle=False)
        qs = [m for m in raw.modules() if "TensorQuantizer" in type(m).__name__]
        for q in qs:
            q.enable_calib()
        print(f"[eval] 用 {args.calibrate} 帧校准 {len(qs)} 个 quantizer 的 amax")
        with torch.no_grad():
            for i, data in enumerate(loader):
                if i >= args.calibrate:
                    break
                try:
                    _MMDP(raw, device_ids=[0])(return_loss=False, **data)
                except Exception as e:
                    print(f"  [skip] 帧 {i}: {type(e).__name__}: {str(e)[:60]}")
        for q in qs:
            if getattr(q, "_calibrator", None) is not None:
                try:
                    q.load_calib_amax()
                except TypeError:
                    q.load_calib_amax(method="percentile", percentile=args.calib_percentile)
            q.disable_calib()
            q.use_fb_fake_quant = True
        # histogram 校准返回的 amax 在 CPU 上，必须搬回 GPU，否则下一步前向报
        # "module must have its parameters and buffers on device cuda:0"
        raw.cuda()
        print("[eval] 校准完成，amax 已载入")
    res = evaluate_model(model, cfg, ann, iou=args.iou, score_thr=args.score_thr,
                         ranges=args.ranges,
                         tag=f"ckpt={os.path.basename(args.checkpoint)} QAT={args.qat}")


if __name__ == "__main__":
    main()
