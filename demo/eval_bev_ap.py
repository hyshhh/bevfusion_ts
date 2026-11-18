"""BEV AP 评测（KITTI 风格：旋转框 BEV IoU 匹配）。

    torchpack dist-run -np 1 python tools/test.py \
        configs/YZdata/camera+lidar_180x180_spconv.yaml \
        runs/yz_z8_ft/epoch_6.pth --out result.pkl

    python demo/eval_bev_ap.py \
        configs/YZdata/camera+lidar_180x180_spconv.yaml \
        result.pkl --iou 0.3 --ranges 0-30 30-50 50-80

所有计算都在 mmdet3d/evaluation/bev_ap.py 里，训练中每 N 轮自动评测
（见 configs 的 bev_ap_eval / mmdet3d/runner/hooks/bev_ap_eval.py）复用的
是同一套实现，因此离线结果和训练日志里的数字可以直接对齐。
"""
import argparse
import pickle

from mmdet3d.evaluation.bev_ap import compute_bev_ap, format_ap_table, load_preds


def _load_cfg(path):
    from mmcv import Config
    from mmdet3d.utils import recursive_eval
    from torchpack.utils.config import configs

    configs.load(path, recursive=True)
    return Config(recursive_eval(configs), filename=path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("result")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--ranges", nargs="+", default=["0-30", "30-50", "50-80"])
    ap.add_argument("--score", type=float, default=0.001)
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    classes = list(cfg.object_classes)
    with open(cfg.ann_val, "rb") as f:
        infos = pickle.load(f)["infos"]
    preds = load_preds(args.result)
    assert len(infos) == len(preds), f"数量不匹配: {len(infos)} 帧 vs {len(preds)} 条预测"

    print(f"配置: {args.config}")
    print(f"验证集: {cfg.ann_val} ({len(infos)} 帧)")
    print(f"匹配阈值: BEV IoU >= {args.iou} | 置信度 >= {args.score}")

    res = compute_bev_ap(infos, classes, preds, iou=args.iou,
                         ranges=args.ranges, score_thr=args.score)
    print("\n" + format_ap_table(classes, res["ranges"], res["ap"],
                                 mAP=res["mAP"]))
    print("\n注：AP 按 all-point 插值；距离取框中心到雷达的距离")


if __name__ == "__main__":
    main()
