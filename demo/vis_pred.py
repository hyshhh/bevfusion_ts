"""预测结果可视化：相机 + BEV 双视图，数据来自 tools/test.py 的 --out 文件。

    torchpack dist-run -np 1 python tools/test.py \
        configs/SeePerSea/camera+lidar_resnet50.yaml \
        runs/seepersea/epoch_96.pth --out result_ss.pkl

    python demo/vis_pred.py \
        configs/SeePerSea/camera+lidar_resnet50.yaml \
        result_ss.pkl --score 0.3 --limit 20
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vis_gt import (bev_window, cam_params, load_cfg, load_infos,  # noqa: E402
                    render_bev, render_camera)


def to_numpy(x):
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    a = np.asarray(x, np.float64)
    return a.reshape(1, -1) if a.ndim == 1 else a


def extract(r, classes):
    if isinstance(r, dict) and "boxes_3d" in r:
        boxes = to_numpy(r["boxes_3d"])
        s = r["scores_3d"]
        scores = np.asarray(s.detach().cpu() if hasattr(s, "detach") else s)
        l = r["labels_3d"]
        labels = np.asarray(l.detach().cpu() if hasattr(l, "detach") else l).astype(int)
        return boxes, scores, labels
    if isinstance(r, list):
        if not r:
            return np.zeros((0, 7)), np.zeros(0), np.zeros(0, int)
        boxes = np.array([[b[0], b[1], b[2], b[3], b[4], b[5], b[6]] for b in r], np.float64)
        scores = np.array([b[7] if len(b) > 7 else 1.0 for b in r], np.float64)
        labels = np.array([classes.index(b[-2]) if len(b) >= 2 and b[-2] in classes else 0
                           for b in r], np.int64)
        return boxes, scores, labels
    return np.zeros((0, 7)), np.zeros(0), np.zeros(0, int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("result")
    ap.add_argument("--out-dir", default="viz/pred")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--score", type=float, default=0.3)
    ap.add_argument("--cam", default="CAM_FRONT")
    ap.add_argument("--y-range", type=float, default=60.0)
    ap.add_argument("--zoom", type=float, default=0.0)
    ap.add_argument("--thickness", type=int, default=3)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    classes = list(cfg.object_classes)
    infos = load_infos(cfg.ann_val)
    pcr = list(cfg.point_cloud_range)
    print(f"配置: {args.config}\n类别: {classes}\n验证集: {cfg.ann_val} ({len(infos)} 帧)")

    res = pickle.load(open(args.result, "rb"))
    if isinstance(res, list):
        assert len(res) == len(infos), f"数量不匹配: {len(infos)} 帧 vs {len(res)} 条预测"
        items = list(zip(infos, res))
    elif isinstance(res, dict):
        by = {it["token"]: it for it in infos}
        items = [(by[k], v) for k, v in res.items() if k in by]
    else:
        raise SystemExit(f"无法识别 result 类型: {type(res)}")

    os.makedirs(args.out_dir, exist_ok=True)
    n_img = n_box = 0
    for info, r in items:
        if args.limit and n_img >= args.limit:
            break
        c = info["cams"].get(args.cam)
        if c is None:
            continue
        img = cv2.imread(c["data_path"])
        if img is None:
            continue

        boxes, scores, labels = extract(r, classes)
        keep = scores >= args.score
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        K, R, t = cam_params(info, args.cam)
        cam, drawn = render_camera(img, boxes, scores, labels, classes, K, R, t,
                                   args.thickness,
                                   title=f"{info['token']}  pred={len(boxes)}  thr={args.score}")
        pts = np.load(info["lidar_path"])
        win = bev_window(pcr, boxes, args.zoom, args.y_range)
        bev = render_bev(pts, boxes, labels, classes, win)
        cv2.imwrite(os.path.join(args.out_dir, f"{info['token']}.png"), np.hstack([cam, bev]))
        n_img += 1
        n_box += drawn
        print(f"  [{n_img}] {info['token']}: 预测 {len(boxes)} 画出 {drawn}")
    print(f"完成 {n_img} 张，共画 {n_box} 个框 -> {args.out_dir}")


if __name__ == "__main__":
    main()
