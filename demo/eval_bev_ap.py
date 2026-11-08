"""BEV AP 评测（KITTI 风格：旋转框 BEV IoU 匹配）。

    torchpack dist-run -np 1 python tools/test.py \
        configs/SeePerSea/camera+lidar_resnet50.yaml \
        runs/seepersea/epoch_96.pth --out result_ss.pkl

    python demo/eval_bev_ap.py \
        configs/SeePerSea/camera+lidar_resnet50.yaml \
        result_ss.pkl --iou 0.5 --ranges 0-30 30-60 60-80
"""
import argparse
import pickle

import numpy as np

sys_path_fix = None


def _load_cfg(path):
    from mmcv import Config
    from mmdet3d.utils import recursive_eval
    from torchpack.utils.config import configs

    configs.load(path, recursive=True)
    return Config(recursive_eval(configs), filename=path)


def load_preds(path):
    r = pickle.load(open(path, "rb"))
    out = []
    if isinstance(r, list):
        for d in r:
            boxes = d["boxes_3d"]
            boxes = boxes.tensor if hasattr(boxes, "tensor") else boxes
            if hasattr(boxes, "detach"):
                boxes = boxes.detach().cpu().numpy()
            s = d["scores_3d"]
            scores = s.detach().cpu().numpy() if hasattr(s, "detach") else np.asarray(s)
            l = d["labels_3d"]
            labels = l.detach().cpu().numpy() if hasattr(l, "detach") else np.asarray(l)
            out.append(dict(boxes=np.asarray(boxes, np.float64),
                            scores=np.asarray(scores, np.float64),
                            labels=np.asarray(labels, np.int64)))
    elif isinstance(r, dict):
        for token, annos in r.items():
            out.append(dict(token=token, annos=annos))
    else:
        raise SystemExit(f"无法识别 {path} 的类型: {type(r)}")
    return out


def bev_5(boxes):
    b = np.asarray(boxes, np.float64)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    if b.size == 0 or b.shape[-1] < 7:
        return np.zeros((0, 5))
    return np.stack([b[:, 0], b[:, 1], b[:, 3], b[:, 4], b[:, 6]], 1)


def _corners(b):
    x, y, dx, dy, yaw = b
    c, s = np.cos(yaw), np.sin(yaw)
    pts = np.array([[dx / 2, dy / 2], [-dx / 2, dy / 2],
                    [-dx / 2, -dy / 2], [dx / 2, -dy / 2]])
    pts = pts @ np.array([[c, -s], [s, c]]).T + np.array([x, y])
    area2 = 0.0
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        area2 += x1 * y2 - x2 * y1
    return pts if area2 > 0 else pts[::-1]


def _area(p):
    a = 0.0
    n = len(p)
    for i in range(n):
        x1, y1 = p[i]
        x2, y2 = p[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) * 0.5


def _clip(subject, clip):
    out = list(subject)
    n = len(clip)
    for i in range(n):
        if not out:
            break
        ax, ay = clip[i]
        bx, by = clip[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        inp, out = out, []
        for j in range(len(inp)):
            px, py = inp[j]
            qx, qy = inp[(j + 1) % len(inp)]
            dp = ex * (py - ay) - ey * (px - ax)
            dq = ex * (qy - ay) - ey * (qx - ax)
            if dp >= 0:
                out.append((px, py))
            if (dp >= 0) != (dq >= 0):
                t = dp / (dp - dq)
                out.append((px + t * (qx - px), py + t * (qy - py)))
    return out


def poly_iou(a, b):
    A = _corners(a)
    B = _corners(b)
    inter = _clip(A, B)
    if len(inter) < 3:
        return 0.0
    ai = _area(inter)
    if ai <= 0:
        return 0.0
    u = _area(A) + _area(B) - ai
    return ai / u if u > 0 else 0.0


def iou_matrix(a5, b5):
    if len(a5) == 0 or len(b5) == 0:
        return np.zeros((len(a5), len(b5)))
    out = np.zeros((len(a5), len(b5)))
    for i in range(len(a5)):
        for j in range(len(b5)):
            out[i, j] = poly_iou(a5[i], b5[j])
    return out


def average_precision(rec, prec):
    if len(rec) == 0:
        return float("nan")
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def parse_range(s):
    a, b = s.split("-")
    return float(a), float(b)


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
    infos = pickle.load(open(cfg.ann_val, "rb"))["infos"]
    preds = load_preds(args.result)
    assert len(infos) == len(preds), f"数量不匹配: {len(infos)} 帧 vs {len(preds)} 条预测"

    print(f"配置: {args.config}")
    print(f"验证集: {cfg.ann_val} ({len(infos)} 帧)")
    print(f"匹配阈值: BEV IoU >= {args.iou} | 置信度 >= {args.score}")

    RANGES = ["all"] + list(args.ranges)
    pars = {r: (parse_range(r) if r != "all" else (-1e9, 1e9)) for r in RANGES}

    gt_cls, gt_dist, gt_5 = [], [], []
    for info in infos:
        b = np.asarray(info["gt_boxes"], np.float64).reshape(-1, 7)
        names = list(info["gt_names"])
        lbl = np.array([classes.index(n) if n in classes else -1 for n in names], np.int64)
        gt_cls.append(lbl)
        gt_dist.append(np.linalg.norm(b[:, :2], axis=1) if len(b) else np.zeros(0))
        gt_5.append(bev_5(b))

    results = np.full((len(classes), len(RANGES)), np.nan)
    for ci, cname in enumerate(classes):
        for ri, rname in enumerate(RANGES):
            lo, hi = pars[rname]
            recs = []
            n_gt = 0
            for k in range(len(infos)):
                gm = (gt_cls[k] == ci) & (gt_dist[k] >= lo) & (gt_dist[k] < hi)
                g5 = gt_5[k][gm]
                n_gt += len(g5)
                d = preds[k]
                if "boxes" not in d:
                    annos = d["annos"]
                    if not len(annos):
                        continue
                    arr = np.array([[a[0], a[1], a[2], a[3], a[4], a[5], a[6]]
                                    for a in annos], np.float64)
                    sc = np.array([a[7] if len(a) > 7 else 1.0 for a in annos], np.float64)
                    nm = [a[-2] if isinstance(a[-2], str) else "" for a in annos]
                    lb = np.array([classes.index(n) if n in classes else -1 for n in nm], np.int64)
                else:
                    arr, sc, lb = d["boxes"], d["scores"], d["labels"]
                pm = (lb == ci) & (sc >= args.score)
                if not pm.any():
                    continue
                b7 = arr[pm]
                pd_ = np.linalg.norm(b7[:, :2], axis=1)
                rm = (pd_ >= lo) & (pd_ < hi)
                if not rm.any():
                    continue
                b7, ps = b7[rm], sc[pm][rm]
                p5 = bev_5(b7)
                order = np.argsort(-ps)
                p5, ps = p5[order], ps[order]
                if len(g5) == 0:
                    recs += [(s, 0) for s in ps]
                    continue
                iou = iou_matrix(p5, g5)
                used = np.zeros(len(g5), bool)
                for i in range(len(ps)):
                    j = int(np.argmax(iou[i]))
                    if iou[i, j] >= args.iou and not used[j]:
                        used[j] = True
                        recs.append((ps[i], 1))
                    else:
                        recs.append((ps[i], 0))
            if n_gt == 0:
                continue
            recs.sort(key=lambda x: -x[0])
            tp = np.cumsum([r[1] for r in recs]) if recs else np.zeros(0)
            fp = np.cumsum([1 - r[1] for r in recs]) if recs else np.zeros(0)
            rec = tp / n_gt
            prec = tp / np.maximum(tp + fp, 1e-9)
            results[ci, ri] = average_precision(rec, prec)

    head = f'{"类别":<14}{"AP(all)":>9}' + "".join(f'{r+"m":>10}' for r in args.ranges)
    print("\n" + head)
    print("-" * len(head))
    for ci, cname in enumerate(classes):
        row = f'{cname:<14}'
        row += f'{results[ci, 0]:9.4f}' if not np.isnan(results[ci, 0]) else f'{"-":>9}'
        for ri in range(1, len(RANGES)):
            v = results[ci, ri]
            row += f'{v:10.4f}' if not np.isnan(v) else f'{"-":>10}'
        print(row)
    print("-" * len(head))
    print(f'{"mAP":<14}{np.nanmean(results[:, 0]):9.4f}')
    print("\n注：AP 按 all-point 插值；距离取框中心到雷达的距离")


if __name__ == "__main__":
    main()
