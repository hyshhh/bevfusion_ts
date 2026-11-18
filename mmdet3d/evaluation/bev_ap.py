"""BEV AP 评测（KITTI 风格：旋转框 BEV IoU 匹配）。

一套实现，两个入口：
  - 离线：``python demo/eval_bev_ap.py <cfg> <result.pkl>``
  - 训练中：``mmdet3d.runner.hooks.BEVAPHook``（每 N 轮自动跑一次）

匹配规则：按类别、按距离区间统计，预测框与 GT 框做 BEV 旋转框 IoU，
IoU >= ``iou`` 且未被占用即视为 TP，AP 用 all-point 插值。
距离取框中心到雷达原点（x, y 的 L2 范数）。
"""
import pickle

import numpy as np


def to_numpy(x):
    """把 tensor / BaseInstance3DBoxes / ndarray 统一成 numpy。"""
    x = getattr(x, "tensor", x)
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def load_preds(path):
    """读入 ``tools/test.py --out`` 产出的 pkl，统一成 numpy 格式。"""
    with open(path, "rb") as f:
        r = pickle.load(f)
    out = []
    if isinstance(r, list):
        for d in r:
            out.append(dict(boxes=np.asarray(to_numpy(d["boxes_3d"]), np.float64),
                            scores=np.asarray(to_numpy(d["scores_3d"]), np.float64),
                            labels=np.asarray(to_numpy(d["labels_3d"]), np.int64)))
    elif isinstance(r, dict):
        for token, annos in r.items():
            out.append(dict(token=token, annos=annos))
    else:
        raise SystemExit(f"无法识别 {path} 的类型: {type(r)}")
    return out


def bev_5(boxes):
    """7 维框 (x, y, z, dx, dy, dz, yaw) -> BEV 5 维 (x, y, dx, dy, yaw)。"""
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
    """``"0-30"`` -> (0, 30)；``"80+"`` -> (80, inf)。"""
    s = str(s).strip()
    if s.endswith("+"):
        return float(s[:-1]), float("inf")
    if s.endswith("-inf"):
        return float(s[:-4]), float("inf")
    a, b = s.split("-")
    return float(a), float(b)


def range_label(r):
    """列头显示名：0-30 -> 0-30m，80+ -> 80m+，all -> all。"""
    if r == "all":
        return "all"
    return f"{r[:-1]}m+" if str(r).endswith("+") else f"{r}m"


def _unpack_pred(d, classes):
    """把单帧预测统一成 (boxes[N,7], scores[N], labels[N])。

    支持三种来源：
      - 训练内 ``single_gpu_test`` 的原始输出：``boxes_3d``/``scores_3d``/``labels_3d``
      - ``load_preds`` 转换后的：``boxes``/``scores``/``labels``
      - ``{token: annos}`` 形式
    """
    if "boxes" in d:
        return d["boxes"], d["scores"], d["labels"]
    if "boxes_3d" in d:
        return (np.asarray(to_numpy(d["boxes_3d"]), np.float64),
                np.asarray(to_numpy(d["scores_3d"]), np.float64),
                np.asarray(to_numpy(d["labels_3d"]), np.int64))
    annos = d.get("annos", None)
    if annos is None:
        raise KeyError(f"无法识别的预测格式，keys={list(d.keys())}")
    if not len(annos):
        return None
    arr = np.array([[a[0], a[1], a[2], a[3], a[4], a[5], a[6]]
                    for a in annos], np.float64)
    sc = np.array([a[7] if len(a) > 7 else 1.0 for a in annos], np.float64)
    nm = [a[-2] if isinstance(a[-2], str) else "" for a in annos]
    lb = np.array([classes.index(n) if n in classes else -1 for n in nm], np.int64)
    return arr, sc, lb


def compute_bev_ap(infos, classes, preds, iou=0.5,
                   ranges=("0-30", "30-50", "50-80"), score_thr=0.001):
    """计算 per-class / per-range BEV AP。

    Args:
        infos (list[dict]): ``ann_val`` 里的 ``infos``，顺序必须与 ``preds`` 一致。
        classes (list[str]): 类别名，下标即 label id。
        preds (list[dict]): 每帧预测，元素含 ``boxes``/``scores``/``labels``
            （或 ``annos`` 形式）。
        iou (float): BEV 旋转框 IoU 匹配阈值。
        ranges (list[str]): 距离区间，形如 ``"0-30"``。
        score_thr (float): 参与统计的最低置信度。

    Returns:
        dict: ``ranges``（含 ``"all"``）、``ap``（shape ``[C, R]``，NaN 表示该
        类别在该区间没有 GT）、``mAP``（每个区间的 nanmean）。
    """
    classes = list(classes)
    run_ranges = ["all"] + [str(r) for r in ranges]
    pars = {r: (parse_range(r) if r != "all" else (-1e9, 1e9)) for r in run_ranges}

    gt_cls, gt_dist, gt_5 = [], [], []
    for info in infos:
        b = np.asarray(info["gt_boxes"], np.float64).reshape(-1, 7)
        names = list(info["gt_names"])
        lbl = np.array([classes.index(n) if n in classes else -1 for n in names], np.int64)
        gt_cls.append(lbl)
        gt_dist.append(np.linalg.norm(b[:, :2], axis=1) if len(b) else np.zeros(0))
        gt_5.append(bev_5(b))

    results = np.full((len(classes), len(run_ranges)), np.nan)
    for ci, _cname in enumerate(classes):
        for ri, rname in enumerate(run_ranges):
            lo, hi = pars[rname]
            recs = []
            n_gt = 0
            for k in range(len(infos)):
                gm = (gt_cls[k] == ci) & (gt_dist[k] >= lo) & (gt_dist[k] < hi)
                g5 = gt_5[k][gm]
                n_gt += len(g5)
                unpacked = _unpack_pred(preds[k], classes)
                if unpacked is None:
                    continue
                arr, sc, lb = unpacked
                pm = (lb == ci) & (sc >= score_thr)
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
                iou_m = iou_matrix(p5, g5)
                used = np.zeros(len(g5), bool)
                for i in range(len(ps)):
                    j = int(np.argmax(iou_m[i]))
                    if iou_m[i, j] >= iou and not used[j]:
                        used[j] = True
                        recs.append((ps[i], 1))
                    else:
                        recs.append((ps[i], 0))
            if n_gt == 0:
                # 该类别在这个距离段没有 GT，不参与 mAP
                continue
            if not recs:
                # 有 GT 但一个预测都没有：AP 记为 0（不能是 nan，否则 nanmean
                # 会把"完全没检出"的类直接丢掉，mAP 会被虚高）
                results[ci, ri] = 0.0
                continue
            recs.sort(key=lambda x: -x[0])
            tp = np.cumsum([r[1] for r in recs])
            fp = np.cumsum([1 - r[1] for r in recs])
            rec = tp / n_gt
            prec = tp / np.maximum(tp + fp, 1e-9)
            results[ci, ri] = average_precision(rec, prec)

    mAP = {}
    for ri, rname in enumerate(run_ranges):
        col = results[:, ri]
        mAP[rname] = float(np.nanmean(col)) if np.isfinite(col).any() else float("nan")
    return dict(ranges=run_ranges, ap=results, mAP=mAP, classes=classes,
                iou=iou, score_thr=score_thr)


def format_ap_table(classes, ranges, ap, mAP=None):
    """把 AP 矩阵渲染成对齐表格；给 ``mAP`` 时在最后追加 mAP 行。"""
    head = f'{"类别":<14}{"AP(all)":>9}' + "".join(
        f'{range_label(r):>10}' for r in ranges[1:])
    lines = [head, "-" * len(head)]
    for ci, cname in enumerate(classes):
        row = f'{cname:<14}'
        row += f'{ap[ci, 0]:9.4f}' if not np.isnan(ap[ci, 0]) else f'{"-":>9}'
        for ri in range(1, len(ranges)):
            v = ap[ci, ri]
            row += f'{v:10.4f}' if not np.isnan(v) else f'{"-":>10}'
        lines.append(row)
    lines.append("-" * len(head))
    if mAP is not None:
        row = f'{"mAP":<14}'
        for ri, r in enumerate(ranges):
            v = mAP.get(r, float("nan"))
            ok = np.isfinite(v) if v is not None else False
            if ri == 0:
                row += f'{v:9.4f}' if ok else f'{"-":>9}'
            else:
                row += f'{v:10.4f}' if ok else f'{"-":>10}'
        lines.append(row)
        lines.append("-" * len(head))
    return "\n".join(lines)
