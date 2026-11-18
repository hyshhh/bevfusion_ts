"""标签（GT）可视化：相机 + BEV 双视图，数据来自 infos pkl。

    python demo/vis_gt.py configs/SeePerSea/camera+lidar_resnet50.yaml --limit 20
    python demo/vis_gt.py configs/YZdata/camera+lidar_resnet50.yaml --limit 20 --out-dir viz/gt_yz

本文件同时提供 vis_pred.py 复用的渲染函数。
"""
import argparse
import os
import pickle

import cv2
import numpy as np

PALETTE = [
    (255, 150, 0), (0, 130, 255), (0, 220, 0), (0, 0, 220), (220, 80, 200),
    (80, 90, 150), (230, 130, 200), (60, 200, 200), (200, 180, 0), (120, 60, 220),
]
EDGES = [(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2),
         (3, 7), (4, 5), (4, 7), (2, 6), (5, 6), (6, 7)]
PANEL_H = 720
BEV_SIZE = 720


def load_cfg(path):
    from mmcv import Config
    from mmdet3d.utils import recursive_eval
    from torchpack.utils.config import configs

    configs.load(path, recursive=True)
    return Config(recursive_eval(configs), filename=path)


def load_infos(pkl_path):
    return pickle.load(open(pkl_path, "rb"))["infos"]


def corners_of(b):
    b = np.asarray(b, np.float64).reshape(-1)
    if b.size < 7:
        return None
    x, y, z, dx, dy, dz, yaw = b[:7]
    c, s = np.cos(yaw), np.sin(yaw)
    base = np.array([[dx / 2, dy / 2], [-dx / 2, dy / 2],
                     [-dx / 2, -dy / 2], [dx / 2, -dy / 2]])
    xy = base @ np.array([[c, -s], [s, c]]).T + np.array([x, y])
    return np.concatenate([np.concatenate([xy, np.full((4, 1), z)], 1),
                           np.concatenate([xy, np.full((4, 1), z + dz)], 1)], 0)


def cam_params(info, cam_name="CAM_FRONT"):
    c = info["cams"][cam_name]
    K = np.asarray(c["cam_intrinsic"], np.float64).reshape(3, 3)
    R_c2l = np.asarray(c["sensor2lidar_rotation"], np.float64).reshape(3, 3)
    t_c2l = np.asarray(c["sensor2lidar_translation"], np.float64).reshape(3)
    return K, R_c2l.T, (-R_c2l.T @ t_c2l).reshape(3, 1)


def render_camera(img, boxes, scores, labels, classes, K, R, t, thickness=3, title=""):
    H, W = img.shape[:2]
    scale = PANEL_H / float(H)
    cam_w = int(round(W * scale))
    cam = cv2.resize(img, (cam_w, PANEL_H), interpolation=cv2.INTER_LINEAR)
    Ks = K.copy()
    Ks[0] *= scale
    Ks[1] *= scale
    n = 0
    for i, b in enumerate(boxes):
        c3 = corners_of(b)
        if c3 is None:
            continue
        cp = c3 @ R.T + t.ravel()
        if (cp[:, 2] <= 0.1).any():
            continue
        uv = (Ks @ cp.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        if uv[:, 0].max() < 0 or uv[:, 1].max() < 0 or \
                uv[:, 0].min() > cam_w or uv[:, 1].min() > PANEL_H:
            continue
        lb = int(labels[i])
        col = PALETTE[lb % len(PALETTE)]
        for a, e in EDGES:
            cv2.line(cam, tuple(np.round(uv[a]).astype(int)),
                     tuple(np.round(uv[e]).astype(int)), col, thickness, cv2.LINE_AA)
        name = classes[lb] if lb < len(classes) else str(lb)
        txt = f"{name} {scores[i]:.2f}" if scores is not None else name
        tx = int(np.clip(uv[:, 0].min(), 2, cam_w - 90))
        ty = int(np.clip(uv[:, 1].min(), 14, PANEL_H - 4))
        cv2.putText(cam, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
        n += 1
    if title:
        cv2.putText(cam, title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return cam, n


def render_bev(pts, boxes, labels, classes, window, size=BEV_SIZE):
    x0, x1, y0, y1 = window
    img = np.zeros((size, size, 3), np.uint8)
    span = max(x1 - x0, y1 - y0)
    sc = size / span
    cxw, cyw = (x0 + x1) / 2.0, (y0 + y1) / 2.0

    def to_px(x, y):
        return int(size / 2 + (y - cyw) * sc), int(size / 2 - (x - cxw) * sc)

    if pts is not None and len(pts):
        p = pts[:, :3]
        m = (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)
        p = p[m]
        if len(p):
            u = (size / 2 + (p[:, 1] - cyw) * sc).astype(int)
            v = (size / 2 - (p[:, 0] - cxw) * sc).astype(int)
            ok = (u >= 0) & (u < size) & (v >= 0) & (v < size)
            u, v = u[ok], v[ok]
            r = np.linalg.norm(p[ok][:, :2], axis=1)
            t = np.clip(r / max(abs(x1), abs(y1), 1e-3), 0, 1)
            val = (255 * (1 - 0.65 * t)).astype(np.uint8)
            col = np.stack([val, val, val], 1)
            img[v, u] = col
            if size >= 600:
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        img[np.clip(v + dy, 0, size - 1),
                            np.clip(u + dx, 0, size - 1)] = col

    for i, b in enumerate(boxes):
        b = np.asarray(b, np.float64).reshape(-1)
        if b.size < 7:
            continue
        x, y, dx, dy, yaw = b[0], b[1], b[3], b[4], b[6]
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            continue
        c, sn = np.cos(yaw), np.sin(yaw)
        base = np.array([[dx / 2, dy / 2], [-dx / 2, dy / 2],
                         [-dx / 2, -dy / 2], [dx / 2, -dy / 2]])
        xy = base @ np.array([[c, -sn], [sn, c]]).T + np.array([x, y])
        uv = np.array([to_px(px, py) for px, py in xy], np.int32)
        lb = int(labels[i])
        col = PALETTE[lb % len(PALETTE)]
        cv2.polylines(img, [uv], True, col, 3, cv2.LINE_AA)
        front = (xy[0] + xy[3]) / 2.0
        cv2.line(img, to_px(x, y), to_px(front[0], front[1]), col, 2, cv2.LINE_AA)

    ex, ey = to_px(0, 0)
    cv2.drawMarker(img, (int(ex), int(ey)), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 14, 2)
    cv2.putText(img, f"x {x0:.0f}~{x1:.0f}m   y ±{y1:.0f}m", (10, size - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1, cv2.LINE_AA)
    return img


def bev_window(pcr, boxes, zoom=0.0, y_range=60.0, fwd=None):
    if fwd and fwd > 0:
        k = float(fwd) / float(pcr[3]) if pcr[3] > 0 else 1.0
    elif zoom > 0:
        k = float(zoom)
    else:
        if len(boxes) >= 2:
            d = np.linalg.norm(np.asarray(boxes)[:, :2], axis=1)
            need = float(np.percentile(d, 90)) * 1.25
        elif len(boxes) == 1:
            need = float(np.linalg.norm(np.asarray(boxes)[0, :2])) * 1.4
        else:
            need = 0.0
        base = max(abs(pcr[3]), abs(pcr[0]), abs(pcr[4]), abs(pcr[1]))
        k = float(np.clip(need / base, 0.15, 1.0)) if need > 0 else 1.0
    return (pcr[0] * k, pcr[3] * k, -abs(y_range), abs(y_range))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out-dir", default="viz/gt")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--cam", default="CAM_FRONT")
    ap.add_argument("--y-range", type=float, default=60.0)
    ap.add_argument("--zoom", type=float, default=0.0)
    ap.add_argument("--thickness", type=int, default=3)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    classes = list(cfg.object_classes)
    infos = load_infos(cfg.ann_val)
    pcr = list(cfg.point_cloud_range)
    print(f"配置: {args.config}\n类别: {classes}\n标签: {cfg.ann_val} ({len(infos)} 帧)")

    os.makedirs(args.out_dir, exist_ok=True)
    n = 0
    for info in infos:
        if args.limit and n >= args.limit:
            break
        c = info["cams"].get(args.cam)
        if c is None:
            continue
        img = cv2.imread(c["data_path"])
        if img is None:
            continue
        boxes = np.asarray(info["gt_boxes"], np.float64).reshape(-1, 7)
        names = list(info["gt_names"])
        labels = np.array([classes.index(t) if t in classes else 0 for t in names], np.int64)
        scores = None

        K, R, t = cam_params(info, args.cam)
        cam, drawn = render_camera(img, boxes, scores, labels, classes, K, R, t,
                                   args.thickness,
                                   title=f"{info['token']}  gt={len(boxes)}")
        pts = np.load(info["lidar_path"])
        win = bev_window(pcr, boxes, args.zoom, args.y_range)
        bev = render_bev(pts, boxes, labels, classes, win)
        cv2.imwrite(os.path.join(args.out_dir, f"{info['token']}.png"), np.hstack([cam, bev]))
        n += 1
        print(f"  [{n}] {info['token']}: GT {len(boxes)} 框")
    print(f"完成 {n} 张 -> {args.out_dir}")


if __name__ == "__main__":
    main()
