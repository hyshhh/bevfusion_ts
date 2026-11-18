"""把 LiDAR 3D 框投影到相机图像（OpenCV projectPoints）。

在你的版本基础上：
  1. 保留逐边近平面裁剪（跨越相机平面的框也能画）
  2. 加入 归一化半径 r <= 0.7 的畸变有效域判断：
     域内用 K+D，域外整框退回纯针孔
  3. 支持 --limit / --tokens / --dist

标签格式（每行）：x y z length width height yaw class_name   （z 为框中心）
标定文件：LiDAR->camera 的 R、t，3x3 内参 K，5 个 OpenCV 畸变系数。

    python project_boxes_opencv.py --root data/data_old/custom_3000 --limit 5
    python project_boxes_opencv.py --root ... --tokens 000005 000376
    python project_boxes_opencv.py --root ... --dist   # 图像未去畸变时才需要
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

R_MAX = 0.7          # 畸变多项式的有效域（图像角点归一化半径约 0.55）

BOX_EDGES = (
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
)


def load_calibration(path):
    text = Path(path).read_text(encoding="utf-8")
    rotation = re.search(r"(?m)^R:\s*\n((?:\s*[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s*\n){3})", text)
    translation = re.search(r"(?m)^t:\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", text)
    intrinsic = re.search(r"(?m)^Intrinsic:\s*\n((?:\s*[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s*\n){3})", text)
    distortion = re.search(r"(?m)^Distortion:\s*\n?\s*\[([^\]]+)\]", text)
    if not (rotation and translation and intrinsic and distortion):
        raise ValueError(f"Could not parse: {path}")
    R = np.array([[float(v) for v in l.split()] for l in rotation.group(1).splitlines()], np.float64)
    t = np.array([float(v) for v in translation.groups()], np.float64).reshape(3, 1)
    K = np.array([[float(v) for v in l.split()] for l in intrinsic.group(1).splitlines()], np.float64)
    dist = np.array([float(v.strip()) for v in distortion.group(1).split(",")], np.float64).reshape(-1, 1)
    return R, t, K, dist


def box_corners_lidar(box):
    """[x,y,z,length,width,height,yaw]，z 为框中心。"""
    x, y, z, length, width, height, yaw = box
    corners = np.array([
        [-length / 2, -width / 2, -height / 2],
        [-length / 2, -width / 2,  height / 2],
        [-length / 2,  width / 2, -height / 2],
        [-length / 2,  width / 2,  height / 2],
        [ length / 2, -width / 2, -height / 2],
        [ length / 2, -width / 2,  height / 2],
        [ length / 2,  width / 2, -height / 2],
        [ length / 2,  width / 2,  height / 2],
    ], np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)
    return corners @ Rz.T + np.array([x, y, z], np.float64)


def project_boxes(image, label_path, calib_path, output_path=None, *,
                  use_dist=False, thickness=2, near=1e-3, verbose=False):
    R, t, K, dist = load_calibration(calib_path)
    rvec, _ = cv2.Rodrigues(R)
    rendered = image.copy()
    stats = {"总框": 0, "画出": 0, "域外退针孔": 0, "跳过": 0}

    for line_no, line in enumerate(Path(label_path).read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 7:
            raise ValueError(f"{label_path}:{line_no}: 至少需要 7 个数")
        values = [float(v) for v in fields[:7]]
        if any(v <= 0 for v in values[3:6]):
            stats["跳过"] += 1
            continue
        stats["总框"] += 1
        name = fields[7] if len(fields) > 7 else "3D"

        corners_lidar = box_corners_lidar(values)
        corners_camera = corners_lidar @ R.T + t.reshape(1, 3)

        # 归一化半径决定是否套畸变（域外整框退回针孔）
        dist_use = None
        if use_dist:
            depth = np.clip(corners_camera[:, 2], 1e-6, None)
            rmax = float(np.linalg.norm(corners_camera[:, :2] / depth[:, None], axis=1).max())
            dist_use = dist if rmax <= R_MAX else None
            if dist_use is None and rmax > R_MAX:
                stats["域外退针孔"] += 1

        drew = False
        for a, b in BOX_EDGES:
            p1, p2 = corners_camera[a].copy(), corners_camera[b].copy()
            if p1[2] <= near and p2[2] <= near:
                continue
            if p1[2] <= near:
                p1 = p1 + (p2 - p1) * ((near - p1[2]) / (p2[2] - p1[2]))
            elif p2[2] <= near:
                p2 = p2 + (p1 - p2) * ((near - p2[2]) / (p1[2] - p2[2]))
            pts, _ = cv2.projectPoints(np.vstack((p1, p2)),
                                       np.zeros((3, 1)), np.zeros((3, 1)), K, dist_use)
            q1, q2 = np.rint(pts.reshape(-1, 2)).astype(int)
            if max(abs(q1[0]), abs(q1[1]), abs(q2[0]), abs(q2[1])) > 20000:
                continue
            cv2.line(rendered, tuple(q1), tuple(q2), (40, 220, 240), thickness, cv2.LINE_AA)
            drew = True
        if drew:
            stats["画出"] += 1

        center, _ = cv2.projectPoints(np.array([values[:3]], np.float64), rvec, t, K, dist_use)
        cx, cy = np.rint(center.reshape(2)).astype(int)
        if 0 <= cx < rendered.shape[1] and 0 <= cy < rendered.shape[0]:
            cv2.putText(rendered, name, (cx + 3, cy - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (40, 220, 240), 2, cv2.LINE_AA)
        if verbose:
            print(f"    {name:>12} rmax={float(np.linalg.norm(corners_camera[:, :2] / np.clip(corners_camera[:, 2:], 1e-6, None), axis=1).max()):.2f}"
                  f" dist={'K+D' if dist_use is not None else 'panhole'}")

    cv2.putText(rendered, f"boxes={stats['总框']} drawn={stats['画出']} pinhole={stats['域外退针孔']}",
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), rendered)
    return rendered, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("/home/lab-203/hys/BEV/bevfusion/data/data_old/custom_3000"))
    ap.add_argument("--result-dir", type=Path, default=Path("viz/opencv"))
    ap.add_argument("--tokens", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=0, help="0 表示全部")
    ap.add_argument("--img-dir", default="images_undist",
                    help="用哪个图像目录：images_undist（默认，训练用）或 images（原图）")
    ap.add_argument("--dist", action="store_true",
                    help="投影时叠加畸变（仅 --img-dir images 时需要）")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    img_root = args.root / args.img_dir
    if not img_root.is_dir():
        print(f"[警告] {img_root} 不存在，回退到 images/（原图，需要 --dist）")
        img_root = args.root / "images"
        args.img_dir = "images"
    print(f"图像目录: {img_root}  畸变: {'叠加' if args.dist else '不叠加'}")
    imgs = sorted(p for p in img_root.glob("*")
                  if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    if args.tokens:
        want = set(args.tokens)
        imgs = [p for p in imgs if p.stem in want]
    if args.limit:
        imgs = imgs[: args.limit]

    ok = miss = 0
    for img_path in imgs:
        lab = args.root / "labels_points" / f"{img_path.stem}.txt"
        cal = args.root / "calib" / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None or not lab.is_file() or not cal.is_file():
            miss += 1
            continue
        _, stats = project_boxes(img, lab, cal, args.result_dir / img_path.name,
                                 use_dist=args.dist, verbose=args.verbose)
        ok += 1
        print(f"{img_path.stem}: 框 {stats['总框']} 画 {stats['画出']} "
              f"退针孔 {stats['域外退针孔']} -> {args.result_dir / img_path.name}")
    print(f"共 {ok} 张，跳过 {miss} 张")


if __name__ == "__main__":
    main()
