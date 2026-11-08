"""把 custom_3000（KITTI 风格 + 逐帧标定 + ASCII PCD）转成 BEVFusion 的 info pkl。

用法:
    python configs/YZdata/convert_custom3000.py --src data/data_old/custom_3000 \
        --out-pkl data/data_pkl

做三件事:
  1. points/*.pcd (ASCII, 全零补齐) -> lidar_npy/*.npy (float32 Nx4, 去补齐)
  2. calib/*.txt (逐帧, lidar->camera) -> sensor2lidar (camera->lidar)
  3. labels_points/*.txt (几何中心) -> gt_boxes (底面中心)
"""
import argparse
import os
import pickle
from multiprocessing import Pool

import numpy as np

# BEVFusion 的 label id 顺序 = 这个列表的顺序
CLASSES = [
    "boat",
    "pier",
    "pillar",
    "shore",
    "pontoon",
    "bridge",
    "buoy",
    "architecture",
    "islands",
]


def parse_calib(path):
    """返回 (K, R_l2c, t_l2c, D)。文件里是 lidar->camera。"""
    txt = open(path).read()
    L = txt.split("\n")
    R = np.array([[float(x) for x in L[i].split()] for i in (2, 3, 4)], dtype=np.float32)
    t = np.array([float(x) for x in L[5].replace("t:", "").split()], dtype=np.float32)
    K = np.array([[float(x) for x in L[i].split()] for i in (8, 9, 10)], dtype=np.float32)
    j = txt.find("Distortion:")
    D = np.array([float(x) for x in txt[j + 11:].strip().strip("[]").split(",")],
                 dtype=np.float32) if j >= 0 else None
    return K, R, t, D


def convert_pcd(args):
    src, dst = args
    raw = open(src, "rb").read()
    i = raw.index(b"DATA ascii") + len(b"DATA ascii") + 1
    p = np.fromstring(raw[i:].decode("ascii"), sep=" ").reshape(-1, 4).astype(np.float32)
    p = p[(p[:, 0] != 0) | (p[:, 1] != 0) | (p[:, 2] != 0)]
    np.save(dst, p)
    return len(p)


def count_pts(points, boxes):
    out = []
    p = points[:, :3]
    for cx, cy, cz, dx, dy, dz, yaw in boxes:
        c, s = np.cos(-yaw), np.sin(-yaw)
        d = p - np.array([cx, cy, cz], dtype=np.float32)
        x = d[:, 0] * c - d[:, 1] * s
        y = d[:, 0] * s + d[:, 1] * c
        z = d[:, 2]
        m = (np.abs(x) <= dx / 2) & (np.abs(y) <= dy / 2) & (np.abs(z) <= dz / 2)
        out.append(int(m.sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/data_old/custom_3000")
    ap.add_argument("--out-pkl", default="data/data_pkl/custom_3000")
    ap.add_argument("--npy-dir", default=None, help="默认 <src>/lidar_npy")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true", help="只统计不写文件")
    args = ap.parse_args()

    src = args.src
    npy_dir = args.npy_dir or os.path.join(src, "lidar_npy")
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(args.out_pkl, exist_ok=True)

    splits = {}
    for name in ("train", "val"):
        f = os.path.join(src, "ImageSets", f"{name}.txt")
        splits[name] = [l.strip() for l in open(f) if l.strip()]

    # ---- 1. PCD -> NPY（多进程）----
    jobs = []
    for name, ids in splits.items():
        for fid in ids:
            jobs.append(
                (os.path.join(src, "points", f"{fid}.pcd"),
                 os.path.join(npy_dir, f"{fid}.npy"))
            )
    print(f"待转换点云: {len(jobs)} 个")
    if not args.dry_run:
        with Pool(args.workers) as pool:
            ns = pool.map(convert_pcd, jobs, chunksize=8)
        print(f"点云转换完成, 平均有效点 {int(np.mean(ns))}")

    # ---- 2. 组装 info ----
    for name, ids in splits.items():
        infos = []
        cls_count, miss_lab, miss_img, drop_cls = {}, 0, 0, {}
        for fid in ids:
            K, R_l2c, t_l2c, D = parse_calib(os.path.join(src, "calib", f"{fid}.txt"))
            cam_rel = f"images/{fid}.png"
            if not os.path.exists(os.path.join(src, cam_rel)):
                miss_img += 1
                continue
            # 文件是 lidar->camera, pkl 要 camera->lidar
            R_c2l = R_l2c.T
            t_c2l = -R_l2c.T @ t_l2c

            lab_path = os.path.join(src, "labels_points", f"{fid}.txt")
            boxes, names = [], []
            if os.path.exists(lab_path):
                for line in open(lab_path):
                    v = line.split()
                    if len(v) < 8:
                        continue
                    x, y, z, dx, dy, dz, yaw = (float(a) for a in v[:7])
                    cname = v[7]
                    if cname not in CLASSES:
                        drop_cls[cname] = drop_cls.get(cname, 0) + 1
                        continue
                    boxes.append([x, y, z - dz / 2, dx, dy, dz, yaw])  # 几何中心 -> 底面中心
                    names.append(cname)
                    cls_count[cname] = cls_count.get(cname, 0) + 1
            else:
                miss_lab += 1

            pts_path = os.path.join(npy_dir, f"{fid}.npy")
            pts = np.load(pts_path) if os.path.exists(pts_path) else np.zeros((0, 4), np.float32)
            npts = [max(1, n) for n in count_pts(pts, boxes)] if boxes else []

            infos.append(
                dict(
                    token=fid,
                    timestamp=int(fid),
                    lidar_path=os.path.abspath(pts_path),
                    sweeps=[],
                    cams={
                        "CAM_FRONT": dict(
                            data_path=os.path.abspath(os.path.join(src, cam_rel)),
                            cam_intrinsic=K,
                            distortion=D,
                            sensor2lidar_rotation=R_c2l.astype(np.float32),
                            sensor2lidar_translation=t_c2l.astype(np.float32),
                            sensor2ego_rotation=[1.0, 0, 0, 0],
                            sensor2ego_translation=[0.0, 0, 0],
                        )
                    },
                    lidar2ego_rotation=[1.0, 0, 0, 0],
                    lidar2ego_translation=[0.0, 0, 0],
                    ego2global_rotation=[1.0, 0, 0, 0],
                    ego2global_translation=[0.0, 0, 0],
                    gt_boxes=np.array(boxes, dtype=np.float32).reshape(-1, 7),
                    gt_names=np.array(names),
                    gt_velocity=np.zeros((len(boxes), 2), dtype=np.float32),
                    num_lidar_pts=np.array(npts, dtype=np.int64),
                )
            )

        out = os.path.join(args.out_pkl, f"infos_{name}.pkl")
        if not args.dry_run:
            with open(out, "wb") as f:
                pickle.dump({"infos": infos, "metadata": {"version": "v1.0-custom3000"}}, f)
        print(
            f"{name}: {len(infos)} 帧 -> {out}"
            f" | 无标注文件 {miss_lab}, 缺图 {miss_img}"
            f" | 类别 {cls_count}"
            + (f" | 丢弃未知类 {drop_cls}" if drop_cls else "")
        )


if __name__ == "__main__":
    main()
