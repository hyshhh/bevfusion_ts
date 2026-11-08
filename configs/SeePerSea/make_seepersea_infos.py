"""把 SeePerSea 三个序列合并成一份 BEVFusion infos pkl。

    python configs/YZdata/make_seepersea_infos.py

输出 data/data_pkl/seepersea/infos_{train,val}.pkl
"""
import glob
import os
import pickle
from pathlib import Path

import numpy as np

ROOT = Path("/home/lab-203/hys/BEV/bevfusion/data/data_old/seepersea")
OUT = Path("/home/lab-203/hys/BEV/bevfusion/data/data_pkl/seepersea")
CLASSES = ["ship", "buoy", "other"]
VAL_RATIO = 0.2
# 数据清洗：point_cloud_range 内点数少于该值的帧直接丢弃
# （空点云会让体素化 kernel 报 CUDA invalid configuration argument）
PCR = np.array([-80.0, -80.0, -5.0, 80.0, 80.0, 15.0])
MIN_INRANGE_PTS = 30
# 数据清洗：框内激光点少于该值的标注直接丢弃（打不到就没有监督信息）
MIN_BOX_PTS = 1


def pts_in_box(pts, x, y, z, l, w, h, ry):
    """框（中心 z）内的点数。"""
    if not len(pts):
        return 0
    cs, sn = np.cos(-ry), np.sin(-ry)
    t = pts - np.array([x, y, z], dtype=pts.dtype)
    loc = np.stack([t[:, 0] * cs - t[:, 1] * sn, t[:, 0] * sn + t[:, 1] * cs], 1)
    m = (np.abs(loc[:, 0]) <= l / 2) & (np.abs(loc[:, 1]) <= w / 2) & (np.abs(t[:, 2]) <= h / 2)
    return int(m.sum())


def read_calib(p):
    d = {}
    for line in open(p):
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = np.array([float(x) for x in v.split()])
    return d


def main():
    seqs = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and p.name != "ImageSets")
    infos_all = []
    n_drop = [0]
    n_box_drop = 0
    n_frame_empty = 0
    for seq in seqs:
        for lab in sorted((ROOT / seq / "label").glob("*.txt")):
            stem = lab.stem
            cal = read_calib(ROOT / seq / "calib" / f"{stem}.txt")
            P2 = cal["P2"].reshape(3, 4)
            Tr = cal["Tr_velo_to_cam"].reshape(3, 4)     # lidar -> camera

            K = P2[:3, :3].astype(np.float32)     # 3x3，数据集要求
            R_l2c, t_l2c = Tr[:, :3], Tr[:, 3]
            R_c2l = R_l2c.T
            t_c2l = -R_c2l.T @ t_l2c

            raw_boxes = []
            for line in open(lab):
                f = line.split()
                if len(f) < 15:
                    continue
                h, w, l = float(f[8]), float(f[9]), float(f[10])
                x, y, z = float(f[11]), float(f[12]), float(f[13])
                ry = float(f[14])
                raw_boxes.append((x, y, z, l, w, h, ry, f[0]))

            pts_all = np.load(ROOT / seq / "points" / f"{stem}.npy")
            inr = (((pts_all[:, 0] >= PCR[0]) & (pts_all[:, 0] <= PCR[3])) &
                   ((pts_all[:, 1] >= PCR[1]) & (pts_all[:, 1] <= PCR[4])) &
                   ((pts_all[:, 2] >= PCR[2]) & (pts_all[:, 2] <= PCR[5]))).sum()
            if inr < MIN_INRANGE_PTS:
                n_drop[0] += 1
                continue

            p3 = pts_all[:, :3]
            boxes, names = [], []
            for (x, y, z, l, w, h, ry, name) in raw_boxes:
                if pts_in_box(p3, x, y, z, l, w, h, ry) < MIN_BOX_PTS:
                    n_box_drop += 1
                    continue
                boxes.append([x, y, z - h / 2, l, w, h, ry])   # 中心 -> 底面中心
                names.append(name)
            if not boxes:
                n_frame_empty += 1
                continue

            infos_all.append(dict(
                token=f"{seq}_{stem}",
                timestamp=len(infos_all),
                lidar_path=os.path.abspath(ROOT / seq / "points" / f"{stem}.npy"),
                sweeps=[],
                cams={"CAM_FRONT": dict(
                    data_path=os.path.abspath(ROOT / seq / "image" / f"{stem}.png"),
                    cam_intrinsic=K,
                    distortion=np.zeros(5, np.float32),      # 图像已去畸变
                    sensor2lidar_rotation=R_c2l.astype(np.float32),
                    sensor2lidar_translation=t_c2l.astype(np.float32),
                    sensor2ego_rotation=[1.0, 0, 0, 0],
                    sensor2ego_translation=[0.0, 0, 0],
                )},
                lidar2ego_rotation=[1.0, 0, 0, 0],
                lidar2ego_translation=[0.0, 0, 0],
                ego2global_rotation=[1.0, 0, 0, 0],
                ego2global_translation=[0.0, 0, 0],
                gt_boxes=np.asarray(boxes, np.float32).reshape(-1, 7),
                gt_names=np.array(names),
                gt_velocity=np.zeros((len(boxes), 2), np.float32),
                num_lidar_pts=np.ones(len(boxes), np.int64),
            ))
        print(f"{seq}: 累计 {len(infos_all)} 帧")

    print(f"丢弃范围内点数 <{MIN_INRANGE_PTS} 的帧: {n_drop[0]}")
    print(f"丢弃框内点数 <{MIN_BOX_PTS} 的标注: {n_box_drop}")
    print(f"清理后无目标而丢弃的帧: {n_frame_empty}")

    # 每个序列最后 20% 作 val（保持场景多样性）
    train, val = [], []
    for seq in seqs:
        idx = [i for i, it in enumerate(infos_all) if it["token"].startswith(seq + "_")]
        k = max(1, int(len(idx) * (1 - VAL_RATIO)))
        train += [infos_all[i] for i in idx[:k]]
        val += [infos_all[i] for i in idx[k:]]
    for i, it in enumerate(train + val):
        it["timestamp"] = i

    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("train", train), ("val", val)):
        p = OUT / f"infos_{name}.pkl"
        pickle.dump(dict(infos=data, metadata=dict(version="seepersea", classes=CLASSES)), open(p, "wb"))
        nb = sum(len(it["gt_boxes"]) for it in data)
        print(f"  {p.name}: {len(data)} 帧, {nb} 个框")

    from collections import Counter
    c = Counter(n for it in train + val for n in it["gt_names"])
    print("类别统计:", dict(c))


if __name__ == "__main__":
    main()
