"""把 SeePerSea 全部序列整理成 KITTI 风格（雷达坐标系），清理无用文件。

    python configs/SeePerSea/prepare_seepersea.py --workers 12

输入每个序列：annotations/(雷达系 3D 框) calib/ undist_images/CAM_FRONT/ point_clouds/
输出：<dst>/<seq>/{image,points,calib,label,label_2d}/

删除：motion_data、timestamp、camera_label、kitti_label(相机系)、empty_label_files.txt
优化：PCD(ASCII 65536 行) -> npy(只留非零点)
"""
import argparse
import json
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np

W, H = 640, 480


def read_pcd(path):
    raw = open(path, "rb").read()
    if b"DATA ascii" in raw:
        i = raw.index(b"DATA ascii") + len(b"DATA ascii") + 1
        a = np.fromstring(raw[i:].decode("ascii", errors="ignore"), sep=" ", dtype=np.float32)
        a = a.reshape(-1, 4)
    else:
        i = raw.index(b"DATA binary") + len(b"DATA binary") + 1
        hdr = raw[:i].decode("ascii", errors="ignore")
        n = int([l for l in hdr.split("\n") if l.startswith("POINTS")][0].split()[1])
        nf = len([l for l in hdr.split("\n") if l.startswith("FIELDS")][0].split()) - 1
        a = np.frombuffer(raw[i:i + n * nf * 4], dtype=np.float32).reshape(-1, nf)
    nz = (a[:, 0] != 0) | (a[:, 1] != 0) | (a[:, 2] != 0)
    return a[nz][:, :4]


def read_calib(path):
    d = {}
    for line in open(path):
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = np.array([float(x) for x in v.split()])
    return d


def project(b, P2, Tr):
    """雷达系 3D 框 -> 图像 2D 框；不可见返回 None。"""
    w, l_, h = b["dimension"]["width"], b["dimension"]["length"], b["dimension"]["height"]
    l_, w = w, l_                      # 该数据集 length/width 与 KITTI 相反
    x, y, z = b["location"]["x"], b["location"]["y"], b["location"]["z"]
    yaw = b["orientation"]["rotationYaw"]
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    cor = np.array([
        [-l_/2, -w/2, -h/2], [-l_/2, -w/2, h/2], [-l_/2, w/2, -h/2], [-l_/2, w/2, h/2],
        [l_/2, -w/2, -h/2], [l_/2, -w/2, h/2], [l_/2, w/2, -h/2], [l_/2, w/2, h/2],
    ]) @ Rz.T + np.array([x, y, z])
    pc = cor @ Tr[:, :3].T + Tr[:, 3]
    if (pc[:, 2] <= 0.1).any():
        return None
    uv = (P2 @ np.concatenate([pc, np.ones((8, 1))], 1).T)
    uv = uv[:2] / uv[2]
    box = [float(uv[0].min()), float(uv[1].min()), float(uv[0].max()), float(uv[1].max())]
    if box[2] < 0 or box[0] > W or box[3] < 0 or box[1] > H:
        return None
    return [max(box[0], 0), max(box[1], 0), min(box[2], W), min(box[3], H)]


def work(job):
    seq, stem, src, dst = job
    s, d = Path(src) / seq, Path(dst) / seq
    img = s / "undist_images" / "CAM_FRONT" / f"{stem}.png"
    if not img.exists():
        return 0, 0
    shutil.copy2(img, d / "image" / f"{stem}.png")

    pcd = s / "point_clouds" / f"{stem}.pcd"
    if pcd.exists():
        np.save(d / "points" / f"{stem}.npy", read_pcd(pcd))

    cal = read_calib(s / "calib" / f"{stem}.txt")
    P2, Tr = cal["P2"].reshape(3, 4), cal["Tr_velo_to_cam"].reshape(3, 4)
    with open(d / "calib" / f"{stem}.txt", "w") as f:
        f.write("P2: " + " ".join(f"{v:.6f}" for v in P2.ravel()) + "\n")
        f.write("Tr_velo_to_cam: " + " ".join(f"{v:.6f}" for v in Tr.ravel()) + "\n")

    ann = s / "annotations" / f"{stem}.json"
    lines3d, lines2d = [], []
    if ann.exists() and ann.stat().st_size > 0:
        try:
            data = json.load(open(ann))
        except Exception:
            data = {}
        for lb in data.get("labels", []):
            b = lb["box3d"]
            w, l_, h = b["dimension"]["width"], b["dimension"]["length"], b["dimension"]["height"]
            l_, w = w, l_
            x, y, z = b["location"]["x"], b["location"]["y"], b["location"]["z"]
            yaw = b["orientation"]["rotationYaw"]
            b2 = project(b, P2, Tr)
            s2 = " ".join(f"{v:.1f}" for v in b2) if b2 else "-1 -1 -1 -1"
            if b2:
                lines2d.append(f"{lb['category']} {s2}")
            lines3d.append(f"{lb['category']} 0 0 0 {s2} {h:.3f} {w:.3f} {l_:.3f} "
                           f"{x:.3f} {y:.3f} {z:.3f} {yaw:.3f}")
    (d / "label" / f"{stem}.txt").write_text("\n".join(lines3d) + ("\n" if lines3d else ""))
    (d / "label_2d" / f"{stem}.txt").write_text("\n".join(lines2d) + ("\n" if lines2d else ""))
    return 1, len(lines3d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/ss_sep/full_data/separate")
    ap.add_argument("--dst", default="/home/lab-203/hys/BEV/bevfusion/data/data_old/seepersea")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    seqs = sorted(p.name for p in src.iterdir() if p.is_dir() and (p / "annotations").is_dir())
    print(f"{len(seqs)} 个序列")

    jobs = []
    for seq in seqs:
        d = dst / seq
        for sub in ("image", "points", "calib", "label", "label_2d"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        for cal in sorted((src / seq / "calib").glob("*.txt")):
            jobs.append((seq, cal.stem, str(src), str(dst)))
    print(f"{len(jobs)} 帧")

    total_f = total_b = 0
    with Pool(args.workers) as pool:
        for i, (nf, nb) in enumerate(pool.imap_unordered(work, jobs, chunksize=16), 1):
            total_f += nf
            total_b += nb
            if i % 2000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    print(f"完成: {total_f} 帧, {total_b} 个 3D 框")


if __name__ == "__main__":
    main()
