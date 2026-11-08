"""用官方 cv2.undistort 对全部图像去畸变，逐帧使用自己的 K / D。

    python configs/YZdata/undistort_images.py --workers 10
    python configs/YZdata/undistort_images.py --dry-run

输出 images_undist/，内参保持不变（new_K = K），因此 pkl 里的 cam_intrinsic 无需修改。
去畸变后的图像符合针孔模型，LSS 不需要任何额外处理。
"""
import argparse
import glob
import os
import pickle
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = "/home/lab-203/hys/BEV/bevfusion/data/data_old/custom_3000"
SRC = os.path.join(ROOT, "images")
CALIB = os.path.join(ROOT, "calib")
DST = os.path.join(ROOT, "images_undist")
PKL_DIR = "/home/lab-203/hys/BEV/bevfusion/data/data_pkl/custom_3000"


def read_kd(tok):
    L = open(os.path.join(CALIB, f"{tok}.txt")).read().split("\n")
    K = np.array([[float(x) for x in L[i].split()] for i in (8, 9, 10)], np.float32)
    txt = "\n".join(L)
    j = txt.find("Distortion:")
    D = np.array([float(x) for x in txt[j + 11:].strip().strip("[]").split(",")], np.float32)
    return K, D


def work(name):
    tok = os.path.splitext(name)[0]
    dst = os.path.join(DST, name)
    if os.path.exists(dst):
        return name, "skip"
    K, D = read_kd(tok)
    img = cv2.imread(os.path.join(SRC, name))
    # new_K = K：内参不变，边缘内容会被推出画面（训练中心裁剪本来也不用）
    cv2.imwrite(dst, cv2.undistort(img, K, D, None, K))
    return name, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    names = sorted(os.listdir(SRC))
    print(f"待处理 {len(names)} 张")
    if args.dry_run:
        for tok in (names[0].split(".")[0], names[3000].split(".")[0]):
            K, D = read_kd(tok)
            print(f"  {tok}: fx={K[0,0]:.1f}  D={np.round(D,4).tolist()}")
        return

    os.makedirs(DST, exist_ok=True)
    with Pool(args.workers) as pool:
        done = 0
        for name, st in pool.imap_unordered(work, names, chunksize=32):
            done += 1
            if done % 1000 == 0:
                print(f"  {done}/{len(names)}", flush=True)
    print("图像完成")

    for p in sorted(glob.glob(os.path.join(PKL_DIR, "infos_*.pkl"))):
        data = pickle.load(open(p, "rb"))
        n = 0
        for info in data["infos"]:
            for cam in info["cams"].values():
                old = cam["data_path"]
                new = old.replace("/images/", "/images_undist/")
                if new != old:
                    cam["data_path"] = new
                    n += 1
        pickle.dump(data, open(p, "wb"))
        print(f"  {os.path.basename(p)}: 更新 {n} 条路径")


if __name__ == "__main__":
    main()
