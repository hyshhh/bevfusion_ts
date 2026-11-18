"""训练中周期性跑验证集 BEV AP 的 hook。

和 ``demo/eval_bev_ap.py`` 用的是同一套 AP 实现（``mmdet3d.evaluation.bev_ap``），
区别只是预测结果不再落盘再由命令行读取，而是在训练进程里直接推理后统计。

在 ``mmdet3d/apis/train.py`` 里按配置注册：

.. code-block:: yaml

    bev_ap_eval:
      interval: 5
      iou: 0.3
      ranges: ["0-30", "30-50", "50-80"]
      score_thr: 0.001

评测结果：
  - 打印到训练日志（表格 + mAP 一行）
  - 写 ``${run_dir}/eval/epoch_XXX_ap.txt`` 和 ``${run_dir}/eval/eval_summary.csv``
  - 通过 ``log_buffer`` 进 TensorBoard（``train/bevAP_all`` 等）

评测失败（OOM、数据异常等）只打 warning，不会中断训练。
"""
import json
import os
import os.path as osp
import pickle
import time
import traceback

import numpy as np
import torch
from mmcv.runner import HOOKS, Hook

from mmdet3d.evaluation.bev_ap import (compute_bev_ap, format_ap_table,
                                      range_label)


@HOOKS.register_module()
class BEVAPHook(Hook):
    """每 ``interval`` 轮在验证集上算一次 BEV AP。

    Args:
        dataloader (DataLoader): 验证集 dataloader（``apis/train.py`` 里已经
            按 ``test_mode=True`` 建好，顺序与 ``ann_file`` 的 infos 一致）。
        interval (int): 每多少轮评测一次。
        start (int | None): 从第几轮开始评测，``None`` 表示从第一轮算起。
        iou (float): BEV 旋转框 IoU 匹配阈值。
        ranges (list[str]): 距离区间，如 ``["0-30", "30-50", "50-80"]``。
        score_thr (float): 参与统计的最低置信度。
        classes (list[str]): 类别名，默认取 ``cfg.object_classes``。
        ann_file (str): 验证集 info 的 pkl，默认取 ``cfg.ann_val``。
        out_dir (str): 结果目录，默认 ``${run_dir}/eval``。
        save_preds (bool): 是否把每轮预测另存 pkl（几十 MB/次）。
        save_best (bool): mAP 刷新时是否复制一份 ``best_mAP.pth``（约 400 MB）。
    """

    def __init__(self,
                 dataloader,
                 interval=5,
                 start=None,
                 by_epoch=True,
                 iou=0.3,
                 ranges=("0-30", "30-50", "50-80"),
                 score_thr=0.001,
                 classes=None,
                 ann_file=None,
                 out_dir=None,
                 save_preds=False,
                 save_best=False):
        self.dataloader = dataloader
        self.interval = int(interval)
        self.start = start
        self.by_epoch = by_epoch
        self.iou = float(iou)
        self.ranges = [str(r) for r in ranges]
        self.score_thr = float(score_thr)
        self.classes = list(classes) if classes is not None else None
        self.ann_file = ann_file
        self.out_dir = out_dir
        self.save_preds = save_preds
        self.save_best = save_best

        self._infos = None
        self._best = None

    # ------------------------------------------------------------------ utils
    def _get_infos(self, runner):
        if self._infos is None:
            path = self.ann_file
            if path is None:
                cfg = getattr(runner, "cfg", None)
                path = cfg.ann_val if cfg is not None else None
            if path is None:
                raise ValueError("BEVAPHook 需要 ann_file（或 cfg.ann_val）")
            with open(path, "rb") as f:
                self._infos = pickle.load(f)["infos"]
            runner.logger.info(f"[BEVAP] 验证集 {path} 共 {len(self._infos)} 帧")
        return self._infos

    def _get_classes(self, runner):
        if self.classes is None:
            cfg = getattr(runner, "cfg", None)
            if cfg is None:
                raise ValueError("BEVAPHook 需要 classes（或 cfg.object_classes）")
            self.classes = list(cfg.object_classes)
        return self.classes

    def _result_dir(self, runner):
        out = self.out_dir or osp.join(runner.work_dir, "eval")
        os.makedirs(out, exist_ok=True)
        return out

    def _infer(self, runner):
        """在验证集上跑一遍推理，返回与 ``tools/test.py --out`` 相同的列表。"""
        # 延迟 import：mmdet3d.apis 会反过来 import mmdet3d.runner，放在模块
        # 顶层会形成循环 import。
        from mmcv.parallel import MMDataParallel
        from mmdet3d.apis import single_gpu_test

        model = runner.model
        if runner.world_size > 1:
            from mmdet.apis import multi_gpu_test

            tmpdir = osp.join(self._result_dir(runner), "_tmp")
            return multi_gpu_test(model, self.dataloader, tmpdir, gpu_collect=False)

        raw_model = model.module if hasattr(model, "module") else model
        was_training = raw_model.training
        try:
            # 必须用 MMDataParallel 包一层：dataloader 出的是 DataContainer，
            # 裸模型不会做 scatter，voxelize 里会报 "not iterable"。
            infer_model = MMDataParallel(raw_model, device_ids=[0])
            results = single_gpu_test(infer_model, self.dataloader)
        finally:
            raw_model.train(was_training)
        return results

    # ------------------------------------------------------------- main entry
    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return
        epoch = runner.epoch + 1
        if self.start is not None and epoch < self.start:
            return
        if not self.every_n_epochs(runner, self.interval):
            return
        try:
            self._run(runner, epoch)
        except Exception as e:  # 评测挂了也不能停训练
            runner.logger.warning(f"[BEVAP] 第 {epoch} 轮评测失败，已跳过：{e}")
            runner.logger.warning(traceback.format_exc())
        finally:
            torch.cuda.empty_cache()

    def _run(self, runner, epoch):
        logger = runner.logger
        infos = self._get_infos(runner)
        classes = self._get_classes(runner)
        logger.info(f"[BEVAP] 第 {epoch} 轮开始评测：{len(infos)} 帧，"
                    f"IoU={self.iou}，score>={self.score_thr}")

        t0 = time.time()
        # 训练把显存缓存池撑得很大时（bs 大尤其明显），先把缓存还给驱动，
        # 避免推理这边因为拿不到连续的显存块而 OOM。
        torch.cuda.empty_cache()
        results = self._infer(runner)
        infer_time = time.time() - t0
        torch.cuda.empty_cache()

        if runner.rank != 0:
            return
        if len(results) != len(infos):
            raise RuntimeError(
                f"预测数 {len(results)} 与验证集帧数 {len(infos)} 不一致，"
                "请检查 data.val.ann_file 是否和 ann_val 一致")

        t1 = time.time()
        res = compute_bev_ap(infos, classes, results, iou=self.iou,
                             ranges=self.ranges, score_thr=self.score_thr)
        ap_time = time.time() - t1

        table = format_ap_table(classes, res["ranges"], res["ap"],
                                mAP=res["mAP"])
        map_str = "  ".join(f"{range_label(k)}: {v:.4f}"
                            for k, v in res["mAP"].items() if np.isfinite(v))
        logger.info(f"\n[BEVAP] epoch {epoch} （推理 {infer_time:.1f}s + "
                    f"统计 {ap_time:.1f}s）\n{table}")
        logger.info(f"[BEVAP] mAP  {map_str}")

        self._dump(runner, epoch, res, results, table, map_str)
        self._log_metrics(runner, res)

    # ---------------------------------------------------------------- outputs
    def _dump(self, runner, epoch, res, results, table, map_str):
        out = self._result_dir(runner)
        keys = list(res["ranges"])
        classes = list(res["classes"])
        ap = res["ap"]          # [C, R]，R 的第 0 列是 all

        def _fmt(v):
            return f"{v:.6f}" if np.isfinite(v) else "nan"

        txt_path = osp.join(out, f"epoch_{epoch:03d}_ap.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"epoch {epoch}  iou {self.iou}  score_thr {self.score_thr}  "
                    f"n_frames {len(self._infos)}\n")
            f.write(table + "\n")

        # 每轮每类的完整 AP 矩阵，方便后面按类画曲线 / 做表
        json_path = osp.join(out, f"epoch_{epoch:03d}_ap.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(dict(
                epoch=epoch,
                iou=self.iou,
                score_thr=self.score_thr,
                n_frames=len(self._infos),
                ranges=keys,
                mAP={k: (None if not np.isfinite(v) else float(v))
                     for k, v in res["mAP"].items()},
                AP={c: {k: (None if not np.isfinite(ap[ci, ri]) else float(ap[ci, ri]))
                        for ri, k in enumerate(keys)}
                    for ci, c in enumerate(classes)},
            ), f, indent=2, ensure_ascii=False)

        # 汇总 csv：mAP(各距离段) + 每类 AP(all)
        csv_path = osp.join(out, "eval_summary.csv")
        header = (["epoch"] + [f"mAP_{k}" for k in keys]
                  + [f"AP_{c}" for c in classes])
        row = ([str(epoch)] + [_fmt(res["mAP"][k]) for k in keys]
               + [_fmt(ap[ci, 0]) for ci in range(len(classes))])
        new_file = not osp.exists(csv_path)
        with open(csv_path, "a", encoding="utf-8") as f:
            if new_file:
                f.write(",".join(header) + "\n")
            f.write(",".join(row) + "\n")

        if self.save_preds:
            import mmcv

            mmcv.dump(results, osp.join(out, f"epoch_{epoch:03d}_preds.pkl"))

        cur = res["mAP"]["all"]
        if np.isfinite(cur) and (self._best is None or cur > self._best[1]):
            self._best = (epoch, float(cur))
            runner.logger.info(f"[BEVAP] 新最佳 mAP：epoch {epoch} -> {cur:.4f}")
            with open(osp.join(out, "best.json"), "w", encoding="utf-8") as f:
                json.dump(dict(epoch=epoch, mAP=float(cur),
                               ranges={k: (None if not np.isfinite(v) else float(v))
                                       for k, v in res["mAP"].items()},
                               AP={c: (None if not np.isfinite(ap[ci, 0]) else float(ap[ci, 0]))
                                   for ci, c in enumerate(classes)}),
                          f, indent=2, ensure_ascii=False)
            if self.save_best:
                src = osp.join(runner.work_dir, f"epoch_{epoch}.pth")
                if osp.exists(src):
                    import shutil

                    shutil.copyfile(src, osp.join(runner.work_dir, "best_mAP.pth"))
                    runner.logger.info(f"[BEVAP] 已复制 {src} -> best_mAP.pth")

    def _log_metrics(self, runner, res):
        """把 mAP 塞进 log_buffer，让 TextLogger/TensorBoard hook 顺带打出来。"""
        metrics = {}
        for k, v in res["mAP"].items():
            if not np.isfinite(v):
                continue
            key = "bevAP_all" if k == "all" else f"bevAP_{k.replace('-', '_')}m"
            metrics[key] = float(v)
        if not metrics:
            return
        runner.log_buffer.update(metrics, 1)
        runner.log_buffer.average()
