import math

import torch
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import (
    DistSamplerSeedHook,
    EpochBasedRunner,
    GradientCumulativeFp16OptimizerHook,
    Fp16OptimizerHook,
    OptimizerHook,
    build_optimizer,
    build_runner,
    wrap_fp16_model,
)
from mmdet3d.runner import CustomEpochBasedRunner
from mmdet3d.runner.hooks.bev_ap_eval import BEVAPHook

from mmcv.runner.hooks import HOOKS  # noqa: E402


@HOOKS.register_module()
class NanSafeOptimizerHook(OptimizerHook):
    """先正常反向（保证 DDP 所有参数都拿到梯度），再根据 loss 是否有限决定要不要丢弃梯度。

    直接跳过反向会让 DDP 报 "Expected to have finished reduction"，
    所以这里必须让 backward 跑完，然后清掉 nan 梯度，使这一步等价于不更新。
    """

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()

        loss_val = runner.outputs['loss'].detach()
        bad = not torch.isfinite(loss_val).all()
        if bad:
            # 丢弃这一步的梯度，权重不变
            runner.optimizer.zero_grad()
            runner.log_buffer.update({'nan_skip': 1.0},
                                     runner.outputs['num_samples'])
            return

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step()


@HOOKS.register_module()
class NanSafeFp16OptimizerHook(Fp16OptimizerHook):
    """Fp16OptimizerHook + nan 保护（buffer 回滚 + 参数快照回滚）。

    fp16 下这个模型会出现两类崩溃，实测都遇到过：
      1. forward 出 nan/inf 时，BN 的 running_mean/var 被污染且不会自愈，
         之后每个 forward 都是 nan（epoch 15 那次）；
      2. 权重先缓慢涨大（检查点里出现 |w|>100），在 fp16 下溢出成 inf/nan，
         GradScaler 只能拦住 nan 梯度，拦不住已经写进权重的 nan（epoch 6 那次）。

    所以这里：
      - 每次迭代前快照浮点 buffer；每 ``snapshot_interval`` 次迭代快照一次全部参数；
      - 该步 loss 非有限、或梯度范数非有限、或参数里出现非有限值时，把 buffer 和
        参数一起回滚到快照，这一步不计入更新（记 nan_skip / nan_rollback）；
      - **连续**回滚次数超过 ``max_rollbacks``（一旦某步正常就清零）就存一份
        ``nan_debug.pth`` 并报错退出；累计次数只用于日志，不会触发退出，
        避免一整晚都在空转。
    """

    def __init__(self, *args, snapshot_interval=20, max_rollbacks=30, **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshot_interval = int(snapshot_interval)
        self.max_rollbacks = int(max_rollbacks)
        self._param_snapshot = None
        self._buf_snapshot = None
        self._rollbacks = 0        # 累计（仅统计/日志）
        self._rollbacks_streak = 0  # 连续（用于退出判据）

    @staticmethod
    def _unwrap(model):
        return model.module if hasattr(model, "module") else model

    def _snapshot(self, runner):
        model = self._unwrap(runner.model)
        self._param_snapshot = {
            name: p.detach().clone() for name, p in model.named_parameters()
        }
        self._buf_snapshot = {
            name: b.detach().clone()
            for name, b in model.named_buffers()
            if b.is_floating_point()
        }

    def _restore(self, runner):
        model = self._unwrap(runner.model)
        with torch.no_grad():
            for name, p in model.named_parameters():
                snap = self._param_snapshot.get(name) if self._param_snapshot else None
                if snap is not None and snap.shape == p.shape:
                    p.copy_(snap)
            for name, b in model.named_buffers():
                snap = self._buf_snapshot.get(name) if self._buf_snapshot else None
                if snap is not None and snap.shape == b.shape:
                    b.copy_(snap)

    def _has_bad_params(self, runner):
        model = self._unwrap(runner.model)
        for p in model.parameters():
            if p.is_floating_point() and not torch.isfinite(p).all():
                return True
        return False

    def before_train_epoch(self, runner):
        self._snapshot(runner)

    def before_train_iter(self, runner):
        if (self._param_snapshot is None
                or runner.iter % self.snapshot_interval == 0):
            self._snapshot(runner)

    def after_train_iter(self, runner):
        runner.model.zero_grad()
        runner.optimizer.zero_grad()

        loss = runner.outputs['loss']
        bad = not bool(torch.isfinite(loss.detach()).all())

        self.loss_scaler.scale(loss).backward()
        self.loss_scaler.unscale_(runner.optimizer)

        grad_norm = None
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        if grad_norm is not None and not math.isfinite(float(grad_norm)):
            bad = True

        # 溢出时 GradScaler 内部会跳过这一步的 step()，并自动降低 loss scale
        self.loss_scaler.step(runner.optimizer)
        self.loss_scaler.update(self._scale_update_param)

        if not bad and self._has_bad_params(runner):
            bad = True
            runner.log_buffer.update({'nan_weight': 1.0},
                                     runner.outputs['num_samples'])

        if bad:
            self._rollbacks += 1
            self._rollbacks_streak += 1
            self._restore(runner)
            runner.optimizer.zero_grad()
            runner.log_buffer.update({'nan_skip': 1.0},
                                     runner.outputs['num_samples'])
            runner.logger.warning(
                'nan/inf at iter %d (rollback #%d, 连续 %d)，rollback to snapshot',
                runner.iter, self._rollbacks, self._rollbacks_streak)
            if self._rollbacks_streak > self.max_rollbacks:
                import os.path as osp
                debug_path = osp.join(runner.work_dir, 'nan_debug.pth')
                torch.save({'state_dict': self._unwrap(runner.model).state_dict(),
                            'epoch': runner.epoch, 'iter': runner.iter},
                           debug_path)
                raise RuntimeError(
                    f'连续 nan 回滚超过 {self.max_rollbacks} 次，已保存 {debug_path}，'
                    '请降低学习率或排查数据后重跑')

        if not bad:
            self._rollbacks_streak = 0

        runner.meta.setdefault(
            'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()


from mmdet3d.utils import get_root_logger
from mmdet.core import DistEvalHook
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor


def train_model(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
):
    logger = get_root_logger()

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            None,
            dist=distributed,
            seed=cfg.seed,
        )
        for ds in dataset
    ]

    # 真正启用 fp16：原仓库从来没调用过 wrap_fp16_model，
    # 导致 fp16 配置只选了个 Fp16OptimizerHook，模型其实还是 fp32
    if cfg.get("fp16", None) is not None:
        wrap_fp16_model(model)
        get_root_logger().info("fp16: model converted to half precision")

    # put model on gpus
    find_unused_parameters = cfg.get("find_unused_parameters", False)
    # Sets the `find_unused_parameters` parameter in
    # torch.nn.parallel.DistributedDataParallel
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters,
    )

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.run_dir,
            logger=logger,
            meta={},
        ),
    )
    
    if hasattr(runner, "set_dataset"):
        runner.set_dataset(dataset)

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        if "cumulative_iters" in cfg.optimizer_config:
            optimizer_config = GradientCumulativeFp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed
            )
        elif cfg.get("nan_guard", True):
            # fp16 一旦出现 nan，BN running stats 会被永久污染（实测 epoch 15 崩过），
            # 用带 buffer 回滚的版本兜住
            optimizer_config = NanSafeFp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed
            )
            logger.info("[fp16] nan guard enabled (NanSafeFp16OptimizerHook)")
        else:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed
            )
    elif distributed and "type" not in cfg.optimizer_config:
        optimizer_config = NanSafeOptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
    )
    if isinstance(runner, EpochBasedRunner):
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
        )
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
        eval_hook = DistEvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

        # 每 N 轮在验证集上跑一次 BEV AP（KITTI 风格 IoU 匹配），
        # 和 demo/eval_bev_ap.py 共用 mmdet3d.evaluation.bev_ap 的实现。
        bev_ap_cfg = cfg.get("bev_ap_eval", None)
        if bev_ap_cfg is not None:
            bev_ap_cfg = dict(bev_ap_cfg)
            bev_ap_cfg.setdefault("classes", list(cfg.object_classes))
            bev_ap_cfg.setdefault("ann_file", cfg.ann_val)
            runner.register_hook(
                BEVAPHook(dataloader=val_dataloader, **bev_ap_cfg))
            logger.info("[BEVAP] 周期性验证集评测已开启：每 "
                        f"{bev_ap_cfg.get('interval', 5)} 轮一次，"
                        f"IoU={bev_ap_cfg.get('iou', 0.3)}")

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, [("train", 1)])
