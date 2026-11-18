"""YZ 数据集专用数据增强。

针对水面场景的实际数据特点设计：
- 点云 x 范围 [-30, 210] 严重不对称，不能做前后翻转（会丢掉 45% 的点）
- 三个采集批次相机色温差异大（k1=-0.81/-0.40/-0.08），需要颜色增强
- 单帧点云很稀疏（约 1900 点），丢点比例要小
"""
from typing import Dict

import cv2
import numpy as np
import torch
from PIL import Image

from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class YZFlip3D:
    """只做左右翻转（y 轴），不做前后翻转（x 轴）。

    mmdet3d 自带的 RandomFlip3D 会独立地以 0.5 概率翻转 x 轴，
    但本数据 x 范围是 [-30, 210]，翻转后 x 变成 [-210, 30]，
    和 point_cloud_range 只有 [-30, 30] 的交集，等于丢掉 45% 的点
    和所有 x > 30 的标注框。

    Args:
        prob (float): 左右翻转的概率。
    """

    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob

    def __call__(self, data: Dict) -> Dict:
        if np.random.rand() > self.prob:
            return data

        if "points" in data:
            data["points"].flip("horizontal")
        if "gt_bboxes_3d" in data:
            data["gt_bboxes_3d"].flip("horizontal")

        # 同步到 lidar_aug_matrix，保证相机和点云的增强一致
        if "lidar_aug_matrix" in data:
            rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
            data["lidar_aug_matrix"][:3, :] = rot @ data["lidar_aug_matrix"][:3, :]

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(prob={self.prob})"


@PIPELINES.register_module()
class PointsJitter:
    """给点云 xyz 加高斯噪声，模拟测距误差。

    本数据单帧约 2000~5500 点，最小的 buoy 类框内中位 182 点，
    噪声不宜过大：sigma=0.02 m 相当于 80 m 处 0.025% 的距离误差。

    Args:
        sigma (float): 高斯噪声标准差（米）。
        prob (float): 施加该增强的概率。
    """

    def __init__(self, sigma: float = 0.02, prob: float = 0.5) -> None:
        self.sigma = sigma
        self.prob = prob

    def __call__(self, data: Dict) -> Dict:
        if np.random.rand() > self.prob or "points" not in data:
            return data

        points = data["points"]
        noise = torch.randn(points.tensor.shape[0], 3) * self.sigma
        points.tensor[:, :3] += noise.to(points.tensor.device)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sigma={self.sigma}, prob={self.prob})"


@PIPELINES.register_module()
class PointsDropout:
    """随机丢弃一部分点，提升对稀疏回波的鲁棒性。

    本数据最小目标 buoy 框内中位 182 点，drop_ratio=0.05 时
    一个框损失超过一半点的概率极低（二项分布尾部 < 1e-30），非常安全。

    Args:
        drop_ratio (float): 每个点被丢弃的概率。
        prob (float): 施加该增强的概率。
    """

    def __init__(self, drop_ratio: float = 0.05, prob: float = 0.5) -> None:
        self.drop_ratio = drop_ratio
        self.prob = prob

    def __call__(self, data: Dict) -> Dict:
        if np.random.rand() > self.prob or "points" not in data:
            return data

        points = data["points"]
        num = points.tensor.shape[0]
        if num == 0:
            return data

        keep = torch.rand(num) > self.drop_ratio
        if keep.sum() < 10:  # 防止丢成空点云
            return data
        data["points"] = points[keep]
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(drop_ratio={self.drop_ratio}, prob={self.prob})"


@PIPELINES.register_module()
class MultiViewColorJitter:
    """多视角图像的光度增强：亮度 / 对比度 / 饱和度 / 色调。

    本数据三个采集批次的相机色温差异很大（k1 = -0.81 / -0.40 / -0.08），
    水面反光也随光照变化剧烈，颜色增强能显著缓解跨批次域差异。

    输入是 LoadMultiViewImageFromFiles + ImageAug3D 之后的 PIL 列表，
    输出仍是 PIL 列表，供后面的 ImageNormalize（torchvision.ToTensor）使用。

    Args:
        brightness_delta (float): 亮度偏移上限（0~255 尺度）。
        contrast_range (tuple): 对比度缩放范围。
        saturation_range (tuple): 饱和度缩放范围。
        hue_delta (float): 色调偏移上限（OpenCV 的 H 通道 0~180）。
        prob (float): 施加该增强的概率。
    """

    def __init__(
        self,
        brightness_delta: float = 24.0,
        contrast_range=(0.75, 1.25),
        saturation_range=(0.75, 1.25),
        hue_delta: float = 6.0,
        prob: float = 0.5,
    ) -> None:
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_delta = hue_delta
        self.prob = prob

    def __call__(self, data: Dict) -> Dict:
        if np.random.rand() > self.prob or "img" not in data:
            return data

        new_imgs = []
        for img in data["img"]:
            arr = np.array(img, dtype=np.float32)  # RGB, 0~255

            # 亮度
            if np.random.rand() < 0.5:
                arr = arr + np.random.uniform(-self.brightness_delta, self.brightness_delta)

            # 对比度
            if np.random.rand() < 0.5:
                alpha = np.random.uniform(*self.contrast_range)
                mean = arr.mean()
                arr = (arr - mean) * alpha + mean

            arr = np.clip(arr, 0, 255).astype(np.uint8)

            # 饱和度 + 色调（OpenCV HSV：H ∈ [0,180)）
            if np.random.rand() < 0.5:
                hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
                if np.random.rand() < 0.5:
                    beta = np.random.uniform(*self.saturation_range)
                    hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * beta, 0, 255)
                if self.hue_delta > 0 and np.random.rand() < 0.5:
                    shift = np.random.uniform(-self.hue_delta, self.hue_delta)
                    hsv[..., 0] = (hsv[..., 0].astype(np.float32) + shift) % 180
                arr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            # 必须返回 PIL：下游 ImageNormalize 用 torchvision.ToTensor，
            # 它对 float 数组不会除以 255（只对 uint8 做），传 numpy float 会漏归一化
            new_imgs.append(Image.fromarray(arr.astype(np.uint8), mode="RGB"))

        data["img"] = new_imgs
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(brightness={self.brightness_delta}, "
            f"contrast={self.contrast_range}, saturation={self.saturation_range}, "
            f"hue={self.hue_delta}, prob={self.prob})"
        )
