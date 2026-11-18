#!/bin/bash
# 用法: bash run_export.sh /path/to/ckpt.pth [fp16|int8] [config] [num_cams] [per_class_topk]
#   ckpt 可以是训练产出的 {state_dict,...} 或已转换的整模型
#   默认配置: configs/YZdata/camera+lidar_180x180_spconv.yaml
#   默认相机数: 3
#   per_class_topk 可选（逗号分隔），不填则用模型自带（yaml 里的值）
set -e
CKPT=${1:?用法: bash run_export.sh /path/to/ckpt.pth [fp16|int8] [config] [num_cams]}
MODE=${2:-fp16}
CFG=${3:-configs/YZdata/camera+lidar_180x180_spconv.yaml}
NCAMS=${4:-3}
PCTOPK=${5:-}
EXTRA=""
[ "$MODE" = "fp16" ] && EXTRA="--fp16"

# 训练产出的 checkpoint 是 {'state_dict':..., 'meta':...}，qat 脚本需要整模型对象，
# 先自动转一份（同时生成 N 相机的 example-data）
echo ">>> 转换 checkpoint（顺便生成 ${NCAMS} 相机的 example-data）"
python tools/to_deploy_ckpt.py "$CFG" "$CKPT" --num-cams "$NCAMS"

DEPLOY_CKPT=$(ls -t qat/ckpt/deploy_*.pth | head -1)
echo ">>> 使用 $DEPLOY_CKPT"

echo ">>> camera backbone + vtransform"
python qat/export-camera.py --ckpt=$DEPLOY_CKPT $EXTRA --device cpu

echo ">>> fuser + head.bbox"
# per_class_topk 覆盖（不传则用 pickle 里烘进去的 yaml 值）
PCTOPK_ARG=""
[ -n "$PCTOPK" ] && PCTOPK_ARG="--per-class-topk=$PCTOPK"
python qat/export-transfuser.py --ckpt=$DEPLOY_CKPT $EXTRA --device cpu $PCTOPK_ARG

echo ">>> lidar scn"
python qat/export-scn.py --ckpt=$DEPLOY_CKPT --save=qat/onnx_$MODE/lidar.backbone.onnx --device cpu
# export-scn 会追加 .xyz / .zyx 后缀，这里补一个不带后缀的副本方便引用
SCN_OUT=$(ls -t qat/onnx_$MODE/lidar.backbone.*.onnx 2>/dev/null | head -1)
[ -n "$SCN_OUT" ] && cp "$SCN_OUT" qat/onnx_$MODE/lidar.backbone.onnx

tar czf deploy_$MODE.tgz -C qat/onnx_$MODE .
echo ">>> 已打包 deploy_$MODE.tgz"
tar tzf deploy_$MODE.tgz
