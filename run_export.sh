#!/bin/bash
# 用法: bash run_export.sh /path/to/ckpt.pth [fp16|int8] [config] [num_cams] [per_class_topk]
#   ckpt 可以是训练产出的 {state_dict,...} 或已转换的整模型
#   默认配置: configs/YZdata/camera+lidar_180x180_spconv.yaml
#   默认相机数: 3
#   per_class_topk 可选（逗号分隔），不填则用模型自带（yaml 里的值）
#   int8：先导 fp32 ONNX，再用 onnxruntime 静态 PTQ 出 QDQ 图（不依赖 pytorch_quantization）
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

# per_class_topk 覆盖（不传则用 pickle 里烘进去的 yaml 值）
PCTOPK_ARG=""
[ -n "$PCTOPK" ] && PCTOPK_ARG="--per-class-topk=$PCTOPK"

# int8 要用标准 LayerNorm 导出（nv::CustomLayerNormalization 只有 TensorRT 认，ORT 不认）
LN_ARG=""
[ "$MODE" = "int8" ] && LN_ARG="--std-layernorm"

echo ">>> camera backbone + vtransform"
python qat/export-camera.py --ckpt=$DEPLOY_CKPT $EXTRA --device cpu

echo ">>> fuser + head.bbox"
python qat/export-transfuser.py --ckpt=$DEPLOY_CKPT $EXTRA --device cpu $PCTOPK_ARG $LN_ARG

echo ">>> lidar scn"
python qat/export-scn.py --ckpt=$DEPLOY_CKPT --save=qat/onnx_$MODE/lidar.backbone.onnx --device cpu
# export-scn 会追加 .xyz / .zyx 后缀，这里补一个不带后缀的副本方便引用
SCN_OUT=$(ls -t qat/onnx_$MODE/lidar.backbone.*.onnx 2>/dev/null | head -1)
[ -n "$SCN_OUT" ] && cp "$SCN_OUT" qat/onnx_$MODE/lidar.backbone.onnx

# int8：对 ORT 能解析的模块做静态 PTQ（QDQ）；spconv 自定义算子会被自动跳过
if [ "$MODE" = "int8" ]; then
  echo ">>> INT8 量化（onnxruntime 静态 PTQ，校准样本 ${CALIB_N:-8}）"
  # fuser 用 ORT PTQ 会崩（实测相对 L2 >90%），默认跳过 → 打包里保持 fp32，部署端交给 TensorRT 自己校准
  INT8_EXCLUDE=${INT8_EXCLUDE:-fuser.onnx}
  QFILES=$(ls qat/onnx_int8/*.onnx 2>/dev/null | grep -v '\.int8\.onnx$' \
           | grep -v "/${INT8_EXCLUDE}$" || true)
  # 有真实校准数据时：CALIB_NPZ=calib.npz bash run_export.sh ... int8  （npz 键名=输入名）
  if [ -n "${CALIB_NPZ:-}" ]; then
    python qat/quantize_onnx_int8.py $QFILES --calib=${CALIB_N:-8} --calib-npz="$CALIB_NPZ"
  else
    python qat/quantize_onnx_int8.py $QFILES --calib=${CALIB_N:-8}
  fi
fi

tar czf deploy_$MODE.tgz -C qat/onnx_$MODE .
echo ">>> 已打包 deploy_$MODE.tgz"
tar tzf deploy_$MODE.tgz
