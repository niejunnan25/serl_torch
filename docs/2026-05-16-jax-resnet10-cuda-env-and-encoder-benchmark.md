# JAX ResNet10 卷积崩溃与 ResNet 编码器测速记录

日期：2026-05-16

## 结论

这次 JAX ResNet10 初始化时崩在卷积，不是 SERL 的 ResNet10 代码本身有问题，而是 `serl_jax` 环境里的 JAX/CUDA 动态库版本不自洽。

修复前，`jax/jaxlib/jax-cuda12-plugin` 是 `0.4.35`，它的 CUDA 插件路径对应 CUDA 12.3 系列；但环境里 pip 安装的 `nvidia-*` 包已经混到了 CUDA 12.9 和 cuDNN 9.22。当前机器驱动是 `555.42.06`，`nvidia-smi` 显示 CUDA 12.5。这个组合可能让 `import jax` 和普通算子通过，但在 XLA 编译卷积时段错误。

用清华源把 `serl_jax` 里的 CUDA pip 包钉回 CUDA 12.3 系列后，SERL 的 JAX ResNet10 已经可以正常初始化和计时。

## 现象

最初跑 JAX ResNet10 fake benchmark 时，程序会在 ResNet 第一层卷积附近段错误。这个位置不是 Python 异常，而是底层 XLA/CUDA/cuDNN 编译或执行阶段直接崩溃。

另外还发现过一个独立的小问题：缺少 `CUDA_ROOT` 时，`import jax` 会因为 `cuda_nvcc.__file__` 是 `None` 报错。设置 `CUDA_ROOT=/usr/local/cuda` 可以绕过这个 import 问题，但不能解决卷积段错误。

## 修复

修复前先备份了环境包列表：

```bash
/vla/users/niejunnan/envs/serl_jax_env_debug/pip-freeze-before-cuda-pin-20260516_002207.txt
```

然后在 `serl_jax` 环境中使用清华源重新安装匹配 JAX 0.4.35 的 CUDA 12.3 系列包：

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate /vla/users/niejunnan/envs/serl_jax

python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --force-reinstall --no-cache-dir \
  nvidia-cuda-runtime-cu12==12.3.101 \
  nvidia-cuda-cupti-cu12==12.3.101 \
  nvidia-cuda-nvcc-cu12==12.3.107 \
  nvidia-cuda-nvrtc-cu12==12.3.107 \
  nvidia-nvjitlink-cu12==12.3.101 \
  nvidia-cublas-cu12==12.3.4.1 \
  nvidia-cusparse-cu12==12.3.1.170 \
  nvidia-cusolver-cu12==11.5.4.101 \
  nvidia-cufft-cu12==11.0.12.1 \
  nvidia-nccl-cu12==2.19.3 \
  nvidia-cudnn-cu12==9.1.1.17
```

修复后的关键版本：

```text
jax==0.4.35
jax-cuda12-pjrt==0.4.35
jax-cuda12-plugin==0.4.35
jaxlib==0.4.35
nvidia-cublas-cu12==12.3.4.1
nvidia-cuda-cupti-cu12==12.3.101
nvidia-cuda-nvcc-cu12==12.3.107
nvidia-cuda-nvrtc-cu12==12.3.107
nvidia-cuda-runtime-cu12==12.3.101
nvidia-cudnn-cu12==9.1.1.17
nvidia-cufft-cu12==11.0.12.1
nvidia-cusolver-cu12==11.5.4.101
nvidia-cusparse-cu12==12.3.1.170
nvidia-nccl-cu12==2.19.3
nvidia-nvjitlink-cu12==12.3.101
```

## 验证

修复后，最小 SERL ResNet10 GPU 初始化和前向已经通过：

```text
init done
block done (1, 256)
```

随后完整运行了 `test/benchmark_resnet_encoder_compare.py` 的 JAX ResNet10 分支，batch 128、两路 224 图像，可以完成 frozen 前向、frozen 前向加 head 反向、全量前向加反向三种计时。

## 公平测速结果

测速条件：

- 同一张 GPU：GPU5，NVIDIA H20。
- batch size：128。
- 图像输入：两路相机，每路 224x224 RGB。
- 权重：随机初始化，避免预训练权重加载差异影响比较。
- 计时：排除编译和 warmup，只统计稳定循环。

| 路径 | frozen 前向 | frozen 前向加 head 反向 | 全量前向加反向 |
|---|---:|---:|---:|
| JAX ResNet10 | 16.14 ms | 17.52 ms | 46.82 ms |
| Torch ResNet18 未编译 | 31.43 ms | 32.11 ms | 98.38 ms |
| Torch ResNet18 编译后 | 21.52 ms | 22.20 ms | 68.90 ms |

由此可见：

- 未编译 Torch ResNet18 大约是 JAX ResNet10 的 1.8 到 2.1 倍耗时。
- 编译后的 Torch ResNet18 差距缩小到大约 1.27 到 1.47 倍。
- ResNet18 本身确实比 ResNet10 更重，但它不是端到端训练吞吐差距的唯一来源。

## 对训练吞吐的解释

SERL 老路径快，主要不只是因为 ResNet10 更轻，还因为 JAX 训练路径可以把 high-UTD 更新用 JIT 和 `lax.scan` 融合成更粗粒度的计算图。

当前 PyTorch 路径慢一些，主要叠加了这些因素：

- ResNet18 编码器比 ResNet10 更重。
- high-UTD critic 更新仍然有 Python 循环和多次优化器调度。
- critic ensemble 如果不缓存特征，会重复做视觉编码。
- replay sampling、batch 拼接、日志指标同步等系统层开销会进入端到端时间。

所以这次修复解决的是 JAX ResNet10 基准无法运行的问题；测速结果说明 encoder 差异真实存在，但继续追 SERL 的整体吞吐，还需要同时优化 PyTorch high-UTD 更新路径和系统流水线。

## 复现命令

JAX ResNet10：

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate /vla/users/niejunnan/envs/serl_jax

CUDA_VISIBLE_DEVICES=5 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
PYTHONPATH=/vla/users/niejunnan/codebase/serl:/vla/users/niejunnan/codebase/serl/serl_launcher:${PYTHONPATH:-} \
python -u /vla/users/niejunnan/codebase/serl_torch/test/benchmark_resnet_encoder_compare.py \
  --backend jax_resnet10 \
  --batch-size 128 \
  --num-views 2 \
  --image-size 224 \
  --warmup 10 \
  --iterations 50 \
  --include-full-backward
```

Torch ResNet18 未编译：

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch

CUDA_VISIBLE_DEVICES=5 \
PYTHONPATH=/vla/users/niejunnan/codebase/serl_torch:/vla/users/niejunnan/codebase/serl_torch/serl_launcher:${PYTHONPATH:-} \
python -u /vla/users/niejunnan/codebase/serl_torch/test/benchmark_resnet_encoder_compare.py \
  --backend torch_resnet18 \
  --batch-size 128 \
  --num-views 2 \
  --image-size 224 \
  --warmup 10 \
  --iterations 50 \
  --include-full-backward \
  --random-init
```

Torch ResNet18 编译后：

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch

CUDA_VISIBLE_DEVICES=5 \
PYTHONPATH=/vla/users/niejunnan/codebase/serl_torch:/vla/users/niejunnan/codebase/serl_torch/serl_launcher:${PYTHONPATH:-} \
python -u /vla/users/niejunnan/codebase/serl_torch/test/benchmark_resnet_encoder_compare.py \
  --backend torch_resnet18 \
  --batch-size 128 \
  --num-views 2 \
  --image-size 224 \
  --warmup 10 \
  --iterations 50 \
  --include-full-backward \
  --random-init \
  --torch-compile \
  --torch-compile-fullgraph
```
