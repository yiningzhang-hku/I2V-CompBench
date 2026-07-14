# AutoDL模型部署与对比实验需求

> 面向AI Coding Agent的独立执行手册  
> 适用场景：在AutoDL GPU服务器上完成图像清晰度增强与尺寸适配的深度学习模型验证实验  
> 文档日期：2026-07-14

---

## 1. 概述

### 1.1 任务定位

本文档定义在AutoDL远程GPU服务器上部署深度学习超分辨率模型（Real-ESRGAN、SwinIR）及生成式补全模型（Stable Diffusion Inpainting/Outpainting），对I2V-CompBench数据集的首帧图像执行正式对比实验。

当前已完成的基线方案（C0+C1: Lanczos + Unsharp Mask）仅使用传统图像处理方法，Laplacian方差从39.33提升至59.87（+90.4%），但无法恢复真实纹理细节。论文第5章需要与深度学习超分方法的定量对比数据，以证明所选方案的合理性或升级为更优方案。

### 1.2 与论文第5章的关系

论文第5章《高质量评测数据集的构建方法》包含以下研究问题：

| 研究问题 | 对应实验 | 本文档覆盖 |
|----------|----------|------------|
| RQ3: 图像清晰度增强效果 | C0-C4对比 | 核心任务 |
| RQ4: 尺寸适配效果 | D0-D5对比 | 核心任务 |
| RQ6: 组合消融实验 | 处理层叠加效果 | 数据支撑 |

### 1.3 预期产出

1. **定量对比数据**：C2(Real-ESRGAN)、C3(SwinIR) 与已有 C0/C1 基线的全维度指标对比
2. **尺寸适配数据**：D5(Outpainting) 与已有 D0-D4 方案的对比
3. **决策依据**：是否需要将生产管线从 C1(Unsharp) 升级为 C2(Real-ESRGAN)
4. **论文素材**：可直接嵌入第5章的表格、统计检验结果和代表性可视化

---

## 2. 环境准备

### 2.1 SSH连接信息

```bash
# 连接命令
ssh -p 35206 root@connect.westd.seetacloud.com

# 或使用SSH配置 (~/.ssh/config)
Host autodl
    HostName connect.westd.seetacloud.com
    Port 35206
    User root
```

### 2.2 存储规划

| 存储位置 | 用途 | 特性 | 容量 |
|----------|------|------|------|
| `/root/autodl-tmp/` | 代码、conda环境、实验数据、运行产物 | 本地SSD，关机保留 | ~150GB |
| `/root/autodl-fs/` | 模型权重（共享） | 网络NAS，跨实例共享 | 大容量 |
| `/root/` | 临时脚本 | 系统盘，关机丢失 | 30GB |

**规划原则**：
- 模型权重存放于 `/root/autodl-fs/models/`（跨实例复用，避免重复下载）
- 实验代码和数据存放于 `/root/autodl-tmp/i2v-compbench/`
- conda环境存放于 `/root/autodl-tmp/miniconda3/envs/`

### 2.3 Conda环境配置

**方案：新建独立环境 `i2v-quality`**（不复用hyvideo，避免依赖冲突）

```bash
# 激活学术网络加速
source /etc/network_turbo

# 创建新环境
conda create -n i2v-quality python=3.10 -y
conda activate i2v-quality

# 验证Python版本
python --version  # 应为 3.10.x
```

**选择Python 3.10的理由**：
- Real-ESRGAN官方支持3.8-3.10
- basicsr依赖在3.10最稳定
- 与PyTorch 2.x兼容性最佳

### 2.4 依赖安装

```bash
conda activate i2v-quality

# Step 1: 安装PyTorch (CUDA 12.1版本，适配RTX 6000 Ada)
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# Step 2: 安装Real-ESRGAN及其依赖
pip install basicsr==1.4.2
pip install facexlib==0.3.0
pip install gfpgan==1.3.8
pip install realesrgan==0.3.0

# Step 3: 安装SwinIR依赖
pip install timm==0.9.16
pip install einops==0.7.0

# Step 4: 安装Stable Diffusion Inpainting依赖（D5实验）
pip install diffusers==0.28.0
pip install transformers==4.41.0
pip install accelerate==0.30.0

# Step 5: 安装评价指标依赖
pip install pyiqa==0.1.10        # NIQE/BRISQUE等无参考指标
pip install scikit-image==0.22.0  # 传统图像质量指标

# Step 6: 安装DINOv2（主体特征相似度）
pip install timm==0.9.16  # 已装则跳过

# Step 7: 通用工具
pip install opencv-python==4.9.0.80
pip install pillow==10.3.0
pip install numpy==1.26.4
pip install tqdm==4.66.4
pip install pyyaml==6.0.1
pip install matplotlib==3.8.4
pip install scipy==1.13.0

# Step 8: 验证安装
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
python -c "from realesrgan import RealESRGANer; print('Real-ESRGAN OK')"
python -c "from basicsr.archs.swinir_arch import SwinIR; print('SwinIR OK')"
python -c "import pyiqa; print('PyIQA OK')"
```

### 2.5 数据传输方案

**方案A：SCP直传（推荐，适合120-180张图像）**

```powershell
# 在本地Windows PowerShell中执行
# 先准备样本包（在本地Python中执行抽样脚本后）

# 上传清晰度实验样本
scp -P 35206 -r D:\projects\I2V-CompBench\data\benchmark_dataset\quality_experiments\sample_120\ root@connect.westd.seetacloud.com:/root/autodl-tmp/i2v-compbench/data/sample_120/

# 上传尺寸实验样本
scp -P 35206 -r D:\projects\I2V-CompBench\data\benchmark_dataset\quality_experiments\sample_60\ root@connect.westd.seetacloud.com:/root/autodl-tmp/i2v-compbench/data/sample_60/
```

**方案B：打包后传输（文件多时效率更高）**

```powershell
# 本地打包
cd D:\projects\I2V-CompBench\data\benchmark_dataset\quality_experiments
tar -czf experiment_samples.tar.gz sample_120/ sample_60/

# 上传
scp -P 35206 experiment_samples.tar.gz root@connect.westd.seetacloud.com:/root/autodl-tmp/i2v-compbench/data/

# 远程解压
ssh -p 35206 root@connect.westd.seetacloud.com "cd /root/autodl-tmp/i2v-compbench/data && tar -xzf experiment_samples.tar.gz"
```

**方案C：AutoDL网盘上传（大批量时使用）**

通过AutoDL控制台的"文件存储"功能上传到 `/root/autodl-fs/`，再复制到工作目录。

---

## 3. 待部署模型清单

### 3.1 Real-ESRGAN x4plus（实验组C2）

| 属性 | 值 |
|------|------|
| **模型名称** | RealESRGAN_x4plus |
| **用途** | 4倍超分辨率，恢复真实纹理细节 |
| **架构** | RRDB (Residual-in-Residual Dense Block) |
| **下载源（国内）** | ModelScope: `damo/cv_rrdb_image-super-resolution` 或 GitHub Release |
| **权重文件** | `RealESRGAN_x4plus.pth` |
| **权重大小** | ~64MB |
| **VRAM需求** | ~2GB（单张854×480推理） |
| **推理速度** | ~0.3s/张（RTX 6000 Ada，854×480输入） |
| **输入要求** | 任意尺寸，输出为4倍放大 |
| **使用策略** | 先缩小到1/4再超分，或直接超分后下采样到目标尺寸 |

**下载命令**：
```bash
# 方法1: 使用realesrgan自带下载（推荐）
python -c "
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
# 首次调用会自动下载权重到 ~/.cache 或指定路径
"

# 方法2: 手动下载到共享存储
mkdir -p /root/autodl-fs/models/realesrgan/
wget -O /root/autodl-fs/models/realesrgan/RealESRGAN_x4plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

# 方法3: ModelScope（国内加速）
pip install modelscope
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('damo/cv_rrdb_image-super-resolution', cache_dir='/root/autodl-fs/models/realesrgan/')
"
```

### 3.2 SwinIR Large（实验组C3）

| 属性 | 值 |
|------|------|
| **模型名称** | SwinIR-Large (Real-World SR x4) |
| **用途** | 基于Swin Transformer的超分辨率，与ESRGAN对比 |
| **架构** | Swin Transformer + Residual |
| **下载源** | GitHub Release / HuggingFace |
| **权重文件** | `003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth` |
| **权重大小** | ~136MB |
| **VRAM需求** | ~4GB（单张854×480推理） |
| **推理速度** | ~0.8s/张（RTX 6000 Ada，854×480输入） |
| **输入要求** | 任意尺寸，window_size=8的倍数padding |

**下载命令**：
```bash
mkdir -p /root/autodl-fs/models/swinir/

# 下载Real-World SR Large模型
wget -O /root/autodl-fs/models/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth \
  https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth
```

### 3.3 GFPGAN v1.4（实验组C4，可选）

| 属性 | 值 |
|------|------|
| **模型名称** | GFPGANv1.4 |
| **用途** | 人脸增强消融实验（Real-ESRGAN + 人脸修复） |
| **架构** | StyleGAN2-based face restoration |
| **下载源** | GitHub Release |
| **权重文件** | `GFPGANv1.4.pth` |
| **权重大小** | ~332MB |
| **VRAM需求** | ~3GB（含人脸检测） |
| **推理速度** | ~0.5s/张（含检测+修复） |
| **适用条件** | 仅对含人脸的样本子集（约40张）有效 |

**下载命令**：
```bash
mkdir -p /root/autodl-fs/models/gfpgan/

wget -O /root/autodl-fs/models/gfpgan/GFPGANv1.4.pth \
  https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth

# 人脸检测模型（GFPGAN依赖）
wget -O /root/autodl-fs/models/gfpgan/detection_Resnet50_Final.pth \
  https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth
wget -O /root/autodl-fs/models/gfpgan/parsing_parsenet.pth \
  https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth
```

### 3.4 Stable Diffusion Inpainting（实验组D5-Outpainting）

| 属性 | 值 |
|------|------|
| **模型名称** | stable-diffusion-2-inpainting |
| **用途** | 生成式边缘补全（Outpainting），替代高斯模糊填充 |
| **架构** | UNet + CLIP Text Encoder + VAE |
| **下载源** | HuggingFace / ModelScope |
| **权重总大小** | ~5.2GB |
| **VRAM需求** | ~8GB（fp16推理，512×512 patch） |
| **推理速度** | ~3-5s/张（20步DDIM，RTX 6000 Ada） |
| **使用策略** | 对模糊填充区域做Inpainting，保持中心主体不变 |

**下载命令**：
```bash
mkdir -p /root/autodl-fs/models/sd-inpainting/

# 方法1: HuggingFace (配合学术加速)
python -c "
from diffusers import StableDiffusionInpaintPipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    'stabilityai/stable-diffusion-2-inpainting',
    cache_dir='/root/autodl-fs/models/sd-inpainting/'
)
print('SD Inpainting downloaded')
"

# 方法2: ModelScope（国内加速，推荐）
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(
    'AI-ModelScope/stable-diffusion-2-inpainting',
    cache_dir='/root/autodl-fs/models/sd-inpainting/'
)
print('SD Inpainting downloaded from ModelScope')
"
```

### 3.5 DINOv2（评价指标用）

| 属性 | 值 |
|------|------|
| **模型名称** | DINOv2 ViT-L/14 |
| **用途** | 计算主体特征相似度（身份保真评价） |
| **权重大小** | ~1.2GB |
| **VRAM需求** | ~2GB |
| **推理速度** | ~0.05s/张 |

**下载命令**：
```bash
# DINOv2通过torch.hub自动下载
python -c "
import torch
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
print('DINOv2 loaded')
"
```

### 3.6 模型资源汇总

| 模型 | 实验组 | 权重大小 | VRAM | 速度/张 | 优先级 |
|------|--------|----------|------|---------|--------|
| Real-ESRGAN x4plus | C2 | 64MB | 2GB | 0.3s | **必须** |
| SwinIR-L | C3 | 136MB | 4GB | 0.8s | **必须** |
| GFPGAN v1.4 | C4 | 332MB | 3GB | 0.5s | 可选 |
| SD-2 Inpainting | D5 | 5.2GB | 8GB | 3-5s | **必须** |
| DINOv2 ViT-L | 评价 | 1.2GB | 2GB | 0.05s | **必须** |
| **合计** | — | **~7GB** | **峰值~12GB** | — | — |

> 48GB VRAM充裕，所有模型可按需加载，无OOM风险。

---

## 4. 实验任务详细设计

### 4.1 实验C：图像清晰度对比（核心）

#### 样本选取

从1500条final benchmark中**分层抽取120张**（参考`configs/quality_experiments.yaml`中`clarity.sample_size: 120`）：

```
抽样策略：
- 5个维度 × 24张/维度 = 120张
- 每维度内按主体Tier分层（T1:14, T2:6, T3:3, T4:1）
- 优先包含人脸子集约40张（用于C4人脸消融）
- 使用固定随机种子 seed=20260712 保证可复现
```

**抽样脚本**（本地执行后上传）：
```python
# scripts/sample_for_autodl.py（本地Windows执行）
import json, random
from pathlib import Path
from collections import defaultdict

SEED = 20260712
BENCHMARK = Path("data/benchmark_dataset/final_benchmark_1500.jsonl")
OUTPUT_DIR = Path("data/benchmark_dataset/quality_experiments/sample_120")

random.seed(SEED)
data = [json.loads(l) for l in open(BENCHMARK, encoding="utf-8")]

# 按维度分组
by_dim = defaultdict(list)
for row in data:
    by_dim[row["dimension"]].append(row)

selected = []
for dim, rows in by_dim.items():
    sampled = random.sample(rows, min(24, len(rows)))
    selected.extend(sampled)

# 输出样本清单和复制图像...
```

#### 对比组设计

| 组别 | 方法 | 说明 | 执行位置 |
|------|------|------|----------|
| C0 | Lanczos resize only | 纯插值放大到854长边，不做锐化 | 已有基线数据 |
| C1 | Lanczos + Adaptive Unsharp Mask | 当前生产方案（kernel=5, σ=1.5, 自适应amount） | 已有基线数据 |
| C2 | Real-ESRGAN x4plus | 4倍超分后下采样到854长边 | **AutoDL执行** |
| C3 | SwinIR-Large (Real-World x4) | Transformer超分后下采样到854长边 | **AutoDL执行** |
| C4 | Real-ESRGAN + GFPGAN | C2基础上对人脸区域额外修复（仅人脸子集） | **AutoDL执行（可选）** |

#### C2/C3处理流程

```
输入: 原始首帧 (e.g. 224×126)
  → 超分模型4倍放大 → 896×504
  → Lanczos下采样到目标长边854 → 854×480（或保持比例）
  → 输出: enhanced_C2_xxx.png / enhanced_C3_xxx.png
```

**关键约束**：超分后必须下采样回目标尺寸，确保与C0/C1输出分辨率一致，指标可比。

#### 评价指标

| 指标 | 类别 | 说明 | 实现方式 |
|------|------|------|----------|
| Laplacian Variance | 清晰度 | 边缘锐利度，值越大越清晰 | `cv2.Laplacian(gray, cv2.CV_64F).var()` |
| Tenengrad | 清晰度 | 梯度幅值，另一清晰度指标 | Sobel梯度均方 |
| NIQE | 无参考质量 | 自然图像质量评估，值越低越好 | `pyiqa.create_metric('niqe')` |
| BRISQUE | 无参考质量 | 盲参考质量，值越低越好 | `pyiqa.create_metric('brisque')` |
| DINOv2 Cosine Sim | 身份保真 | 超分前后主体特征余弦相似度 | DINOv2 ViT-L提取CLS token |
| 人脸身份相似度 | 身份保真 | ArcFace特征余弦相似度（人脸子集） | facexlib或insightface |
| 伪纹理率 | 伪影 | 人工判断是否引入不存在的纹理 | 人工标注（后续补充） |

#### 胜出规则

```
优先级: 身份保持 > 清晰度提升 > 自然度

判定逻辑:
1. DINOv2主体相似度 ≥ 0.90（非劣门槛）→ 通过
2. 在通过身份门槛的方法中，选择NIQE最低者
3. 若C2/C3的DINOv2 < 0.90，则判定C1（传统方法）仍为最优
4. McNemar配对检验验证差异显著性 (p < 0.05)
```

### 4.2 实验D：尺寸适配对比

#### 样本选取

从1500条中按**宽高比分桶**抽取60张（参考`configs/quality_experiments.yaml`中`aspect.ratio_stage_sample_size: 60`）：

```
6个宽高比桶 × 10张/桶 = 60张

桶定义（源图宽高比 src_ratio = w/h）：
- 桶1: src_ratio ∈ [0.5, 0.75)  — 竖版窄图
- 桶2: src_ratio ∈ [0.75, 1.0)  — 接近正方形偏窄
- 桶3: src_ratio ∈ [1.0, 1.33)  — 4:3附近
- 桶4: src_ratio ∈ [1.33, 1.6)  — 介于4:3和16:9之间
- 桶5: src_ratio ∈ [1.6, 1.95)  — 接近16:9
- 桶6: src_ratio ∈ [1.95, 2.6]  — 超宽图
```

#### 对比组设计

| 组别 | 方法 | 说明 | 执行位置 |
|------|------|------|----------|
| D0 | 非等比缩放（Stretch） | 强制拉伸到854×480，主体变形 | **AutoDL执行** |
| D1 | Center Crop | 中心裁剪到16:9区域 | **AutoDL执行** |
| D2 | Letterbox（黑边） | 等比缩放+黑色边框填充 | **AutoDL执行** |
| D3 | Blur Padding（当前方案） | 等比缩放+高斯模糊背景填充 | 已有基线数据 |
| D4 | Saliency-aware Crop | 基于显著性检测的智能裁剪 | **AutoDL执行** |
| D5 | Outpainting（生成式补全） | SD Inpainting补全边缘区域 | **AutoDL执行** |

#### D5 Outpainting处理流程

```
输入: 原始首帧 (e.g. 300×400, 竖版)
  → 等比缩放使高度=480 → 360×480
  → 创建854×480画布，将缩放图居中放置
  → 中心区域为原图内容，左右为需补全区域
  → 创建mask：中心区域=0（保护），边缘=255（待补全）
  → SD Inpainting以"natural background extension"为prompt补全
  → 输出: adapted_D5_xxx.png
```

#### 评价指标

| 指标 | 类别 | 说明 |
|------|------|------|
| 主体bbox保留比例 | 内容保真 | 主体在输出中的面积占比（D1裁剪可能丢失） |
| DINOv2主体相似度 | 特征保真 | 适配前后主体区域的DINOv2余弦相似度 |
| 接缝可见度 | 伪影 | 填充区域与原图边界的过渡自然度（梯度不连续性） |
| 语义一致性 | 伪影 | Outpainting补全内容是否引入不存在的语义对象 |
| 黑边占比 | 覆盖率 | Letterbox方案中黑色像素占总面积的比例 |
| 重复纹理率 | 伪影 | 自相关检测补全区域是否出现明显重复模式 |
| 人工总体可用率 | 综合 | 人工判断：该图作为I2V输入是否可接受 |

#### D4 Saliency-aware Crop策略

```python
# 使用DINOv2 attention map作为显著性图
# 裁剪时最大化显著区域保留
策略:
1. 提取DINOv2最后一层attention map作为saliency
2. 计算所有可能16:9裁剪窗口中saliency总量
3. 选择saliency最大的裁剪窗口
4. 缩放到854×480
```

### 4.3 小规模I2V视频探针（可选）

**前提**：已有`hyvideo`环境中部署了HunyuanVideo-I2V模型。

| 项目 | 说明 |
|------|------|
| 样本量 | 从120张清晰度样本中选30张（6张/维度） |
| 模型 | HunyuanVideo-I2V (fp8, 720P) |
| 目的 | 验证不同清晰度/适配方案对生成视频质量的影响 |
| 对比 | 同一prompt，不同输入图→生成视频→VQA评估 |

**执行条件**：仅在C/D实验完成且有余力时执行，不阻断主实验。

---

## 5. 数据传输与目录规划

### 5.1 AutoDL目录结构

```
/root/autodl-tmp/i2v-compbench/
├── data/
│   ├── sample_120/                    # 清晰度实验120张原始图
│   │   ├── originals/                 # 原始分辨率首帧
│   │   ├── c0_lanczos/               # C0基线（Lanczos放大）
│   │   ├── c1_unsharp/               # C1基线（+Unsharp Mask）
│   │   ├── c2_realesrgan/            # C2实验结果
│   │   ├── c3_swinir/                # C3实验结果
│   │   └── c4_gfpgan/                # C4实验结果（可选）
│   ├── sample_60/                     # 尺寸实验60张
│   │   ├── originals/                 # 原始首帧
│   │   ├── d0_stretch/               # D0拉伸
│   │   ├── d1_center_crop/           # D1中心裁剪
│   │   ├── d2_letterbox/             # D2黑边
│   │   ├── d3_blur_padding/          # D3模糊填充（已有）
│   │   ├── d4_saliency_crop/         # D4显著性裁剪
│   │   └── d5_outpainting/           # D5生成式补全
│   ├── results/                       # 实验结果JSON
│   │   ├── clarity_experiment_results.json
│   │   ├── aspect_experiment_results.json
│   │   └── comparison_report.md
│   └── sample_manifest.json           # 样本清单（含qid、维度、原始路径）
├── scripts/
│   ├── run_realesrgan.py             # C2推理脚本
│   ├── run_swinir.py                 # C3推理脚本
│   ├── run_gfpgan.py                 # C4推理脚本（可选）
│   ├── run_outpainting.py            # D5推理脚本
│   ├── run_aspect_baselines.py       # D0-D2/D4推理脚本
│   ├── evaluate_clarity.py           # 清晰度指标计算
│   ├── evaluate_aspect.py            # 尺寸适配指标计算
│   ├── compute_dino_similarity.py    # DINOv2相似度计算
│   └── generate_report.py            # 生成对比报告
├── configs/
│   └── experiment_config.yaml         # 实验参数配置
└── logs/                              # 运行日志
    └── experiment_YYYYMMDD.log

/root/autodl-fs/models/
├── realesrgan/
│   └── RealESRGAN_x4plus.pth         # 64MB
├── swinir/
│   └── 003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth  # 136MB
├── gfpgan/
│   ├── GFPGANv1.4.pth                # 332MB
│   ├── detection_Resnet50_Final.pth
│   └── parsing_parsenet.pth
├── sd-inpainting/
│   └── stable-diffusion-2-inpainting/ # ~5.2GB
└── dinov2/
    └── dinov2_vitl14_pretrain.pth     # ~1.2GB (torch.hub缓存)
```

### 5.2 本地准备工作

在执行AutoDL实验前，需在本地Windows完成以下准备：

1. **抽样**：运行抽样脚本生成 `sample_120/` 和 `sample_60/`
2. **C0/C1基线**：将已有的Lanczos和Unsharp结果复制到对应目录
3. **D3基线**：将已有的blur_padding结果复制到对应目录
4. **元数据**：生成 `sample_manifest.json`（包含每张图的qid、维度、原始尺寸等）

---

## 6. 执行流程（Step-by-Step）

### Step 0: 本地准备（Windows）

```powershell
# 在本地执行抽样脚本（需事先编写）
cd D:\projects\I2V-CompBench
python scripts/sample_for_autodl.py

# 验证样本数量
(Get-ChildItem data\benchmark_dataset\quality_experiments\sample_120\originals\*.png).Count
# 应输出 120

(Get-ChildItem data\benchmark_dataset\quality_experiments\sample_60\originals\*.png).Count
# 应输出 60
```

### Step 1: SSH连接与环境初始化

```bash
# 连接AutoDL
ssh -p 35206 root@connect.westd.seetacloud.com

# 启用学术网络加速
source /etc/network_turbo

# 创建工作目录
mkdir -p /root/autodl-tmp/i2v-compbench/{data,scripts,configs,logs}
mkdir -p /root/autodl-tmp/i2v-compbench/data/{sample_120,sample_60,results}
mkdir -p /root/autodl-tmp/i2v-compbench/data/sample_120/{originals,c0_lanczos,c1_unsharp,c2_realesrgan,c3_swinir,c4_gfpgan}
mkdir -p /root/autodl-tmp/i2v-compbench/data/sample_60/{originals,d0_stretch,d1_center_crop,d2_letterbox,d3_blur_padding,d4_saliency_crop,d5_outpainting}
```

### Step 2: 环境配置

```bash
# 创建conda环境
conda create -n i2v-quality python=3.10 -y
conda activate i2v-quality

# 安装PyTorch
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# 安装核心依赖
pip install basicsr==1.4.2 facexlib==0.3.0 gfpgan==1.3.8 realesrgan==0.3.0
pip install timm==0.9.16 einops==0.7.0
pip install diffusers==0.28.0 transformers==4.41.0 accelerate==0.30.0
pip install pyiqa==0.1.10 scikit-image==0.22.0
pip install opencv-python==4.9.0.80 pillow==10.3.0 numpy==1.26.4
pip install tqdm==4.66.4 pyyaml==6.0.1 matplotlib==3.8.4 scipy==1.13.0

# 验证GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB')"
# 预期输出: CUDA: True, GPU: NVIDIA RTX 6000 Ada Generation, VRAM: 48.0GB
```

### Step 3: 模型下载

```bash
conda activate i2v-quality
source /etc/network_turbo

# Real-ESRGAN
mkdir -p /root/autodl-fs/models/realesrgan/
wget -O /root/autodl-fs/models/realesrgan/RealESRGAN_x4plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

# SwinIR
mkdir -p /root/autodl-fs/models/swinir/
wget -O /root/autodl-fs/models/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth \
  https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth

# GFPGAN（可选）
mkdir -p /root/autodl-fs/models/gfpgan/
wget -O /root/autodl-fs/models/gfpgan/GFPGANv1.4.pth \
  https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
wget -O /root/autodl-fs/models/gfpgan/detection_Resnet50_Final.pth \
  https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth
wget -O /root/autodl-fs/models/gfpgan/parsing_parsenet.pth \
  https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth

# SD Inpainting（使用Python下载，支持断点续传）
python -c "
from diffusers import StableDiffusionInpaintPipeline
import torch
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    'stabilityai/stable-diffusion-2-inpainting',
    torch_dtype=torch.float16,
    cache_dir='/root/autodl-fs/models/sd-inpainting/'
)
print('SD Inpainting downloaded successfully')
"

# DINOv2（通过torch.hub）
python -c "
import torch
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
print('DINOv2 downloaded successfully')
"

# 验证所有模型文件
echo "=== Model Verification ==="
ls -lh /root/autodl-fs/models/realesrgan/
ls -lh /root/autodl-fs/models/swinir/
ls -lh /root/autodl-fs/models/gfpgan/
ls -lh /root/autodl-fs/models/sd-inpainting/
```

### Step 4: 数据上传

```bash
# 在本地Windows PowerShell中执行（另开终端）：
scp -P 35206 -r D:\projects\I2V-CompBench\data\benchmark_dataset\quality_experiments\sample_120\originals\* root@connect.westd.seetacloud.com:/root/autodl-tmp/i2v-compbench/data/sample_120/originals/

scp -P 35206 -r D:\projects\I2V-CompBench\data\benchmark_dataset\quality_experiments\sample_60\originals\* root@connect.westd.seetacloud.com:/root/autodl-tmp/i2v-compbench/data/sample_60/originals/

# 上传已有基线数据
scp -P 35206 -r D:\projects\I2V-CompBench\data\benchmark_dataset\quality_experiments\sample_120\c0_lanczos\* root@connect.westd.seetacloud.com:/root/autodl-tmp/i2v-compbench/data/sample_120/c0_lanczos/

scp -P 35206 -r D:\projects\I2V-CompBench\data\benchmark_dataset\quality_experiments\sample_120\c1_unsharp\* root@connect.westd.seetacloud.com:/root/autodl-tmp/i2v-compbench/data/sample_120/c1_unsharp/

# 上传元数据
scp -P 35206 D:\projects\I2V-CompBench\data\benchmark_dataset\quality_experiments\sample_manifest.json root@connect.westd.seetacloud.com:/root/autodl-tmp/i2v-compbench/data/

# 在AutoDL上验证
ssh -p 35206 root@connect.westd.seetacloud.com "ls /root/autodl-tmp/i2v-compbench/data/sample_120/originals/ | wc -l"
# 应输出 120
```

### Step 5: 运行实验C（清晰度对比）

```bash
conda activate i2v-quality
cd /root/autodl-tmp/i2v-compbench

# === C2: Real-ESRGAN ===
python scripts/run_realesrgan.py \
  --input data/sample_120/originals/ \
  --output data/sample_120/c2_realesrgan/ \
  --model_path /root/autodl-fs/models/realesrgan/RealESRGAN_x4plus.pth \
  --target_long_edge 854 \
  --gpu_id 0

# 验证C2输出
ls data/sample_120/c2_realesrgan/ | wc -l  # 应为120

# === C3: SwinIR ===
python scripts/run_swinir.py \
  --input data/sample_120/originals/ \
  --output data/sample_120/c3_swinir/ \
  --model_path /root/autodl-fs/models/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth \
  --target_long_edge 854 \
  --gpu_id 0

# 验证C3输出
ls data/sample_120/c3_swinir/ | wc -l  # 应为120

# === C4: GFPGAN（可选，仅人脸子集） ===
python scripts/run_gfpgan.py \
  --input data/sample_120/originals/ \
  --output data/sample_120/c4_gfpgan/ \
  --model_path /root/autodl-fs/models/gfpgan/GFPGANv1.4.pth \
  --bg_upsampler realesrgan \
  --bg_model_path /root/autodl-fs/models/realesrgan/RealESRGAN_x4plus.pth \
  --target_long_edge 854 \
  --face_subset_only \
  --gpu_id 0
```

### Step 6: 运行实验D（尺寸适配）

```bash
# === D0-D2, D4: 传统方法 ===
python scripts/run_aspect_baselines.py \
  --input data/sample_60/originals/ \
  --output_root data/sample_60/ \
  --target_w 854 \
  --target_h 480 \
  --methods stretch,center_crop,letterbox,saliency_crop \
  --gpu_id 0

# === D5: Outpainting ===
python scripts/run_outpainting.py \
  --input data/sample_60/originals/ \
  --output data/sample_60/d5_outpainting/ \
  --model_path /root/autodl-fs/models/sd-inpainting/ \
  --target_w 854 \
  --target_h 480 \
  --num_inference_steps 20 \
  --guidance_scale 7.5 \
  --prompt "natural background extension, seamless continuation" \
  --gpu_id 0

# 验证所有D组输出
for d in d0_stretch d1_center_crop d2_letterbox d4_saliency_crop d5_outpainting; do
  echo "$d: $(ls data/sample_60/$d/ | wc -l) files"
done
# 每个应为60
```

### Step 7: 计算评价指标

```bash
# === 清晰度指标 ===
python scripts/evaluate_clarity.py \
  --data_root data/sample_120/ \
  --methods c0_lanczos,c1_unsharp,c2_realesrgan,c3_swinir \
  --originals_dir originals \
  --output data/results/clarity_experiment_results.json \
  --metrics laplacian,tenengrad,niqe,brisque \
  --gpu_id 0

# === DINOv2主体相似度（清晰度） ===
python scripts/compute_dino_similarity.py \
  --reference data/sample_120/originals/ \
  --candidates data/sample_120/c2_realesrgan/,data/sample_120/c3_swinir/ \
  --output data/results/clarity_dino_similarity.json \
  --gpu_id 0

# === 尺寸适配指标 ===
python scripts/evaluate_aspect.py \
  --data_root data/sample_60/ \
  --methods d0_stretch,d1_center_crop,d2_letterbox,d3_blur_padding,d4_saliency_crop,d5_outpainting \
  --originals_dir originals \
  --output data/results/aspect_experiment_results.json \
  --target_w 854 --target_h 480 \
  --gpu_id 0

# === DINOv2主体相似度（尺寸） ===
python scripts/compute_dino_similarity.py \
  --reference data/sample_60/originals/ \
  --candidates data/sample_60/d0_stretch/,data/sample_60/d1_center_crop/,data/sample_60/d4_saliency_crop/,data/sample_60/d5_outpainting/ \
  --output data/results/aspect_dino_similarity.json \
  --gpu_id 0
```

### Step 8: 生成对比报告

```bash
python scripts/generate_report.py \
  --clarity_results data/results/clarity_experiment_results.json \
  --aspect_results data/results/aspect_experiment_results.json \
  --clarity_dino data/results/clarity_dino_similarity.json \
  --aspect_dino data/results/aspect_dino_similarity.json \
  --output data/results/comparison_report.md

# 查看报告摘要
head -50 data/results/comparison_report.md
```

### Step 9: 结果回传

```bash
# 在本地Windows PowerShell中执行
# 下载结果文件
scp -P 35206 -r root@connect.westd.seetacloud.com:/root/autodl-tmp/i2v-compbench/data/results/ D:\projects\I2V-CompBench\data\benchmark_dataset\quality_experiments\autodl_results\

# 下载增强后的图像（用于论文可视化）
scp -P 35206 -r root@connect.westd.seetacloud.com:/root/autodl-tmp/i2v-compbench/data/sample_120/c2_realesrgan/ D:\projects\I2V-CompBench\data\benchmark_dataset\quality_experiments\sample_120\c2_realesrgan\
scp -P 35206 -r root@connect.westd.seetacloud.com:/root/autodl-tmp/i2v-compbench/data/sample_120/c3_swinir/ D:\projects\I2V-CompBench\data\benchmark_dataset\quality_experiments\sample_120\c3_swinir\
```

---

## 7. 验收标准

### 7.1 基础验收（MUST）

| 检查项 | 标准 | 验证命令 |
|--------|------|----------|
| GPU可用 | CUDA可用，识别RTX 6000 Ada | `python -c "import torch; assert torch.cuda.is_available()"` |
| Real-ESRGAN推理 | 120张全部成功，无OOM | `ls c2_realesrgan/ \| wc -l` == 120 |
| SwinIR推理 | 120张全部成功，无OOM | `ls c3_swinir/ \| wc -l` == 120 |
| D0-D5生成 | 60张×6组全部成功 | 每组目录包含60个PNG文件 |
| Laplacian计算 | 所有方法的Laplacian方差已计算 | JSON中每条记录含`laplacian_var`字段 |
| NIQE计算 | 所有方法的NIQE分数已计算 | JSON中每条记录含`niqe`字段 |
| BRISQUE计算 | 所有方法的BRISQUE分数已计算 | JSON中每条记录含`brisque`字段 |
| DINOv2相似度 | C2/C3与原图的余弦相似度已计算 | JSON中每对含`dino_cosine_sim`字段 |
| 输出分辨率一致 | 所有C组输出长边=854px | 批量验证图像尺寸 |
| 结果JSON完整 | `clarity_experiment_results.json` 结构正确 | JSON可被Python正常加载 |

### 7.2 质量验收（SHOULD）

| 检查项 | 标准 |
|--------|------|
| C2 DINOv2相似度 | 均值 ≥ 0.85（身份未严重偏移） |
| C3 DINOv2相似度 | 均值 ≥ 0.85 |
| C2 Laplacian提升 | 相对C0均值提升 > 50% |
| D5无明显语义入侵 | Outpainting未引入主体不存在的大面积新对象 |
| 统计检验完成 | McNemar/Wilcoxon p-value已计算 |
| 报告生成 | `comparison_report.md` 包含完整对比表格 |

### 7.3 可选验收（MAY）

| 检查项 | 标准 |
|--------|------|
| C4 GFPGAN | 人脸子集40张处理完成 |
| 视频探针 | 30张样本的I2V视频生成并有VQA评分 |
| 人脸身份相似度 | ArcFace余弦相似度已计算（人脸子集） |

---

## 8. 输出产物

### 8.1 核心产物

| 文件 | 内容 | 格式 |
|------|------|------|
| `clarity_experiment_results.json` | C0-C4每张图的全部指标（Laplacian、Tenengrad、NIQE、BRISQUE） | JSON |
| `aspect_experiment_results.json` | D0-D5每张图的全部指标（主体保留、DINOv2、接缝评分） | JSON |
| `clarity_dino_similarity.json` | C2/C3与原图的DINOv2余弦相似度 | JSON |
| `aspect_dino_similarity.json` | D0-D5与原图的DINOv2余弦相似度 | JSON |
| `comparison_report.md` | 可直接嵌入论文的对比分析报告（含统计检验） | Markdown |

### 8.2 JSON结构示例

**clarity_experiment_results.json**:
```json
{
  "experiment_date": "2026-07-XX",
  "config": {
    "sample_size": 120,
    "target_long_edge": 854,
    "seed": 20260712
  },
  "summary": {
    "C0_lanczos": {
      "laplacian_mean": 39.33,
      "laplacian_std": 12.5,
      "niqe_mean": 5.2,
      "brisque_mean": 35.0
    },
    "C1_unsharp": {
      "laplacian_mean": 59.87,
      "laplacian_std": 18.3,
      "niqe_mean": 4.8,
      "brisque_mean": 32.0
    },
    "C2_realesrgan": {
      "laplacian_mean": null,
      "niqe_mean": null,
      "brisque_mean": null,
      "dino_similarity_mean": null
    },
    "C3_swinir": {
      "laplacian_mean": null,
      "niqe_mean": null,
      "brisque_mean": null,
      "dino_similarity_mean": null
    }
  },
  "statistical_tests": {
    "C1_vs_C2": {
      "method": "Wilcoxon signed-rank",
      "metric": "niqe",
      "p_value": null,
      "significant": null,
      "effect_size": null
    }
  },
  "per_image": [
    {
      "filename": "act_single_0001.png",
      "dimension": "action_binding",
      "qid": "act_single_0001",
      "original_size": "224x126",
      "C0": {"laplacian": 35.2, "niqe": 5.5, "brisque": 38.1},
      "C1": {"laplacian": 55.3, "niqe": 5.0, "brisque": 34.2},
      "C2": {"laplacian": null, "niqe": null, "brisque": null, "dino_sim": null},
      "C3": {"laplacian": null, "niqe": null, "brisque": null, "dino_sim": null}
    }
  ]
}
```

### 8.3 图像产物

| 目录 | 内容 | 数量 |
|------|------|------|
| `sample_120/c2_realesrgan/` | Real-ESRGAN增强图 | 120张 |
| `sample_120/c3_swinir/` | SwinIR增强图 | 120张 |
| `sample_120/c4_gfpgan/` | GFPGAN增强图（可选） | ~40张 |
| `sample_60/d0_stretch/` | 拉伸适配图 | 60张 |
| `sample_60/d1_center_crop/` | 中心裁剪图 | 60张 |
| `sample_60/d2_letterbox/` | 黑边填充图 | 60张 |
| `sample_60/d4_saliency_crop/` | 显著性裁剪图 | 60张 |
| `sample_60/d5_outpainting/` | 生成式补全图 | 60张 |

### 8.4 论文数据更新

实验完成后需更新以下论文内容：
- 第5章 表5-3: 清晰度增强方法对比（补充C2/C3列）
- 第5章 表5-4: 尺寸适配策略对比（补充D4/D5列）
- 第5章 图5-X: 代表性样本可视化（选择4-6张典型对比图）

---

## 9. 风险与注意事项

### 9.1 VRAM管理

| 场景 | 预估VRAM | 风险 |
|------|----------|------|
| Real-ESRGAN单张推理 | ~2GB | 无风险 |
| SwinIR单张推理 | ~4GB | 无风险 |
| SD Inpainting fp16推理 | ~8GB | 无风险 |
| DINOv2特征提取 | ~2GB | 无风险 |
| 同时加载Real-ESRGAN + DINOv2 | ~4GB | 无风险 |
| 所有模型同时加载 | ~16GB | 无风险（48GB充裕） |

**策略**：逐模型加载→推理→释放→加载下一个。48GB VRAM即使全部同时加载也不会OOM。

### 9.2 Outpainting语义入侵风险

**问题**：SD Inpainting可能在填充区域生成不存在于原始场景中的语义对象（如人物、物体），干扰I2V模型的理解。

**缓解措施**：
1. 使用保守的prompt：`"natural background extension, same environment, no new objects"`
2. 设置较低的guidance_scale（5-7），减少过度生成
3. 对生成结果做CLIP评分，检测新增语义
4. 如果语义入侵率>20%，则判定D5不可用

### 9.3 身份改变的决策规则

```
IF DINOv2_similarity(enhanced, original) < 0.85:
    标记为"身份偏移"，该方法在该样本上判负
    
IF 某方法的身份偏移率 > 15%:
    该方法整体判定为"不可接受"
    论文中报告为"因身份保持不达标而被排除"
```

### 9.4 时间估算

| 任务 | 预估时间 | 说明 |
|------|----------|------|
| 环境配置 | 30min | conda + pip install |
| 模型下载 | 30-60min | 取决于网速，SD最大 |
| 数据上传 | 10-20min | 120+60张PNG |
| C2 Real-ESRGAN推理 | ~40s | 120张 × 0.3s |
| C3 SwinIR推理 | ~100s | 120张 × 0.8s |
| D5 Outpainting推理 | ~5min | 60张 × 5s |
| D0-D4生成 | ~2min | 传统方法，快 |
| 指标计算 | ~10min | NIQE/BRISQUE较慢 |
| DINOv2相似度 | ~2min | 批量提取特征 |
| **总计** | **~2-3小时** | 含调试时间 |

### 9.5 AutoDL特别注意

1. **关机保护**：确保所有产物存储在 `/root/autodl-tmp/`，不要放系统盘
2. **网络加速**：每次新SSH连接后执行 `source /etc/network_turbo`
3. **实例空闲**：无操作超时会自动关机，长时间推理用 `nohup` 或 `tmux`
4. **磁盘空间**：SD模型~5GB + 增强图像~2GB + 缓存~3GB ≈ 10GB，autodl-tmp空间充裕

```bash
# 使用tmux防止SSH断连导致中断
tmux new -s experiment
# 在tmux中执行实验命令
# 断开: Ctrl+B, D
# 重连: tmux attach -t experiment
```

---

## 10. 与论文第5章的对接

### 10.1 RQ3（图像清晰度）数据填充

**论文表5-3: 图像清晰度增强方法对比**

| 方法 | Laplacian↑ | NIQE↓ | BRISQUE↓ | DINOv2 Sim↑ | 身份保持率 |
|------|-----------|-------|----------|-------------|-----------|
| C0: Lanczos | 39.33 | — | — | 1.00 (ref) | 100% |
| C1: +Unsharp Mask | 59.87 | — | — | ~0.99 | ~100% |
| C2: Real-ESRGAN | **待填** | **待填** | **待填** | **待填** | **待填** |
| C3: SwinIR | **待填** | **待填** | **待填** | **待填** | **待填** |

**论文论述逻辑**：
- 若C2/C3显著优于C1且身份保持：推荐升级生产管线
- 若C2/C3优于C1但身份下降：论文讨论trade-off，保守选择C1
- 若C2/C3与C1无显著差异：验证C1作为"足够好"的方案

### 10.2 RQ4（尺寸适配）数据填充

**论文表5-4: 16:9尺寸适配策略对比**

| 方法 | 主体保留↑ | DINOv2↑ | 接缝/伪影↓ | 计算成本 | 总体可用率↑ |
|------|-----------|---------|-----------|----------|------------|
| D0: Stretch | 100% | **待填** | 0 | 极低 | 低（变形） |
| D1: Center Crop | **待填** | **待填** | 0 | 极低 | **待填** |
| D2: Letterbox | 100% | **待填** | 黑边 | 极低 | 低（黑边伪影） |
| D3: Blur Padding | ~92% | ~0.95 | 低 | 低 | ~95% |
| D4: Saliency Crop | **待填** | **待填** | 0 | 中 | **待填** |
| D5: Outpainting | **待填** | **待填** | **待填** | 高 | **待填** |

### 10.3 图表更新计划

| 图表编号 | 内容 | 数据来源 |
|----------|------|----------|
| 图5-X | 清晰度增强可视化对比（4张典型样本×4方法） | C0-C3输出图像 |
| 图5-Y | 尺寸适配可视化对比（3张典型样本×6方法） | D0-D5输出图像 |
| 图5-Z | Laplacian方差箱线图（C0-C3） | clarity_experiment_results.json |
| 图5-W | DINOv2相似度分布直方图 | clarity_dino_similarity.json |
| 表5-3 | 清晰度方法定量对比 | clarity_experiment_results.json |
| 表5-4 | 尺寸适配策略定量对比 | aspect_experiment_results.json |
| 表5-5 | 统计显著性检验结果 | comparison_report.md |

### 10.4 结论映射

实验结果将直接映射到论文第5章以下结论段落：

1. **§5.3.3 清晰度增强方案选择**：基于C0-C3对比数据，论证最终采用方案的合理性
2. **§5.4.3 尺寸适配方案选择**：基于D0-D5对比数据，论证Blur Padding优于其他方案（或发现更优方案）
3. **§5.6 组合消融实验**：清晰度和尺寸方案的联合贡献量化

---

## 附录A: 核心脚本参考实现

### A.1 run_realesrgan.py 骨架

```python
"""C2: Real-ESRGAN超分辨率推理脚本"""
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--target_long_edge", type=int, default=854)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    # 初始化模型
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4, model_path=args.model_path, model=model,
        tile=0, tile_pad=10, pre_pad=0, half=True,
        gpu_id=args.gpu_id
    )

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.png"))
    print(f"Processing {len(files)} images with Real-ESRGAN...")

    for img_path in tqdm(files):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        # 超分
        output, _ = upsampler.enhance(img, outscale=4)
        # 下采样到目标长边
        h, w = output.shape[:2]
        long_edge = max(h, w)
        if long_edge > args.target_long_edge:
            scale = args.target_long_edge / long_edge
            new_w, new_h = int(w * scale), int(h * scale)
            output = cv2.resize(output, (new_w, new_h),
                              interpolation=cv2.INTER_LANCZOS4)
        # 保存
        cv2.imwrite(str(output_dir / img_path.name), output)

    print(f"Done! {len(files)} images saved to {output_dir}")


if __name__ == "__main__":
    main()
```

### A.2 evaluate_clarity.py 骨架

```python
"""清晰度评价指标计算脚本"""
import argparse
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import pyiqa


def compute_laplacian(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def compute_tenengrad(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx**2 + gy**2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--methods", type=str, required=True)
    parser.add_argument("--originals_dir", type=str, default="originals")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--metrics", type=str, default="laplacian,tenengrad,niqe,brisque")
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}")
    methods = args.methods.split(",")
    metrics_list = args.metrics.split(",")

    # 初始化IQA模型
    iqa_models = {}
    if "niqe" in metrics_list:
        iqa_models["niqe"] = pyiqa.create_metric("niqe", device=device)
    if "brisque" in metrics_list:
        iqa_models["brisque"] = pyiqa.create_metric("brisque", device=device)

    data_root = Path(args.data_root)
    results = {"per_image": [], "summary": {}}

    # 获取文件列表
    orig_dir = data_root / args.originals_dir
    files = sorted(orig_dir.glob("*.png"))

    for method in methods:
        method_dir = data_root / method
        method_scores = []
        
        for img_path in tqdm(files, desc=f"Evaluating {method}"):
            enhanced_path = method_dir / img_path.name
            if not enhanced_path.exists():
                continue
            
            scores = {"filename": img_path.name, "method": method}
            
            if "laplacian" in metrics_list:
                scores["laplacian"] = compute_laplacian(enhanced_path)
            if "tenengrad" in metrics_list:
                scores["tenengrad"] = compute_tenengrad(enhanced_path)
            if "niqe" in metrics_list:
                scores["niqe"] = float(iqa_models["niqe"](str(enhanced_path)).item())
            if "brisque" in metrics_list:
                scores["brisque"] = float(iqa_models["brisque"](str(enhanced_path)).item())
            
            method_scores.append(scores)
            results["per_image"].append(scores)
        
        # 汇总统计
        if method_scores:
            results["summary"][method] = {
                m: {
                    "mean": float(np.mean([s[m] for s in method_scores if m in s])),
                    "std": float(np.std([s[m] for s in method_scores if m in s])),
                    "median": float(np.median([s[m] for s in method_scores if m in s])),
                }
                for m in metrics_list if any(m in s for s in method_scores)
            }

    # 保存
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
```

---

## 附录B: 快速验证命令（Smoke Test）

在正式运行全部120张前，先用5张做smoke test验证流程：

```bash
conda activate i2v-quality
cd /root/autodl-tmp/i2v-compbench

# 取5张作为测试
mkdir -p data/smoke_test/originals
cp $(ls data/sample_120/originals/*.png | head -5) data/smoke_test/originals/

# Smoke test: Real-ESRGAN
python scripts/run_realesrgan.py \
  --input data/smoke_test/originals/ \
  --output data/smoke_test/c2_test/ \
  --model_path /root/autodl-fs/models/realesrgan/RealESRGAN_x4plus.pth \
  --target_long_edge 854

# 验证输出
python -c "
from PIL import Image
from pathlib import Path
for f in Path('data/smoke_test/c2_test').glob('*.png'):
    img = Image.open(f)
    print(f'{f.name}: {img.size}')
    assert max(img.size) == 854, f'Long edge != 854: {img.size}'
print('SMOKE TEST PASSED')
"

# Smoke test: NIQE计算
python -c "
import pyiqa, torch
niqe = pyiqa.create_metric('niqe', device=torch.device('cuda:0'))
from pathlib import Path
for f in Path('data/smoke_test/c2_test').glob('*.png'):
    score = niqe(str(f))
    print(f'{f.name}: NIQE={score.item():.3f}')
print('NIQE SMOKE TEST PASSED')
"
```

---

## 附录C: 故障排除

| 问题 | 解决方案 |
|------|----------|
| `CUDA out of memory` | 减小tile_size或使用`--tile 512`分块推理 |
| `wget`下载失败 | 先执行`source /etc/network_turbo`，或使用ModelScope国内源 |
| SSH连接超时 | 检查AutoDL实例是否在运行状态 |
| `ModuleNotFoundError: basicsr` | 确认已激活i2v-quality环境：`conda activate i2v-quality` |
| SwinIR内存不足 | 使用`--tile_size 256`分块处理大图 |
| Outpainting生成全黑 | 检查mask是否正确（保护区域=0，生成区域=255） |
| pyiqa计算NaN | 检查输入图像是否为空或损坏 |
| DINOv2下载失败 | 手动下载权重到`~/.cache/torch/hub/`，或设置代理 |
