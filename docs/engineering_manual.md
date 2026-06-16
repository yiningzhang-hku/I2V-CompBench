# TIP-I2V-Prior-Analysis 工程手册

## 一、工程概述

本工程是 I2V-CompBench（图像到视频评测基准）的**先验分析子系统**。其核心任务是：从 TIP-I2V 真实 I2V 数据集中，联合分析 prompt 文本和首帧图像，提取结构化的"先验包"(Prior Package)，为下游 LLM 合成评测题目提供数据支撑。

## 二、工程目录结构

```
D:\projects\I2V-CompBench\tip_i2v_data_analysis\   ← 代码工程（仅代码）
├── main.py                          # CLI 入口，支持分步/全量执行
├── requirements.txt                 # Python 依赖
├── configs/
│   └── config.yaml                  # 全局配置（路径、模型、采样、先验包参数）
├── prompts/
│   ├── vlm_image_parse.txt          # VLM 图像分析的 prompt 模板
│   └── llm_text_parse.txt           # LLM 文本分析的 prompt 模板
└── src/
    ├── step1_manifest.py            # Step 1: 数据获取与清洗
    ├── step2_image_analysis.py      # Step 2: VLM 图像结构分析
    ├── step3_text_analysis.py       # Step 3: LLM 文本意图/语义槽位分析
    ├── step4_joint_analysis.py      # Step 4: 联合分析 + 深度先验提取
    ├── step5_pool_and_report.py     # Step 5: Prior Package 打包 + 报告生成
    └── utils/
        ├── api_client.py            # SiliconFlow API 封装（VLM/LLM 异步调用）
        ├── io_utils.py              # I/O 工具（JSONL/CSV/YAML 读写、JSON 解析）
        └── schema.py                # Pydantic 数据模型定义

E:\I2V-CompBench\                                   ← 数据目录（外接硬盘）
├── raw/                             # 原始数据（备用本地加载）
├── manifest/
│   ├── manifest.jsonl               # 全量清单（含 bad samples）
│   ├── manifest_clean.jsonl         # 有效样本清单
│   ├── bad_samples.jsonl            # 坏样本（empty_prompt 等）
│   └── images/                      # 首帧缩略图（224px，~13KB/张）
└── outputs/
    ├── pipeline.log                 # 运行日志
    ├── image_analysis/              # Step 2 输出
    ├── text_analysis/               # Step 3 输出
    ├── joint_analysis/              # Step 4 输出（含先验数据）
    │   ├── seed_examples/           # 每维度种子样例
    │   └── ...
    └── reports/                     # Step 5 输出
        ├── prior_package.json       # ★ 核心交付物：完整先验包
        └── summary.md               # 可读报告
```

**路径配置**: 所有路径由 `configs/config.yaml` 中的 `paths` 节控制，支持相对路径和绝对路径。如需切换存储位置，修改 `manifest_dir` 和 `output_dir` 即可。

## 三、Pipeline 运行逻辑

整个流程分 5 个 Step，顺序执行：

```
Step 1 (manifest)    → Step 2 (image)     → Step 3 (text)
   数据获取与清洗         VLM 图像分析         LLM 文本分析
         ↓                    ↓                    ↓
                    Step 4 (joint)
                联合分析 + 深度先验提取
                         ↓
                    Step 5 (report)
                Prior Package 打包 + 报告
```

### Step 1 — Manifest Build（~30 秒/1000 样本）
- 从 HuggingFace `tipi2v/TIP-I2V` 流式加载数据（或本地 parquet/jsonl）
- 验证图像和 prompt，提取并清除 Pika 参数（-camera、-motion、-neg 等）
- 输出 manifest.jsonl、manifest_clean.jsonl、bad_samples.jsonl

### Step 2 — VLM Image Analysis（~25 秒/样本，异步 5 并发）
- 对每张首帧图像调用 VLM，提取主体信息（名称、属性、姿态、位置）、主体关系、背景信息（元素、光照、天气）、运镜基线（镜头类型、角度、景深）
- 支持断点续传（checkpoint resume）

### Step 3 — LLM Text Analysis（~30 秒/样本，异步 5 并发）
- 对每条 prompt 调用 LLM，分类意图、提取 6 个维度的语义槽位
- 提取词性标注（名词、动词、形容词）
- 支持断点续传

### Step 4 — Joint Analysis + Prior Extraction（秒级）
- 文本+图像 inner join → 逐样本评估 6 维度可评测性
- 按维度提取概念分布、视觉构成先验、句式模板、种子样例
- 整合 Pika 运镜参数分布
- 计算维度共现矩阵

### Step 5 — Prior Package Generation（秒级）
- 组装 prior_package.json（核心交付物）
- 生成可读 summary.md 报告和 CSV 统计表

## 四、如何运行

### 环境准备
```bash
cd D:\projects\I2V-CompBench\tip_i2v_data_analysis
pip install -r requirements.txt
```

### 设置 API 密钥（PowerShell）
```powershell
$env:SILICONFLOW_API_KEY="你的API密钥"
```

### 运行命令

```powershell
# 全量运行（5 步顺序执行）
python main.py --step all

# 分步运行（推荐，便于断点续传和监控）
python main.py --step manifest   # Step 1
python main.py --step image      # Step 2（最耗时）
python main.py --step text       # Step 3（最耗时）
python main.py --step joint      # Step 4（秒级）
python main.py --step report     # Step 5（秒级）

# 指定自定义配置文件
python main.py --config path/to/config.yaml --step all
```

### 关键配置项（config.yaml）

```yaml
sampling:
  max_samples: 1000    # 处理样本数，null=全部
  hf_split: "Eval"     # "Eval"(10K) 或 "Full"(1.7M)

prior_package:
  seed_examples_per_dim: 10   # 每维度种子样例数
  top_n_concepts: 30          # 概念分布 Top-N
  template_min_count: 2       # 句式模板最低出现次数

api:
  batch_size: 5        # VLM/LLM 并发数
  timeout: 300         # 单次请求超时（秒）
```

## 五、如何监控进度

### 查看 VLM 进度（Step 2）
```powershell
$c = (Select-String -Path "E:\I2V-CompBench\outputs\image_analysis\image_parse.jsonl" -Pattern "sample_id" | Measure-Object).Count
Write-Output "VLM: $c / 989"
```

### 查看 LLM 进度（Step 3）
```powershell
$c = (Select-String -Path "E:\I2V-CompBench\outputs\text_analysis\text_parse.jsonl" -Pattern "sample_id" | Measure-Object).Count
Write-Output "LLM: $c / 989"
```

### 查看实时日志
```powershell
Get-Content "E:\I2V-CompBench\outputs\pipeline.log" -Tail 20
```

### 检查 Python 进程是否存活
```powershell
Get-Process -Name python -ErrorAction SilentlyContinue
```

## 六、断点续传机制

Step 2 和 Step 3 具有自动断点续传能力：
- 每个样本处理完后立即追加写入 JSONL 文件
- 重启时自动读取已处理的 sample_id，跳过已完成的样本
- 无论是意外中断还是主动中断，重新运行同一 step 即可续传

## 七、技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| API | SiliconFlow (OpenAI 兼容) |
| VLM/LLM | Qwen/Qwen3.5-397B-A17B |
| 数据模型 | Pydantic v2 |
| 异步 | asyncio + Semaphore |
| 数据源 | HuggingFace `tipi2v/TIP-I2V` |
| 日志 | loguru |
| 数据处理 | pandas |
