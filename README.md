# I2V-CompBench

**Image-to-Video Compositional Benchmark**——基于先验分析驱动的 I2V 组合能力评测数据集构建框架。

本仓库包含两个核心阶段：

| 阶段 | 功能 | 核心产出 |
|---|---|---|
| **Phase 1** | 先验数据准备：联合分析 TIP-I2V Eval 的 prompt + 首帧图，生成 Prior Package | `prior_package.json` / `candidate_recipes.jsonl` / `reference_bank/` |
| **Phase 2** | Benchmark 数据集合成：消费 Phase 1 产出，自动合成评测题目 | `phase3_manifest.jsonl` / `by_dimension/` 分层打包 |

---

## 七维度评测框架

| # | 维度 | 评测目标 |
|---|---|---|
| 1 | Attribute Binding | 单主体属性/状态编辑（如“狗变黑”） |
| 2 | Action Binding | 单主体语义动作执行（如“狗坐下”） |
| 3 | Motion Binding | 单主体绝对方向运动（如“狗向左移动”） |
| 4 | Spatial Composition | 多主体间相对空间关系编辑 |
| 5 | Background Dynamics | 背景元素动态变化（如“白天变夜晚”） |
| 6 | View Transformation | 视角/运镜变换（如“镜头拉近”） |
| 7 | Interaction Reasoning | 主体间交互因果推理（v2 预留） |

---

## 环境准备

```bash
# 1. 安装依赖
cd i2v_compbench/
pip install -r requirements.txt

# 2. 配置 API Key（二选一）
#    方式 A：创建 .env 文件（推荐）
cp .env.example .env
#    编辑 .env，填入真实 SiliconFlow API Key

#    方式 B：系统环境变量
$Env:SILICONFLOW_API_KEY = "sk-your-real-key"   # PowerShell
export SILICONFLOW_API_KEY="sk-your-real-key"   # Bash
```

### .env 可选覆盖项

`.env` 默认只放 API Key。想临时切换模型，取消注释即可覆盖 yaml 默认值：

```env
SILICONFLOW_API_KEY=sk-xxx
# VLM_MODEL=Qwen/Qwen3-VL-30B-A3B-Instruct
# LLM_MODEL=deepseek-ai/DeepSeek-V3
# T2I_MODEL=black-forest-labs/FLUX.1-schnell
```

---

## Phase 1：先验数据准备

### 概述

联合分析 TIP-I2V Eval 的 prompt 文本和首帧图片，提取先验特征，为 Phase 2 合成提供基座。

### 流水线步骤

| 步骤 | 命令 | 功能 | 调用模型 |
|---|---|---|---|
| manifest | `--step manifest` | 扫描原始数据目录，生成样本清单 | 无 |
| image | `--step image` | VLM 解析首帧图片（主体/场景/布局） | VLM |
| text | `--step text` | LLM 解析 prompt 文本（动词/名词/属性） | LLM |
| joint | `--step joint` | 联合分析：质量分数 + 维度 routing + 可行性 | 无 |
| report | `--step report` | 汇聚统计报告 | 无 |
| patch | `--step patch` | 数据补丁（修正失败/不完整样本） | 无 |
| align | `--step align` | 对齐图像/文本实体 | 无 |
| refbank | `--step refbank` | 构建 reference_bank（参考图资产库） | 无 |
| priors2 | `--step priors2` | 增强先验：频率层级 / 共现矩阵 / prior_package | 无 |
| recipes | `--step recipes` | 生成 candidate_recipes.jsonl | 无 |
| p1audit | `--step p1audit` | Phase 1 整体质检 | 无 |

### 运行

```bash
# 一键执行全部 Phase 1
python main.py --config configs/phase1.yaml --step phase1

# 或单步调试
python main.py --config configs/phase1.yaml --step image
```

### Phase 1 产出

```
E:/I2V-CompBench/outputs/phase1/
├── manifest/sample_manifest.jsonl        # 样本清单
├── image_parse/image_parse_v2.jsonl      # VLM 图像解析结果
├── text_parse/text_parse_v2.jsonl        # LLM 文本解析结果
├── joint/joint_analysis.jsonl            # 联合分析（维度 routing / 质量分）
├── priors/
│   ├── prior_package.json                 # 先验包（词表分布 / 句式模板 / 种子样例）
│   └── frequency_tiers.json               # 频率分层（head/torso/long_tail）
├── reference_bank/                        # 参考图资产库
└── candidate_recipes.jsonl                # 候选配方（Phase 2 采样池）
```

---

## Phase 2：Benchmark 数据集合成

### 概述

消费 Phase 1 产出，自动合成评测题目：生成配额→采样→规划→首帧生成→VQA质检→Prompt定稿→导出→审计→打包。

### 流水线步骤

| 步骤 | 命令 | 功能 | 调用模型 |
|---|---|---|---|
| quota | `--step quota` | 按 7 维×难度×罕见度生成配额计划 | 无 |
| sample | `--step sample` | 从 candidate_recipes 采样填桶 | 无 |
| plan | `--step plan` | 为每题生成 question_plan（输入计划 / 目标计划 / 约束） | 无 |
| construct | `--step construct` | 生成首帧图（T2I）/ 复制参考图 | T2I |
| verify | `--step verify` | VQA-QC 校验首帧与题目一致性 | VLM |
| finalize | `--step finalize` | LLM 润色定稿最终 I2V prompt | LLM |
| export | `--step export` | 导出 phase3_manifest + 按维度切分 samples/ | 无 |
| audit | `--step audit` | 生成 dataset_card.md 质量报告 | 无 |
| package | `--step package` | 按维度+按题目打包到 by_dimension/ | 无 |

### 运行

```bash
# 一键执行全部 Phase 2
python main.py --config configs/phase2.yaml --step phase2

# 或单步调试
python main.py --config configs/phase2.yaml --step verify
```

### Phase 2 产出

```
data/benchmark_dataset/
├── phase3_manifest.jsonl              # ★ Phase 3 评测输入清单
├── dataset_card.md                     # 数据集卡片
├── samples/                            # 按维度切分的 JSONL
│   ├── attribute_binding.jsonl
│   ├── action_binding.jsonl
│   └── ...
├── by_dimension/                       # ★ 人读友好的分层打包
│   ├── attribute_binding/
│   │   ├── attr_single_0001/
│   │   │   ├── prompt.json
│   │   │   └── first_frame.png
│   │   └── ...
│   ├── action_binding/
│   └── ...
├── first_frames/                       # 原始首帧图（T2I 生成）
├── prompts/final_prompts.jsonl         # 定稿后的 prompt
├── qc_reports/                         # 每题的 VQA-QC 检查报告
├── question_plans.jsonl                # 规划阶段中间产物
├── sampled_recipes.jsonl               # 采样出的配方
└── quota_plan.json                     # 配额计划
```

---

## 模型配置

| 角色 | 默认模型 (yaml) | 用于 | 平台 |
|---|---|---|---|
| VLM | `Qwen/Qwen3-VL-30B-A3B-Instruct` | Phase 1 图像解析 / Phase 2 VQA 校验 | SiliconFlow |
| LLM | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Phase 1 文本解析 / Phase 2 prompt 定稿 | SiliconFlow |
| T2I | `Kwai-Kolors/Kolors` | Phase 2 首帧生成 | SiliconFlow |

所有模型均可通过 `configs/phase1.yaml` / `configs/phase2.yaml` 更换，或通过 `.env` 中的 `VLM_MODEL` / `LLM_MODEL` / `T2I_MODEL` 环境变量临时覆盖。

---

## 目录结构

```
i2v_compbench/
├── main.py                       # CLI 入口（--step 分发）
├── configs/
│   ├── phase1.yaml               # Phase 1 配置
│   ├── phase2.yaml               # Phase 2 配置
│   └── templates/                # 7 维度 YAML 模板
├── prompts/
│   ├── prompt_polish.txt         # LLM 定稿模板
│   └── vqa_qc/                   # 7 份 VQA 质检提示词
├── src/i2vcompbench/
│   ├── phase2/                   # Phase 2 各步骤实现
│   ├── schemas/                  # Pydantic 数据模型
│   └── utils/                    # 共用工具（API 客户端 / IO / 模板）
├── data/benchmark_dataset/       # Phase 2 产出（自动生成）
├── .env.example                  # 环境变量模板
├── .gitignore
├── requirements.txt
└── README.md                     # ← 本文件
```

---

## 设计文档

- [各阶段产物与用途说明](../三阶段实现需求/各阶段产物与用途说明.md)
- [Phase 2 Benchmark 数据集合成](../三阶段实现需求/Phase2_Benchmark数据集合成.md)
- [Phase 2 首帧来源策略讨论纪要](../三阶段实现需求/Phase2首帧来源与Prompt定稿策略讨论纪要.md)
- [Phase 1 先验数据准备](../三阶段实现需求/Phase1_先验数据准备.md)
