# Phase 3 评测代码实施报告（喂给 AI Coding Agent）

> **目的**：把 Phase 3「模型评测」阶段的全部工程要求，连同上游 Phase 1 / Phase 2 的实际产出契约，一次性交付给 AI Coding Agent；Agent 据此在 T2V-CompBench 开源代码基础上完成 Phase 3 的全部代码实现。
>
> **本报告的三层信息**：① 现状盘点（已有 / 未有的代码与产物） ② 契约对齐（上游 schema 与下游期望的 7 处差距） ③ 实施清单（模块 / 函数签名 / 公式 / KILLER 约束 / CLI / 验收）。
>
> **Agent 工作根目录**：`d:/projects/I2V-CompBench/`

---

## §0 一句话执行摘要

在 `I2V-CompBench/`（T2V-CompBench fork）下落地一个新 Python 包 `i2v_eval/`，提供 7 个维度（attribute / action / motion / spatial / background / view / interaction）+ 1 个 multi_image 专用的 identity_binding 检查，**所有维度统一采用执行门控乘法评分 `S = E·(0.6·P + 0.4·C)`**，复用 T2V-CompBench 的 `spatial.py` / `motion.py` / GroundingDINO / 视频帧工具，并严格遵守 3 条硬约束（KILLER-1 Depth Anything 反向语义、KILLER-2 Motion fps=8、KILLER-3 frame→video 双轨评分）。

---

## §1 任务边界

### 1.1 必须做（In-Scope）

1. 在 `I2V-CompBench/` 子目录拉取 T2V-CompBench 上游代码（**当前仅有 `.gitattributes`，源码尚未克隆**），按 §3.1 fork 清单改造若干文件。
2. 在 `I2V-CompBench/i2v_eval/` 下从零实现 Phase 3 的 11 个模块（§4）。
3. §2.3 的 7 处衔接适配层已在 Phase 1/2 文档全面落实（Phase 1 §5.6.4 + Phase 2 §6.3/§6.4/§6.6/§6.7/§8/附录 A/附录 C）；Phase 3 代码**只需直接消费 BenchmarkSample**，遇到违规字段一律 fail-fast，不允许在 Phase 3 内部写运行时适配代码。
4. 提供端到端 CLI（§7）+ pytest 单测（§8）。

### 1.2 不要做（Out-of-Scope）

- 不要修改 `tip_i2v_data_analysis/src/step1~5*.py`（这是 Phase 1 上游分析脚本，已稳定）。
- 不要重训任何模型；所有 evaluator 走「冻结预训练模型 + 工具化推理」路线。
- 不要在 Phase 3 内做 prompt 生成 / 数据集合成；只做评测。
- 不要引入 GPT-4V / Gemini 等闭源 API 依赖（VLM 走开源 LLaVA-OneVision / Qwen2-VL）。

### 1.3 当前现状盘点（关键）

| 模块 | 路径 | 现状 | Phase 3 是否依赖 |
|---|---|---|---|
| Phase 1 上游 step1-5 | `tip_i2v_data_analysis/src/step{1..5}_*.py` | **已实施且能跑** | 间接（产物喂 Phase 2） |
| Phase 1 schema v2 | `tip_i2v_data_analysis/src/utils/schema_phase1.py` | **类已定义，但尚未被 step1-5 使用** | 间接 |
| Phase 1 增强模块（mock_geometry / align_instances / reference_bank / recipes / audit） | `tip_i2v_data_analysis/src/phase1/` | **目录不存在，未实施** | 间接 |
| Phase 2 全部模块 | `i2v_compbench/src/i2vcompbench/phase2/` | **已部分实施** | **直接，必须先有 BenchmarkSample 产物才能评测** |
| T2V-CompBench fork | `I2V-CompBench/` | **只有 .gitattributes，源码未拉取** | **直接，必须先克隆** |
| Phase 3 全部代码 | `I2V-CompBench/i2v_eval/` | **未实施** | **直接** |

**给 Agent 的提示**：如果你被指派只做 Phase 3，那么你需要在 `data/benchmark_dataset/`（约定路径，见 §2.2）放一份**人工或脚本伪造的 BenchmarkSample 样例 JSONL**（≥10 条覆盖 7 维 + 双 input_mode + contrastive_pair），用来跑通 Phase 3 流水线。**真实 BenchmarkSample 由 Phase 2 产出**，但这是另一组任务。

---

## §2 上下游契约对齐（最重要章节）

### 2.1 Phase 1 实际产物（已经存在，路径见 `configs/config.yaml`）

**配置中 `output_dir = E:/I2V-CompBench/outputs`（外接硬盘备选 `data/outputs`）**

```
{output_dir}/
├── text_analysis/text_parse.jsonl              # TextAnalysisResult，每行 1 条
├── image_analysis/image_parse.jsonl            # ImageAnalysisResult，每行 1 条
├── joint_analysis/
│   ├── joint_analysis.jsonl                    # JointAnalysisResult
│   ├── dimension_priors.jsonl                  # 6 维 DimensionPrior（注意：旧 6 维命名）
│   ├── global_distributions.jsonl
│   ├── global_visual_prior.json
│   ├── pika_distributions.json
│   └── dimension_cooccurrence.json
└── reports/
    ├── prior_package.json                      # 核心交付（喂 LLM few-shot）
    ├── summary.md
    ├── dataset_overview.csv
    ├── dimension_analysis_summary.csv
    └── dimension_gap_analysis.csv
```

**两个隐性陷阱**：

1. step5 的 `DIMENSIONS` 仍是旧 **6 维**命名（`spatial_relation / scene_dynamics / camera_transformation`），与 Phase 1 设计文档承诺的 **7 维 V2**（`spatial_composition / background_dynamics / view_transformation / +interaction_reasoning`）不一致——`schema_phase1.LEGACY_TO_V2_DIM` 已定义映射但尚未启用。**Phase 3 应只信任 7 维 V2 命名，并在 §2.3 适配层强制转换。**
2. `ImageAnalysisResult.subjects[i].bbox` 字段已在 schema 里挂好，但只有当 Phase 1 增强模块 `mock_geometry` 实施后才会被填充；当前实际产物里 bbox 多半为 None。**Phase 3 spatial / interaction 维度对 bbox 的依赖必须有 fallback：bbox=None 时回退到 GroundingDINO 现场重检测。**

### 2.2 Phase 2 应产出的 BenchmarkSample（Phase 3 唯一入口）

**约定路径**：`data/benchmark_dataset/`（与 Phase 2 输出目录一致）

```
data/benchmark_dataset/
├── phase3_manifest.jsonl                      # Phase 3 唯一入口，每行 1 条 BenchmarkSample
├── first_frames/                               # 单图模式 / 多图模式都会有的“实际首帧”
│   └── {question_id}.png
├── ref_images/                                 # 仅 multi_image 模式：参考图集合
│   └── {question_id}_ref{k}.png
├── contrastive_pairs.jsonl                     # 配对索引（pair_id → original_qids + baseline_qids）
└── manifest.json                               # 数据集元信息 + evaluator_version 哈希
```

**Phase 3 直接消费 `BenchmarkSample`，字段 schema 见 §2.4（已经按 7 处适配层落地后的版本）。**

### 2.3 七处衔接适配层（已落实于 Phase 1/2 文档，Phase 3 直接消费）

> 以下 7 处衔接缺口已于本仓库 `三阶段实现需求/Phase1_先验数据准备.md` §5.6.4 与 `Phase2_Benchmark数据集合成.md` §6.3/§6.4/§6.6/§6.7/§8/附录 A/附录 C 全面修订落实。**Phase 3 代码在全部 Phase 2 输出上只需直接消费，不应再出现任何运时适配代码**；benchmark loader 遇到下表任一违规现象均应 fail-fast。

| # | 衔接点 | Phase 3 期望 | 落实位置 |
|---|---|---|---|
| 1 | 目录命名 | `first_frames/` 与 `ref_images/`，后缀 `.png` | Phase 2 §6.4 step 5（PNG）+ step 6 路径约定（禁止 `images/` `references/`） |
| 2 | 字段扁平化 | 17 个顶层字段、`prompt` 不是 `i2v_prompt`、禁止 `metadata.*` 嵌套 | Phase 2 §6.6 finalize_prompts（`prompt`） + §6.7 step 4（顶层拼装） |
| 3 | target_subjects 稳定 id 与 ref_image_idx | `[{id:"s1", noun:..., ref_image_idx:k}, ...]` | Phase 2 §6.3 step 6（赋值） + §6.4 step 6（与 ref 文件名 k 对齐） + §6.7 step 4（透传） |
| 4 | contrastive 顶层透传 | 顶层 `contrastive_pair_id` + `contrastive_role` + `contrastive_pairs.jsonl` | Phase 1 §5.6.4（配对透传约束）+ Phase 2 §6.7 step 4/5（顶层透传与 pairs 索引产出） |
| 5 | evaluator_tools 强枚举 | ⊂ 9 项工具枚举 | Phase 2 §6.3 step 7 + 附录 A.1 `_TOOLS_BY_DIM` |
| 6 | expected_failure_modes 默认填充 | ⊂ §2.5 的 15 种 FAILURE_MODES | Phase 2 §6.3 step 8 + 附录 A.2 `_FAILURES_BY_DIM` |
| 7 | source_type 词汇转换 | ⊂ `{tip_i2v_real, tip_i2v_synthetic_first_frame, external_real, external_synthetic}` | Phase 2 §6.7 step 4（映射写入）+ 附录 C `_LEGACY_SOURCE_MAP` |

**约束**：Phase 3 代码读到 BenchmarkSample 后**只允许**用顶层字段；遇到任何未适配字段（嵌套 `metadata`、旧字段名 `i2v_prompt`、旧目录 `images/` / `references/`、旧 source_type 词汇、`evaluator_tools` 越枚举、`expected_failure_modes` 为空、顶层出现 `_audit` 等）应**直接报错**，不能容忍。`_audit.*` 是 Phase 2 调试子节点，Phase 3 evaluator **禁止**读取。

### 2.4 BenchmarkSample 最终 schema（Phase 3 入口）

```json
{
  "question_id": "spatial_0042",
  "dimension": "spatial_composition",
  "input_mode": "multi_image",                 // single_image | multi_image
  "first_frame_path": "first_frames/spatial_0042.png",
  "input_image_paths": ["ref_images/spatial_0042_ref0.png",
                        "ref_images/spatial_0042_ref1.png"],   // multi_image 时 ≥2，single_image 时为空数组
  "prompt": "The cat jumps on the chair while looking at the camera",
  "target_subjects": [
    {"id": "s1", "noun": "cat",   "ref_image_idx": 0},
    {"id": "s2", "noun": "chair", "ref_image_idx": 1}
  ],
  "target_relation": {"type": "spatial", "value": "on", "subj": "s1", "obj": "s2"},
  "preservation_set": [
    {"scope": "s2", "constraint": "appearance"},
    {"scope": "background", "constraint": "global"}
  ],
  "contrastive_pair_id": "spatial_pair_0021",
  "contrastive_role": "original",              // 或 baseline_subject_swap（曾用 "inverse"，已废弃）
  "evaluator_tools": ["grounding", "depth", "vlm_relation"],
  "expected_failure_modes": ["wrong_relation", "object_missing"],
  "subtype": "static_relation",
  "difficulty": "medium",
  "semantic_rarity": "common",
  "source_type": "tip_i2v_real"
}
```

## 2.5 失败模式枚举（共 15 种，全局共享）

```
static_copy | global_filter | camera_pan_cheat | object_missing |
wrong_attribute | wrong_direction | wrong_relation | wrong_camera |
identity_lost | non_target_drift | background_drift | artifact_severe |
timing_wrong | identity_unbound | tool_uncertain
```

词汇变更说明（与早期草案差异）：

- `identity_drift` → `identity_lost`（与 identity_binding 软因子的判失语义对齐）
- `wrong_motion_direction` → `wrong_direction`（缩短 + 涵盖运动/视角方向）
- `view_collapse` → `wrong_camera`（直接表达"相机意图错"）
- `wrong_action` 拆解：由 `static_copy`（动作未发生）或 `timing_wrong`（动作时序错）覆盖
- 新增 `artifact_severe / timing_wrong / identity_unbound`（覆盖伪影 / 时序错 / 多图身份未绑定）
- 移除 `low_confidence / invalid_input`（已并入 `tool_uncertain` 单一不确定标签）

---

## §3 T2V-CompBench fork 与项目结构

### 3.1 fork 改造清单（**必须沿用 T2V-CompBench 的工程组织，不要另起炉灶**）

> Agent 第一步：`git clone https://github.com/KaiyueSun98/T2V-CompBench.git I2V-CompBench/_t2v_upstream` —— 然后按下表把需要复用的文件复制到 `I2V-CompBench/i2v_eval/tools_t2v/` 并打 patch。**保留 git history**。

| T2V-CompBench 上游文件 | 关键行号（已确认） | Phase 3 复用 / 改造点 |
|---|---|---|
| `MLLM_eval/spatial.py` | `spatial_judge` L222 / `pick_max_2d` L259 / `pick_max_3d` L295 / `combine_frame` L304 / `combine_csv_and_cal_model_score` L357 | 复用为 `i2v_eval/dimensions/spatial.py` 的 P 项；**`combine_csv_and_cal_model_score` 输出我们丢弃**（轨道 A 兼容用），主分走我们自己的 E·(0.6P+0.4C) |
| `MLLM_eval/motion.py` | `foreground` L632 / `background` L498 / `object_score` L347 / `model_score` L450 | 复用 fg/bg detector + DOT motion 分类，**强制 fps=8 重采样后再喂入**（KILLER-2） |
| `utils/video_utils.py` | `extract_frames` / `convert_video_to_grid` / `convert_video_to_standard_video` | 复用 + 在 grid 函数处加 `grid_layout='3x2'` 选项（默认 16 帧均匀采样→3×2 拼图取 6 关键帧） |
| `GroundingDINO/groundingdino_demo.py` | `load_model` L54 / `get_grounding_output` L65 | 复用做主体检测；新加 `text_prompt_from_subjects` helper |
| `requirements.txt` | - | 追加 `dinov2`, `transformers>=4.40`, `torch>=2.1`, `decord`, `loguru`, `pydantic>=2`, `tabulate`, `scipy`（Spearman） |

### 3.2 Phase 3 目标项目结构

```
I2V-CompBench/
├── _t2v_upstream/                              # 原样保留 T2V-CompBench 上游（参考用）
├── i2v_eval/
│   ├── __init__.py
│   ├── schemas.py                              # BenchmarkSample / EvalResult / DimensionScore (pydantic v2)
│   ├── enums.py                                # DIMENSIONS_V2 / FAILURE_MODES / TOOL_NAMES / SOURCE_TYPES
│   ├── cli.py                                  # 入口：`python -m i2v_eval.cli <subcommand>`
│   ├── pipeline/
│   │   ├── generate_videos.py                  # 调用 I2V 模型批量生成；不在本期内实现（占位 stub），但定义接口
│   │   ├── preprocess_videos.py                # KILLER-2 fps=8 + KILLER-3 frame↔video 转换
│   │   ├── extract_tool_features.py            # 一次性跑 grounding / depth / dot_motion / optical_flow / dinov2 / clip 并落盘
│   │   ├── evaluate.py                         # 七维 + identity_binding 主分计算
│   │   ├── human_validation.py                 # 抽样人工标注 + Spearman ρ
│   │   ├── aggregate.py                        # lambda ablation 扫描 / contrastive pair 配对统计
│   │   └── report.py                           # 模型对比报告（含 tool_uncertain 占比、E 命中率）
│   ├── dimensions/                             # 一维一文件，每个文件提供 evaluate_one(sample, video_features) -> DimensionScore
│   │   ├── attribute_binding.py
│   │   ├── action_binding.py
│   │   ├── motion_binding.py
│   │   ├── spatial_composition.py              # 调 tools_t2v/spatial.py
│   │   ├── background_dynamics.py
│   │   ├── view_transformation.py
│   │   ├── interaction_reasoning.py
│   │   └── identity_binding.py                 # multi_image 软惩罚因子（不计入 7 维主分，作为 E 系数乘子）
│   ├── tools/
│   │   ├── grounding.py                        # GroundingDINO 封装；输入 subjects→bbox/mask/conf
│   │   ├── depth.py                            # Depth Anything V2；KILLER-1 注释 + 工具内自动反向归一
│   │   ├── dot_motion.py                       # DOT 前景运动方向分类（fps=8 输入）
│   │   ├── optical_flow.py                     # RAFT 备选（背景稳定性 / 全局滤镜检测）
│   │   ├── vlm.py                              # LLaVA-OneVision / Qwen2-VL 统一接口；提供 disentangled VQA 三轮 priming
│   │   ├── dinov2.py                           # 身份保持
│   │   └── clip.py                             # 身份保持兜底
│   ├── tools_t2v/                              # 从 T2V-CompBench 复用并打 patch 的代码
│   │   ├── spatial.py
│   │   ├── motion.py
│   │   ├── video_utils.py
│   │   └── grounding_dino_helper.py
│   ├── utils/
│   │   ├── preservation.py                     # identity / non_target_drift / background_drift（七维共用）
│   │   ├── coherence.py                        # frame_yes_ratio / signal_stable / temporal_order
│   │   ├── scoring.py                          # gated_multiplicative_score(E, P, C, lam=0.6)
│   │   ├── frame_sampling.py                   # 16 帧 / 3×2 grid 6 帧 / 末帧 三种采样策略
│   │   ├── io.py                               # JSONL 读写、视频路径管理、evaluator_version 哈希
│   │   └── stats.py                            # Spearman / Kendall / per-dim 一致性
│   ├── baselines/
│   │   ├── static_copy.py
│   │   ├── random_motion.py
│   │   ├── global_filter.py
│   │   ├── camera_pan_cheat.py
│   │   └── run_all.py                          # 独立 CLI：`python -m i2v_eval.baselines.run_all`
│   └── tests/
│       ├── test_scoring.py                     # E·(0.6P+0.4C) 边界
│       ├── test_killer1_depth.py               # 必跑：放一张已知 GT 的图，断言"值大=离相机近"
│       ├── test_killer2_fps.py                 # 必跑：1 个 30fps 视频→重采样后 fps==8
│       ├── test_killer3_frame_video.py
│       ├── test_baselines.py                   # 4 个动态 baseline 必须显著低于 original（paired_E_diff > 0）
│       └── fixtures/                           # 10 条 BenchmarkSample mock + 几个 5s 视频
├── configs/
│   ├── evaluator_weights.yaml                  # 7 维 P/C 子项权重 + λ + 软惩罚阈值（哈希进 evaluator_version）
│   ├── tools_versions.yaml                     # 模型 ckpt 版本号（GroundingDINO / Depth Anything / DOT / VLM / DINOv2 / CLIP）
│   └── lambda_ablation.yaml                    # {0.4, 0.5, 0.6, 0.7}
├── requirements.txt
└── README.md
```

---

## §4 模块详规（11 个模块的最小实现）

### 4.1 `i2v_eval/schemas.py`

```python
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict
from .enums import DIMENSIONS_V2, FAILURE_MODES, TOOL_NAMES, SOURCE_TYPES

class TargetSubject(BaseModel):
    id: str                                  # "s1", "s2", ...
    noun: str
    ref_image_idx: Optional[int] = None      # multi_image 时必填

class TargetRelation(BaseModel):
    type: Literal["spatial","temporal","interaction","none"]
    value: str
    subj: Optional[str] = None
    obj: Optional[str] = None

class BenchmarkSample(BaseModel):
    question_id: str
    dimension: Literal[*DIMENSIONS_V2]
    input_mode: Literal["single_image","multi_image"]
    first_frame_path: str
    input_image_paths: List[str] = Field(default_factory=list)
    prompt: str
    target_subjects: List[TargetSubject]
    target_relation: TargetRelation
    preservation_set: List[Dict[str, str]] = Field(default_factory=list)
    contrastive_pair_id: Optional[str] = None
    contrastive_role: Optional[str] = None
    evaluator_tools: List[Literal[*TOOL_NAMES]]
    expected_failure_modes: List[Literal[*FAILURE_MODES]] = Field(default_factory=list)
    subtype: str = ""
    difficulty: Literal["easy","medium","hard"] = "medium"
    semantic_rarity: Literal["common","rare"] = "common"
    source_type: Literal[*SOURCE_TYPES]

class DimensionScore(BaseModel):
    question_id: str
    dimension: str
    E: float                  # 0/1 门控（识别期 + 输入有效性 + identity 软因子≥0.5）
    P: float                  # 主项分 [0,1]
    C: float                  # 一致性分 [0,1]
    S: float                  # 主分 = E·(0.6P + 0.4C)
    track_a_t2v_score: Optional[float] = None   # 兼容 T2V-CompBench CSV 的旧分（轨道 A）
    failure_modes: List[str] = Field(default_factory=list)
    tool_status: Literal["valid","tool_uncertain"]
    debug: Dict = Field(default_factory=dict)   # 工具命中率、子项分、识别失败原因等
```

### 4.2 `i2v_eval/utils/scoring.py`

**唯一打分函数**（禁止任何线性求和）：

```python
def gated_multiplicative_score(E: float, P: float, C: float, lam: float = 0.6) -> float:
    """主分 = E · (lam·P + (1-lam)·C)；禁止 E·P + (1-E)·C 等线性组合。"""
    assert 0.0 <= E <= 1.0 and 0.0 <= P <= 1.0 and 0.0 <= C <= 1.0
    assert 0.4 <= lam <= 0.7, "λ ablation 范围"
    return E * (lam * P + (1 - lam) * C)
```

### 4.3 `i2v_eval/utils/frame_sampling.py`（VLM 输入分流）

| 维度 | 采样策略 | 说明 |
|---|---|---|
| attribute_binding / background_dynamics | **3×2 grid 6 帧** | 16 帧均匀→挑 6 个关键帧拼成 3×2 大图 |
| action_binding / motion_binding / interaction_reasoning | **16 帧原生序列** | 不拼图，VLM 按视频/序列输入 |
| spatial_composition / view_transformation | **末帧单图** | 仅取最后 1 帧 |

实现签名：

```python
def sample_for_dimension(video_path: str, dimension: str) -> Union[Path, List[Path]]:
    """返回一张拼图路径 / 帧路径列表 / 单帧路径"""
```

### 4.4 `i2v_eval/utils/preservation.py`（七维共享）

```python
def identity_factor(ref_imgs: List[Path], video_frames: List[Path],
                    bbox_per_frame: List[Bbox]) -> float:
    """DINOv2 ⊕ CLIP cosine; 三档软惩罚返回 ∈ {0.5, 0.7, 1.0}"""

def non_target_drift(video_frames, target_bboxes, non_target_bboxes) -> float:
    """非目标主体的位置/外观漂移；越大越坏；阈值化后 → 失败模式 non_target_drift"""

def background_drift(first_frame_bg_mask, video_frames_bg) -> float:
    """背景稳定性；与 view_transformation 互斥（view 维度 background_drift 是好事）"""
```

### 4.5 `i2v_eval/utils/coherence.py`

```python
def frame_yes_ratio(per_frame_yes: List[bool]) -> float:
    """逐帧 VLM Yes/No → ratio[0,1]"""
def signal_stable(signal_per_frame: List[float], expected: str) -> float:
    """signal 一致性（如运动方向不应中途反向）"""
def temporal_order(events: List[Tuple[str,int]]) -> float:
    """因果链评分：A 推 B → B 倒，事件先后必须吻合"""
```

### 4.6 七维 evaluator 实现要点（每文件 1 个 `evaluate_one`）

> 统一接口：`evaluate_one(sample: BenchmarkSample, feat: ToolFeatures) -> DimensionScore`
> `feat` 是 §4.7 一次性提取的工具特征字典。

#### A. attribute_binding

- **E**：GroundingDINO 在末帧能定位到所有 target_subjects（IoU≥0.3）→ 1，否则 0；`identity_factor` < 0.5 时强制 E=0。
- **P**：VLM 多选题 `What is the {attr_type} of the {noun}? A/B/C/D` × 重复 3 次取 majority；正确率。
- **C**：`frame_yes_ratio(逐帧 VLM 二值"Is the {attr} of {noun} correct? Yes/No")`。
- **失败模式**：wrong_attribute / object_missing / identity_lost。

#### B. action_binding

- **E**：GroundingDINO 命中 target_subject + 该主体在 16 帧上有>3 帧出现。
- **P**：VLM 16 帧序列 + 拆问 `Is the {subj} performing {action_verb}? Yes/No` × 3。
- **C**：`signal_stable(per_frame_action_logit, expected=action_verb)` + `non_target_drift` 约束。
- **失败模式**：static_copy / object_missing / timing_wrong。

#### C. motion_binding

- **预处理硬约束**：必须先经过 KILLER-2 fps=8 重采样。
- **E**：DOT 在前景能输出有效运动向量 + 强度 > 阈值。
- **P**：DOT 方向分类 vs `motion_slot.direction`（leftward / rightward / upward / ...）。
- **C**：方向稳定性（`signal_stable`）+ baseline `random_motion` 必须 P 显著低于 original。
- **失败模式**：wrong_direction / static_copy / camera_pan_cheat（如果 fg motion 与 bg motion 高度相关）。

#### D. spatial_composition

- **E**：调用 `tools_t2v/spatial.py:spatial_judge`（L222）确认两个主体都在末帧被定位。
- **P**：复用 `pick_max_2d`（L259, 用于 left/right/above/below）+ `pick_max_3d`（L295, 用于 in_front_of / behind 需要 depth）；**对 `pick_max_3d` 必须传入 KILLER-1 修正后的 depth**（值大=近）。
- **C**：`combine_frame`（L304）逐帧聚合后取均值（**注意：丢弃 `combine_csv_and_cal_model_score` L357 的输出，那是轨道 A**）。
- **失败模式**：wrong_relation / object_missing / identity_lost。

#### E. background_dynamics

- **E**：`background_drift` > 阈值（背景"应该"变化）+ 前景主体 identity 不漂移。
- **P**：VLM 3×2 grid 提问 `Is the {bg_change_type} happening (e.g., raining → not raining)? Yes/No` × 3。
- **C**：optical_flow 背景区域有持续运动信号 + 前景区域运动不显著。
- **失败模式**：background_drift / non_target_drift / global_filter（前景背景被一起改色调）。

#### F. view_transformation

- **E**：grounding 确认主体在首帧/末帧都能找到（容许位置变化）。
- **P**：VLM 末帧 + 首帧对比 `Has the camera performed {camera_command}? Yes/No`；同时光流全局向量必须与 camera_command 方向一致。
- **C**：optical_flow 全局运动方向稳定（`signal_stable`）。
- **失败模式**：wrong_camera / camera_pan_cheat / identity_lost。

#### G. interaction_reasoning

- **E**：grounding 命中 agent + patient 主体；两主体在 16 帧序列上至少有 ≥3 帧重叠或邻接（IoU 或距离阈值）。
- **P**：VLM 16 帧序列 + disentangled 三轮：① agent 在做什么 ② patient 状态变化 ③ 因果链是否成立。
- **C**：`temporal_order(events, expected=interaction_slot.expected_outcome)`。
- **失败模式**：wrong_relation / timing_wrong / object_missing。

#### H. identity_binding（仅 multi_image，不计入 7 维主分）

- 计算 `identity_factor(ref_imgs, video_frames)`。
- 三档软惩罚：≥0.7 → 1.0；[0.5, 0.7) → 0.7；<0.5 → 0.5。
- **作为 E 的乘子**：所有 7 维 evaluator 的 `E_final = E_raw × identity_factor_band`。
- 同时挂在 `DimensionScore.debug["identity_factor"]`。

### 4.7 `i2v_eval/pipeline/extract_tool_features.py`

**只跑一遍**，把所有视频的工具输出全部落盘成单一 npz/parquet，避免每个 evaluator 重跑模型：

```
{output}/tool_features/{question_id}__{model_name}.parquet
  - frames_16: List[np.ndarray]
  - frames_grid_6: np.ndarray
  - last_frame: np.ndarray
  - grounding: {subject_id: [bbox_per_frame, conf_per_frame]}
  - depth: [H,W,T]            # 已经做过 KILLER-1 反向归一
  - optical_flow: [H,W,2,T-1]
  - dot_motion: {fg_dir, fg_mag_per_frame, bg_dir, bg_mag_per_frame}  # fps=8 input
  - dinov2_features: [T, D]
  - clip_features: [T, D]
  - vlm_outputs: {prompt_template_id: List[str]}    # 缓存 VLM 重复 3 次的输出
```

### 4.8 `i2v_eval/pipeline/evaluate.py`

```python
def main():
    samples = load_jsonl("data/benchmark_dataset/phase3_manifest.jsonl", BenchmarkSample)
    for model_name in cfg.models_to_eval:
        results = []
        for s in samples:
            feat = load_features(s.question_id, model_name)
            ev = DIMENSION_EVALUATORS[s.dimension]
            score = ev.evaluate_one(s, feat)
            # multi_image 注入 identity 因子
            if s.input_mode == "multi_image":
                idf = identity_binding.evaluate_one(s, feat).P
                score.E *= bandify(idf)
                score.S = gated_multiplicative_score(score.E, score.P, score.C, cfg.lam)
            results.append(score)
        save_jsonl(f"results/{model_name}/eval.jsonl", results)
```

### 4.9 `i2v_eval/pipeline/aggregate.py`

- λ ablation：从 `configs/lambda_ablation.yaml` 读 {0.4, 0.5, 0.6, 0.7}，对每个 λ 重算 S，输出按维度 ranking 是否稳定。
- contrastive pair：按 `contrastive_pair_id` 聚合，要求 `E_original − E_baseline > 0` 在 ≥80% pair 上成立（**KILLER 验收**）。

### 4.10 `i2v_eval/pipeline/human_validation.py`

- 每模型每维抽 ≥50 样本（共 7×50=350 条起），common/rare 按 80/20 分层抽样。人工打分 0/0.5/1 三级，每样本 ≥3 名标注员取中位数。
- 计算 Spearman ρ(human, S) per-dim；**所有 7 维 ρ ≥ 0.5 才算评测系统达标**（`§8 验收门槛`）。
- **执行 lambda ablation**：在人评子集上扫 `lambda ∈ {0.4, 0.5, 0.6, 0.7}`，取使 Spearman ρ(S, human_S) 最大者作为最终 lambda。
- Human validation **前置于 aggregate**——必须先通过 Spearman ρ ≥ 0.5 的可信门控，aggregate 才被允许把该维度写入论文主表。

### 4.11 `i2v_eval/baselines/`

4 种动态合成基线（由评测端生成，不进入 `phase3_manifest.jsonl`）：

| baseline 名 | 视频生成方式 | 期望失败 |
|---|---|---|
| static_copy | 把首帧直接复制 16 帧 | static_copy |
| random_motion | 加噪声光流 / 随机平移 | wrong_direction |
| global_filter | 给原视频套全局色调滤镜 | global_filter |
| camera_pan_cheat | 把首帧平移生成假“运镜” | camera_pan_cheat |

第 5 种对照 `baseline_subject_swap`（交换 ref_images 顺序 / 主体，multi_image 专属）由 Phase 2 在数据合成阶段已落盘为独立 BenchmarkSample 行，通过 `contrastive_pairs.jsonl` 索引与 original 配对，Phase 3 直接读取计算 `paired_E_diff`，不需动态合成。

每个动态 baseline 必须有独立 CLI：`python -m i2v_eval.baselines.run_all --models <m1,m2>`。

---

## §5 三大 KILLER 硬约束（必须有单测兜底）

### KILLER-1：Depth Anything 反直觉语义

> Depth Anything V2 输出**反向深度图**：值越大=离相机越近，与传统深度图语义相反。

**强制做法**：在 `i2v_eval/tools/depth.py` 加载后**立刻**做：
```python
depth_normalized = depth_raw / depth_raw.max()        # ∈ [0,1]
proximity = depth_normalized                           # 命名改为 proximity 避免误用
# 注释中显式写："proximity[i,j] = 1 表示该像素距相机最近"
```
所有下游（包括 `pick_max_3d`）只接受 `proximity`，**禁止接受 raw depth**。

**单测**：`test_killer1_depth.py` 放一张前景近、背景远的已知 GT 图，断言 `proximity[fg_pixels].mean() > proximity[bg_pixels].mean()`。

### KILLER-2：Motion fps=8 重采样

> DOT 训练分布要求 fps=8。直接把 30fps 视频喂入会让方向预测严重漂移。

**强制做法**：`preprocess_videos.py` 对所有视频：
```python
target_fps = 8
duration = num_frames / src_fps
target_num_frames = int(round(duration * target_fps))   # 不足/超出都重采样
```
**单测**：`test_killer2_fps.py` 把一个 30fps、5s 共 150 帧的视频丢进去，断言重采样后 frames=40。

### KILLER-3：frame→video 双轨评分

- **轨道 A（兼容 T2V-CompBench CSV）**：`combine_csv_and_cal_model_score` 输出落到 `results/{model}/track_a_t2v_compatible.csv`，**仅供论文对比**，不参与主分。
- **轨道 B（我们的主分）**：`E·(0.6P+0.4C)`，输出到 `results/{model}/eval.jsonl`，包含 `tool_uncertain` 标记。

**单测**：`test_killer3_frame_video.py` 跑同一个视频 + 同一个 sample，验证两个轨道 CSV/JSONL 同时产生且字段不重叠。

---

## §6 配置文件占位（必填项）

### 6.1 `configs/evaluator_weights.yaml`

```yaml
# 主分 lambda
scoring:
  lambda: 0.6                  # P 项权重；ablation 走 lambda_ablation.yaml

# 七维子项权重（如果某维有多个 P 子项，用此处加权；和必须=1.0）
dimensions:
  attribute_binding:
    P_subitems: {vlm_mcq: 1.0}
    C_subitems: {frame_yes_ratio: 1.0}
  action_binding:
    P_subitems: {vlm_yesno_3rep: 1.0}
    C_subitems: {signal_stable: 0.7, non_target_drift: 0.3}
  motion_binding:
    P_subitems: {dot_direction_match: 1.0}
    C_subitems: {direction_stable: 0.6, fg_bg_decoupled: 0.4}
  spatial_composition:
    P_subitems: {pick_max_2d: 0.5, pick_max_3d: 0.5}
    C_subitems: {combine_frame_mean: 1.0}
  background_dynamics:
    P_subitems: {vlm_yesno_3rep: 1.0}
    C_subitems: {bg_optical_flow: 0.6, fg_static: 0.4}
  view_transformation:
    P_subitems: {vlm_first_last: 0.5, global_flow_dir: 0.5}
    C_subitems: {global_flow_stable: 1.0}
  interaction_reasoning:
    P_subitems: {vlm_disentangled_3turn: 1.0}
    C_subitems: {temporal_order: 1.0}

# multi_image identity 软惩罚阈值
identity_binding:
  bands: [{min: 0.7, factor: 1.0}, {min: 0.5, factor: 0.7}, {min: 0.0, factor: 0.5}]
  weight_dinov2: 0.6
  weight_clip: 0.4

# evaluator_version 哈希源（写入 manifest.json）
hash_inputs:
  - this_file
  - configs/tools_versions.yaml
```

### 6.2 `configs/tools_versions.yaml`

```yaml
grounding_dino:    {ckpt: "GroundingDINO_SwinT_OGC.pth", version: "1.0.0"}
depth_anything:    {ckpt: "depth_anything_v2_vitl.pth", version: "2.0"}
dot:               {ckpt: "dot_2024_03.pth", version: "0.1.0", required_fps: 8}
optical_flow:      {model: "RAFT", ckpt: "raft-things.pth"}
vlm:               {model: "llava-onevision-qwen2-7b-ov", repeat: 3, temperature: 0.0}
dinov2:            {model: "dinov2_vitl14"}
clip:              {model: "ViT-L-14"}
```

---

## §7 CLI 流水线（按顺序）

```
# 1) Phase 2 已经产出 phase3_manifest.jsonl
python -m i2v_eval.cli generate_videos     --models cogvideox-i2v,seine,dynamicrafter --benchmark data/benchmark_dataset/phase3_manifest.jsonl --out videos/

# 2) KILLER-2 + KILLER-3 预处理
python -m i2v_eval.cli preprocess_videos   --in videos/ --out videos_8fps/ --target_fps 8

# 3) 一次性提取所有工具特征
python -m i2v_eval.cli extract_tool_features --videos videos_8fps/ --bench data/benchmark_dataset/phase3_manifest.jsonl --out tool_features/

# 4) 主分评测（七维 + identity）
python -m i2v_eval.cli evaluate            --bench data/benchmark_dataset/phase3_manifest.jsonl --features tool_features/ --models all --out results/

# 5) 抽样人工标注 + Spearman 计算
python -m i2v_eval.cli human_validation    --results results/ --sample_per_dim 50 --out results/human/

# 6) λ ablation + contrastive pair 聚合
python -m i2v_eval.cli aggregate           --results results/ --lambdas 0.4,0.5,0.6,0.7 --out results/aggregated/

# 7) 报告
python -m i2v_eval.cli report              --aggregated results/aggregated/ --human results/human/ --out reports/

# 旁路：跑 4 种动态 baseline
python -m i2v_eval.baselines.run_all       --bench data/benchmark_dataset/phase3_manifest.jsonl --models all --out results_baseline/
```

---

## §8 验收门槛（不通过=Phase 3 不算交付）

| 门槛 | 验收方式 | 阈值 |
|---|---|---|
| 7 维 human-vs-S Spearman ρ | `human_validation.py` | **每维 ρ ≥ 0.5** |
| 5 baseline 配对差异 | `aggregate.py` `paired_E_diff > 0` 占比 | **≥ 80%** |
| KILLER-1 单测 | `pytest tests/test_killer1_depth.py` | 必须 PASS |
| KILLER-2 单测 | `pytest tests/test_killer2_fps.py` | 必须 PASS |
| KILLER-3 单测 | `pytest tests/test_killer3_frame_video.py` | 必须 PASS |
| evaluator_version 哈希 | `manifest.json` 存在且与 weights/tools yaml 哈希一致 | 必填 |
| tool_uncertain 占比 | 评测样本中 tool_status="tool_uncertain" 占比 | **< 10%** |
| λ ablation 排名稳定性 | 模型按 S 的排名在 λ∈{0.4,0.5,0.6,0.7} 上的 Kendall τ | **≥ 0.8** |

---

## §9 实施优先级（P0 / P1 / P2）

### P0（最小可跑通骨架）

1. `git clone` T2V-CompBench 到 `I2V-CompBench/_t2v_upstream/`，按 §3 复制并打 patch。
2. 写好 `schemas.py` + `enums.py` + `utils/scoring.py` + `utils/io.py`。
3. 实现 `pipeline/preprocess_videos.py`（KILLER-2）+ `tools/depth.py`（KILLER-1）+ 三个 KILLER 单测。
4. 实现 `pipeline/extract_tool_features.py` 串通 grounding + vlm + depth + dot + optical_flow + dinov2 + clip 七个工具。
5. 先实现 **2 个维度**：`spatial_composition`（最复杂，验证 fork 复用）+ `attribute_binding`（最简单，验证 VLM/grounding 链路）。
6. 写 `tests/fixtures/` 10 条假 BenchmarkSample + 5 个短视频，端到端跑通 `evaluate → aggregate → report`。

### P1（功能完整）

7. 补完剩余 5 个维度 + `identity_binding`。
8. 实现 4 种动态 baseline + `baselines/run_all.py`。
9. 实现 `human_validation.py` + Spearman 计算。
10. 实现 `aggregate.py` 的 λ ablation 和 contrastive pair 聚合。

### P2（论文级）

11. 实现 `report.py` 模型对比表（含 tool_uncertain 占比可视化）。
12. 实现轨道 A T2V-CompBench 兼容 CSV 导出。
13. **应急兜底**（默认不启用，Phase 2 已全面落实适配）：如遇 Phase 2 未交付需赶工跑通 Phase 3 水线，可使用 §C.2 的 `scripts/adapt_phase2_to_phase3.py`；Phase 2 交付后必须从流线中移除该脚本。

---

## §10 给 Agent 的工作守则

1. **任何评分函数禁止线性求和**——只准用 `gated_multiplicative_score`。代码 review 时全文 grep `E\s*\*\s*P\s*\+|E\s*\+\s*P` 必须为空。
2. **任何调用 Depth Anything 的地方必须经过 `tools/depth.py:get_proximity()`**，禁止直接读 raw depth 值。
3. **任何调用 DOT 的地方必须先经过 `preprocess_videos.py` 输出（fps=8）**，禁止直接喂原始视频。
4. **VLM 调用必须 repeat=3 取 majority/mean**，单次调用一律视为非法。
5. **每个 evaluator 都必须显式给出 failure_modes**（即使为空数组也要赋值）；遇到工具不可用返回 `tool_status="tool_uncertain"` + S=0，**不要伪造分数**。
6. **不要在 `i2v_eval/` 内编辑 `_t2v_upstream/`**，所有改造走复制+patch；保留上游 git history 便于追溯。
7. **不要创建 `images/` `references/` 这类旧名目录**；目录命名严格按 §2.2。
8. **所有 yaml 配置改动必须同步更新 `evaluator_version` 哈希**；hash 写入 `data/benchmark_dataset/manifest.json` 与 `results/{model}/eval.jsonl` 顶部一行。
9. 遇到 BenchmarkSample 字段缺失或类型不符——**直接 raise**，不要静默跳过；这是 Phase 2 没做适配层的强信号，必须暴露。
10. **遇到 Phase 2 还未实施**（默认不会发生，Phase 2 §6.3/§6.7 已落实）：如需赶工，可按 §9 P2 第 13 项描述暂用 §C.2 应急脚本。Phase 3 代码库本身**不允许**调用该脚本；遇到任何不符 §2.4 schema 的 BenchmarkSample 只能 fail-fast。

---

## §A 附录：维度 → evaluator_tools / expected_failure_modes 强制映射表（§2.3 #5 #6）

> 本附录与 Phase 2 `Phase2_Benchmark数据集合成.md` 附录 A.1 / A.2 **逐字一致**，是上下游共享的权威枚举表。Phase 2 §6.3 step 7/8 在写入 BenchmarkSample 前已按表强制覆写，Phase 3 只需验证取值仍⊂表内枚举。

### A.1 `_TOOLS_BY_DIM`（required evaluator_tools）

| dimension | required evaluator_tools |
|-----------|--------------------------|
| attribute_binding | `grounding`, `vlm_attribute`, `vlm_existence` |
| action_binding | `grounding`, `vlm_existence`, `optical_flow` |
| motion_binding | `grounding`, `dot_motion`, `optical_flow` |
| spatial_composition | `grounding`, `depth`, `vlm_relation` |
| background_dynamics | `grounding`, `optical_flow`, `vlm_existence` |
| view_transformation | `depth`, `optical_flow`, `vlm_existence` |
| interaction_reasoning | `grounding`, `vlm_relation`, `optical_flow` |
| identity_binding (multi_image 加挂) | `dinov2`, `clip`, `grounding` |

**multi_image 加挂**：当 `input_mode == multi_image` 时，无论 dimension 为何，都额外追加 `dinov2, clip, grounding`（身份保持 + 外观一致性度量）。

**强枚举值域**：`{grounding, depth, dot_motion, optical_flow, vlm_existence, vlm_attribute, vlm_relation, dinov2, clip}` 共 9 项。

### A.2 `_FAILURES_BY_DIM`（默认 expected_failure_modes）

| dimension | default expected_failure_modes |
|-----------|--------------------------------|
| attribute_binding | `wrong_attribute`, `object_missing`, `identity_lost` |
| action_binding | `static_copy`, `object_missing`, `timing_wrong` |
| motion_binding | `wrong_direction`, `static_copy`, `camera_pan_cheat` |
| spatial_composition | `wrong_relation`, `object_missing`, `identity_lost` |
| background_dynamics | `background_drift`, `non_target_drift`, `global_filter` |
| view_transformation | `wrong_camera`, `camera_pan_cheat`, `identity_lost` |
| interaction_reasoning | `wrong_relation`, `timing_wrong`, `object_missing` |

**强枚举值域**：必须 ⊂ §2.5 FAILURE_MODES 15 种枚举。LLM 在 Phase 2 §6.6 polish 阶段可追加更具体的失败模式，但仍须保持在 15 项内。

---

## §B 附录：完整 enums

```python
DIMENSIONS_V2 = [
    "attribute_binding", "action_binding", "motion_binding",
    "spatial_composition", "background_dynamics",
    "view_transformation", "interaction_reasoning",
]
TOOL_NAMES = [
    "grounding", "depth", "dot_motion", "optical_flow",
    "vlm_existence", "vlm_attribute", "vlm_relation",
    "dinov2", "clip",
]
FAILURE_MODES = [
    "static_copy", "global_filter", "camera_pan_cheat", "object_missing",
    "wrong_attribute", "wrong_direction", "wrong_relation", "wrong_camera",
    "identity_lost", "non_target_drift", "background_drift", "artifact_severe",
    "timing_wrong", "identity_unbound", "tool_uncertain",
]
SOURCE_TYPES = [
    "tip_i2v_real", "tip_i2v_synthetic_first_frame",
    "external_real", "external_synthetic",
]
```

---

## §C 附录：`_LEGACY_SOURCE_MAP` + 应急适配脚本（§2.3 #7）

### C.1 `_LEGACY_SOURCE_MAP`（Phase 1 词汇 → Phase 3 枚举）

与 Phase 2 `Phase2_Benchmark数据集合成.md` 附录 C 一致。Phase 2 §6.7 step 4 已按表写入 BenchmarkSample 顶层 `source_type`；Phase 3 loader 遇到任何旧词汇直接 fail-fast。

| Phase 1 source_type（旧） | Phase 3 source_type（顶层枚举） |
|--------------------------|----------------------------------|
| `observed_single_image` | `tip_i2v_real` |
| `derived_single_image` | `tip_i2v_synthetic_first_frame` |
| `derived_multi_reference` | `tip_i2v_synthetic_first_frame` |
| `external_real` | `external_real` |
| `external_synthetic` | `external_synthetic` |

**Phase 3 顶层 source_type 强枚举值域**：`{tip_i2v_real, tip_i2v_synthetic_first_frame, external_real, external_synthetic}` 共 4 项。

### C.2 应急适配脚本（§9 P2 兜底，**默认不启用**）

> Phase 2 已在 §6.3/§6.7 完成 7 处适配，**本脚本仅在 Phase 2 交付之前需要赶工跑通 Phase 3 水线时使用**；一旦 Phase 2 产出可用，需从流线中移除，Phase 3 代码库不允许在运行时调用。

```python
# scripts/adapt_phase2_to_phase3.py  （应急，默认不启用）
from i2v_eval.enums import REQUIRED_TOOLS_BY_DIM, DEFAULT_FAILURES_BY_DIM, LEGACY_SOURCE_MAP

def adapt(phase2_raw: dict) -> dict:
    out = {**phase2_raw.pop("metadata", {}), **phase2_raw}    # 字段扁平化（#2）
    out["prompt"] = out.pop("i2v_prompt", out.get("prompt", ""))
    # target_subjects 补 id + ref_image_idx（#3）
    for i, s in enumerate(out.get("target_subjects", [])):
        s.setdefault("id", f"s{i+1}")
        if out.get("input_mode") == "multi_image":
            s.setdefault("ref_image_idx", i)
    # contrastive 透传（#4）
    out["contrastive_pair_id"] = out.get("contrastive_pair_id") or _alloc_pair_id(out)
    out["contrastive_role"] = out.get("contrastive_role", "original")
    # evaluator_tools 强制（#5） — 表与 §A.1 一致
    out["evaluator_tools"] = REQUIRED_TOOLS_BY_DIM[out["dimension"]]
    # expected_failure_modes 默认填充（#6） — 表与 §A.2 一致
    out.setdefault("expected_failure_modes", DEFAULT_FAILURES_BY_DIM[out["dimension"]])
    # source_type 词汇映射（#7） — 表与 §C.1 一致
    out["source_type"] = LEGACY_SOURCE_MAP.get(out.get("source_type"), "tip_i2v_real")
    # 目录名（#1）由 export 时统一处理，不在此脚本
    return out
```

---

**报告结束。Agent 收到本文件后，按 §9 优先级 P0 → P1 → P2 顺序逐项交付，每完成一项即跑 §8 对应单测/验收门槛。**
