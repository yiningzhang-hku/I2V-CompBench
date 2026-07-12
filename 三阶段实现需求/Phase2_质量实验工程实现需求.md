# Phase 2质量实验与正式数据集筛选工程实现需求

> 面向AI Coding Agent的工程规格文档  
> 适用范围：I2V-CompBench数据集构建方（Phase 1 + Phase 2）  
> 最终目标：完成质量对比实验，并从五维候选池中筛选每维300条、共1500条正式样本

---

## 0. 文档约定

本文使用以下强制级别：

- **MUST**：违反即不得进入下一阶段；
- **SHOULD**：默认实现，若不实现必须在run manifest中说明原因；
- **MAY**：资源允许时实现的增强项。

Coding Agent不得将本文中的“候选方案”直接写成实验结论。所有提升数值必须来自实际运行产物。

## 1. 工程目标

### 1.1 核心目标

在不覆盖现有Phase 2原始产物的前提下，建立一条可断点续跑、可审计、可复现的质量实验流水线，完成：

1. 对3517条五维候选执行统一质量审计；
2. 修复或剔除空主体、空目标变化和无效fallback；
3. 比较Prompt低频词治理方案；
4. 比较图像清晰度增强方案；
5. 比较4:3/16:9与尺寸适配方案；
6. 完成主体Tier标注和难度重标；
7. 生成开发集、验证集、人工标注任务与统计报告；
8. 从质量合格候选中按五维等量原则筛选1500条正式数据集；
9. 输出论文第5章可直接引用的表格、JSON统计和失败案例索引。

### 1.2 正式范围

正式维度仅包括：

```python
FORMAL_DIMENSIONS = [
    "attribute_binding",
    "action_binding",
    "motion_binding",
    "background_dynamics",
    "view_transformation",
]
```

`spatial_composition` 和 `interaction_reasoning` **MUST NOT** 进入正式1500条样本。

### 1.3 非目标

本工程不负责：

- 实现Phase 3的E/P/C视频评分器；
- 构建多模型Leaderboard；
- 研究自动视频指标与人类判断的相关性；
- 实现Grounding、DINOv2或光流作为正式视频评分器；
- 用人工拼词或替换主体扩充不存在的recipe；
- 为凑足300条而降低硬质量门槛。

Grounding、DINOv2、CLIP等工具只允许作为**输入数据质量信号**使用。

## 2. 当前基线与已知阻断项

### 2.1 当前候选池

| 维度 | 候选数 | 正式目标 |
|---|---:|---:|
| attribute_binding | 517 | 300 |
| action_binding | 1013 | 300 |
| motion_binding | 676 | 300 |
| background_dynamics | 946 | 300 |
| view_transformation | 365 | 300 |
| **总计** | **3517** | **1500** |

### 2.2 无模型审计结果

当前审计已确认：

- 3517条 `target_subjects[].noun` 缺失；
- 3517条使用泛化主体描述 `the subject`；
- 3517条缺少可供下游直接消费的结构化目标变化；
- 121条prompt字数异常；
- 113条存在重复冠词；
- 99条已有 `failed_check`；
- 18条存在空槽位模式；
- 117条使用fallback。

因此，旧 `phase3_manifest.jsonl` **MUST NOT** 直接作为正式数据集发布。

### 2.3 已修复但尚需重跑的代码问题

仓库已修改：

- Phase 2支持读取Phase 1真实字段 `aligned_subjects`；
- Phase 2支持顶层 `attribute_change_slots`、`action_slots`、`motion_slots`、`background_change_slots`、`camera_movement_slots`；
- Phase1Bundle支持 `image_parse_v2.jsonl` 和 `text_parse_v2.jsonl`；
- View维度从“禁止运镜词”修正为“必须包含运镜词”；
- finalize新增空槽位、重复冠词和失败重试；
- export新增结构化目标和failed prompt阻断。

Coding Agent **MUST NOT** 只对旧question plan重跑export。优先重跑：

```text
plan → finalize → export → audit
```

若Phase 1 bundle不可用，才允许走§9的旧产物迁移修复路径。

## 3. 工程原则

### 3.1 原始产物不可覆盖

以下文件视为只读：

```text
data/benchmark_dataset/phase3_manifest.jsonl
data/benchmark_dataset/question_plans.jsonl
data/benchmark_dataset/prompts/final_prompts.jsonl
data/benchmark_dataset/samples/*.jsonl
data/benchmark_dataset/by_dimension/**
```

实验变体和修复结果 **MUST** 写入独立run目录。

### 3.2 幂等与断点续跑

每个步骤必须：

- 输入相同且配置相同时输出确定；
- 已完成记录默认跳过；
- 支持 `--resume`；
- 对API或模型调用逐条落盘；
- 单条失败不得中断整批任务；
- 失败原因使用强枚举，不写自由文本作为主状态。

### 3.3 API安全

涉及VLM/LLM调用的命令默认dry-run。只有显式传入 `--allow-api` 才能调用外部API。

必须支持：

- `--limit N`：仅处理N条；
- `--max-api-calls N`：API调用上限；
- `--estimated-cost-only`：只输出预计调用数；
- `--resume`：跳过已有有效响应；
- 记录模型名、temperature、prompt hash和原始响应。

### 3.4 依赖与模型权重

代码不得在运行时静默下载大模型。缺少权重时返回 `missing_model_weight` 并给出路径说明。

所有图像增强和视觉工具权重需写入配置，并在run manifest记录文件hash。

## 4. 推荐工程目录

```text
configs/
  quality_experiments.yaml
  quality_models.yaml

src/i2vcompbench/quality/
  __init__.py
  cli.py
  schemas.py
  paths.py
  hashing.py
  audit.py
  split.py
  target_repair.py
  prompt_rules.py
  prompt_experiment.py
  image_metrics.py
  image_variants.py
  clarity_experiment.py
  aspect_experiment.py
  subject_tier.py
  difficulty.py
  orthogonal.py
  ablation.py
  cost_decision.py
  human_annotation.py
  statistics.py
  final_selection.py
  report.py

scripts/
  audit_final_candidates.py       # 保留为兼容wrapper，内部调用quality.audit

tests/quality/
  fixtures/
  test_audit.py
  test_split.py
  test_target_repair.py
  test_prompt_rules.py
  test_image_paths.py
  test_subject_tier.py
  test_difficulty.py
  test_final_selection.py
  test_quality_cli_smoke.py

data/benchmark_dataset/quality_experiments/
  <run_id>/
    run_manifest.json
    logs/
    audit/
    splits/
    target_repair/
    prompt/
    clarity/
    aspect/
    subject/
    difficulty/
    orthogonal/
    ablation/
    decision/
    human/
    selection/
    report/
```

## 5. 配置文件规格

新增 `configs/quality_experiments.yaml`：

```yaml
run:
  run_id: null                    # null时生成 YYYYMMDD_HHMMSS
  seed: 20260712
  output_root: data/benchmark_dataset/quality_experiments
  overwrite: false

input:
  benchmark_root: data/benchmark_dataset
  phase1_bundle_dir: "E:/I2V-CompBench/outputs/phase1"
  manifest: phase3_manifest.jsonl
  question_plans: question_plans.jsonl
  final_prompts: prompts/final_prompts.jsonl
  formal_dimensions:
    - attribute_binding
    - action_binding
    - motion_binding
    - background_dynamics
    - view_transformation

split:
  development_per_dimension: 50
  validation_per_dimension: 50
  stratify_by: [difficulty, semantic_rarity]

target_repair:
  prefer_phase1_rebuild: true
  allow_vlm_migration: true
  min_confidence: 0.80
  manual_review_below: 0.90

prompt:
  min_words: 8
  max_words: 25
  zipf_threshold: 3.5
  do_not_replace_pos: [NOUN, PROPN]
  max_llm_retries: 3
  semantic_consistency_min: 0.95

clarity:
  sample_size: 120
  face_subset_size: 40
  methods: [lanczos, unsharp, realesrgan, swinir]
  optional_methods: [realesrgan_gfpgan]
  video_probe_size: 30

aspect:
  ratio_stage_sample_size: 60
  strategy_stage_sample_size: 120
  ratios: ["4:3", "16:9"]
  methods: [stretch, center_crop, letterbox, blur_padding, saliency_crop]
  optional_methods: [outpainting]

subject_tier:
  ratios:
    T1_common: 0.55
    T2_longtail: 0.28
    T3_finegrained: 0.12
    T4_rare_fictional: 0.05

difficulty:
  weights:
    target_complexity: 0.30
    subject_localization: 0.25
    background_interference: 0.20
    semantic_rarity: 0.15
    judgeability: 0.10
  initial_thresholds: [0.33, 0.66]

selection:
  per_dimension: 300
  difficulty_ratio: {easy: 0.40, medium: 0.35, hard: 0.25}
  rarity_ratio: {common: 0.70, rare: 0.30}
  reject_old_fallback: true
  require_image_exists: true
  require_schema_complete: true
  require_dimension_verified: true

orthogonal:
  enabled: true
  quadrant_sizes:
    A_control: 80
    B_lexical_probe: 60
    C_subject_probe: 60
    D_compound_probe: 40
  zipf_threshold_lexical: 4.0
  subject_tier_split: ["T1_common", "T2_longtail"]  # 归入common的tier

ablation:
  enabled: true
  evaluation_set: "validation"  # 在验证集上运行
  conditions: ["baseline", "text_only", "image_only", "sampling_only", "text_image", "full"]
  statistical_test: "wilcoxon"
  alpha: 0.05
  correction: "bonferroni"

decision:
  weights:
    quality: 0.60
    time: 0.25
    complexity: 0.15
  max_total_budget_hours: 8
  prefer_deterministic: true
```

配置加载后必须执行比例求和、路径存在性、方法枚举和数值范围校验。

## 6. 核心Schema

所有Schema使用Pydantic v2或dataclass实现。禁止不同模块自行定义同名字段。

### 6.1 QualityCandidate

```python
class QualityCandidate(BaseModel):
    question_id: str
    dimension: Literal[
        "attribute_binding",
        "action_binding",
        "motion_binding",
        "background_dynamics",
        "view_transformation",
    ]
    difficulty_old: Literal["easy", "medium", "hard"]
    semantic_rarity: Literal["common", "rare"]
    prompt: str
    first_frame_path: str
    source_manifest_hash: str
    source_plan_hash: str | None = None
    quadrant: Literal[
        "A_control", "B_lexical_probe", "C_subject_probe", "D_compound_probe"
    ] | None = None
```

### 6.2 TargetRepairResult

```python
class TargetRepairResult(BaseModel):
    question_id: str
    target_subjects: list[SubjectRef]
    target_relation: TargetRelation
    reviewed_dimension: str
    dimension_consistent: bool
    confidence: float
    repair_source: Literal["phase1_rebuild", "vlm_migration", "manual"]
    status: Literal[
        "pass", "needs_manual_review", "rejected", "api_failed", "invalid_response"
    ]
    raw_response_path: str | None = None
```

五个维度的 `target_relation.type` 固定为：

```python
TARGET_TYPE_BY_DIMENSION = {
    "attribute_binding": "attribute",
    "action_binding": "action",
    "motion_binding": "motion",
    "background_dynamics": "background",
    "view_transformation": "view",
}
```

### 6.3 PromptVariantResult

```python
class PromptVariantResult(BaseModel):
    question_id: str
    method: Literal[
        "A0", "A1", "A2", "A3", "A4", "A5", "A6",
        "B0", "B1", "B2", "B3",
    ]
    source_layer: Literal["P1-Source-Off", "P1-Source-On"] = "P1-Source-Off"
    prompt_before: str
    prompt_after: str
    word_count: int
    rare_modifier_hits: list[str]
    structural_issues: list[str]
    semantic_consistency: float | None = None
    dimension_consistent: bool | None = None
    rare_modifier_rate: float | None = None
    mean_zipf_score: float | None = None
    clip_sim_delta: float | None = None
    perplexity_mean: float | None = None
    semantic_preservation_rate: float | None = None
    api_calls: int = 0
    status: Literal["pass", "failed", "needs_manual_review"]
```

### 6.4 ImageVariantResult

```python
class ImageVariantResult(BaseModel):
    question_id: str
    experiment: Literal["clarity", "aspect"]
    method: str
    source_path: str
    output_path: str
    source_sha256: str
    output_sha256: str
    source_size: tuple[int, int]
    output_size: tuple[int, int]
    metrics: dict[str, float | None]
    status: Literal["pass", "failed", "missing_dependency", "missing_weight"]
```

### 6.5 HumanAnnotation

```python
class HumanAnnotation(BaseModel):
    annotation_id: str
    question_id: str
    experiment: str
    method: str
    annotator_id: str
    dimension_correct: bool | None = None
    target_consistent: bool | None = None
    prompt_naturalness: int | None = None   # 1..5
    subject_complete: bool | None = None
    identity_changed: bool | None = None
    artifact_present: bool | None = None
    overall_usable: bool | None = None
    comment: str = ""
```

### 6.6 SelectionDecision

```python
class SelectionDecision(BaseModel):
    question_id: str
    dimension: str
    eligible: bool
    blocking_reasons: list[str]
    prompt_method: str | None
    image_method: str | None
    subject_tier: str | None
    difficulty_new: str | None
    quality_rank_score: float | None
    selected: bool = False
    bucket_assignments: dict[str, str] = {}
```

## 7. CLI规格

统一入口：

```bash
python -m i2vcompbench.quality.cli \
  --config configs/quality_experiments.yaml \
  --run-id thesis_v1 \
  <command>
```

必须支持的命令：

```text
audit
split
repair-targets
prompt-variants
prompt-metrics
clarity-variants
clarity-metrics
aspect-variants
aspect-metrics
tag-subjects
calibrate-difficulty
orthogonal-assign
orthogonal-analyze
ablation-run
ablation-analyze
cost-estimate
decision-matrix
prepare-human
import-human
select-final
report
all-local
```

通用参数：

```text
--resume
--limit N
--only-qid QUESTION_ID
--dry-run
--allow-api
--max-api-calls N
--log-level INFO|DEBUG
```

`all-local`只能执行不需要API和大模型权重的步骤，不得隐式发起网络调用。

## 8. Step 1：审计与划分

### 8.1 audit.py

输入：旧manifest、question plans、final prompts、图像目录。

必须检查：

- 是否属于五个正式维度；
- question_id是否唯一；
- prompt是否为空；
- 是否存在重复冠词、空槽位、占位符、长度异常；
- final prompt是否带 `failed_check`；
- target_subjects是否为空或泛化；
- target_relation是否为空；
- preservation_set是否为空；
- 图像路径是否存在；
- Windows反斜杠路径是否可转换；
- 同一源样本是否产生重复候选。

图像路径解析优先级：

1. manifest中原始路径；
2. `first_frames/{qid}.png`；
3. `by_dimension/{dimension}/{qid}/first_frame.png`；
4. 旧版兼容目录。

输出：

```text
audit/candidate_quality_rows.jsonl
audit/candidate_quality_summary.json
audit/blocking_examples.jsonl
```

### 8.2 split.py

开发集和验证集各按维度50条，按 `(difficulty_old, semantic_rarity)` 比例分层。

要求：

- 两个集合不得重叠；
- 每维数量必须精确；
- 同一 `source_sample_id` SHOULD 不跨集合；
- 输出包含原始完整行，不只输出qid；
- 输出split hash。

## 9. Step 2：结构化目标修复

### 9.1 首选路径：Phase 1重建

若Phase 1 bundle可用：

1. 验证 `image_parse_v2.jsonl`、`text_parse_v2.jsonl`、`aligned_instances.jsonl`；
2. 运行修正后的build_question_plan；
3. 检查目标主体、noun和target relation；
4. 将新plan与旧question_id建立稳定映射；
5. 输出repair overlay，不覆盖旧plan。

若重建后关键字段仍为空，则进入VLM迁移路径。

### 9.2 VLM迁移路径

VLM输入：首帧、final prompt、旧dimension、旧target plan和VLM caption。

VLM必须返回严格JSON：

```json
{
  "target_subjects": [
    {"id": "s1", "noun": "elephant", "description": "cartoon elephant"}
  ],
  "target_relation": {
    "type": "action",
    "value": "lifts its trunk upward",
    "subj": "s1",
    "obj": null
  },
  "reviewed_dimension": "action_binding",
  "dimension_consistent": false,
  "confidence": 0.94,
  "rationale": "The prompt describes a body action rather than an attribute change."
}
```

硬规则：

- 不允许只根据final prompt猜主体，必须同时观察图像；
- noun不得为 `subject`、`object`、`thing`；
- target relation value不得为空；
- reviewed_dimension不一致时不得自动改写原dimension，进入人工队列；
- confidence < 0.80直接reject；
- 0.80 ≤ confidence < 0.90进入人工复核；
- API原始响应必须保存。

输出：

```text
target_repair/repairs.jsonl
target_repair/manual_review_queue.jsonl
target_repair/rejected.jsonl
target_repair/raw_responses/{qid}.txt
```

### 9.3 验收

- repair结果行数与目标输入数一致；
- `status=pass`的Schema完整率100%；
- 开发集人工主体准确率≥95%；
- 开发集目标变化准确率≥95%；
- 维度冲突样本不得自动通过。

## 10. Step 3：Prompt实验

### 10.1 prompt_rules.py

实现无模型规则：

- 空prompt；
- 未解析 `{slot}`；
- `the the`、`a a`、`an an`；
- 冠词后直接标点；
- 字数范围；
- 维度显式禁词；
- View必须包含运镜线索；
- Prompt必须包含可识别变化谓词；
- 低频词检测时排除NOUN和PROPN。

规则函数必须是纯函数，并有单元测试。

### 10.2 A0—A6（逐层消融）

设计原则：每层严格包含上一层，可通过逐层对比量化每个策略的增量贡献。

- **A0**：原prompt（基线，不做任何修改）；
- **A1**：仅修改Polish指令模板（在`prompt_polish.txt`中增加词汇约束规则#9–#11，属于软约束）；
- **A2**：A1 + 确定性同义词映射表替换（`RARE_TO_COMMON`映射，如`ethereal→glowing`）；
- **A3**：A2 + Zipf词频硬门控（wordfreq Zipf < 3.5 触发retry）；
- **A4**：A3 + RareWordBlocker解码层硬拒绝（`LogitsProcessor`级别禁止rare token生成）；
- **A5**：A4 + CLIP-Sim反向验证（polish前后CLIP-Sim下降>0.02则回退）；
- **A6**：A5 + GPT-2 Perplexity过滤（perplexity > 50触发retry）。

#### 源头预洗消融因子

- **P1-Source-Off**：不对Phase 1 VLM输出做预洗（当前状态）；
- **P1-Source-On**：在`_resolve_slots`前插入`_prewash_slot_value`，对VLM返回的修饰词做前置清洗。

总计形成 **7（A层）× 2（源头层）= 14个实验条件**。

通过消融可回答：

1. 仅改指令有多大效果？（A1 vs A0）
2. 同义词映射的增量是多少？（A2 vs A1）
3. Zipf硬门控的增量是多少？（A3 vs A2）
4. 解码层硬拒绝在Zipf之上还有多少增量？（A4 vs A3）
5. CLIP-Sim后验检查能进一步降低多少？（A5 vs A4）
6. Perplexity过滤的额外贡献？（A6 vs A5）
7. 源头预洗的独立贡献是多少？（Source-On vs Source-Off在各A层的差异）

#### A系列禁止修改项

A2的确定性映射表禁止：

- 修改主体noun；
- 修改动作核心动词；
- 修改方向词；
- 修改相机指令；
- 删除否定词。

A3/A4/A5/A6的LLM输入必须包含不可变字段：目标主体、目标关系、目标维度和preservation set。

### 10.3 B0—B3

对所有117条旧fallback运行：

- B0：旧prompt；
- B1：确定性模板；
- B2：LLM重写；
- B3：B2 + 自动检查 + 人工复核队列。

若关键槽位为空，B1必须返回 `missing_required_slot`，不得生成“The subject ...”占位句。

### 10.4 输出

```text
prompt/variants.jsonl
prompt/metrics_by_method.json
prompt/fallback_comparison.json
prompt/manual_annotation_tasks.jsonl
prompt/failure_examples.jsonl
prompt/source_layer_comparison.json
```

每条`variants.jsonl`必须包含以下额外指标字段：

| 指标 | 含义 | 目标 |
|------|------|------|
| `rare_modifier_rate` | 低频修饰词比例 | <1% |
| `mean_zipf_score` | prompt中非名词词汇的平均Zipf分数 | >4.5 |
| `clip_sim_delta` | polish前后CLIP-Sim变化 | ≥-0.02 |
| `perplexity_mean` | GPT-2平均困惑度 | <50 |
| `semantic_preservation_rate` | 与原prompt的语义相似度（sentence-transformers cosine） | ≥0.95 |

### 10.5 胜出规则

先应用硬门槛：

- 目标语义保持率（`semantic_preservation_rate`）≥95%；
- 维度纯度≥95%；
- 无效prompt率为0；
- `clip_sim_delta` ≥ -0.02（不得显著偏离原始语义）；
- `perplexity_mean` < 50（生成文本必须流畅自然）。

满足硬门槛后，按以下优先级排序选择正式方法：

1. `rare_modifier_rate`最低（主目标）；
2. `mean_zipf_score`最高；
3. `semantic_preservation_rate`最高；
4. API成本最低。

若多个方法均满足硬门槛且质量差异在95% CI内不显著，优先选择Pareto前沿上处理时间更短、确定性更高的方法（参见§16.4）。

## 11. Step 4：图像清晰度实验

### 11.1 样本选择

从开发集选择120张，分层因素：

- 五个维度；
- 人物/非人物；
- 原始长边区间；
- Laplacian模糊程度；
- 横图/方图/竖图。

人脸子集40张。

### 11.2 处理方法

- C0：Lanczos；
- C1：Lanczos + 轻度Unsharp；
- C2：Real-ESRGAN；
- C3：SwinIR；
- C4：Real-ESRGAN + GFPGAN，仅可选消融；
- C5：Real-ESRGAN + Unsharp Masking（sigma=1.5, amount=0.5）；
- C6：Real-ESRGAN + CLAHE（clipLimit=2.0, tileGridSize=8×8）；
- C7：Real-ESRGAN + Unsharp + CLAHE（完整组合流水线，对应分析报告§2.5.6推荐方案）。

每个输出路径：

```text
clarity/images/{method}/{question_id}.png
```

不得覆盖原图。保存方法参数和源/输出hash。

### 11.3 指标

实现：

- Laplacian variance；
- Tenengrad；
- NIQE；
- BRISQUE；
- DINOv2全图与主体区域相似度；
- 人脸身份相似度（仅人脸子集）；
- Grounding目标检测成功率；
- `enhancement_chain_contribution`：每个后处理环节的增量贡献（用NIQE差值量化，仅适用于C5/C6/C7组合流水线）。

缺少bbox时，主体区域指标记为null，不得使用整图指标假装主体指标。

### 11.4 视频探针边界

本模块只导出：

```text
clarity/video_probe_manifest.jsonl
```

包含30条×入选方法的首帧和prompt。视频可由现有共享adapter生成；质量包只导入人工清晰度、身份和伪影标注，不实现Phase 3评分。

### 11.5 胜出规则

身份改变率不得显著高于C0；DINO主体相似度不得超过配置容差下降。满足保真门槛后，选择人工清晰度和主体检测成功率最高的方法。

若多个方法均满足硬门槛且质量差异在95% CI内不显著，优先选择Pareto前沿上处理时间更短、确定性更高的方法（参见§16.4）。

### 11.6 组合消融设计

目标：量化C5/C6/C7中锐化和对比度增强各自的增量贡献，以及组合是否存在边际递减。

消融路径：

```text
路径A：C2 (SR-only) → C5 (SR+Sharpen) → C7 (SR+Sharpen+CLAHE)
路径B：C2 (SR-only) → C6 (SR+CLAHE) → C7 (SR+Sharpen+CLAHE)
```

对每个环节报告：

| 对比 | 量化指标 | 含义 |
|------|---------|------|
| C5 - C2 | ΔNIQE, ΔLaplacian | Unsharp Masking的独立增量 |
| C6 - C2 | ΔNIQE, ΔBRISQUE | CLAHE的独立增量 |
| C7 - C5 | ΔNIQE, ΔBRISQUE | 在已有USM基础上叠加CLAHE的边际收益 |
| C7 - C6 | ΔNIQE, ΔLaplacian | 在已有CLAHE基础上叠加USM的边际收益 |

若 `C7 - C5` 的增量小于 `C6 - C2` 的独立增量，则存在边际递减，需在报告中标注。

输出：

```text
clarity/ablation_chain_metrics.json
clarity/ablation_chain_plot.png
```

## 12. Step 5：尺寸与比例实验

### 12.1 比例阶段

从六类宽高比区间各取10张，共60张。使用同一保守方法输出4:3和16:9版本。

输出：

```text
aspect/ratio/{ratio}/{question_id}.png
aspect/ratio_metrics.json
aspect/ratio_probe_manifest.jsonl
```

### 12.2 策略阶段

在冻结比例下比较：

- D0 stretch；
- D1 center_crop；
- D2 letterbox；
- D3 blur_padding或reflection_padding；
- D4 saliency_crop；
- D5 outpainting，仅可选上界。

### 12.3 混合策略整体验证

新增实验臂：

- **D6_hybrid_conservative**：stretch(≤4%) + saliency_crop(4–20%) + blur_padding(>20%)；
- **D7_hybrid_aggressive**：stretch(≤4%) + saliency_crop(4–30%) + outpainting(>30%)。

对比设计：

- D6/D7 vs 最佳单一方法（从D0–D5中选出的winner）；
- 使用验证集120张全量运行，按宽高比分桶报告（近方图/中等比/极端比三桶）；
- 每桶至少30张有效样本。

输出：

```text
aspect/hybrid_comparison.json
aspect/hybrid_per_bucket.csv
```

### 12.4 阈值敏感性分析

对混合路由器的两个阈值进行网格搜索：

- `stretch_threshold`：[0.02, 0.04, 0.06, 0.08]
- `saliency_threshold`：[0.15, 0.20, 0.25, 0.30, 0.35]

每组合在60张ratio阶段样本上运行，报告：

| 指标 | 说明 |
|------|------|
| 主体保留率 | GroundingDINO recall（IoU≥0.5） |
| CLIP-Sim(image, prompt) | 图文对齐 |
| NIQE分数 | 无参考图像质量 |
| 各档位实际触发比例 | stretch/saliency_crop/padding各路由实际被触发的比例 |

输出：

```text
aspect/threshold_sensitivity_heatmap.json
aspect/threshold_sensitivity_plot.png
```

最终阈值选择标准：在主体保留率≥95%前提下，选CLIP-Sim最高的组合。

### 12.5 Saliency Crop要求

必须：

- 使用目标主体bbox或显著性区域；
- 裁剪框包含完整主体bbox及安全边距；
- 无bbox时不得假装主体感知，回退到padding并记录原因；
- 输出裁剪前后主体保留率。

### 12.6 Outpainting限制

Outpainting输出不得进入正式主流程，除非人工验证和语义保持实验明确通过。默认仅作对比上界。

### 12.7 正式混合策略

Coding Agent实现可配置路由器：

```python
if ratio_error <= 0.04:
    method = "minor_resize_or_crop"
elif ratio_error <= 0.30 and subject_bbox_available:
    method = "saliency_crop"
else:
    method = "blur_or_reflection_padding"
```

阈值必须来自配置，不得硬编码在多处。

## 13. Step 6：主体Tier

### 13.1 标注信号

按优先级组合：

1. TIP-I2V内部主体频率；
2. COCO/LVIS本地类别表；
3. 细粒度类别资源；
4. WordNet hypernym；
5. wordfreq兜底。

资源文件放在：

```text
resources/quality/
  coco80.json
  lvis1203.json
  subtlex_top_nouns.json
  finegrained_categories.json
```

不允许运行时从网络静默下载。

### 13.2 多词主体

必须同时保存：

```json
{
  "surface_form": "skeleton king",
  "head_noun": "king",
  "canonical_form": "skeleton king",
  "tier": "T4_rare_fictional",
  "evidence": ["not_in_lvis", "zipf_below_3"]
}
```

不得仅用head noun将“skeleton king”归为常见人物。

### 13.3 输出与验收

```text
subject/subject_tiers.jsonl
subject/distribution_natural.json
subject/distribution_stratified_preview.json
subject/unknown_subjects.jsonl
```

随机抽查每Tier至少30条，报告人工准确率。

## 14. Step 7：难度重标

### 14.1 特征

每项归一化到 `[0,1]`：

- target_complexity；
- subject_localization；
- background_interference；
- semantic_rarity；
- judgeability。

综合分：

```python
D = (
    0.30 * target_complexity
    + 0.25 * subject_localization
    + 0.20 * background_interference
    + 0.15 * semantic_rarity
    + 0.10 * judgeability
)
```

权重从配置读取。

### 14.2 禁止混入质量缺陷

以下因素不得增加difficulty：

- 图像模糊；
- prompt无效；
- 空主体；
- 图文错配；
- 结构字段缺失。

这些问题必须在质量门控中修复或剔除。

### 14.3 人工标定

开发集250条生成匿名人工难度标注任务。至少两名标注者独立给出easy/medium/hard。

比较：

- G0旧标签；
- G1单目标复杂度；
- G2五因素固定权重；
- G3人工标定阈值或有序模型。

输出：

```text
difficulty/features.jsonl
difficulty/human_tasks.jsonl
difficulty/calibration.json
difficulty/labels_new.jsonl
difficulty/confusion_matrix.csv
```

## 14bis. Step 7b：词汇×主体正交诊断实验

### 14bis.1 目的

验证"生僻词影响"和"生僻主体影响"是否可独立分离，为论文第5章提供正交诊断能力的数据支撑。本实验解耦两类失败模式：

- 文本编码器对low-frequency modifier的鲁棒性缺失；
- 视觉概念层对长尾主体的识别能力不足。

### 14bis.2 四象限定义

| 象限 | 提示词 | 主体 | 测量能力 |
|------|--------|------|---------|
| **A (控制组)** | Common（所有修饰词Zipf≥4.0） | T1/T2 (common) | 纯组合能力基线 |
| **B (词汇探针)** | Rare（含≥1个Zipf<4.0修饰词） | T1/T2 (common) | 文本编码器对rare token的鲁棒性 |
| **C (主体探针)** | Common（所有修饰词Zipf≥4.0） | T3/T4 (rare) | 视觉概念层对长尾主体的识别能力 |
| **D (复合探针)** | Rare | T3/T4 (rare) | 交互效应（仅诊断用，不入总分） |

判定规则：

- **Common prompt**：prompt中所有非名词/专名词汇的Zipf分数均≥4.0；
- **Rare prompt**：prompt中存在至少1个非名词/专名词汇Zipf<4.0；
- **Common主体**：subject tier为T1_common或T2_longtail；
- **Rare主体**：subject tier为T3_finegrained或T4_rare_fictional。

### 14bis.3 样本分配

从3517条候选中按以下规则分配：

| 象限 | 最低数量 | 抽样策略 |
|------|---------|---------|
| A控制组 | ≥80条 | 从quality合格池中随机抽样，确保prompt全为common词 |
| B词汇探针 | ≥60条 | 保留或人工注入rare修饰词，主体限T1/T2 |
| C主体探针 | ≥60条 | prompt做common化处理，主体限T3/T4 |
| D复合探针 | ≥40条 | 保留rare prompt + T3/T4主体 |

总计≥240条作为正交诊断子集，从正式1500条中划出或作为附加探针集。

### 14bis.4 与P1/P4的关系

- P1的A系列实验在**A+B象限子集**上运行，验证词汇简化对B象限得分的提升是否显著高于A象限（预期：B象限提升>A象限提升）；
- P4的分层采样在**A+C象限子集**上验证，确认T3/T4主体在C象限的得分下降是否独立于词汇因素。

### 14bis.5 统计分析

使用2×2 ANOVA：

- **主效应1（词汇）**：B-A 和 D-C 的平均差异；
- **主效应2（主体）**：C-A 和 D-B 的平均差异；
- **交互效应**：(D-C) - (B-A) 是否显著≠0。

报告要求：

- effect size（Cohen's d）和95% CI；
- 若交互效应显著（p<0.05），说明两类因素不可独立分离，在论文中需讨论混淆路径；
- 若交互效应不显著，两类治理（P1词汇治理、P4主体采样）可独立优化。

### 14bis.6 输出

```text
orthogonal/quadrant_assignments.jsonl
orthogonal/anova_results.json
orthogonal/effect_sizes.json
orthogonal/interaction_plot.png
```

### 14bis.7 Schema扩展

在QualityCandidate中增加：

```python
quadrant: Literal["A_control", "B_lexical_probe", "C_subject_probe", "D_compound_probe"] | None = None
```

### 14bis.8 与正式1500条的关系

正交诊断子集可以是正式1500条的子集（标注quadrant字段），也可以是额外探针集。规则：

- A象限样本同时计入正式样本配额；
- D象限样本**不入总分计算**，仅在论文中作为诊断证据；
- 若作为子集，不得因诊断需要降低正式样本的质量门槛。

## 15. Step 8：人工标注工具

### 15.1 导出

`prepare-human`输出JSONL和CSV，字段包含相对图像路径、prompt、方法匿名编码和标注问题。

方法名必须随机匿名为M1/M2/M3，避免标注者知道模型或处理方案。

### 15.2 导入

`import-human`必须检查：

- annotation_id唯一；
- 标注者ID非空；
- 量表范围合法；
- 每个实验样本是否满足最少标注人数；
- 是否存在缺失方法组。

### 15.3 一致性

实现Cohen's Kappa/Fleiss' Kappa和bootstrap 95% CI。若SciPy等依赖缺失，应给出清晰错误，不得静默跳过统计。

## 16. Step 9：正式1500条筛选

### 16.1 硬门控

样本必须同时满足：

- 属于五个正式维度；
- 图像文件存在；
- target subjects、noun和target relation完整；
- dimension review通过；
- prompt结构检查通过；
- 不带failed_check；
- 旧fallback只有在重新修复后才可进入；
- 图文一致性通过；
- 图像质量方案已冻结并成功产出；
- 无人工reject标记。

### 16.2 目标边际配额

每维300条：

- difficulty：40% / 35% / 25%；
- semantic rarity：70% / 30%；
- subject tier：55% / 28% / 12% / 5%作为参考边际。

不要求三类变量完整笛卡尔积硬配额，避免桶爆炸。

### 16.3 选择优先级

```text
硬质量门槛
  > 每维总量300
  > difficulty边际
  > common/rare边际
  > subject tier边际
  > 质量排序分
```

同层候选质量排序权重必须在配置中定义。不得把任务难度作为质量低的惩罚项。

### 16.4 成本-效果联合决策框架

#### 目的

在质量胜出规则之外引入工程可行性维度，帮助选择在时间/计算约束下的最优方案。

#### 成本维度定义

每个实验方法需记录：

```python
class MethodCost(BaseModel):
    method_id: str
    experiment: Literal["prompt", "clarity", "aspect"]
    requires_gpu: bool
    requires_api: bool
    gpu_vram_gb: float | None
    api_calls_per_sample: float
    api_cost_per_call_usd: float | None
    processing_time_per_sample_sec: float  # 实测或估算
    total_time_4092_samples_min: float
    model_weight_size_gb: float | None
    additional_dependencies: list[str]
    deterministic: bool  # 是否每次运行结果相同
```

#### 预填成本估算表

要求在实验运行前预填估算，运行后更新实测值：

| 方法 | GPU | API | 单张耗时 | 4092张总时 | 额外依赖 | 确定性 |
|------|-----|-----|---------|-----------|---------|--------|
| A0 (原prompt) | - | - | 0s | 0min | - | ✓ |
| A1 (改指令) | - | ✓ | 2s | 137min | - | ✗ |
| A2 (映射表) | - | - | 0.01s | 1min | wordfreq | ✓ |
| A3 (+Zipf门控) | - | ✓ | 2s | 137min | wordfreq | ✗ |
| A4 (+RareWordBlocker) | ✓ | - | 3s | 205min | transformers | ✗ |
| A5 (+CLIP-Sim) | ✓ | - | 1s | 68min | clip | ✗ |
| A6 (+Perplexity) | ✓ | - | 0.5s | 34min | transformers | ✓ |
| C0 (Lanczos) | - | - | 0.05s | 3min | PIL | ✓ |
| C2 (Real-ESRGAN) | ✓ | - | 0.5s | 34min | realesrgan | ✓ |
| C7 (SR+USM+CLAHE) | ✓ | - | 0.6s | 41min | realesrgan,opencv | ✓ |
| D3 (blur_padding) | - | - | 0.05s | 3min | PIL | ✓ |
| D4 (saliency_crop) | ✓ | - | 0.2s | 14min | grounding-dino | ✓ |
| D5 (outpainting) | ✓ | - | 5s | 341min | diffusers | ✗ |

#### 决策矩阵

方案选择使用加权评分：

```python
def method_score(quality_gain: float, cost: MethodCost, weights: dict) -> float:
    """
    quality_gain: 相对基线的质量提升（归一化到[0,1]）
    weights: 从配置读取
    """
    time_penalty = min(1.0, cost.total_time_4092_samples_min / 480)  # 8h为最大预算
    determinism_bonus = 0.1 if cost.deterministic else 0.0
    
    score = (
        weights["quality"] * quality_gain
        - weights["time"] * time_penalty
        - weights["complexity"] * len(cost.additional_dependencies) * 0.05
        + determinism_bonus
    )
    return score
```

权重配置（见§5 `configs/quality_experiments.yaml`）：

```yaml
decision:
  weights:
    quality: 0.60
    time: 0.25
    complexity: 0.15
  max_total_budget_hours: 8
  prefer_deterministic: true
```

#### Pareto前沿分析

对每个问题域（prompt/clarity/aspect），绘制质量提升 vs 处理时间的散点图，标出Pareto前沿上的方法。只有Pareto前沿上的方法才是合理候选。

输出：

```text
decision/method_costs.jsonl
decision/pareto_frontier.json
decision/decision_matrix.csv
decision/pareto_plots/{experiment}.png
```

#### 与胜出规则的关系

§10.5/§11.5/§12.7的胜出规则增加一条：

> 若多个方法均满足硬门槛且质量差异在95% CI内不显著，优先选择Pareto前沿上处理时间更短、确定性更高的方法。

### 16.5 View供给风险

View候选只有365条。若硬门控后不足300条：

- 输出shortfall；
- 优先人工修复可修复样本；
- 不降低Schema、维度正确率或图像存在性门槛；
- 不从其他维度补齐；
- 不自动生成新题。

### 16.6 输出

```text
selection/final_1500_manifest.jsonl
selection/final_ids.txt
selection/selection_decisions.jsonl
selection/quota_actual_vs_target.json
selection/shortfall_report.json
selection/rejected_examples.jsonl
```

正式manifest路径必须为POSIX风格相对路径，不写Windows反斜杠和机器绝对路径。

## 16bis. Step 9b：组合消融实验

### 16bis.1 目的

验证各治理层的独立贡献和联合收益，回答"如果只能选一个方案，应该优先做什么"。

### 16bis.2 消融条件

| 条件 | Text治理 | Image增强 | Sampling策略 | 说明 |
|------|----------|-----------|------------|------|
| **Baseline** | A0 (原prompt) | C0 (Lanczos) | 无Tier配额 | 当前产物原始状态 |
| **+Text** | A_winner | C0 | 无Tier配额 | 仅修复prompt |
| **+Image** | A0 | C_winner | 无Tier配额 | 仅增强图像 |
| **+Sampling** | A0 | C0 | Tier配额+难度校准 | 仅改采样策略 |
| **+Text+Image** | A_winner | C_winner | 无Tier配额 | 文本+图像联合 |
| **Full** | A_winner | C_winner + D_winner | Tier配额+难度校准 | 所有治理层叠加 |

其中`A_winner`、`C_winner`、`D_winner`分别为§10、§11、§12实验的胜出方法。

### 16bis.3 样本与评估

在验证集250条上运行所有6个条件。每条件输出：

| 指标类别 | 具体指标 |
|---------|---------|
| Prompt质量 | rare_rate, fk_grade, perplexity |
| 图像质量 | NIQE, Laplacian variance, DINOv2 sim |
| 结构完整率 | Schema pass rate |
| 综合可用率 | 同时通过所有硬门控的比例 |

### 16bis.4 统计检验

- **配对比较**：每对条件使用McNemar检验（pass/fail二分类）或Wilcoxon符号秩检验（连续分数）；
- **多重比较校正**：Bonferroni或FDR（Benjamini-Hochberg）；
- 报告每对比较的p-value和effect size；
- 至少报告以下关键对比的显著性：

| 对比 | 验证目标 |
|------|---------|
| Baseline vs Full | 总效果 |
| +Text vs Baseline | 文本治理独立效果 |
| +Image vs Baseline | 图像增强独立效果 |
| +Text+Image vs +Text | 图像在文本已修复基础上的增量 |
| Full vs +Text+Image | 采样策略在前两者之上的增量 |

### 16bis.5 输出

```text
ablation/conditions.jsonl
ablation/per_sample_scores.jsonl
ablation/pairwise_tests.json
ablation/ablation_summary_table.csv
ablation/contribution_bar_chart.png
```

## 17. Step 10：论文报告生成

`report`必须输出：

```text
report/quality_experiment_report.md
report/tables/*.csv
report/figures/*.png
report/method_winners.json
report/final_dataset_card.md
```

至少包含：

1. 3517条候选的基线问题统计；
2. A0—A6 Prompt逐层消融对比（含源头预洗因子）；
3. B0—B3 fallback修复率；
4. C0—C7清晰度与身份保持（含组合消融链）；
5. 4:3/16:9和D0—D7尺寸策略对比（含混合策略与阈值敏感性）；
6. 自然抽样与主体分层抽样对比；
7. G0—G3难度标定对比；
8. Baseline、+Text、+Image、+Sampling、Full组合消融；
9. 正式1500条的五维、难度、罕见度和主体分布；
10. 失败案例与限制；
11. 正交诊断2×2 ANOVA结果与交互效应图；
12. 组合消融贡献柱状图与配对检验结果；
13. 各问题域Pareto前沿图与最终方案选择依据。

报告生成器只读取结构化结果，不重新计算模型指标。

## 18. 测试要求

### 18.1 单元测试

必须覆盖：

- Windows/Unix路径解析；
- 重复冠词和空槽位；
- View运镜线索；
- target relation五维映射；
- 分层抽样总数与不重叠；
- 最大余数配额；
- subject多词短语；
- 难度分数边界；
- 硬门控原因；
- View不足300时shortfall而非降级。

### 18.2 集成测试

构造每维2条、共10条fixture，运行：

```text
audit → split → local prompt rules → tag subjects → difficulty → selection preview → report
```

不得调用网络。

### 18.3 API smoke test

在显式 `--allow-api --limit 2 --max-api-calls 2` 下验证target repair或prompt rewrite。默认CI不得调用API。

### 18.4 回归测试

必须为以下已知bug建立测试：

- `aligned_subjects`字段被正确读取；
- `image_parse_v2.jsonl`和`text_parse_v2.jsonl`可发现；
- View prompt含zoom/pan时不被错误拒绝；
- “The the subject.”被拒绝；
- failed fallback不进入正式manifest。

## 19. 分阶段开发任务

### Task 0：环境与输入体检

工作：检查依赖、Phase 1 bundle、图像目录和现有产物。

完成标准：输出 `environment_report.json`，不修改数据。

### Task 1：质量包骨架

工作：创建目录、Schema、配置加载、paths、hash和CLI。

完成标准：`all-local --limit 10`可以运行空骨架并生成run manifest。

### Task 2：审计与划分迁移

工作：将现有 `scripts/audit_final_candidates.py` 逻辑迁入package，保留兼容wrapper。

完成标准：审计计数与当前基线一致；开发/验证集各250条且不重叠。

### Task 3：结构化目标修复

工作：实现Phase 1重建和VLM迁移两条路径、overlay与人工队列。

完成标准：开发集Schema完整率100%，人工目标准确率达到门槛。

### Task 4：Prompt与fallback实验

工作：实现A0—A6逐层消融（含源头预洗因子）、B0—B3、规则、指标和人工任务。

完成标准：117条fallback均有明确pass/reject结果；不存在无效prompt通过；14个消融条件均有完整指标。

### Task 5：图像实验

工作：实现图像变体（含C5—C7组合流水线和D6—D7混合策略）、指标、清晰度消融链和尺寸阈值敏感性实验。

完成标准：所有变体不覆盖原图；缺少模型时可恢复并清晰报告；组合消融链和阈值热力图输出可生成。

### Task 6：主体与难度

工作：实现Tier、多词主体、难度特征和人工标定。

完成标准：所有eligible候选都有Tier和新难度或明确失败原因。

### Task 7：正式筛选与报告

工作：硬门控、边际配额、1500条选择、数据卡和论文表格。

完成标准：满足§20验收；不足时输出shortfall而非伪造完成。

### Task 7b：正交诊断

工作：实现象限分配（`orthogonal-assign`）、2×2 ANOVA分析（`orthogonal-analyze`）和效应量报告。

完成标准：至少240条分配到四象限；ANOVA结果含主效应和交互效应的p值与Cohen's d。

### Task 8：组合消融与决策

工作：实现消融条件运行器（`ablation-run`）、统计检验（`ablation-analyze`）、成本估算（`cost-estimate`）和Pareto分析（`decision-matrix`）。

完成标准：6个条件在验证集上有完整分数；配对检验结果和决策矩阵可生成；每个问题域Pareto前沿图可输出。

## 20. 最终验收标准

### 20.1 P0验收

- 审计覆盖3517条五维候选；
- 开发集和验证集各250条且无重叠；
- `status=pass`样本结构化字段完整率100%；
- 无failed prompt和旧无效fallback进入eligible池；
- 所有外部调用可断点续跑并有预算保护；
- 每条处理结果可追溯到原始qid和输入hash。

### 20.2 P1验收

- Prompt、fallback、清晰度、尺寸至少各完成一个验证集对比；
- 方法胜出规则在看验证集结果之前冻结；
- 人工标注一致性有统计报告；
- 图像方法同时报告清晰度和身份保持，不只报告锐度；
- 所有实验结果可由结构化产物重新生成报告；
- 正交诊断ANOVA报告含主效应p值和effect size；
- 组合消融至少6个条件有完整结果；
- 每个问题域的方案选择有Pareto依据而非仅质量单指标。

### 20.3 正式集验收

- 五个维度各300条，总计1500条；
- qid唯一；
- 图像路径全部存在且可移植；
- 结构化目标、保持约束和prompt一致；
- 正式分布和配额偏差有完整报告；
- 最终ID列表、配置、代码版本和输入hash被冻结；
- 若任一维不足300条，不得标记任务完成，必须输出shortfall与修复建议。

## 21. Coding Agent禁止事项

1. 不得覆盖原始manifest、prompt或图像；
2. 不得为了配额把其他维度样本重标后直接补入；
3. 不得把模糊、空槽位等质量缺陷标成hard；
4. 不得从final prompt单独猜测结构化目标；
5. 不得对全部低频名词做无条件替换；
6. 不得默认启用GFPGAN或Outpainting进入主数据；
7. 不得只根据NIQE、Laplacian等单指标选图像方案；
8. 不得静默下载模型权重；
9. 不得默认调用付费API；
10. 不得实现或宣称Phase 3评测系统为本工程贡献；
11. 不得在未完成实验时把目标阈值写成已获得结果；
12. 不得在View不足时降低硬质量门槛凑到300条。

## 22. Coding Agent第一轮建议任务

第一轮只完成不依赖外部模型的闭环：

1. 创建 `quality/` package与Schema；
2. 实现配置和run manifest；
3. 迁移audit与split；
4. 实现prompt纯规则检查；
5. 实现路径解析和输入hash；
6. 生成开发/验证集与人工标注模板；
7. 为已知bug补回归测试；
8. 输出第一次environment/audit报告。

第一轮验收通过后，再进入VLM结构修复和图像模型实验，避免在基础数据接口仍不稳定时产生大量API与GPU成本。
