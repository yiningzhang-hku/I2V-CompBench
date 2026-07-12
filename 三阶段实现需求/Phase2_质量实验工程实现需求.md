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
    method: Literal["A0", "A1", "A2", "A3", "A4", "B0", "B1", "B2", "B3"]
    prompt_before: str
    prompt_after: str
    word_count: int
    rare_modifier_hits: list[str]
    structural_issues: list[str]
    semantic_consistency: float | None = None
    dimension_consistent: bool | None = None
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

### 10.2 A0—A4

- A0：原prompt；
- A1：仅修改polish指令；
- A2：常见同义词映射；
- A3：异常prompt定向LLM简化；
- A4：A2 + A3 + 结构化目标复核。

A2禁止：

- 修改主体noun；
- 修改动作核心动词；
- 修改方向词；
- 修改相机指令；
- 删除否定词。

A3/A4的LLM输入必须包含不可变字段：目标主体、目标关系、目标维度和preservation set。

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
```

### 10.5 胜出规则

先应用硬门槛：

- 目标语义保持率≥95%；
- 维度纯度≥95%；
- 无效prompt率为0。

满足硬门槛后，按低频修饰词率、自然度和API成本选择正式方法。

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
- C4：Real-ESRGAN + GFPGAN，仅可选消融。

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
- Grounding目标检测成功率。

缺少bbox时，主体区域指标记为null，不得使用整图指标假装主体指标。

### 11.4 视频探针边界

本模块只导出：

```text
clarity/video_probe_manifest.jsonl
```

包含30条×入选方法的首帧和prompt。视频可由现有共享adapter生成；质量包只导入人工清晰度、身份和伪影标注，不实现Phase 3评分。

### 11.5 胜出规则

身份改变率不得显著高于C0；DINO主体相似度不得超过配置容差下降。满足保真门槛后，选择人工清晰度和主体检测成功率最高的方法。

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

### 12.3 Saliency Crop要求

必须：

- 使用目标主体bbox或显著性区域；
- 裁剪框包含完整主体bbox及安全边距；
- 无bbox时不得假装主体感知，回退到padding并记录原因；
- 输出裁剪前后主体保留率。

### 12.4 Outpainting限制

Outpainting输出不得进入正式主流程，除非人工验证和语义保持实验明确通过。默认仅作对比上界。

### 12.5 正式混合策略

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

### 16.4 View供给风险

View候选只有365条。若硬门控后不足300条：

- 输出shortfall；
- 优先人工修复可修复样本；
- 不降低Schema、维度正确率或图像存在性门槛；
- 不从其他维度补齐；
- 不自动生成新题。

### 16.5 输出

```text
selection/final_1500_manifest.jsonl
selection/final_ids.txt
selection/selection_decisions.jsonl
selection/quota_actual_vs_target.json
selection/shortfall_report.json
selection/rejected_examples.jsonl
```

正式manifest路径必须为POSIX风格相对路径，不写Windows反斜杠和机器绝对路径。

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
2. A0—A4 Prompt对比；
3. B0—B3 fallback修复率；
4. C0—C4清晰度与身份保持；
5. 4:3/16:9和D0—D5尺寸策略对比；
6. 自然抽样与主体分层抽样对比；
7. G0—G3难度标定对比；
8. Baseline、+Text、+Image、+Sampling、Full组合消融；
9. 正式1500条的五维、难度、罕见度和主体分布；
10. 失败案例与限制。

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

工作：实现A0—A4、B0—B3、规则、指标和人工任务。

完成标准：117条fallback均有明确pass/reject结果；不存在无效prompt通过。

### Task 5：图像实验

工作：实现图像变体、指标、清晰度和尺寸实验。

完成标准：所有变体不覆盖原图；缺少模型时可恢复并清晰报告。

### Task 6：主体与难度

工作：实现Tier、多词主体、难度特征和人工标定。

完成标准：所有eligible候选都有Tier和新难度或明确失败原因。

### Task 7：正式筛选与报告

工作：硬门控、边际配额、1500条选择、数据卡和论文表格。

完成标准：满足§20验收；不足时输出shortfall而非伪造完成。

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
- 所有实验结果可由结构化产物重新生成报告。

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
