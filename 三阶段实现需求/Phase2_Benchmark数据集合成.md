# Phase 2：Benchmark 数据集合成

## 1. 定位与约束

Phase 2 接收 Phase 1 产出的 `phase1_bundle/`（`prior_package.json` / `compatibility_matrix.json` / `assets.jsonl` / `candidate_recipes.jsonl`），通过配额规划 → 配方采样 → 题目计划 → 输入构造 → QC 验收 → Prompt 定稿 → 数据集导出 → 审计的 8 步流水线，产出最终评测数据集 `benchmark_dataset/`，并交付 Phase 3 唯一入口 `phase3_manifest.jsonl`。

核心约束：

- **零拼词表**：所有题目必须从 Phase 1 `candidate_recipes.jsonl` 采样而来，禁止 Phase 2 自行启发式拼接组合。
- **七维齐全**：`attribute_binding / action_binding / motion_binding / spatial_composition / background_dynamics / view_transformation / interaction_reasoning`。
- **多图为主维度**：`multi_image` 不再是 stress-only，与 `single_image` 同等地位计入主维度评测样本。
- **Phase 3-ready**：每条样本必须落定 `target_subjects / target_relation / preservation_set / evaluator_tools / expected_failure_modes` 等结构化字段，Phase 3 不允许通过 NLU 反向解析自然语言 prompt 推断评测目标。
- **可审计性**：每条样本必须能反查至 Phase 1 的 `recipe_id` / `phase1_sample_ids` / `phase1_asset_ids`。

### 1.1 与 SOTA Benchmark 的差异化定位

| 维度 | T2V-CompBench (NeurIPS'24) | VBench-I2V | 本框架 (Phase 2 产出) |
|------|----------------------------|------------|---------------------|
| 题目来源 | VidProM 167 万 prompt + WordNet 分类 + GPT-4 生成 | 人工设计主题词 | TIP-I2V 真实分布采样 + Phase 1 recipe，零拼词表 |
| 输入形态 | T2V（仅文本） | I2V 单图 | I2V 单图 + **多图作为主维度**（非 stress） |
| 评测维度 | 7 类（含 Numeracy） | 16 类（混合质量与指令遵循） | 7 维纯指令遵循，显式不引入 Numeracy（详见 Phase 1 §5） |
| 对照设计 | static_copy 一项 | 无 | static_copy / random_motion / global_filter / camera_pan_cheat / **subject_swap** 共 5 项 |
| 评分公式 | 加权平均 | 维度指标拼接 | 执行门控乘法 `S = E·(0.6P + 0.4C)` |

设计取舍说明：本框架不复制 T2V-CompBench 的词表拼接路径（Phase 1 §10 决策记录），也不堆维度凑数量（VBench 的 16 类多与质量耦合，无法独立归因到指令遵循）。

---

## 2. 上游输入：Phase 1 产物

Phase 2 完全建立在 Phase 1 输出的 `phase1_bundle/` 之上，该目录由 `tools/pack_phase1_bundle.py` 从 Phase 1 原始输出树（Phase 1 §7）重组而成：

```
data/phase1_bundle/
├── prior_package.json            # Step 5 组装的全局先验（频率/视觉/共现/种子）
├── compatibility_matrix.json     # 7×7 维度兼容矩阵（priors_enhance 输出）
├── manifest_clean.jsonl          # construct_inputs 读取首帧物理路径
├── image_parse_v2.jsonl          # build_question_plan 三层槽位融合之一
├── text_parse_v2.jsonl           # build_question_plan 三层槽位融合之二
├── aligned_instances.jsonl       # build_question_plan 三层槽位融合之三
├── candidate_recipes.jsonl       # 候选配方，Phase 2 唯一题目来源
└── reference_bank/
    ├── assets.jsonl              # 多图素材的元数据（asset_id / quality / provenance）
    ├── subject/                  # 主体资产图像
    ├── attribute/                # 属性资产图像
    ├── object/                   # 物体资产图像
    └── scene_reference_original/ # 场景原图
```

各文件在 Phase 2 中的消费位置：

| Phase 1 产物 | 消费模块 | 消费方式 |
|--------------|---------|---------|
| `candidate_recipes.jsonl` | sample_recipes (§6.2) | 桶化索引 + 配额抽样；bucket key 中的 `subtype / difficulty / semantic_rarity / quality_flags` 字段由 Phase 1 §5.6.7 产出 |
| `assets.jsonl` | sample_recipes / construct_inputs (§6.2 / §6.4) | 多图体检（quality_score / clean_background）+ 实际取图路径 |
| `prior_package.json` | build_question_plan (§6.3) | 参与 §6.3 步骤 5b/5c：模板槽位 fallback（`dimension_priors[*].seed_examples`）与谓词/动词概念查询（`dimension_priors[*].concept_distributions`）、全局名/动/形词表（`global_distributions`）供 LLM polish 参考 |
| `compatibility_matrix.json` | sample_recipes (§6.2) | 校验 recipe 的 dimension_isolation 兼容性 |
| `image_parse_v2.jsonl` / `text_parse_v2.jsonl` / `aligned_instances.jsonl` | build_question_plan (§6.3) | 三层槽位融合（image → aligned → text）的语义来源 |
| `manifest_clean.jsonl` | construct_inputs (§6.4) | 单图首帧的真实路径 |

> **未消费产物说明**：Phase 1 `priors/frequency_tiers.json` `priors/subject_pair_distribution.json` `priors/multi_reference_priors.json` 在 Phase 1 §5.6.7 推导 `semantic_rarity / quality_flags` 时已被消费完毕，产物已序列化进入 candidate_recipes 顶层字段；Phase 2 **不重复读这三份原始频率文件**，避免 "同一词两阶段被赋予不同档位" 的漂移。

---

## 3. 工程结构

```
i2v_compbench/
├── configs/
│   ├── phase2.yaml
│   ├── dimensions.yaml
│   └── templates/
│       ├── attribute.yaml
│       ├── action.yaml
│       ├── motion.yaml
│       ├── spatial.yaml
│       ├── background.yaml
│       ├── view.yaml
│       └── interaction.yaml
├── data/
│   ├── phase1_bundle/         # Phase 1 输出，作为 Phase 2 输入
│   └── benchmark_dataset/     # Phase 2 产出（详见 §7）
└── src/i2vcompbench/
    ├── phase2/
    │   ├── build_quota.py            # P2.1 配额规划
    │   ├── sample_recipes.py         # P2.2 配方采样
    │   ├── build_question_plan.py    # P2.3 题目计划
    │   ├── construct_inputs.py       # P2.4 输入构造
    │   ├── verify_inputs.py          # P2.5 输入验收
    │   ├── finalize_prompts.py       # P2.6 Prompt 定稿
    │   ├── export_dataset.py         # P2.7 数据集导出
    │   └── audit_phase2.py           # P2.8 审计与数据卡
    ├── schemas/
    │   └── phase2.py                 # QuotaPlan / SampledRecipe / QuestionPlan / BenchmarkSample 等
    └── utils/
        ├── api_client.py             # Phase2SiliconFlowClient（VLM / LLM / T2I）
        ├── io.py                     # JSONL/JSON 读写、路径管理
        ├── ids.py                    # question_id / pair_id 生成
        ├── image.py                  # PIL resize / PNG 落盘
        └── templates.py              # 模板渲染、forbidden 词与字数检测
```

---

## 4. 模块调用关系

| 步骤 | 主脚本 | 依赖的内部模块 / 客户端 | 读取 | 写入 |
|------|--------|----------------------|------|------|
| build_quota | `build_quota.py` | `utils.io`、内部 `_round_split` | `configs/phase2.yaml` | `quota_plan.json` |
| sample_recipes | `sample_recipes.py` | `utils.io`、`utils.ids` | `quota_plan.json`、`candidate_recipes.jsonl`、`assets.jsonl` | `sampled_recipes.jsonl`、`quota_unfilled_report.json` |
| build_question_plan | `build_question_plan.py` | `utils.io`、`utils.ids`、`utils.templates`、`schemas.phase2.QuestionPlan` | `sampled_recipes.jsonl`、`image_parse_v2.jsonl`、`text_parse_v2.jsonl`、`aligned_instances.jsonl`、`configs/templates/{dim}.yaml` | `question_plans.jsonl` |
| construct_inputs | `construct_inputs.py` | `utils.io`、`utils.image`、`utils.api_client.Phase2SiliconFlowClient.call_t2i` | `question_plans.jsonl`、`manifest_clean.jsonl`、`assets.jsonl`、`reference_bank/*.jpg` | `first_frames/*.png`、`ref_images/*.png`、`input_assets_manifest.jsonl` |
| verify_inputs | `verify_inputs.py` | `utils.io`、`utils.api_client.Phase2SiliconFlowClient.call_vqa_structured` | `question_plans.jsonl`、`input_assets_manifest.jsonl`、`first_frames/`、`ref_images/`、模板 `qc_checks` | `qc_reports/{qid}.json`、`qc_failed_to_retry.jsonl`、`manual_review_queue.jsonl` |
| finalize_prompts | `finalize_prompts.py` | `utils.io`、`utils.api_client`、`utils.templates.count_words / find_forbidden_hits`、SpaCy POS | `question_plans.jsonl`、`qc_reports/`、`first_frames/`、`prompts/prompt_polish.txt` | `prompts/final_prompts.jsonl` |
| export_dataset | `export_dataset.py` | `utils.io`、`schemas.phase2.BenchmarkSample`、`_LEGACY_SOURCE_MAP`、`_aggregate_multi_quality` | `question_plans.jsonl`、`input_assets_manifest.jsonl`、`qc_reports/`、`final_prompts.jsonl` | `samples/{dim}.jsonl` × 7、`phase3_manifest.jsonl`、`contrastive_pairs.jsonl` |
| audit_phase2 | `audit_phase2.py` | `utils.io`、`schemas.phase2.DIMENSIONS_V2` | `quota_plan.json`、`sampled_recipes.jsonl`、`question_plans.jsonl`、`qc_reports/`、`samples/*.jsonl`、`phase3_manifest.jsonl` | `dataset_card.md` |

CLI 调度（参见 §10）：

```
build_quota → sample_recipes → build_question_plan → construct_inputs
            → verify_inputs → finalize_prompts → export_dataset → audit_phase2
```

每一步的输出都是文件落盘，下一步重新读取——保证任意步骤崩溃后可单独重跑而不必从头开始。

---

## 5. 全局枚举与边界规则

### 5.1 Dimension 枚举

```python
DIMENSIONS = [
    "attribute_binding",
    "action_binding",
    "motion_binding",
    "spatial_composition",
    "background_dynamics",
    "view_transformation",
    "interaction_reasoning",
]
```

### 5.2 InputMode 枚举

```python
INPUT_MODES = ["single_image", "multi_image"]
```

### 5.3 Motion / Spatial 边界（Phase 2 模板与 prompt 定稿强制执行）

- 含主体位移 → `motion_binding`。
- 静态布局组合 → `spatial_composition`。
- Spatial 多图 recipe 必须为 static layout，**不得包含** `move / shift / reposition` 等位移动词。

### 5.4 Action / Interaction 边界（Phase 2 prompt 定稿强制执行）

```
single-subject body movement     → action_binding
multi-object causal/social/functional event → interaction_reasoning
```

### 5.5 工具与失败模式强枚举（详见附录 A）

- `evaluator_tools` 取值必须 ⊂ `{grounding, depth, dot_motion, optical_flow, vlm_existence, vlm_attribute, vlm_relation, dinov2, clip}` 共 9 项。
- `expected_failure_modes` 取值必须 ⊂ Phase 3 §2.5 FAILURE_MODES 共 15 项。
- `source_type` 取值必须 ⊂ `{tip_i2v_real, tip_i2v_synthetic_first_frame, external_real, external_synthetic}` 共 4 项；Phase 1 旧词汇必须经 `_LEGACY_SOURCE_MAP`（附录 C）转换。

---

## 6. 各模块技术原理

### 6.1 build_quota — 抽象配额拆解为可执行桶

**输入**：`configs/phase2.yaml`，包含 `mode`（pilot / full）、`num_per_dimension`（pilot=20、full=200）、`input_mode_ratio` / `subtype_ratio` / `difficulty_ratio` / `rarity_ratio`。

**处理**：

1. 对每个维度依次按四级分裂：`mode → subtype → difficulty → rarity`。
2. 每级用最大余数法 `_round_split` 把整数配额按比例分到子桶，保证总和不漂、误差落在余数最大的几个桶上。
3. 七维度共用一份 `difficulty_ratio` / `rarity_ratio`，但 `input_mode_ratio` 与 `subtype_ratio` 按维度独立配置。
4. 每个叶节点拼出 `bucket_id = {dim}__{mode}__{difficulty}__{rarity}`；motion 维度仅有 subtype 概念，bucket_id 退化为 `{dim}__{subtype}__{difficulty}__{rarity}`。

**输出**：

- `benchmark_dataset/quota_plan.json`：`QuotaPlan { mode, total_target, buckets: [QuotaBucket] }`。
- 每个 `QuotaBucket` 含 `bucket_id / dimension / input_mode / subtype / difficulty / rarity / target_count / contrastive_pair_required`。

**关键设计与底层原理**：

- **为什么用最大余数法而不是四舍五入**：整数配额按比例切分时，简单舍入会让总和漂移（例如 0.33/0.33/0.34 切 20 时舍入会出现 7+7+7=21 越界）。最大余数法在每一级先取整再把误差按"余数排序"分配，保证 `target_count` 求和精确等于 `num_per_dimension × 7`，下游 audit 才能用减法直接得到缺口。
- **为什么把 bucket_id 提前固化到 quota_plan**：评测的真实分组键就是这五元组，预先生成稳定主键让 sample / export / audit 能用 join 串起来，下游模块只需理解"桶"概念，不必再读 yaml。
- **为什么七维共用 difficulty/rarity 但 input_mode_ratio 维度独立**：难度与稀有度是评测体系横向可比的统一概念；输入模式分布与维度真实先验强耦合（view 95% 单图源自"运镜数据天然只有一张起始帧"，spatial 80% 多图源自"空间组合天然需要多个参照"），共用比例反而失真。
- **为什么 motion 退化为 subtype 分桶**：motion 在 TIP-I2V 真实分布的分类轴是 `type_a_absolute_single / type_b_relative_single / type_c_multi_motion`，并非 single/multi 二元。强行套 input_mode 会让"绝对位移单图"与"相对位移单图"挤进同一桶，违反 §5.3 边界规则。

### 6.2 sample_recipes — 在配额下从 Phase 1 真实候选抽题

**输入**：`quota_plan.json`、Phase 1 的 `candidate_recipes.jsonl`（唯一题源）、`assets.jsonl`。

**处理**：

1. **建索引**：candidate_recipes 按 `(dimension, input_mode/subtype, difficulty, rarity)` 桶化；assets 按 `asset_id` 建查找表。
2. **基础过滤**：剔除带阻断标志的 recipe，阻断集合 `_BLOCKING_QUALITY_FLAGS = {"low_alignment", "missing_inpainted_scene", "subject_not_visible", "evaluator_infeasible"}`。
3. **多图体检**：对 `input_mode=multi_image` 或 motion 的 `_MULTI_IMAGE_SUBTYPES = {"type_c_multi_motion"}`，逐条校验每个 `reference_asset_id`：
   - 资产存在且 `quality_score ≥ 0.4`；
   - 该资产 `is_clean_background=True`；
   - 至少 2 个有效 reference，否则该 recipe 进 `quality_below_threshold` 缺口。
4. **配额抽样**：每桶按 `target_count` 抽样；contrastive 桶按 `source_sample_id` 分组成对消费 A/B（落单时降级为 `A_only`，并在缺口报告记 `contrastive_unpaired`）。
5. **缺口诊断**：四类原因写入 `quota_unfilled_report.json`：
   - `no_candidate`：桶里根本没有匹配 recipe；
   - `no_reference_asset`：多图 recipe 但没绑定足量资产；
   - `quality_below_threshold`：资产 quality_score 或 clean_background 不达标；
   - `blocking_flag`：候选全被 quality_flags 阻断。

**输出**：

- `benchmark_dataset/sampled_recipes.jsonl`：`SampledRecipe { recipe_id, bucket_id, contrastive_pair_id, contrastive_role, reference_asset_ids[] }`。
- `benchmark_dataset/quota_unfilled_report.json`：每桶 `{ bucket_id, target, sampled, gap, reasons[] }`。

**关键设计与底层原理**：

- **为什么绝不允许 Phase 2 自己拼词表**：benchmark 可信度建立在"题目源自真实 I2V 分布"——一旦 Phase 2 用启发式拼新组合，reviewer 第一个质疑就是"如何证明这条题目对应真实用户场景"。让 candidate_recipes 成为唯一题源，每条样本反向追到一条 TIP 视频，省去整本论文里反复辩护"分布合理性"的成本。
- **为什么阈值定在 quality_score ≥ 0.4 且 is_clean_background**：低质量素材会在 Phase 3 变成捷径——背景里残留杂物会让"看背景反推空间关系"成为模型偷懒解法。把这层闸门放在抽样阶段（而非 QC 阶段）是用最便宜的方式先剔除大批不可用 recipe，节省后续 T2I 与 VQA 调用。
- **为什么多图至少 2 张达标**：multi_image 的本质是"组合多个独立线索"，1 张 reference 退化成单图任务，违背维度设计。这里硬卡比下游再降级更早、更便宜。
- **为什么 A/B 必须原子化配对**：contrastive consistency 是 anti-shortcut 的关键证据，需要"同一题型正反方向各跑一次"。只采到 A 没采到 B 时，这条数据在 contrastive 表直接作废，比起多采几道无关题，"配对完整性"对评测意义更大。
- **为什么缺口要分四类原因而不是只给 gap 数字**：reviewer 关心的不是"差几条"，而是"差在哪"——四类原因分别提示数据扩充方向、补 Phase 1 资产、调阈值、或修 Phase 1 patch。粒度化分类让缺口直接转成 actionable 工单。

### 6.3 build_question_plan — recipe 翻译为可执行题目计划

**输入**：`sampled_recipes.jsonl`、Phase 1 的 `image_parse_v2.jsonl` / `text_parse_v2.jsonl` / `aligned_instances.jsonl`、`configs/templates/{dimension}.yaml` 模板库。

**处理**：

1. **三层槽位融合**：按 `image_parse → aligned_instances → text_parse` 顺序叠加，后者覆盖前者，得到维度专属 `slot_dict`（subjects / attributes / relations / camera_baseline 等）。
2. **subtype 解析**：优先用 recipe 显式声明的 subtype；缺失时按 `input_mode` 在模板库 `subtypes` 列表里匹配（如 `multi_image → attribute_transfer`），再缺则退到该维度第一个可用 subtype。
3. **question_id 生成**：`{dim_short}_{mode_short}_{seq:04d}`，其中 `DIM_SHORT` 把长维度名缩成 attr / action / motion / spatial / bg / view / inter，`mode_short` 是 single / multi。
4. **模板渲染**：用 `prompt_pattern`（Jinja-like）渲染 `prompt_draft`；槽位缺失导致渲染失败时回退到 recipe 自带的 `base_prompt_draft`，并在 `risk_flags` 记 `template_render_failed`。
5. **结构化字段填充**：从模板拷贝 `target_plan / preserve_plan / dimension_isolation / evaluator_tools / expected_failure_modes`，把槽位占位符替换成实际值（`ref:target_subject` 等指代保持原样，等 §6.4 落实到 asset_id）。
   - **`preserve_plan` 结构转换**：Phase 1 CandidateRecipe 中的 `preserve_constraints: [{target, aspect, note}]` 需转为 Phase 2 / Phase 3 统一的 `[{scope, constraint}]` 字典列表：
     ```
     {"target": "s2", "aspect": "appearance", "note": ...}
         → {"scope": "s2", "constraint": "appearance"}
     {"target": "background", "aspect": "global", "note": ...}
         → {"scope": "background", "constraint": "global"}
     ```
     `note` 字段仅限 `_audit` 保留，不进入顶层 `preservation_set`。
   - **`dimension_isolation` 语义转换（维度名 → 禁词）**：Phase 1 `dimension_isolation: {primary_dimension, forbidden_dimensions: [str], leakage_risk_notes}` 表达“什么维度不能被污染”，Phase 2 输出 `dimension_isolation: {forbidden_words: [str], camera_constraint: str}` 表达“prompt 不允许出现哪些词 / 镜头运动可限制到什么程度”。转换表 `_FORBIDDEN_WORDS_BY_DIM`：
     ```python
     _FORBIDDEN_WORDS_BY_DIM = {
         "motion_binding":        ["pan", "zoom", "tilt", "dolly", "orbit"],          # 禁镜头伪造运动
         "spatial_composition":   ["move", "shift", "reposition", "slide"],            # spatial 是静态组装
         "background_dynamics":   ["the subject moves", "the actor walks"],            # 防主体变化偏转
         "view_transformation":   [],                                                  # 该维度允许镜头词
         "attribute_binding":     ["move", "walk", "run", "pan", "zoom"],              # 只考属性变化
         "action_binding":        ["pan", "zoom", "tilt"],                              # 防镜头偏转
         "interaction_reasoning": ["pan", "zoom", "tilt"],
     }
     _CAMERA_CONSTRAINT_BY_DIM = {
         "view_transformation": "required_active",       # 镜头动作为主任务
         "motion_binding":      "forbidden",              # 禁镜头运动
         "_default":            "static_or_minor",        # 其余维度允许轻微镜头
     }
     ```
     合成逻辑：`forbidden_words = _FORBIDDEN_WORDS_BY_DIM[primary_dimension]`；`camera_constraint = _CAMERA_CONSTRAINT_BY_DIM.get(primary_dimension, _CAMERA_CONSTRAINT_BY_DIM["_default"])`。处于中间表位置的 Phase 1 `forbidden_dimensions: [str]` 仅作 `_audit.dimension_isolation_phase1` 保留供查补，不入 `phase3_manifest`。
   - **模板槽位 fallback 使用 prior_package**：某个槽位（如 `attribute_value`）从 `image_parse_v2 / text_parse_v2 / aligned_instances` 三层都拿不到值时，按顺序尝试 fallback： (a) `prior_package.dimension_priors[primary].seed_examples` 随机抽一条同主体例子提取 ；(b) `prior_package.dimension_priors[primary].concept_distributions` 取 head 档高频词 ；(c) 还不行才走 §6.3 步骤 4 的 `template_render_failed` fallback。
6. **target_subjects 稳定 id 与 ref_image_idx 分配**（Phase 3 衔接硬约束）：`target_plan.target_subjects[]` 必须按 plan 内出现顺序补 `id = "s{i+1}"`（s1, s2, …）；当 `input_mode == multi_image` 时，每个 target_subject 必须额外带 `ref_image_idx`，与 §6.4 写入 `ref_images/{question_id}_ref{k}.png` 的下标一一对齐（同一角色多张参考时取该角色的首张 ref_image_idx）。Phase 3 evaluator 直接通过 `target_subjects[i].ref_image_idx` 索引参考图。
7. **evaluator_tools 强枚举映射**：依据 `target_dimension` 查 `_TOOLS_BY_DIM` 表（附录 A.1），结果必须 ⊂ 9 项工具枚举；任何模板自由文本会被强制覆写。`multi_image` 样本额外追加 `dinov2, clip, grounding`（用于 Phase 3 identity_binding 软因子）。
8. **expected_failure_modes 默认填充**：依据 `target_dimension` 查 `_FAILURES_BY_DIM` 表（附录 A.2）取默认值；若 LLM 在 §6.6 polish 阶段进一步标注更具体的失败模式可追加，但取值必须 ⊂ Phase 3 §2.5 的 15 项 FAILURE_MODES 枚举（`static_copy / global_filter / camera_pan_cheat / object_missing / wrong_attribute / wrong_direction / wrong_relation / wrong_camera / identity_lost / non_target_drift / background_drift / artifact_severe / timing_wrong / identity_unbound / tool_uncertain`）。

**输出**：

- `benchmark_dataset/question_plans.jsonl`：每条 `QuestionPlan` 包含 question_id、recipe_id、维度五元组、`input_plan` / `target_plan`（含 `target_subjects[]` 稳定 id + 多图模式 ref_image_idx）/ `preserve_plan` / `dimension_isolation` / `evaluator_tools`（强枚举）/ `expected_failure_modes`（15 种枚举之子集）/ `prompt_draft`。

**关键设计与底层原理**：

- **为什么三层叠加且后者覆盖前者**：三个数据源各自承载不同语义——`image_parse` 给"图里实际有什么"（视觉真值），`aligned_instances` 给"文本指代到底绑到了哪个实例"（图文对齐），`text_parse` 给"用户到底想要什么变化"（意图）。后者覆盖前者体现"意图 > 对齐 > 现状"，最终 prompt 是为意图服务的。
- **为什么 subtype 走三级回退而不是直接抛错**：模板库不可能枚举所有真实分布的 subtype 组合；让 input_mode 一致即可，再不行退到该维度第一个可用 subtype。"宽松匹配"比"严格匹配整桶失败"更利于 pilot 跑通。
- **为什么 question_id 用 DIM_SHORT + mode_short 而非哈希**：人工排查时第一眼能从 ID 看出维度与模式（`attr_multi_0001` 一眼就知是属性绑定多图题），比纯哈希友好；短前缀避免文件名超长，也方便按前缀做 grep / 分桶统计。
- **为什么 prompt_pattern 渲染失败要回退到 base_prompt_draft 而非丢弃**：渲染失败的真因往往是某个槽位缺值。recipe 自带的 `base_prompt_draft` 已经是 Phase 1 用真实 caption 拼出的可读句子，留作兜底比让整道题失败更划算——QC 仍能跑、polish 仍能改写，只是 risk_flags 多一条提示人工抽查。
- **为什么 evaluator_tools 必须在 Phase 2 落定且强枚举**：如果 Phase 3 才反向解析自然语言 prompt 推工具集，会引入 NLU 误差且不可复现——同一道题在不同次评测里调用的工具栈可能不同，分数无法横向比。把工具集合编码进 `evaluator_tools`（受 `_TOOLS_BY_DIM` 强约束）是把"评测目标"从自然语言搬到结构化字段，让 Phase 3 变成纯执行器。**注意原稿中的 `evaluator_plan` 自由文本字段已废弃**——Phase 3 只读 `evaluator_tools` 强枚举列表。
- **为什么 expected_failure_modes 必须在 Phase 2 写定**：Phase 3 baseline 验收门槛要求"5 种 baseline 配对差异 ≥ 80%"，意味着每条原题必须**预先告诉评测器**它期望暴露哪些失败模式，否则 baseline 命中后无法对照。`_FAILURES_BY_DIM` 默认表（附录 A.2）保证每条样本至少有 2 个 expected_failure_modes，避免空数组导致 baseline 配对失效。
- **为什么 target_subjects 必须有稳定 id**：Phase 3 的 `target_relation.subj/obj` 需要引用主体（如 `subj="s1", obj="s2"`），依赖稳定 id 而非 noun（noun 可能重复，例如两只 cat）。在 Phase 2 落定 `s1/s2/...` 顺序，Phase 3 才能正确做 grounding 与配对。
- **为什么 multi_image 必须强制 ref_image_idx**：Phase 3 的 identity_binding 软因子按主体逐个比对 ref_image vs 视频帧；若 `ref_image_idx` 缺失或与文件名顺序不一致，会把 cat 的参考图误比给 chair 的视频帧，DINOv2 cosine 必然崩盘。这是 multi_image 最容易翻车且最难调试的地方，Phase 2 必须保证文件名下标 = ref_image_idx。
- **为什么 prompt_draft 不能直接当 final**：draft 是结构化模板渲染的产物，并未"看图说话"过。如果首帧实际状态与模板槽位有出入（例如模板写"a woman"但首帧实际是"两个人，左边是女士"），prompt 与图就会错位。最终 prompt 必须经过 §6.6 的 VLM 描述对齐才能保证 prompt 与起始状态自洽。

### 6.4 construct_inputs — 题目计划落实为真正的输入图像

**输入**：`question_plans.jsonl`、Phase 1 的 `manifest_clean.jsonl`（首帧路径）、`assets.jsonl`（参考资产）、`reference_bank/{asset_type}/*.jpg`。

**处理**：

1. **角色 → 资产类型映射**：用 `_ROLE_TO_ASSET_TYPE` 把 `target_subject → subject`、`attribute_reference → attribute`、`scene_reference → scene_reference_inpainted | scene_reference_original` 等。
2. **首帧（single_image）**：直接复用 Phase 1 manifest 中 source_sample 对应的 `image_path`，不重新生成。
3. **多图参考（multi_image）**：按 recipe 的 `reference_asset_ids` 取资产；同一角色多个候选时挑 `quality_score` 最高的那个。
4. **`source_preference` 链式回退**：模板优先序 `tip_derived_reference → t2i_generated → external`：
   - 优先从 reference_bank 拿真实 crop；
   - 缺位时调 `Phase2SiliconFlowClient.call_t2i`（Kwai-Kolors/Kolors，1024×1024）按角色描述合成；
   - T2I 失败则在 `quality.notes` 写 `t2i_generated` 或 `t2i_failed`，由 QC 决定是否丢弃。
5. **统一规格化（双轨产物）**：每个资产同时产出两份 PNG：
   - **主产物（`<name>.png`）**：PIL `resize_long_edge(..., enlarge=True)` 强制等比放大到 `long_edge=1280`（720P 长边），**保留原始宽高比**，不裁剪不 letterbox。用途：Phase 3 evaluator 读取作为 P 维度逐帧对齐 / identity_binding 的真值参照。
   - **16:9 推理伴生件（`<name>_16x9.png`）**：PIL `to_16x9_720p()` 产 **严格 1280×720**，供需要 16:9 输入的 I2V 生成模型（SVD / CogVideoX / HunyuanVideo 等）调用。适配策略按原始比例自动选取：16:9 ±4% 直接 resize；更扁横屏 center crop 左右；偏方/竖屏 letterbox 黑边（保留全部内容不丢主体）。
   - 16:9 伴生件在 `_audit` 中记录 `inference_strategy ∈ {"resize", "crop", "letterbox"}`，供 dataset_card 统计。`resize_long_edge(..., enlarge=True)` 强制放大低分辨率上游图（TIP-I2V 抽帧常为 224×126 / 224×224 / 126×224）。
6. **路径约定**（Phase 3 衔接硬约束）：单图首帧 `first_frames/{question_id}.png`；多图参考 `ref_images/{question_id}_ref{k}.png`，其中 `k` **等于** `target_subjects[i].ref_image_idx`（在 §6.3 已绑定）。**禁止使用旧名 `images/` `references/`**。
   - 同一角色多张参考时的 `{question_id}_ref{k}_v{j}.png` 后缀**当前为设计预留**，pilot 实现仅生成 `_ref{k}.png` 主参考；如未来需要多版本兼容，Phase 3 默认只读 v0（即不带 `_v{j}` 后缀的文件）。
7. **资产质量元数据**：每张图记录 `identity_visibility / crop_leakage_risk / resolution_ok`，这些字段在 QC 与最终多图聚合中被反复消费。

**输出**：

- `benchmark_dataset/first_frames/{question_id}.png`、`benchmark_dataset/ref_images/{question_id}_ref{k}.png`：实际图像文件。
- `benchmark_dataset/input_assets_manifest.jsonl`：每行一个 question_id 对应的 `assets[]`，含 `asset_id / role / path / source_type / source_ref_id / quality`。

**关键设计与底层原理**：

- **为什么 `_ROLE_TO_ASSET_TYPE` 做间接映射**：题目语义层用 role（`target_subject` / `attribute_reference`），资产存储层用 asset_type（`subject` / `attribute` / `scene_reference_inpainted`）。两层 schema 各自演化，中间留映射表能让任一层调整时不污染另一层；硬编码合并会导致一处改动牵连两套 schema。
- **为什么单图直接复用 TIP 首帧而不是合成**：这是 benchmark 真实性的根基——单图任务的"起始状态"应当是真实视频首帧，否则评测的就不是"模型对真实首帧的处理能力"，与论文论点错位。Phase 1 已经付出代价做过 manifest 清洗，没必要在这里推倒重来。
- **为什么同角色多候选挑 quality_score 最高**：quality_score 是 Phase 1 综合 bbox 面积、可见度、完整度算出的代理指标，已融合多个质量信号；用它做启发式比"随机选一张"或"按时间序选最早"都更稳定，且无额外计算成本。
- **为什么链式回退顺序是 tip → t2i → external**：三级递降的不是只看"清洁度"，更重要的是"reviewer 可审计度"——TIP 资产可追溯到原视频帧（最强）、T2I 可追溯到 prompt+seed（中等）、external 需要单独 provenance 字段（最弱）。把审计成本作为优先级权重，而不是单纯的视觉质量。
- **为什么 long_edge=1280 + PNG**：1280 是 720P 标准长边（1280×720 progressive），与 SVD / CogVideoX / Sora-class 等主流 I2V 模型常用的 720P 推理档位对齐；再大会让 Phase 3 推理开销陡增；PNG 无损保存避免 JPEG 在边缘产生 ringing 伪影——这些伪影会干扰 grounding 评测器的边缘检测，让 P 评分受 codec 噪声影响。
- **为什么允许放大（enlarge=True）**：TIP-I2V 上游视频抽帧原始分辨率多为 224×126 等低码率规格，必须等比放大到 720P 长边才能与下游 evaluator（grounding / depth / optical_flow）的常用输入尺寸对齐。承担 5.7× 双线性插值的放大噪声，换取尺寸统一与下游一致性。
- **为什么双轨产物（`<name>.png` + `<name>_16x9.png`）**：I2V 生成模型多有硬编码的 16:9 输入约束（SVD 1024×576、CogVideoX 多分辨率、HunyuanVideo 1280×720 等），而 benchmark 原生首帧频繁出现正方形 / 竖屏 等非 16:9 比例。双轨设计把两个冲突需求解耦：evaluator 读原始等比图作 P 维度真值（零形变零裁剪零黑边）；生成器读 16:9 伴生件。这样生成模型拿到的是合法输入，评测期却不损失原始画面信息。选 letterbox 而非 crop 处理竖屏是因为：crop 会丢 56% 内容含主体，letterbox 仅带黑边且 SVD / CogVideoX 训练数据中本就存在大量 letterbox 样本，模型有处理黑边的先验。
- **为什么 T2I 失败不抛异常**：流水线 8 步，任意一步在单题失败如果中断会让上游全部白跑。让 QC 统一裁决，整体吞吐更稳。
- **为什么把 quality 元数据存进 manifest 而不是临时计算**：QC（§6.5）和 export（§6.7）都要消费这些字段，存一次比每步重算更稳定，也避免不同步骤算法略有差异导致同一字段两次算出不同结果。

### 6.5 verify_inputs — 用结构化 VQA 给输入图把关

**输入**：`question_plans.jsonl`、`input_assets_manifest.jsonl`、对应 PNG 图像、`configs/templates/{dimension}.yaml` 中每个 subtype 声明的 `qc_checks[]`（每条 check 含 `name / question / hard_check`）。

**处理**：

1. **逐题逐项跑 VQA**：对每张输入图按模板 `qc_checks` 顺序调 `Phase2SiliconFlowClient.call_vqa_structured`，模型返回 `{answer: bool, confidence: float, rationale: str}`。
2. **HARD_CHECK 名单**：`_HARD_CHECK_NAMES` 含 11 项强制检查：
   - `has_target_subject_visible / single_target_subject / all_required_subjects_visible`
   - `target_subject_static / no_action_already_in_progress / no_motion_blur_present`
   - `interaction_not_yet_started / no_target_attribute_already`
   - `scene_has_subject_room / resolution_ok` 等。
3. **三态聚合 `_aggregate_status`**：
   - 任何 hard_check 返回 `answer=False` 且 `confidence ≥ 0.7` → `fail`；
   - 任何 check `confidence < 0.7` → `needs_manual_review`；
   - 全部通过 → `pass`。
4. **双队列分流**：
   - `fail` 写入 `qc_failed_to_retry.jsonl`，由配置控制是否触发 §6.4 重采资产；
   - `needs_manual_review` 写入 `manual_review_queue.jsonl`，等待人工裁决，**默认不入 Phase 3**。

**输出**：

- `benchmark_dataset/qc_reports/{question_id}.json`：`QCReport { question_id, qc_status, checks[], risk_flags }`。
- `benchmark_dataset/qc_failed_to_retry.jsonl`、`benchmark_dataset/manual_review_queue.jsonl`。

**关键设计与底层原理**：

- **为什么用 VLM 而不是规则匹配**：题目合规性是语义级判断（"主体是否还静止"、"目标属性是否还没出现"），传统 CV 规则无法覆盖；VLM 是当前能给出语义级判定的最低成本工具。
- **为什么 confidence ≥ 0.7 才算 fail**：VLM 在边界样本上有抖动，单次低置信度的"否"很可能是模型不自信而非真否决。卡在 0.7 把"明确否决"（≥0.7 的 False）与"模糊判断"（<0.7）分开——前者直接进 retry，后者进人工，两者用不同处理成本对应不同确定度。
- **为什么 < 0.7 直接进人工而非再 retry**：retry 的本质是换一张图重新构造，但 VLM 不确定的题目即使换图也大概率再次抖动；让人裁决一次的边际成本，比反复跑 VLM API 既便宜又终局。
- **为什么 HARD_CHECK 集中在常量而非分散在模板**："哪些 check 一票否决"是评测设计的一等决策，模板作者不应能随意提升某个 check 的权重——否则不同维度的"严格度"会随作者风格漂移。集中常量也避免拼写错误造成静默退化。
- **为什么有 `no_target_attribute_already` / `no_action_already_in_progress` 这类反向检查**：这些是防"题目自我作弊"——若首帧已包含目标终态，模型完全静止也能在 E 评分里拿满分。这种"起始即终态"的样本必须从源头剔除，否则 Static Copy baseline 会与正常模型并列高分，anti-shortcut 实验直接破产。
- **为什么 needs_manual_review 默认不进 Phase 3**：multi-image 维度本来就稀缺，但与其放进去拉低评测可信度，不如等人裁决。论文 reviewer 关心的是"过滤了多少（怎么保证质量）"而非"保留了多少（怎么凑数）"，所以默认走严格路径。

### 6.6 finalize_prompts — 看着真实首帧把 prompt 写定

**输入**：`question_plans.jsonl`、`qc_reports/`、`input_assets_manifest.jsonl`、首帧 PNG、模板 `prompts/prompt_polish.txt`、`Phase2SiliconFlowClient`（VLM 描述 + LLM 改写）。

**处理**（仅处理 `qc_status=pass` 的题目）：

1. **VLM 描述首帧**：调用 VLM 把首帧描述成中性 caption，作为定稿时的视觉锚点。
2. **LLM polish**：把 caption + question_plan 的 `target_plan / preserve_plan / dimension_isolation` 拼进 `prompt_polish.txt`，要求模型返回 JSON：`{prompt, reasoning}`（**字段名 `prompt`，不是 `i2v_prompt`**，与 Phase 3 BenchmarkSample 顶层字段名一致）。
3. **解析与校验**：先剥 fenced ``` 代码块再做括号扫描提取 JSON；然后对 `prompt` 做：
   - **禁词检查**：命中 `dimension_isolation.forbidden_words` 任一词即失败；
   - **字数检查**：常规维度 8–25 词，`interaction_reasoning` 放宽到 30 词；
   - **active verb 硬约束**（除 `view_transformation` 外的六个维度）：prompt 必须至少包含一个 active verb（SpaCy POS 标为 `VERB` 且 lemma 不在隐含静态集 `{be, have, exist, remain, stay, look, seem, appear}` 中）。`view_transformation` 题仅要求包含 camera 动作词（`zoom` / `pan` / `tilt` / `dolly` / `orbit` 任一）。未命中记 `failed_check="missing_active_verb"`；
   - **指代检查**：多图必须能解析出 `image 1 / image 2` 或 `the reference X` 这类显式角色指代。
4. **重试与回退**：最多尝试 N 次（`polish_attempts` 字段记录次数），仍不合格则回退到 `prompt_draft`，并在 `risk_flags` 加 `prompt_polish_fallback`。
5. **前缀清理**（入库前统一执行，无论是否 fallback）：`prompt = re.sub(r'^\[dim=[^\]]+\]\s*', '', prompt)`。Phase 1 `base_prompt_draft` 携带的 `[dim={primary_dimension}]` 调试前缀（Phase 1 §5.6.6）不能进入最终 BenchmarkSample.prompt；fallback path 下尤其需要该清理。

**输出**：

- `benchmark_dataset/prompts/final_prompts.jsonl`：`FinalPromptEntry { question_id, prompt, polish_attempts, used_fallback, vlm_caption }`（**字段名 `prompt`，不是 `i2v_prompt`**）。

**关键设计与底层原理**：

- **为什么先 VLM caption 再 LLM polish 两步分工**：VLM 负责"看图说话"提供视觉锚点，LLM 负责"按目标改写"做语言润色。合并到一步让 VLM 直接出 prompt 会让结构化约束（forbidden_words、字数、reference 指代）很难管控——VLM 优于看图但弱于遵循复杂格式约束，LLM 反之，分工各取所长。
- **为什么强制 LLM 返回 JSON 而非裸文本**：裸文本下 LLM 经常自作主张加引号、解释、emoji 或 markdown 标题；JSON 强约束让解析端能严格校验；同时保留 `reasoning` 字段以便事后审查。
- **为什么解析需要 fenced ``` + 括号扫描双层兜底**：LLM 即使被强制要求输出纯 JSON，也常包在 markdown 代码块里、或在 JSON 前后加注释。双层解析（先剥 fenced，再扫描第一对配平的大括号）让 polish 不会因为格式细节白费一次 API 调用。
- **为什么字数定 8–25 / interaction 30**：TIP-I2V 真实 prompt 中位数在 12 词左右，8–25 覆盖约 80% 真实分布；interaction 题需要描述"主体 A + 主体 B + 因果链"，强行压到 25 词会丢信息，所以专门放宽到 30 词。
- **为什么强制 active verb**：T2I-CompBench 与 T2V-CompBench 都明确指出，不含动作动词的 prompt（"a red ball on the table"）让 I2V 模型退化为静态复制却仍能拿高分，评测出现 false positive。把"至少包含一个动作动词"落到 finalize_prompts 的硬检查，是把评测设计意图（"考查指令是否被执行"）从文档约定变成可执行约束。`view_transformation` 作为例外是因为它考的就是 camera 运动，动作动词会跟主维度冲突。
- **为什么禁词命中即重写而非软警告**：dimension_isolation 是评测可信度的最后防线——motion 题里出现 "pan" 会让模型用相机平移伪装主体位移，spatial 题里出现 "move" 会让 spatial 与 motion 失去边界，§5.3 / §5.4 的硬规则在这里落地。让禁词命中变成强制重写，是把维度边界从设计文档落到可执行约束。
- **为什么 fallback 仍允许 export**：流水线已把 QC 通过的样本送到这一步，因为 polish 失败丢弃整题会让上游 4 步白费。标记 `prompt_polish_fallback` 让审计层（§6.8）能统计 fallback 比例并触发后续修复——比起静默丢数据，留痕的质量警示更利于持续改进。

### 6.7 export_dataset — 四源 join 出最终数据集

**输入**：`question_plans.jsonl`、`input_assets_manifest.jsonl`、`qc_reports/*.json`、`final_prompts.jsonl`。

**处理**：

1. **四源 join**：以 `question_id` 为主键，把 question_plan、input_assets、qc_report、final_prompt 内连接。
2. **门控筛选**：仅保留 `qc_status=pass` 且有非空 `prompt`（原名 `i2v_prompt` 已废弃）且 manifest 非空的样本。
3. **多图质量聚合 `_aggregate_multi_quality`**（worst-case 策略）：
   - `crop_leakage_risk` 取所有 reference 中的最大值（最差）；
   - `identity_visibility` 取最小值（最弱）；
   - `scale_compatibility` 由 bbox 面积比换算后取最小值。
4. **拼装 BenchmarkSample（顶层字段全部扁平，禁止 `metadata` 嵌套）**：
   - **17 个顶层字段**：`question_id / dimension / input_mode / first_frame_path / input_image_paths / prompt / target_subjects / target_relation / preservation_set / contrastive_pair_id / contrastive_role / evaluator_tools / expected_failure_modes / subtype / difficulty / semantic_rarity / source_type` —— 严格对齐 Phase 3 §2.4 schema；
   - `first_frame_path` 指向 `first_frames/{question_id}.png`（**不是 `images/`**）；
   - `input_image_paths` 为 multi_image 时 ≥2、single_image 时为空数组，顺序与 `target_subjects[i].ref_image_idx` 一致；
   - `target_subjects` 透传 §6.3 已分配的 `id`（s1/s2/...）+ `ref_image_idx`；
   - `contrastive_pair_id` / `contrastive_role` 从 SampledRecipe 透传到顶层（**原稿中 metadata 内嵌套位置已废弃**）；
   - `evaluator_tools` 强枚举透传 question_plan；`expected_failure_modes` 透传 question_plan（必 ⊂ 15 种 FAILURE_MODES）；
   - `source_type` 走 `_LEGACY_SOURCE_MAP` 转换（附录 C）：Phase 1 的 `observed_single_image / derived_single_image / derived_multi_reference / external_*` → Phase 3 枚举 `tip_i2v_real / tip_i2v_synthetic_first_frame / external_real / external_synthetic`；
   - **调试辅助字段**（不进顶层 schema，另存于 `_audit` 子节点）：`source_trace`（`recipe_id / phase1_sample_ids / phase1_asset_ids`）、`qc.status / qc.risk_flags`、`multi_reference_quality`——仅供 Phase 2 audit 与人工查验，Phase 3 evaluator **禁止**读取 `_audit.*` 字段。
5. **contrastive_pairs.jsonl 产出**：按 `contrastive_pair_id` 聚合为 `{pair_id, dimension, original_qids:[...], baseline_qids:[{qid, role}]}` 写入 `benchmark_dataset/contrastive_pairs.jsonl`，Phase 3 `aggregate.py` 直接读取该索引计算 pair_E_diff。
   - **`baseline_qids` 取值范围**：仅收入 `contrastive_role == "baseline_subject_swap"` 的落盘行（Phase 1 §5.6.4 说明：subject_swap 是唯一在 Phase 1/2 产出独立 BenchmarkSample 行的 baseline）。每个元素为 `{"qid": str, "role": str}` 字典，`role` 嵌入该字典而非独立字段。其他 4 类通用 baseline（`baseline_static_copy / baseline_random_motion / baseline_global_filter / baseline_camera_pan_cheat`）不占用 qid，由 Phase 3 §4.7 `baselines/*.py` 在评测端动态合成，不进入本索引。
   - **最小成对约束**：Pilot 阶段每条 pair 类型 = ≥1 original_qid + 1 baseline_subject_swap；P1 阶段同 pair 多个 baseline 为理论上允许但当前不产出。
6. **分维度落盘**：按 dimension 写到 `samples/{dimension}.jsonl`（共 7 个文件），同时把每条样本的精简版本（等价于 BenchmarkSample 顶层视图，**不包含 `_audit`**）写入 `phase3_manifest.jsonl`。

**输出**：

- `benchmark_dataset/samples/{dimension}.jsonl` × 7（顶层字段对齐 Phase 3 §2.4，含 `_audit` 子节点供内部查验）。
- `benchmark_dataset/phase3_manifest.jsonl`：Phase 3 唯一入口（不含 `_audit`）。
- `benchmark_dataset/contrastive_pairs.jsonl`：`contrastive_pair_id` 聚合索引，Phase 3 `aggregate.py` 计算 pair_E_diff 使用。

**关键设计与底层原理**：

- **为什么以 question_id 落盘 join 而非内存串接**：四个上游模块各自独立写 jsonl，靠内存对象串接一旦中间崩溃就要从头重跑。落盘 + 文件 join 是流水线最稳的串接方式——任何一步失败可单独重跑该步，前后产物保留可复查。
- **为什么三重门控（pass + 非空 prompt + 非空 manifest）**：三种空值各自代表不同失败模式——`qc_status≠pass` 是质量否决，`prompt 空` 是 polish 失败且未 fallback（兜底），`manifest 空` 是构图失败。任何一项缺失都会让 Phase 3 报错，三重门控能在 export 阶段就把这些样本滤掉，比让 Phase 3 模型跑完才发现节省至少一个量级算力。
- **为什么多图质量按 worst-case 聚合**：评测可信度的关键是"任何一张参考的弱点都会被模型利用"——三张参考里有一张背景脏，模型可能就靠那张背景做空间推断；最高/平均聚合会掩盖这种风险。worst-case 标记最保守，让 audit 能精准识别"看似合格但有薄弱点"的多图样本。
- **为什么 phase3_manifest 是 BenchmarkSample 视图而非独立结构**：保证两份产物 schema 同源——manifest 字段调整时直接从 Sample 派生，不会出现"加了字段但忘了同步 manifest"的下游事故。
- **为什么按 dimension 分文件而非单一大文件**：Phase 3 评测器是按维度调度的（Phase 3 §5 `evaluators/{dimension}.py`），分文件让"只跑某维度"成为零开销操作；论文统计也需要按维度切片，分文件让 wc -l 就能统计每维度规模。
- **为什么 source_trace 必带 phase1_sample_ids 与 phase1_asset_ids**：reviewer 一抽样就要能回到 TIP 原视频与原资产去验证"这道题确实源自真实分布"。这是 benchmark "可审计性"的硬性要求——少了任何一个 ID，"从样本反查 provenance"的链条就断了。

### 6.8 audit_phase2 — 出货前最后体检与数据卡

**输入**：`quota_plan.json`、`sampled_recipes.jsonl`、`question_plans.jsonl`、`qc_reports/*.json`、`samples/*.jsonl`、`phase3_manifest.jsonl`。

**处理**：六类校验

1. **行数一致性**：`samples/*.jsonl` 总行数 = `phase3_manifest.jsonl` 行数；不一致直接 fail。
2. **配额缺口**：将实际产出按 `bucket_id` 聚合，与 `quota_plan.target_count` 对比，列出每桶 gap。
3. **维度统计**：每维度的 single/multi/subtype/difficulty/rarity 分布。
4. **Contrastive 配对**：统计每对 `contrastive_pair_id` 是否同时存在 A 与 B；落单的列入风险。
5. **多图质量直方图**：crop_leakage_risk / identity_visibility / scale_compatibility 三个字段的分布。
6. **QC 状态直方图**：pass / fail / needs_manual_review 比例。

**输出**：

- `benchmark_dataset/dataset_card.md`：六节统计 + 风险摘要 + 验收清单。

**关键设计与底层原理**：

- **为什么单独做 audit 而非在 export 里顺便校验**：审计是"独立见证人"角色——审计逻辑与导出逻辑共用代码会让 bug 互相隐藏。分离两者才能在自检失败时给出可信信号，类似软件工程里"test code 不复用 production code 私有函数"。
- **为什么 audit 不自动修复**：自动修复的本质是"猜测意图"，但每类缺口背后的根因不同——配额缺口可能是 Phase 1 数据不够、可能是阈值太严、也可能是某个上游 step 有 bug。把决策权留给人，比让脚本"按某个默认策略修复"安全得多。
- **为什么六类校验缺一不可**：六个维度对应六类风险——行数对应"完整性"、配额 gap 对应"代表性"、维度统计对应"均衡性"、contrastive 配对对应"实验设计完整性"、质量直方图对应"评测可信度"、QC 直方图对应"成本可控性"。漏掉任一项都会在 Phase 3 暴露成问题。
- **为什么 dataset_card.md 用 Markdown 而非 JSON**：论文素材直接 copy/paste，CI 可读，git diff 友好；JSON 适合机器消费但不适合人审阅。审计输出主要受众是项目 owner 与论文作者，Markdown 是这个语境下最合适的载体。

---

## 7. 全部产出文件结构

```
data/benchmark_dataset/
├── quota_plan.json                  # §6.1 配额规划
├── sampled_recipes.jsonl            # §6.2 配额下抽出的 recipe
├── quota_unfilled_report.json       # §6.2 缺口诊断（四类原因）
├── question_plans.jsonl             # §6.3 题目计划
├── first_frames/
│   └── {question_id}.png            # §6.4 单图首帧（禁止使用旧名 images/）
├── ref_images/
│   └── {question_id}_ref{k}.png     # §6.4 多图参考（禁止使用旧名 references/）
├── input_assets_manifest.jsonl      # §6.4 每题对应资产
├── qc_reports/
│   └── {question_id}.json           # §6.5 VQA 质检报告
├── qc_failed_to_retry.jsonl         # §6.5 QC fail 队列
├── manual_review_queue.jsonl        # §6.5 人工裁决队列
├── prompts/
│   └── final_prompts.jsonl          # §6.6 prompt 定稿
├── samples/                         # §6.7 按维度分文件落盘
│   ├── attribute_binding.jsonl
│   ├── action_binding.jsonl
│   ├── motion_binding.jsonl
│   ├── spatial_composition.jsonl
│   ├── background_dynamics.jsonl
│   ├── view_transformation.jsonl
│   └── interaction_reasoning.jsonl
├── phase3_manifest.jsonl            # §6.7 Phase 3 唯一入口
├── contrastive_pairs.jsonl          # §6.7 配对索引（pair_E_diff 使用）
└── dataset_card.md                  # §6.8 审计与数据卡
```

---

## 8. 核心数据结构

本节字段定义与 `src/i2vcompbench/schemas/phase2.py` 严格一致。Phase 3 evaluator 仅消费 `phase3_manifest.jsonl`（即 `BenchmarkSample` 顶层去掉 `_audit` 子节点的视图）。

### QuotaPlan / QuotaBucket（quota_plan.json）

```python
QuotaPlan = {
    "mode": str,                          # pilot | full
    "num_per_dimension": int,             # 单维度目标数（pilot=20 / full=200）
    "buckets": [QuotaBucket],
}

QuotaBucket = {
    "bucket_id": str,                     # {dim}__{mode_or_subtype}__{difficulty}__{rarity}
    "dimension": str,
    "input_mode_or_subtype": str,         # single_image | multi_image | type_a_absolute_single | ...
    "difficulty": str,                    # easy | medium | hard
    "rarity": str,                        # common | rare
    "target_count": int,
    "contrastive_pair_required": bool,
}
```

说明：motion 维度按 subtype 分桶（§6.1），其它维度按 input_mode 分桶；两种情况共用 `input_mode_or_subtype` 单字段，避免 schema 出现互斥可空字段。

### SampledRecipe（sampled_recipes.jsonl 每行）

```python
SampledRecipe = {
    "bucket_id": str,
    "dimension": str,
    "input_mode": str,                    # single_image | multi_image
    "subtype": str,
    "difficulty": str,                    # easy | medium | hard
    "semantic_rarity": str,               # common | rare
    "contrastive_pair_id": Optional[str], # {dim}_pair_{NNNN}
    "contrastive_role": str,              # original | baseline_static_copy | baseline_random_motion
                                          # | baseline_global_filter | baseline_camera_pan_cheat
                                          # | baseline_subject_swap
    "recipe": dict,                       # 透传 Phase 1 CandidateRecipe 全字段
                                          # 含 recipe_id / source_sample_id / reference_asset_ids 等
}
```

说明：原 Phase 1 字段（`recipe_id` / `source_sample_id` / `reference_asset_ids` / 其他）整体放入 `recipe` 子字典透传，避免 SampledRecipe 与 CandidateRecipe schema 互相污染；下游需要时直接读 `sample.recipe["recipe_id"]` 等。

### QuestionPlan（question_plans.jsonl 每行）

```python
QuestionPlan = {
    "question_id": str,                   # {dim_short}_{mode_short}_{seq:04d}
    "recipe_id": str,
    "dimension": str,
    "input_mode": str,
    "subtype": str,
    "difficulty": str,
    "semantic_rarity": str,
    "contrastive_pair_id": Optional[str],
    "contrastive_role": str,              # ⊂ 6 项 ContrastiveRole 枚举

    "input_plan": {
        "required_images": [
            {
                "role": str,              # first_frame | target_subject | attribute_reference | scene_reference
                "description": str,
                "source_preference": [str],  # ⊂ {tip_derived_reference, t2i_generated, external}
            }
        ],
    },

    "target_plan": {
        "target_subjects": [
            {
                "id": "s1",               # 稳定 id：s1 / s2 / ...，按 plan 出现顺序
                "description": str,
                "noun": Optional[str],
                "ref_image_idx": Optional[int],   # multi_image 必填，与 ref_images/{qid}_ref{k}.png 对齐
            }
        ],
        "target_relation": Optional[{
            "type": str,                  # 关系类别：spatial | interaction | temporal | ...
            "value": str,                 # 具体关系值：on | left_of | hand_to | ...
            "subj": Optional[str],        # 引用 target_subjects[i].id（如 "s1"）
            "obj": Optional[str],         # 引用 target_subjects[i].id（如 "s2"）
        }],
        "operation": str,                 # transform / compose / preserve（默认 transform）
        "attribute_source": Optional[str],# 如 "ref:attribute_reference"
        "expected_final_state": str,      # 自然语言描述目标终态
    },

    "preserve_plan": [
        {
            "scope": str,                 # target_identity | background | camera | s1 | s2 | ...
            "constraint": str,            # preserve | stable | appearance_and_position | ...
        }
    ],

    "dimension_isolation": {
        "forbidden_words": [str],
        "camera_constraint": str,         # fixed | minimal | allowed
    },

    "evaluator_tools": [str],             # ⊂ 9 项强枚举（附录 A.1）
    "expected_failure_modes": [str],      # ⊂ 15 项 FAILURE_MODES（附录 A.2）
    "prompt_draft": str,
}
```

字段名变更说明（与早期草案的差异，合作者请注意）：

- `input_plan` 由 `{first_frame_role, ref_roles[]}` 改为 `{required_images: [{role, description, source_preference}]}`；`first_frame` 也作为 `role` 之一进入 `required_images`。
- `target_plan` 不再使用 `expected_change`；改为三段式：`operation`（操作大类）+ `attribute_source`（属性来源指代）+ `expected_final_state`（目标终态自然语言）。
- `target_relation` 字段名 `predicate` 已废弃；与 Phase 3 §2.6 对齐为 `{type, value, subj, obj}`。
- `preserve_plan` 由 `{preservation_set: [str]}` 改为结构化 `List[PreserveItem]`，每项 `{scope, constraint}`，与 BenchmarkSample.preservation_set 的写盘形态完全一致。
- `dimension_isolation` 不含 `primary_dimension / forbidden_dimensions`（dimension 在样本顶层即唯一确定，无需重复），改为 `{forbidden_words, camera_constraint}`，后者用于 §6.6 finalize_prompts 的相机约束硬检查。
- 不再有 `risk_flags` 字段；模板渲染失败等风险信号写入 `BenchmarkSample._audit.qc.risk_flags`。

### InputAsset（input_assets_manifest.jsonl 每行的 assets[] 元素）

```python
InputAssetItem = {
    "asset_id": str,
    "role": str,                          # first_frame | target_subject | attribute_reference | scene_reference
    "path": str,
    "source_type": str,                   # ⊂ {tip_derived_reference, t2i_generated, external}
    "source_ref_id": Optional[str],       # 来自 reference_bank 时为 phase1 asset_id；T2I 时为 None
    "ref_image_idx": Optional[int],       # multi_image 参考下标，与 SubjectRef.ref_image_idx 对齐
    "quality": {
        "identity_visibility": str,       # high | medium | low | unknown（枚举字符串，非 float）
        "crop_leakage_risk": str,         # low | medium | high | unknown
        "resolution_ok": bool,
        "notes": str,                     # 单字符串备注（非 list）
    },
}

InputAssetManifest = {
    "question_id": str,
    "assets": [InputAssetItem],
}
```

说明：`quality.identity_visibility / crop_leakage_risk` 为四档枚举字符串（与早期 float 约定不同）；这是为了让 §6.7 的 worst-case 聚合直接用 max/min 字典序，无需引入分位数计算。多图聚合到 `BenchmarkSample._audit.multi_reference_quality` 时会做一次 `MultiReferenceQuality` 的派生（含 `scale_compatibility` float）。

### QCCheck / QCReport（qc_reports/{question_id}.json）

```python
QCCheck = {
    "name": str,
    "answer": bool,
    "confidence": float,
    "rationale": str,
}

QCReport = {
    "question_id": str,
    "qc_status": str,                     # pass | fail | needs_manual_review
    "checks": [QCCheck],
    "risk_flags": [str],
    "retry_count": int,
    "notes": str,
}
```

说明：`hard_check` **不**作为 QCCheck 字段存储（避免每条 check 都重复一份硬编码标志）；hard_check 名单由代码常量 `_HARD_CHECK_NAMES` 集中维护（§6.5），聚合时按名查表即可。

### FinalPromptEntry（final_prompts.jsonl 每行）

```python
FinalPromptEntry = {
    "question_id": str,
    "prompt": str,                        # 注：字段名是 prompt，不是 i2v_prompt
    "length_words": int,
    "forbidden_hits": [str],              # 命中的禁词列表
    "polish_attempts": int,
    "used_fallback": bool,
    "vlm_caption": str,
    "failed_check": Optional[str],        # missing_active_verb | forbidden_hit | out_of_range | None
}
```

### BenchmarkSample（samples/{dimension}.jsonl 每行）

顶层 17 个字段全部扁平（**禁止 `metadata` 嵌套**）；调试字段集中放在 `_audit` 子节点，`phase3_manifest.jsonl` 必须剔除 `_audit`。

```python
BenchmarkSample = {
    # 基本元信息
    "question_id": str,
    "dimension": str,
    "input_mode": str,                    # single_image | multi_image

    # 输入图像
    "first_frame_path": str,              # first_frames/{qid}.png
    "input_image_paths": [str],           # multi_image 时 ≥2，single 时 []

    # 评测目标（Phase 3 直接消费）
    "prompt": str,
    "target_subjects": [
        {
            "id": "s1",
            "description": str,
            "noun": Optional[str],
            "ref_image_idx": Optional[int],
        }
    ],
    "target_relation": Optional[{
        "type": str,                      # spatial | interaction | ...
        "value": str,                     # on | left_of | hand_to | ...
        "subj": Optional[str],
        "obj": Optional[str],
    }],
    "preservation_set": [
        {"scope": str, "constraint": str} # 与 QuestionPlan.preserve_plan 完全同形
    ],

    # 对照配对
    "contrastive_pair_id": Optional[str],
    "contrastive_role": str,              # ⊂ 6 项 ContrastiveRole 枚举

    # 评测器配置
    "evaluator_tools": [str],             # ⊂ 9 项
    "expected_failure_modes": [str],      # ⊂ 15 项

    # 分布标签
    "subtype": str,
    "difficulty": str,
    "semantic_rarity": str,
    "source_type": str,                   # ⊂ 4 项 Phase 3 枚举

    # 调试辅助（Phase 3 evaluator 禁止读；phase3_manifest.jsonl 必须剔除）
    "_audit": {
        "source_trace": {
            "recipe_id": str,
            "legacy_source_type": str,    # Phase 1 旧 source_type 原值（审计用）
            "phase1_sample_ids": [str],
            "phase1_asset_ids": [str],
        },
        "qc": {
            "status": str,                # pass | fail | needs_manual_review
            "risk_flags": [str],
        },
        "multi_reference_quality": Optional[{
            "crop_leakage_risk": str,     # worst-case 聚合（four-level 枚举字符串）
            "scene_leakage_risk": str,    # worst-case
            "identity_visibility": str,   # worst-case（min-level）
            "scale_compatibility": float, # bbox 面积比 worst-case 后归一化
        }],
    },
}
```

### ContrastivePair（contrastive_pairs.jsonl 每行）

`contrastive_pairs.jsonl` 由 `export_dataset.py` 直接以字典写盘，schemas/phase2.py 未声明 Pydantic class（两端字段简单且只有读侧消费）。每行结构：

```python
ContrastivePair = {
    "pair_id": str,                       # {dim}_pair_{NNNN}（= contrastive_pair_id）
    "dimension": str,
    "original_qids": [str],               # 该 pair 下 role="original" 的 question_id 列表
    "baseline_qids": [                    # 该 pair 下所有 baseline 落盘行
        {"qid": str, "role": str}         # role ⊂ 5 项 baseline_* 枚举（当前仅产出 baseline_subject_swap）
    ],
}
```

Phase 3 `aggregate.py` 通过 `pair_id` join 回 BenchmarkSample 计算 `pair_E_diff`（Phase 3 §4.8）。

---

## 9. 配置项

`configs/phase2.yaml` 关键参数：

```yaml
mode: pilot                            # pilot | full

paths:
  phase1_bundle: data/phase1_bundle
  benchmark_dataset: data/benchmark_dataset

quota:
  num_per_dimension:
    pilot: 20
    full: 200
  difficulty_ratio:                    # 七维共用
    easy: 0.30
    medium: 0.50
    hard: 0.20
  rarity_ratio:                        # 七维共用
    common: 0.80
    rare: 0.20
  input_mode_ratio:                    # 各维度独立
    attribute_binding:    {single_image: 0.60, multi_image: 0.40}
    action_binding:       {single_image: 0.70, multi_image: 0.30}
    spatial_composition:  {single_image: 0.20, multi_image: 0.80}
    view_transformation:  {single_image: 0.95, multi_image: 0.05}
    background_dynamics:  {single_image: 0.80, multi_image: 0.20}
    interaction_reasoning:{single_image: 0.50, multi_image: 0.50}
  motion_subtype_ratio:                # motion 退化为 subtype 分桶
    type_a_absolute_single: 0.40
    type_b_relative_single: 0.30
    type_c_multi_motion:    0.30

sampling:
  blocking_quality_flags:
    - low_alignment
    - missing_inpainted_scene
    - subject_not_visible
    - evaluator_infeasible
  multi_image_quality_threshold: 0.4
  require_clean_background: true

construct_inputs:
  long_edge: 1280
  format: png
  source_preference:
    - tip_derived_reference
    - t2i_generated
    - external

verify_inputs:
  hard_check_confidence_threshold: 0.7
  manual_review_default_to_drop: true

finalize_prompts:
  word_count_range: {min: 8, max: 25}
  word_count_range_interaction: {min: 8, max: 30}
  polish_max_attempts: 3
  require_active_verb_dimensions:
    - attribute_binding
    - action_binding
    - motion_binding
    - spatial_composition
    - background_dynamics
    - interaction_reasoning
  view_transformation_camera_verbs:
    - zoom
    - pan
    - tilt
    - dolly
    - orbit
```

---

## 10. 运行方式

```bash
# 单步执行
python -m i2vcompbench.phase2.build_quota         --config configs/phase2.yaml --mode pilot
python -m i2vcompbench.phase2.sample_recipes      --config configs/phase2.yaml
python -m i2vcompbench.phase2.build_question_plan --config configs/phase2.yaml
python -m i2vcompbench.phase2.construct_inputs    --config configs/phase2.yaml
python -m i2vcompbench.phase2.verify_inputs       --config configs/phase2.yaml
python -m i2vcompbench.phase2.finalize_prompts    --config configs/phase2.yaml
python -m i2vcompbench.phase2.export_dataset      --config configs/phase2.yaml
python -m i2vcompbench.phase2.audit_phase2        --dataset data/benchmark_dataset

# 全量执行（unified CLI）
i2vcompbench phase2 --config configs/phase2.yaml --mode pilot
```

---

## 11. 实现状态与扩展计划

### 11.1 Pilot 实现优先级

**P0：先跑通闭环**

1. 定义 Pydantic / dataclass schemas（`QuestionPlan`、`InputAsset`、`BenchmarkSample`）。
2. 实现 build_quota / sample_recipes / build_question_plan 的最小版本。
3. 实现 single_image construct_inputs（直接复用 TIP 首帧）。
4. 实现 finalize_prompts 与 export_dataset，输出 7×20=140 条 pilot 样本。

**P1：增强多图主维度**

1. 多图 reference quality 检查。
2. Spatial multi-reference composition。
3. Attribute multi-reference transfer。
4. Background subject-scene replacement。

**P2：质量与可信度**

1. verify_inputs 的结构化 VQA/QC 全量实现。
2. dimension_isolation 禁词与禁义检查。
3. dataset_card.md 自动生成论文图表所需统计。

### 11.2 最小验收标准

#### 11.2.1 体量与覆盖

- pilot 输出 7×20=140 条样本。
- 每个维度同时包含 single_image 和 multi_image，View 可少量 multi_image。
- 每条 multi_image 样本必须落盘 `ref_images/{question_id}_ref{k}.png`（k 与 `target_subjects[i].ref_image_idx` 对齐），并在 `_audit.multi_reference_quality` 留痕。
- `phase3_manifest.jsonl` 与 `samples/{dimension}.jsonl` 总条数一致。

#### 11.2.2 Phase 3 衔接硬约束（7 项必检，违反任一即不可进入 Phase 3）

- **目录命名**：必须使用 `first_frames/` 与 `ref_images/`，**禁止**出现 `images/` 或 `references/`。
- **字段扁平化**：BenchmarkSample 顶层必须包含 §8 列出的 17 个字段全集，**禁止** `metadata.*` 嵌套；`prompt` 字段名是 `prompt`，**不是** `i2v_prompt`。
- **target_subjects 稳定 id**：每条样本的 `target_subjects[i].id` 必须形如 `s1, s2, ...`，按 question_plan 出现顺序赋值；multi_image 样本必须额外带 `ref_image_idx` 且与 `ref_images/{qid}_ref{k}.png` 中的 k 一一对齐。
- **evaluator_tools 强枚举**：取值必须 ⊂ 9 项工具枚举（附录 A.1），由 `_TOOLS_BY_DIM` 按 dimension 强制覆写，禁止自由文本。
- **expected_failure_modes 非空**：必须默认填充（附录 A.2 `_FAILURES_BY_DIM`）且取值 ⊂ Phase 3 §2.5 的 15 种 FAILURE_MODES 枚举。
- **source_type 新词汇**：必须 ⊂ 4 项 Phase 3 枚举，Phase 1 旧词汇必须经 `_LEGACY_SOURCE_MAP`（附录 C）转换。
- **contrastive 顶层透传**：`contrastive_pair_id` / `contrastive_role` 必须出现在顶层；同 pair 的 original 与 baseline 行的 `contrastive_pair_id` 必须一致；`contrastive_pairs.jsonl` 必须存在且每条 pair 至少包含 1 original + ≥1 baseline。
- **_audit 隔离**：`source_trace` / `qc` / `multi_reference_quality` 等调试字段只允许出现在 `_audit` 子节点；`phase3_manifest.jsonl` 中**禁止**包含 `_audit`。

### 11.3 常见错误清单

1. 不要让 Phase 2 绕过 Phase 1 自己随机拼题。
2. 不要把多图样本标成 stress-only；当前主线要求多图进入主维度。
3. 不要把 Spatial prompt 写成 `move/reposition/shift`。Spatial 是静态组装。
4. 不要让 `qc_status=needs_manual_review` 的样本未经人工确认就进入 `phase3_manifest.jsonl`。
5. 不要让 multi_image 样本只填 prompt 而忘记 `ref_images/{qid}_ref{k}.png` 落盘 + `_audit.multi_reference_quality` 留痕。
6. 不要让 Phase 3 反向解析自然语言 prompt 才能知道评测目标——所有结构化字段（target_subjects / target_relation / preservation_set / evaluator_tools / expected_failure_modes）在 Phase 2 必须填好。
7. 不要静默降级 input_mode（例如某维度多图不够时偷偷转成单图），应写入 `quota_unfilled_report.json`。
8. **不要保留 `metadata` 嵌套结构**——Phase 3 §2.4 要求 17 个字段全部扁平到 BenchmarkSample 顶层；`expected_change / preserve_constraints / evaluator_hints` 等旧设计必须拆扁平为 `target_subjects / target_relation / preservation_set / evaluator_tools / expected_failure_modes`。
9. **不要使用 Phase 1 旧 source_type 词汇**——`observed_single_image / derived_single_image / derived_multi_reference / external_real / external_synthetic` 必须经 `_LEGACY_SOURCE_MAP`（附录 C）映射后再写入顶层 `source_type`。
10. **不要在 `evaluator_tools` 中写自由文本**——必须 ⊂ 9 项工具枚举（附录 A.1）；同样**不要把 `expected_failure_modes` 留空**——必须按 dimension 默认填充（附录 A.2）。
11. **不要使用旧目录名 `images/` `references/`**——Phase 3 硬编码读取 `first_frames/` 与 `ref_images/`，旧名会让 evaluator 直接 FileNotFound。
12. **不要把调试字段混入顶层**——`source_trace / qc / multi_reference_quality` 等 Phase 1/2 内部审计字段只能出现在 `_audit` 子节点，且 `phase3_manifest.jsonl` 必须剔除 `_audit`。

### 11.4 Coding Agent 第一步任务建议

1. 创建 schemas：`QuestionPlan`、`InputAsset`、`BenchmarkSample`。
2. 实现 build_quota（pilot 模式：每维度 20 条），从 Phase 1 sampled_recipes 选 21 条做最小验证。
3. 接通 single_image 输入（直接复用 Phase 1 首帧），跑通 export_dataset。
4. 验证 `phase3_manifest.jsonl` schema 合法、metadata 完整。

完成 Phase 2 闭环后，再进入 Phase 3（详见 `Phase3_模型评测.md`）。

---

## 附录 A. dimension → evaluator_tools / expected_failure_modes 映射表

本附录被 §6.3 build_question_plan 与 §6.7 export_dataset 引用，是 Phase 2 → Phase 3 衔接的强枚举权威表。**任何与下表不一致的 evaluator_tools / expected_failure_modes 取值都必须在 §6.3 步骤 7/8 被覆写为下表值**。

### A.1 `_TOOLS_BY_DIM`（dimension → required evaluator_tools）

| dimension | required evaluator_tools |
|-----------|--------------------------|
| attribute_binding | `grounding, vlm_attribute, vlm_existence` |
| action_binding | `grounding, vlm_existence, optical_flow` |
| motion_binding | `grounding, dot_motion, optical_flow` |
| spatial_composition | `grounding, depth, vlm_relation` |
| background_dynamics | `grounding, optical_flow, vlm_existence` |
| view_transformation | `depth, optical_flow, vlm_existence` |
| interaction_reasoning | `grounding, vlm_relation, optical_flow` |

**multi_image 加挂工具**：当 `input_mode == multi_image` 时，无论 dimension 为何，都额外追加 `dinov2, clip, grounding`（用于身份保持与外观一致性度量）。

**强枚举值域**：`{grounding, depth, dot_motion, optical_flow, vlm_existence, vlm_attribute, vlm_relation, dinov2, clip}` 共 9 项。本表为该枚举值域的唯一权威源；Phase 3 §A（环境与模型权重清单）仅记录每项工具的 backbone / weight hash，**不重复定义枚举值域**。

### A.2 `_FAILURES_BY_DIM`（dimension → 默认 expected_failure_modes）

本表与 `src/i2vcompbench/schemas/phase2.py::_FAILURES_BY_DIM` 一致；任何与下表不符的取值在 §6.3 步骤 8 被强制覆写。

| dimension | default expected_failure_modes |
|-----------|--------------------------------|
| attribute_binding | `wrong_attribute, object_missing, identity_lost` |
| action_binding | `static_copy, object_missing, timing_wrong` |
| motion_binding | `wrong_direction, static_copy, camera_pan_cheat` |
| spatial_composition | `wrong_relation, object_missing, identity_lost` |
| background_dynamics | `background_drift, non_target_drift, global_filter` |
| view_transformation | `wrong_camera, camera_pan_cheat, identity_lost` |
| interaction_reasoning | `wrong_relation, timing_wrong, object_missing` |

**强枚举值域**（与 Phase 3 §2.5 FAILURE_MODES 严格一致，共 15 项）：

`static_copy / global_filter / camera_pan_cheat / object_missing / wrong_attribute / wrong_direction / wrong_relation / wrong_camera / identity_lost / non_target_drift / background_drift / artifact_severe / timing_wrong / identity_unbound / tool_uncertain`

词汇变更说明（与早期草案的差异，合作者请注意）：

- `identity_drift` → `identity_lost`（与 Phase 3 identity_binding 软因子的判失语义对齐）
- `wrong_motion_direction` → `wrong_direction`（缩短 + 涵盖运动/视角方向）
- `view_collapse` → `wrong_camera`（直接表达"相机意图错"）
- `wrong_action` 拆解：仍可由 `static_copy`（动作未发生）或 `timing_wrong`（动作时序错）覆盖
- 新增 `artifact_severe / timing_wrong / identity_unbound`（覆盖伪影 / 时序错 / 多图身份未绑定）
- 移除 `low_confidence / invalid_input`（已并入 `tool_uncertain` 单一不确定标签）

LLM 在 §6.6 finalize_prompts polish 阶段如能识别更具体的失败模式可追加，但仍必须在上述 15 项内。

---

## 附录 C. `_LEGACY_SOURCE_MAP`（Phase 1 source_type → Phase 3 source_type）

本附录被 §6.7 export_dataset 步骤 4 引用。Phase 1 的 `source_type` 取值是为先验数据准备阶段设计的（描述"这条 recipe 的图像/题面是怎么来的"），与 Phase 3 evaluator 期望的"数据集分布标签"词汇不一致；Phase 2 export 时必须按下表强制转换：

| Phase 1 source_type（旧） | Phase 3 source_type（新，写入 BenchmarkSample 顶层） |
|--------------------------|----------------------------------------------------|
| `observed_single_image` | `tip_i2v_real` |
| `derived_single_image` | `tip_i2v_synthetic_first_frame` |
| `derived_multi_reference` | `tip_i2v_synthetic_first_frame` |
| `external_real` | `external_real`  _(预留：Phase 1 当前不产出此值，为未来接入第三方真实数据集预留映射)_ |
| `external_synthetic` | `external_synthetic`  _(预留：Phase 1 当前不产出此值，为未来接入 T2I/外部合成素材预留映射)_ |

**Phase 3 顶层 source_type 强枚举值域**：`{tip_i2v_real, tip_i2v_synthetic_first_frame, external_real, external_synthetic}` 共 4 项。任何旧词汇直接透传都视为衔接违规（见 §11.2.2 与 §11.3 第 9 条）。
