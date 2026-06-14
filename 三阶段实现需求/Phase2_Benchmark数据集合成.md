# I2V-CompBench Phase 2 实现需求（Benchmark 数据集合成）

> 本文档面向 AI coding agent，是《I2V-CompBench Phase 1/2/3 实现需求（Coding Agent 版）》的 Phase 2 拆分文档。Phase 1、Phase 3 内容请参见同目录下的另外两份文档。
>
> Phase 2 的目标：消费 Phase 1 的 `phase1_bundle/`，按配额、采样、QC、定稿全流程产出最终 Benchmark 数据集 `benchmark_dataset/`，并提供 Phase 3 可直接消费的 `phase3_manifest.jsonl`。

---

## 0. Phase 2 总目标

```text
Phase 2: phase1_bundle/ -> benchmark_dataset/
```

必须支持：

- 7 个评测维度：attribute, action, motion, spatial, background, view, interaction。
- 2 种输入模式：single_image, multi_image。
- 多图模式是**主维度样本**，不是 stress-only。
- 每道题必须输出 **Phase 3-ready metadata**，Phase 3 不应重新解析自然语言 prompt 才知道评测目标。
- 不允许 Phase 2 绕过 Phase 1 自己随机拼词表。所有题目必须从 Phase 1 的 `candidate_recipes.jsonl` 采样而来。

---

## 1. 推荐目录结构（Phase 2 部分）

```text
i2v_compbench/
  configs/
    phase2.yaml
    dimensions.yaml
    templates/
      attribute.yaml
      action.yaml
      motion.yaml
      spatial.yaml
      background.yaml
      view.yaml
      interaction.yaml
  data/
    phase1_bundle/        # Phase 1 输出，作为 Phase 2 输入
    benchmark_dataset/
  src/
    i2vcompbench/
      phase2/
        build_quota.py
        sample_recipes.py
        build_question_plan.py
        construct_inputs.py
        verify_inputs.py
        finalize_prompts.py
        export_dataset.py
        audit_phase2.py
      schemas/
        phase2.py
      utils/
        io.py
        ids.py
        image.py
        llm.py
        vlm.py
```

---

## 2. 全局枚举与边界规则（Phase 2 必须遵守）

### 2.1 Dimension 枚举

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

### 2.2 InputMode 枚举

```python
INPUT_MODES = ["single_image", "multi_image"]
```

### 2.3 Motion / Spatial 边界（Phase 2 模板与 prompt 定稿强制执行）

硬规则：

- 包含主体位移的题目 → `motion_binding`。
- 静态布局组合题目 → `spatial_composition`。
- Spatial 多图 recipe 必须是 static layout，**不含 `move/shift/reposition`** 等位移动词。

### 2.4 Action / Interaction 边界（Phase 2 prompt 定稿强制执行）

```text
single-subject body movement -> action_binding
multi-object causal/social/functional event -> interaction_reasoning
```

---

## 3. Phase 2 输入与输出

### 3.1 输入

```text
data/phase1_bundle/
  prior_package.json
  compatibility_matrix.json
  reference_bank/assets.jsonl
  candidate_recipes.jsonl
```

### 3.2 输出

```text
data/benchmark_dataset/
  samples/
    attribute_binding.jsonl
    action_binding.jsonl
    motion_binding.jsonl
    spatial_composition.jsonl
    background_dynamics.jsonl
    view_transformation.jsonl
    interaction_reasoning.jsonl
  images/
  references/
  prompts/
  phase3_manifest.jsonl
  qc_reports/
  dataset_card.md
```

---

## 4. 模块详细规格

Phase 2 由 8 个模块串成单向流水线，前一步的输出就是后一步的输入。下面按执行顺序逐个模块说明：**喂进去什么、内部做了什么、产出什么、关键设计取舍**。

```text
build_quota → sample_recipes → build_question_plan → construct_inputs
            → verify_inputs → finalize_prompts → export_dataset → audit_phase2
```

### 4.1 `build_quota.py` — 把抽象配额拆成可执行的桶

**输入**

- `configs/phase2.yaml`：`mode`（pilot / full）、`num_per_dimension`（pilot=20、full=200）、`input_mode_ratio` / `subtype_ratio` / `difficulty_ratio` / `rarity_ratio`。

**处理**

1. 对每个维度依次按四级分裂：`mode → subtype → difficulty → rarity`。
2. 每级用最大余数法 `_round_split` 把整数配额按比例分到子桶，保证总和不漂、误差落在余数最大的几个桶上。
3. 七维度共用同一份 `difficulty_ratio` / `rarity_ratio`，但 `input_mode_ratio` 与 `subtype_ratio` 按维度独立配置。
4. 每个叶节点拼出 `bucket_id = {dim}__{mode}__{difficulty}__{rarity}`；motion 维度因为只有 subtype 概念，bucket_id 退化为 `{dim}__{subtype}__{difficulty}__{rarity}`。

**输出**

- `benchmark_dataset/quota_plan.json`：`QuotaPlan { mode, total_target, buckets: [QuotaBucket] }`。
- 每个 `QuotaBucket` 含 `bucket_id / dimension / input_mode / subtype / difficulty / rarity / target_count / contrastive_pair_required`。

**关键设计**

- 所有比例参数都进入 `quota_plan.json`，避免下游再读 yaml；后面的 sample / audit 直接 join `bucket_id`。
- `target_count` 求和等于 `num_per_dimension × 7`（pilot=140、full=1400）。如果某维度 sub-ratio 本身求和不为 1，会被归一化后再分。

### 4.2 `sample_recipes.py` — 在配额下从 Phase 1 真实候选里抽题

**输入**

- `quota_plan.json`（Step 4.1）。
- Phase 1 的 `candidate_recipes.jsonl`（唯一题源）和 `assets.jsonl`（多图素材）。

**处理**

1. **建索引**：把 candidate_recipes 按 `(dimension, input_mode/subtype, difficulty, rarity)` 桶化；assets 按 `asset_id` 建查找表。
2. **基础过滤**：剔除带阻断标志的 recipe，阻断集合为 `_BLOCKING_QUALITY_FLAGS = {"low_alignment", "missing_inpainted_scene", "subject_not_visible", "evaluator_infeasible"}`。
3. **多图体检**：对 `input_mode=multi_image` 或 motion 的 `_MULTI_IMAGE_SUBTYPES = {"type_c_multi_motion"}`，逐条校验每个 `reference_asset_id`：
   - 资产存在且 `quality_score ≥ 0.4`；
   - 该资产 `is_clean_background=True`（避免引入背景泄漏）；
   - 至少 2 个有效 reference，否则该 recipe 进 `quality_below_threshold` 缺口。
4. **配额抽样**：每个桶按 `target_count` 抽样，contrastive 桶要按 `source_sample_id` 分组成对消费 A/B（落单时降级为 `A_only`，并在缺口报告里记 `contrastive_unpaired`）。
5. **缺口诊断**：四类原因写入 `quota_unfilled_report.json`：
   - `no_candidate`：桶里根本没有匹配 recipe；
   - `no_reference_asset`：多图 recipe 但没绑定足量资产；
   - `quality_below_threshold`：资产 quality_score 或 clean_background 不达标；
   - `blocking_flag`：所有候选都被 quality_flags 阻断。

**输出**

- `benchmark_dataset/sampled_recipes.jsonl`：`SampledRecipe { recipe_id, bucket_id, contrastive_pair_id, contrastive_role, reference_asset_ids[] }`。
- `benchmark_dataset/quota_unfilled_report.json`：每桶 `{ bucket_id, target, sampled, gap, reasons[] }`。

**关键设计**

- **绝不绕过 Phase 1**：题目只能来自 candidate_recipes；缺口靠报告说明而不是降级。
- **多图体检前置**：在抽样阶段就把素材不达标的 recipe 拦下，避免到 construct_inputs 才发现没图可用。
- **A/B 配对原子化**：要么一起入选要么一起退出，避免下游 contrastive 评测时只剩半边。

### 4.3 `build_question_plan.py` — 把 recipe 翻译成可执行的题目计划

**输入**

- `sampled_recipes.jsonl`（Step 4.2）。
- Phase 1 的 `image_parse_v2.jsonl` / `text_parse_v2.jsonl` / `aligned_instances.jsonl`，提供主体、属性、关系、几何等结构化槽位。
- `configs/templates/{dimension}.yaml` 模板库（见 §6 模板格式）。

**处理**

1. **三层槽位融合**：按 `image_parse → aligned_instances → text_parse` 顺序叠加，后者覆盖前者，得到一份维度专属的 `slot_dict`（subjects / attributes / relations / camera_baseline 等）。
2. **subtype 解析**：优先用 recipe 上显式声明的 subtype；缺失时按 `input_mode` 在模板库 `subtypes` 列表里匹配（如 `multi_image → attribute_transfer`），再缺就退到该维度第一个可用 subtype。
3. **question_id 生成**：`{dim_short}_{mode_short}_{seq:04d}`，其中 `DIM_SHORT` 把长维度名缩成 attr / action / motion / spatial / bg / view / inter；`mode_short` 是 single / multi。
4. **模板渲染**：用 `prompt_pattern`（Jinja-like）渲染 `prompt_draft`；若槽位缺失导致渲染失败，回退到 recipe 自带的 `base_prompt_draft`，并在 risk_flags 里记 `template_render_failed`。
5. **结构化字段填充**：从模板拷贝 `target_plan / preserve_plan / dimension_isolation / evaluator_plan`，把里面的槽位占位符替换成实际值（`ref:target_subject` 之类指代保持原样，等 Step 4.4 落实到 asset_id）。

**输出**

- `benchmark_dataset/question_plans.jsonl`：每条 `QuestionPlan` 包含 question_id、recipe_id、维度五元组、`input_plan` / `target_plan` / `preserve_plan` / `dimension_isolation` / `evaluator_plan` / `prompt_draft`。

**关键设计**

- **draft ≠ final**：这里产出的 prompt 仅用于占位与 QC 参考，最终 prompt 由 Step 4.6 看着真实首帧重写。
- **evaluator_plan 必须填齐**：Phase 3 不再回头解析 prompt，所有 E/P/C 工具集合都在这一步定下来。

### 4.4 `construct_inputs.py` — 把题目计划落实成真正的输入图像

**输入**

- `question_plans.jsonl`（Step 4.3）。
- Phase 1 的 `manifest_clean.jsonl`（首帧路径）、`assets.jsonl`（参考资产）、`reference_bank/{asset_type}/*.jpg`。

**处理**

1. **角色 → 资产类型映射**：用 `_ROLE_TO_ASSET_TYPE` 把 `target_subject → subject`、`attribute_reference → attribute`、`scene_reference → scene_reference_inpainted | scene_reference_original` 等。
2. **首帧（single_image）**：直接复用 Phase 1 manifest 中 source_sample 对应的 `image_path`，不重新生成。
3. **多图参考（multi_image）**：按 recipe 的 `reference_asset_ids` 取资产；若同一角色有多个候选，挑 `quality_score` 最高的那个。
4. **`source_preference` 链式回退**：模板里的优先序是 `tip_derived_reference → t2i_generated → external`：
   - 优先从 reference_bank 拿真实 crop；
   - 缺位时调用 `Phase2SiliconFlowClient.call_t2i`（Kwai-Kolors/Kolors，1024×1024）按角色描述合成；
   - T2I 失败则在 `quality.notes` 写 `t2i_generated` 或 `t2i_failed`，由 QC 决定是否丢弃。
5. **统一规格化**：所有图像用 PIL resize 到 `long_edge=1024`，保持宽高比，统一存为 PNG。
6. **路径约定**：单图首帧 `images/{question_id}.png`；多图参考 `references/{question_id}__{role}.png`。
7. **资产质量元数据**：每张图记录 `identity_visibility / crop_leakage_risk / resolution_ok`，这些字段在 QC 与最终多图聚合中被反复消费。

**输出**

- `benchmark_dataset/images/`、`benchmark_dataset/references/`：实际图像文件。
- `benchmark_dataset/input_assets_manifest.jsonl`：每行一个 question_id 对应的 `assets[]`，含 `asset_id / role / path / source_type / source_ref_id / quality`。

**关键设计**

- **真实优先、合成兜底**：尽量用 Phase 1 真实素材保证视觉先验真实；T2I 仅作为缺口补救，避免合成图喧宾夺主。
- **失败不阻断流水线**：T2I 客户端不可用时仅将该题标记，让 QC 统一处理，不在 construct 阶段抛异常。

### 4.5 `verify_inputs.py` — 用结构化 VQA 给输入图把关

**输入**

- `question_plans.jsonl`、`input_assets_manifest.jsonl`、对应的 PNG 图像。
- `configs/templates/{dimension}.yaml` 中每个 subtype 声明的 `qc_checks[]`（每条 check 含 `name / question / hard_check`）。

**处理**

1. **逐题逐项跑 VQA**：对每张输入图，按模板里的 `qc_checks` 顺序调用 `Phase2SiliconFlowClient.call_vqa_structured`，模型返回 `{answer: bool, confidence: float, rationale: str}`。
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
   - `fail` 写入 `qc_failed_to_retry.jsonl`，由配置控制是否触发 Step 4.4 重采资产；
   - `needs_manual_review` 写入 `manual_review_queue.jsonl`，等待人工裁决，**默认不入 Phase 3**。

**输出**

- `benchmark_dataset/qc_reports/{question_id}.json`：`QCReport { question_id, qc_status, checks[], risk_flags }`。
- `benchmark_dataset/qc_failed_to_retry.jsonl`、`benchmark_dataset/manual_review_queue.jsonl`。

**关键设计**

- **soft check 不否决**：低置信度只触发人工复核，避免 VLM 抖动直接误杀样本。
- **HARD_CHECK 集中维护**：所有阻断项集中在 `_HARD_CHECK_NAMES`，模板只要复用名字就自动获得阻断语义，不必在每个模板里重复声明。

### 4.6 `finalize_prompts.py` — 看着真实首帧把 prompt 写定

**输入**

- `question_plans.jsonl`、`qc_reports/`、`input_assets_manifest.jsonl`、首帧 PNG。
- 模板 `prompts/prompt_polish.txt`。
- `Phase2SiliconFlowClient`（VLM 描述 + LLM 改写）。

**处理**（仅处理 `qc_status=pass` 的题目）

1. **VLM 描述首帧**：调用 VLM 把首帧描述成中性 caption，作为定稿时的视觉锚点。
2. **LLM polish**：把 caption + question_plan 的 `target_plan / preserve_plan / dimension_isolation` 拼进 `prompt_polish.txt`，要求模型返回 JSON：`{i2v_prompt, reasoning}`。
3. **解析与校验**：先剥 fenced ``` 代码块再做括号扫描提取 JSON；然后对 `i2v_prompt` 做：
   - **禁词检查**：命中 `dimension_isolation.forbidden_words` 任一词即失败；
   - **字数检查**：常规维度 8–25 词，`interaction_reasoning` 放宽到 30 词；
   - **指代检查**：多图必须能解析出 `image 1 / image 2` 或 `the reference X` 这类显式角色指代。
4. **重试与回退**：最多尝试 N 次（`polish_attempts` 字段记录次数），仍不合格则回退到 `prompt_draft`，并在 `risk_flags` 加 `prompt_polish_fallback`。

**输出**

- `benchmark_dataset/prompts/final_prompts.jsonl`：`FinalPromptEntry { question_id, i2v_prompt, polish_attempts, used_fallback, vlm_caption }`。

**关键设计**

- **看图说话**：先读首帧再写 prompt，避免 prompt 与实际起始状态错位。
- **draft 作为安全网**：极端失败也能拿到一条不犯禁的 prompt，保证 export 不掉数据。

### 4.7 `export_dataset.py` — 四源 join 出最终数据集

**输入**

- `question_plans.jsonl`、`input_assets_manifest.jsonl`、`qc_reports/*.json`、`final_prompts.jsonl`。

**处理**

1. **四源 join**：以 `question_id` 为主键，把 question_plan、input_assets、qc_report、final_prompt 内连接。
2. **门控筛选**：仅保留 `qc_status=pass` 且有非空 `i2v_prompt` 且 manifest 非空的样本。
3. **多图质量聚合 `_aggregate_multi_quality`**（worst-case 策略）：
   - `crop_leakage_risk` 取所有 reference 中的最大值（最差）；
   - `identity_visibility` 取最小值（最弱）；
   - `scale_compatibility` 由 bbox 面积比换算后取最小。
4. **拼装 BenchmarkSample**：
   - 维度五元组直接搬；`input_images[]` 来自 manifest；`i2v_prompt` 来自 final_prompts；
   - `metadata` 由 question_plan 的 target/preserve/evaluator_plan 翻译而来；
   - `source_trace` 含 `recipe_id / source_type / phase1_sample_ids / phase1_asset_ids`，全程可追溯到 TIP 原始首帧；
   - `qc.status / qc.risk_flags` 直接从 QCReport 拷贝。
5. **分维度落盘**：按 dimension 写到 `samples/{dimension}.jsonl`（共 7 个文件），同时把每条样本的精简版本写入 `phase3_manifest.jsonl`。

**输出**

- `benchmark_dataset/samples/{dimension}.jsonl` × 7。
- `benchmark_dataset/phase3_manifest.jsonl`：Phase 3 唯一入口。

**关键设计**

- **worst-case 聚合**：多图质量取最差边界，避免一张糟糕参考被另外两张漂亮的掩盖。
- **manifest 与 samples 一致**：行数严格相等，是 Step 4.8 的关键校验项。

### 4.8 `audit_phase2.py` — 出货前的最后体检与数据卡

**输入**

- `quota_plan.json`、`sampled_recipes.jsonl`、`question_plans.jsonl`、`qc_reports/*.json`、`samples/*.jsonl`、`phase3_manifest.jsonl`。

**处理**：六类校验

1. **行数一致性**：`samples/*.jsonl` 总行数 = `phase3_manifest.jsonl` 行数；不一致直接 fail。
2. **配额缺口**：将实际产出按 `bucket_id` 聚合，与 `quota_plan.target_count` 对比，列出每桶 gap。
3. **维度统计**：每维度的 single/multi/subtype/difficulty/rarity 分布。
4. **Contrastive 配对**：统计每对 `contrastive_pair_id` 是否同时存在 A 与 B；落单的列入风险。
5. **多图质量直方图**：crop_leakage_risk / identity_visibility / scale_compatibility 三个字段的分布。
6. **QC 状态直方图**：pass / fail / needs_manual_review 比例。

**输出**

- `benchmark_dataset/dataset_card.md`：六节统计 + 风险摘要 + 验收清单。

**关键设计**

- **不修复，只报告**：audit 阶段不动数据，只生成数据卡，让人决定是否放行进 Phase 3。
- **dataset_card.md 即论文素材**：六节统计直接对应论文 §dataset 的图表。

---

## 5. Phase 2 CLI 总流程

```bash
python -m i2vcompbench.phase2.build_quota --config configs/phase2.yaml --mode pilot
python -m i2vcompbench.phase2.sample_recipes --config configs/phase2.yaml
python -m i2vcompbench.phase2.build_question_plan --config configs/phase2.yaml
python -m i2vcompbench.phase2.construct_inputs --config configs/phase2.yaml
python -m i2vcompbench.phase2.verify_inputs --config configs/phase2.yaml
python -m i2vcompbench.phase2.finalize_prompts --config configs/phase2.yaml
python -m i2vcompbench.phase2.export_dataset --config configs/phase2.yaml
python -m i2vcompbench.phase2.audit_phase2 --dataset data/benchmark_dataset
```

可以先实现一个 unified CLI：

```bash
i2vcompbench phase2 --config configs/phase2.yaml --mode pilot
```

---

## 6. Phase 2 Pilot 实现优先级

### P0：先跑通闭环

1. 定义 Pydantic/dataclass schemas（`QuestionPlan`、`InputAsset`、`BenchmarkSample`）。
2. 实现 quota / sample_recipes / build_question_plan 的最小版本。
3. 实现 single_image construct_inputs（可先复用 TIP 首帧）。
4. 实现 finalize_prompts 与 export_dataset，输出 7×20=140 条 pilot 样本。

### P1：增强多图主维度

1. 多图 reference quality 检查。
2. Spatial multi-reference composition。
3. Attribute multi-reference transfer。
4. Background subject-scene replacement。

### P2：质量与可信度

1. verify_inputs 的结构化 VQA/QC 全量实现。
2. dimension_isolation 禁词与禁义检查。
3. dataset_card.md 自动生成论文图表所需统计。

---

## 7. Phase 2 最小验收标准

- pilot 输出 7×20=140 条样本。
- 每个维度同时包含 single_image 和 multi_image，View 可少量 multi_image。
- 每条样本有完整 `metadata.expected_change`、`preserve_constraints`、`evaluator_hints`。
- 每条 multi_image 样本有 `reference_assets` 与 `multi_reference_quality`。
- `phase3_manifest.jsonl` 与 samples 数量一致。

---

## 8. Phase 2 常见错误清单

1. 不要让 Phase 2 绕过 Phase 1 自己随机拼题。
2. 不要把多图样本标成 stress-only；当前主线要求多图进入主维度。
3. 不要把 Spatial prompt 写成 "move/reposition/shift"。Spatial 是静态组装。
4. 不要让 `qc_status=needs_manual_review` 的样本未经人工确认就进入 `phase3_manifest.jsonl`。
5. 不要让 multi_image 样本只填 prompt 而忘记 `reference_assets` 与 `multi_reference_quality`。
6. 不要让 Phase 3 反向解析自然语言 prompt 才能知道评测目标——所有 metadata 在 Phase 2 必须填好。
7. 不要静默降级 input_mode（例如某维度多图不够时偷偷转成单图），应写入 `quota_unfilled_report.json`。

---

## 9. Coding Agent 的第一步任务建议（Phase 2）

1. 创建 schemas：`QuestionPlan`、`InputAsset`、`BenchmarkSample`。
2. 实现 build_quota（pilot 模式：每维度 20 条），从 Phase 1 sampled_recipes 选 21 条做最小验证。
3. 接通 single_image 输入（直接复用 Phase 1 首帧），跑通 export_dataset。
4. 验证 `phase3_manifest.jsonl` schema 合法、metadata 完整。

完成 Phase 2 闭环后，再进入 Phase 3（详见 `Phase3_模型评测.md`）。
