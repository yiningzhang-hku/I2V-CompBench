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

### 4.1 Module: `build_quota.py`

功能：

- 生成七维度配额表。
- 支持 full 与 pilot 两种模式。

默认 full：

```yaml
num_per_dimension: 200
rarity:
  common: 0.8
  rare: 0.2
difficulty:
  easy: 0.4
  medium: 0.4
  hard: 0.2
input_mode_ratio:
  attribute_binding: {single_image: 0.6, multi_image: 0.4}
  action_binding: {single_image: 0.6, multi_image: 0.4}
  motion_binding:
    type_a_absolute_single: 0.5
    type_b_relative_single: 0.35
    type_c_multi_motion: 0.15
  spatial_composition: {single_image: 0.2, multi_image: 0.8}
  background_dynamics: {single_image: 0.6, multi_image: 0.4}
  view_transformation: {single_image: 0.95, multi_image: 0.05}
  interaction_reasoning: {single_image: 0.5, multi_image: 0.5}
```

输出 `quota_plan.json`。

验收：

- full 总数 = 1400。
- pilot 可配置，比如每维度 20。
- 所有 dimension/input_mode/subtype 都有明确 count。

### 4.2 Module: `sample_recipes.py`

功能：

- 根据 quota 从 `candidate_recipes.jsonl` 采样。
- **不允许绕过 Phase 1 随机拼词表**。

采样过滤：

- dimension 匹配。
- input_mode/subtype 匹配。
- difficulty/rarity 匹配。
- evaluator_requirements 可满足。
- quality_flags 不含 blocking issue。
- 多图样本 reference requirements 可被 reference_bank 满足。

输出 `sampled_recipes.jsonl`。

验收：

- 每个配额桶缺口要写入 `quota_unfilled_report.json`。
- 不得静默降级 input_mode。

### 4.3 Module: `build_question_plan.py`

功能：

- 将 recipe 转成结构化 `QuestionPlan`。
- 匹配模板，生成 draft prompt 与 evaluator metadata。

`QuestionPlan` schema：

```json
{
  "question_id": "attr_multi_0001",
  "recipe_id": "recipe_attr_0081",
  "dimension": "attribute_binding",
  "input_mode": "multi_image",
  "subtype": "attribute_transfer",
  "difficulty": "medium",
  "semantic_rarity": "common",
  "contrastive_pair_id": "attr_pair_0001",
  "contrastive_role": "A",
  "input_plan": {
    "required_images": [
      {"role": "target_subject", "description": "a woman"},
      {"role": "attribute_reference", "description": "a red jacket"}
    ],
    "source_preference": ["tip_derived_reference", "t2i_generated"]
  },
  "target_plan": {
    "target_subject": "ref:target_subject",
    "operation": "wear_attribute",
    "attribute_source": "ref:attribute_reference",
    "expected_final_state": "the woman wears the red jacket"
  },
  "preserve_plan": [
    {"scope": "target_identity", "constraint": "preserve"},
    {"scope": "background", "constraint": "stable"}
  ],
  "dimension_isolation": {
    "forbidden_words": ["pan", "zoom", "move left"],
    "camera_constraint": "fixed"
  },
  "evaluator_plan": {
    "E_target": "red jacket appears on target woman",
    "P_constraints": ["identity preserved", "no unrelated clothing changes"],
    "C_criteria": ["no flicker", "stable wearing state"],
    "tools": ["grounding", "segmentation", "vlm", "clip"]
  },
  "prompt_draft": "The woman wears the red jacket from the reference image."
}
```

验收：

- 每个 QuestionPlan 必须有 target_plan、preserve_plan、evaluator_plan。
- dimension_isolation 不得为空。

### 4.4 Module: `construct_inputs.py`

功能：

- 为 QuestionPlan 构建最终输入图像。
- single image 可使用 TIP 首帧、T2I 生成首帧或外部真实图。
- multi image 使用 reference_bank 或 T2I 补足。

输出：

- `images/` 与 `references/`
- `input_assets_manifest.jsonl`

`InputAsset` schema：

```json
{
  "question_id": "spatial_multi_0001",
  "assets": [
    {
      "asset_id": "asset_001",
      "role": "object_reference",
      "path": "data/benchmark_dataset/references/spatial_multi_0001_cat.png",
      "source_type": "tip_derived_reference",
      "source_ref_id": "tip_000001_s1_masked",
      "quality": {
        "identity_visibility": "high",
        "crop_leakage_risk": "low",
        "resolution_ok": true
      }
    }
  ]
}
```

验收：

- multi_image 样本至少 2 张输入图。
- 所有 input assets 有 role 和 source_trace。
- reference quality 不达标则重采样。

### 4.5 Module: `verify_inputs.py`

功能：

- 结构化 VQA/QC。
- 检查输入是否满足题目计划。

QC 输出：

```json
{
  "question_id": "spatial_multi_0001",
  "qc_status": "pass",
  "checks": [
    {"name": "has_exactly_one_cat_reference", "answer": true, "confidence": 0.92},
    {"name": "scene_has_enough_space", "answer": true, "confidence": 0.81}
  ],
  "risk_flags": []
}
```

必检项：

- 主体数量。
- 属性/状态可见性。
- 多图每张 reference 是否单一清楚。
- crop/background leakage。
- 场景容量和深度线索。
- 起始状态是否没有提前泄露终态。

验收：

- 只有 `qc_status=pass` 可进入 final dataset。
- `needs_manual_review` 不进入 Phase 3 manifest，除非人工确认。

### 4.6 Module: `finalize_prompts.py`

功能：

- 根据通过 QC 的实际输入定稿 I2V prompt。
- 执行禁词和长度检查。

要求：

- prompt 使用英文。
- 8 到 25 词为主，复杂 interaction 可略长。
- 明确 reference role。
- 不混入非目标维度。
- 多图 reference 指代稳定，例如 image 1 / image 2 / reference jacket。

输出 `final_prompts.jsonl`。

### 4.7 Module: `export_dataset.py`

功能：

- 汇总 QuestionPlan、InputAsset、QC、prompt。
- 导出 `BenchmarkSample` 和 `phase3_manifest.jsonl`。

`BenchmarkSample` schema：

```json
{
  "question_id": "motion_B_0042",
  "dimension": "motion_binding",
  "input_mode": "single_image",
  "subtype": "relative_displacement",
  "difficulty": "medium",
  "semantic_rarity": "common",
  "contrastive_pair_id": "motion_pair_021",
  "contrastive_role": "A",
  "input_images": [
    {
      "path": "data/benchmark_dataset/images/motion_B_0042.png",
      "role": "first_frame",
      "asset_id": "asset_001"
    }
  ],
  "i2v_prompt": "The red ball moves to the right side of the blue box.",
  "metadata": {
    "target_subjects": [
      {"id": "s1", "description": "red ball"}
    ],
    "reference_subjects": [
      {"id": "s2", "description": "blue box"}
    ],
    "expected_change": {
      "type": "relative_displacement",
      "target_relation": "right_of"
    },
    "preserve_constraints": [
      {"scope": "s2", "constraint": "appearance_and_position"},
      {"scope": "background", "constraint": "stable"}
    ],
    "evaluator_hints": {
      "tools": ["grounding", "segmentation", "tracking", "flow"],
      "primary_metric": "relative_displacement"
    }
  },
  "source_trace": {
    "recipe_id": "recipe_motion_0091",
    "source_type": "observed_single_image",
    "phase1_sample_ids": ["tip_000918"]
  },
  "qc": {
    "status": "pass",
    "risk_flags": []
  }
}
```

Multi-image supplement：

```json
{
  "input_mode": "multi_image",
  "reference_assets": [
    {"asset_id": "asset_001", "role": "target_subject"},
    {"asset_id": "asset_002", "role": "scene_reference"}
  ],
  "multi_reference_quality": {
    "crop_leakage_risk": "low",
    "scene_leakage_risk": "medium",
    "identity_visibility": "high",
    "scale_compatibility": 0.82
  }
}
```

验收：

- `phase3_manifest.jsonl` 与 samples 数量一致。
- 每条样本都有 E/P/C 所需 metadata。
- 每条 multi_image 样本都有 reference_assets 与 quality。

### 4.8 Module: `audit_phase2.py`

功能：

- 校验 benchmark_dataset/ 完整性。
- 统计每维度 / input_mode / subtype 的实际产出数量与 quota 缺口。
- 输出 `dataset_card.md` 与质检摘要。

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
