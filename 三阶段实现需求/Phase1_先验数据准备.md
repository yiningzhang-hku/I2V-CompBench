# I2V-CompBench Phase 1 实现需求（先验数据准备）

> 本文档面向 AI coding agent，是《I2V-CompBench Phase 1/2/3 实现需求（Coding Agent 版）》的 Phase 1 拆分文档。Phase 2、Phase 3 内容请参见同目录下的另外两份文档。
>
> Phase 1 的目标：把原始 TIP-I2V 数据加工成结构化的 `phase1_bundle/`，为 Phase 2 的题目合成提供：清洁清单、图像/文本结构化解析、多参考资产库、先验包以及候选 recipe 列表。

---

## 0. Phase 1 总目标

```text
Phase 1: raw TIP-I2V data -> phase1_bundle/
```

必须支持：

- 7 个评测维度：attribute, action, motion, spatial, background, view, interaction。
- 2 种输入模式：single_image, multi_image。
- 多图模式是**主维度样本**，不是 stress-only：Phase 1 必须显式构建 `reference_bank/` 与多图 recipe。
- 所有中间文件使用 JSONL/JSON，保留 provenance。
- 输出的每个 `candidate_recipe` 都必须可被 Phase 2 直接消费，不允许 Phase 2 绕过 Phase 1 自己随机拼题。

---

## 1. 推荐目录结构（Phase 1 部分）

```text
i2v_compbench/
  configs/
    phase1.yaml
    dimensions.yaml
  data/
    raw/
    phase1_bundle/
  src/
    i2vcompbench/
      phase1/
        build_manifest.py
        parse_images.py
        parse_text.py
        align_instances.py
        build_reference_bank.py
        build_priors.py
        build_recipes.py
        audit_phase1.py
      schemas/
        phase1.py
      utils/
        io.py
        ids.py
        image.py
        llm.py
        vlm.py
        geometry.py
```

如果当前项目已有目录结构（例如 `tip_i2v_data_analysis/`），优先适配现有结构，但模块边界应保持一致。

---

## 2. 全局枚举与边界规则（Phase 1 必须遵守）

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

### 2.3 Motion / Spatial 边界（Phase 1 文本解析与对齐时强制执行）

硬规则：

```text
if task_contains_subject_displacement:
    dimension = "motion_binding"
else if task_requires_static_layout_composition:
    dimension = "spatial_composition"
```

不要把"红球移动到蓝盒右边"放到 Spatial。它是 **Motion Binding Type B**。

Spatial 只处理：

- 多参考物体静态放置。
- 已有布局保持/验证。
- 生成后关系正确且稳定。

### 2.4 Action / Interaction 边界（Phase 1 文本解析时强制执行）

```text
single-subject body movement -> action_binding
multi-object causal/social/functional event -> interaction_reasoning
```

例如：

- "the woman waves" -> Action。
- "the woman hands a book to the man" -> Interaction。

---

## 3. Phase 1 输入

配置示例：

```yaml
dataset:
  name: TIP-I2V
  split: eval
  max_samples: 300
  streaming: true
output_dir: data/phase1_bundle
models:
  vlm: qwen-vl-or-compatible
  llm: qwen-or-compatible
```

输入样本至少包含：

```json
{
  "source_id": "uuid",
  "prompt": "raw prompt with optional Pika params",
  "image": "PIL image or image bytes",
  "metadata": {}
}
```

---

## 4. Phase 1 输出

```text
data/phase1_bundle/
  manifest_clean.jsonl
  images/
  image_parse.jsonl
  text_parse.jsonl
  aligned_instances.jsonl
  reference_bank/
    assets.jsonl
    crops/
    masks/
    scenes/
  prior_package.json
  compatibility_matrix.json
  transform_ontology.json
  candidate_recipes.jsonl
  phase1_audit_report.md
```

---

## 5. 模块详细规格

### 5.1 Module: `build_manifest.py`

功能：

- 读取 TIP-I2V 数据。
- 解析 Pika 参数。
- 清洗 prompt。
- 保存首帧图像。
- 输出 `manifest_clean.jsonl`。

CLI：

```bash
python -m i2vcompbench.phase1.build_manifest \
  --config configs/phase1.yaml \
  --output data/phase1_bundle/manifest_clean.jsonl
```

`ManifestSample` schema：

```json
{
  "sample_id": "tip_000001",
  "source_id": "raw_uuid",
  "clean_prompt": "a dog runs across a grassy field",
  "raw_prompt": "... -camera zoom in -motion 2",
  "image_path": "data/phase1_bundle/images/tip_000001.jpg",
  "pika_params": {
    "camera": "zoom in",
    "motion": 2,
    "negative_prompt": null,
    "seed": null,
    "fps": null,
    "aspect_ratio": null
  },
  "status": "ok",
  "errors": []
}
```

验收：

- `status=ok` 样本比例 >= 90%。
- 所有 `ok` 样本图像路径存在。
- prompt 参数剥离后不为空。

### 5.2 Module: `parse_images.py`

功能：

- 对首帧做 VLM 结构化解析。
- 可选接入 detection/segmentation 模型生成 bbox/mask。
- 输出 `image_parse.jsonl`。

`ImageParse` schema：

```json
{
  "sample_id": "tip_000001",
  "subjects": [
    {
      "instance_id": "s1",
      "category": "dog",
      "description": "small brown dog",
      "bbox": [120, 80, 340, 420],
      "mask_path": "reference_bank/masks/tip_000001_s1.png",
      "attributes": {
        "color": "brown",
        "material": null,
        "state": null,
        "wearing": []
      },
      "pose_action": "standing",
      "position_in_frame": "center_right",
      "is_animate": true,
      "visibility": "high",
      "segmentation_quality": 0.87,
      "tracking_feasibility": "good"
    }
  ],
  "subject_relations": [
    {
      "subject_a": "s1",
      "predicate": "left_of",
      "subject_b": "s2",
      "confidence": 0.82
    }
  ],
  "background": {
    "scene_type": "park",
    "elements": ["grass", "trees"],
    "lighting": "daylight",
    "weather": "clear",
    "foreground_background_separability": "high"
  },
  "camera_baseline": {
    "shot_type": "medium_shot",
    "camera_angle": "eye_level",
    "has_rigid_reference_structure": false,
    "depth_cues": "medium"
  },
  "reference_potential": {
    "suitable_for_multi_reference": true,
    "num_extractable_subjects": 2,
    "scene_extractable": true
  },
  "parse_confidence": 0.84
}
```

验收：

- 每个 `ok` manifest 样本有一条 image parse。
- 至少输出 subjects，可以为空但要有原因。
- bbox/mask 缺失时必须标记 `segmentation_quality=0` 或 `tracking_feasibility=poor`。

### 5.3 Module: `parse_text.py`

功能：

- 把 clean prompt 解析成七维度 transform slots。
- 输出 `text_parse.jsonl` 与 `transform_ontology.json`。

`TextParse` schema：

```json
{
  "sample_id": "tip_000001",
  "primary_dimension": "motion_binding",
  "candidate_dimensions": ["motion_binding"],
  "slots": {
    "attribute_change": [],
    "action": [],
    "motion": [
      {
        "target_subject": "dog",
        "direction": "left",
        "target_relation": null,
        "motion_phrase": "moves to the left"
      }
    ],
    "spatial": [],
    "background": [],
    "view": [],
    "interaction": []
  },
  "transform_ontology": {
    "type": "absolute_motion",
    "canonical_label": "move_left",
    "contrastive_transform": "move_right"
  },
  "forbidden_dimension_leakage": ["camera", "attribute", "interaction"],
  "parse_confidence": 0.86,
  "ambiguous": false
}
```

验收：

- primary_dimension 必须属于 7 维度之一或 `ambiguous`。
- motion/spatial 按边界规则分类。
- 每条可评测样本必须有 transform ontology。

### 5.4 Module: `align_instances.py`

功能：

- 将文本里的 target/reference subject 对齐到图像 `subjects[]`。
- 判断可评测性。
- 输出 `aligned_instances.jsonl`。

`AlignedSample` schema：

```json
{
  "sample_id": "tip_000001",
  "dimension": "motion_binding",
  "target_instances": ["s1"],
  "reference_instances": [],
  "preserve_instances": ["s2"],
  "alignment_confidence": 0.78,
  "evaluator_feasibility": {
    "is_evaluable": true,
    "required_tools": ["grounding", "segmentation", "tracking", "flow"],
    "risk_flags": ["small_target"],
    "reason": null
  },
  "dimension_isolation": {
    "suppressed_dimensions": ["attribute_binding", "view_transformation"],
    "forbidden_words": ["pan", "zoom", "turns red"]
  }
}
```

验收：

- `is_evaluable=true` 的样本必须有 target_instances。
- `alignment_confidence < threshold` 不进入 candidate_recipes。

### 5.5 Module: `build_reference_bank.py`

功能：

- 从单首帧提取多图/多参考资产。
- 输出 `reference_bank/assets.jsonl` 与图像文件。

资产类型：

- `subject_reference`
- `attribute_reference`
- `object_reference`
- `scene_reference_original`
- `scene_reference_inpainted`

`ReferenceAsset` schema：

```json
{
  "ref_id": "tip_000001_s1_masked",
  "source_sample_id": "tip_000001",
  "source_instance_id": "s1",
  "role": "subject_reference",
  "category": "dog",
  "path": "data/phase1_bundle/reference_bank/crops/tip_000001_s1_masked.png",
  "mask_path": "data/phase1_bundle/reference_bank/masks/tip_000001_s1.png",
  "bbox": [120, 80, 340, 420],
  "quality": {
    "mask_quality": 0.88,
    "identity_visibility": "high",
    "background_leakage_risk": "low",
    "occlusion_level": "low",
    "resolution_ok": true
  },
  "visual_properties": {
    "dominant_colors": ["brown"],
    "view_angle": "side",
    "lighting": "daylight"
  },
  "usable_for": [
    "attribute_binding",
    "action_binding",
    "motion_binding",
    "spatial_composition",
    "interaction_reasoning"
  ],
  "provenance": {
    "source": "tip_i2v_derived",
    "sample_id": "tip_000001"
  }
}
```

验收：

- 对 `suitable_for_multi_reference=true` 样本，至少提取 1 个 subject_reference。
- 每个 asset 必须有 quality 与 provenance。
- 高污染 asset 不能进入主采样，除非标记为 hard/risk 样本。

### 5.6 Module: `build_priors.py`

功能：

- 聚合 concept distributions。
- 生成 common/rare tiers。
- 构建 compatibility matrix。
- 输出 `prior_package.json` 与 `compatibility_matrix.json`。

必须包含：

```json
{
  "global_distributions": {},
  "dimension_priors": {},
  "frequency_tiers": {},
  "subject_pair_distribution": {},
  "subject_scene_cooccurrence": {},
  "subject_action_cooccurrence": {},
  "subject_attribute_cooccurrence": {},
  "multi_reference_priors": {},
  "dimension_cooccurrence": {},
  "evaluator_feasibility_stats": {}
}
```

验收：

- 每个维度都有 concept distributions。
- common/rare 可直接被 Phase 2 使用。
- multi_reference_priors 统计 reference_bank 的主体、场景、关系和质量分布。

### 5.7 Module: `build_recipes.py`

功能：

- 从 aligned samples、reference_bank 和 priors 构建可采样 recipe。
- 输出 `candidate_recipes.jsonl`。

`CandidateRecipe` schema：

```json
{
  "recipe_id": "recipe_motion_000091",
  "dimension": "motion_binding",
  "input_mode": "single_image",
  "subtype": "relative_displacement",
  "source_type": "observed_single_image",
  "source_sample_ids": ["tip_000918"],
  "difficulty_factors": {
    "binding_load": 1,
    "discrimination_difficulty": "medium",
    "perceptual_complexity": "medium"
  },
  "semantic_rarity": "common",
  "slots": {
    "target_subject": "red ball",
    "reference_subject": "blue box",
    "target_relation": "right_of"
  },
  "reference_requirements": [],
  "preserve_constraints": [
    {"scope": "reference_subject", "constraint": "appearance_and_position"},
    {"scope": "background", "constraint": "stable"}
  ],
  "contrastive": {
    "pairable": true,
    "opposite_transform": {"target_relation": "left_of"}
  },
  "evaluator_requirements": {
    "tools": ["grounding", "segmentation", "tracking", "flow"],
    "min_target_size": 0.03
  },
  "quality_flags": []
}
```

Multi-image recipe requirements：

```json
{
  "input_mode": "multi_image",
  "source_type": "derived_multi_reference",
  "reference_requirements": [
    {"role": "target_subject", "category": "person"},
    {"role": "attribute_reference", "category": "jacket"},
    {"role": "scene_reference", "category": "cafe"}
  ],
  "multi_reference_constraints": {
    "max_crop_leakage_risk": "medium",
    "min_identity_visibility": "medium",
    "require_scene_capacity": true
  }
}
```

验收：

- 每个维度至少生成 pilot 所需 recipe 数的 3 倍候选。
- `source_type=derived_multi_reference` 必须可追溯到 reference_bank。
- Motion Type B 必须属于 motion_binding。
- Spatial 多图 recipe 必须是 static layout，不含 `move/shift/reposition`。

### 5.8 Module: `audit_phase1.py`

功能：

- 校验 phase1_bundle/ 完整性。
- 统计每维度可用 recipe 数与 reference_bank 资产数。
- 输出 `phase1_audit_report.md`，作为进入 Phase 2 的门禁。

---

## 6. Phase 1 CLI 总流程

```bash
python -m i2vcompbench.phase1.build_manifest --config configs/phase1.yaml
python -m i2vcompbench.phase1.parse_images --config configs/phase1.yaml
python -m i2vcompbench.phase1.parse_text --config configs/phase1.yaml
python -m i2vcompbench.phase1.align_instances --config configs/phase1.yaml
python -m i2vcompbench.phase1.build_reference_bank --config configs/phase1.yaml
python -m i2vcompbench.phase1.build_priors --config configs/phase1.yaml
python -m i2vcompbench.phase1.build_recipes --config configs/phase1.yaml
python -m i2vcompbench.phase1.audit_phase1 --bundle data/phase1_bundle
```

可以先实现一个 unified CLI：

```bash
i2vcompbench phase1 --config configs/phase1.yaml
```

---

## 7. Phase 1 Pilot 实现优先级

### P0：先跑通闭环

1. 定义 Pydantic/dataclass schemas（`ManifestSample`、`ImageParse`、`TextParse`、`ReferenceAsset`、`CandidateRecipe`）。
2. 实现 manifest/text_parse/image_parse 的 mock 或轻量版本。
3. 实现 reference_bank 最小版：主体 crop + asset manifest。
4. 实现 candidate_recipes 生成（覆盖 7 个维度，每维度至少 3×pilot 数候选）。

### P1：增强多图主维度

1. 多图 reference quality 检查。
2. 高质量主体 crop / 场景 inpainting。
3. multi_reference_priors 统计完整。

### P2：可信度增强

1. evaluator_feasibility_stats 全量输出。
2. compatibility_matrix 校验。
3. audit_phase1 自动报告论文图表所需统计。

---

## 8. Phase 1 最小验收标准

- `phase1_bundle/` 所有规定文件存在。
- `candidate_recipes.jsonl` 覆盖 7 个维度。
- multi_image recipe 可追溯到 reference_bank。
- Motion Type B 被归入 motion_binding。
- 每个 reference asset 有 quality 与 provenance。

---

## 9. Phase 1 常见错误清单

1. 不要把多图样本标成 stress-only；当前主线要求多图进入主维度。
2. 不要把相对位移任务放进 Spatial。只要移动，就是 Motion。
3. 不要让多图 reference 缺少 provenance，否则论文无法证明多图先验来源。
4. 不要把 Action 与 Interaction 混并；单主体肢体动作归 Action，多主体因果/社交/功能事件归 Interaction。
5. 不要在 align_instances 阶段静默丢弃 `alignment_confidence` 偏低样本，需要写入审计日志。
6. 不要让 `build_recipes` 跳过 reference_bank 直接从 image_parse 凭空生成 multi-image recipe。

---

## 10. Coding Agent 的第一步任务建议（Phase 1）

1. 创建 schemas：`ManifestSample`、`ImageParse`、`TextParse`、`AlignedSample`、`ReferenceAsset`、`CandidateRecipe`。
2. 写 JSONL read/write 工具和 schema validation。
3. 用 20 条 mock/TIP 样本跑 Phase 1，生成 reference_bank 和 candidate_recipes。
4. 跑 `audit_phase1`，确认 7 个维度都有可用 recipe。

完成 Phase 1 闭环后，再进入 Phase 2（详见 `Phase2_Benchmark数据集合成.md`）。
