# Phase 1：先验数据准备

## 1. 定位与约束

Phase 1 接收 Step 1–5 产出的结构化解析（`image_parse.jsonl`、`text_parse.jsonl`、`manifest_clean.jsonl`），在**不调用 VLM / LLM** 的前提下，通过规则推导和启发式计算，补齐评测所需的缺失字段，最终输出 `candidate_recipes.jsonl` 供 Phase 2 直接消费。

核心约束：零 API 调用、纯确定性规则、幂等可重跑。

---

## 2. 工程结构

```
tip_i2v_data_analysis/
├── configs/config.yaml
├── main.py                            # CLI 入口，统一调度
└── src/
    ├── phase1/
    │   ├── patch_existing_outputs.py  # P1.1 字段补齐
    │   ├── mock_geometry.py           # 几何工具库（非独立步骤）
    │   ├── align_instances.py         # P1.2 图文对齐
    │   ├── reference_bank.py          # P1.3 素材抽取
    │   ├── priors_enhance.py          # P1.4 先验增强
    │   ├── recipes.py                 # P1.5 配方生成
    │   └── audit.py                   # P1.6 审计报告
    └── utils/
        ├── schema.py                  # 基础数据结构定义
        ├── schema_phase1.py           # Phase 1 专属结构定义
        └── io_utils.py                # JSONL/JSON/YAML 读写
```

---

## 3. 模块调用关系

| 步骤 | 主脚本 | 依赖的内部模块（函数级） | 读取 | 写入 |
|------|--------|------------------------|------|------|
| patch | `patch_existing_outputs.py` | `mock_geometry.position_to_bbox`、`mock_geometry.estimate_tracking_feasibility`、`io_utils.read_jsonl`、`io_utils.write_jsonl` | image_parse.jsonl、text_parse.jsonl | image_parse_v2.jsonl、text_parse_v2.jsonl |
| align | `align_instances.py` | `io_utils.read_jsonl`、`io_utils.write_jsonl`、`io_utils.ensure_dir` | image_parse_v2.jsonl、text_parse_v2.jsonl | aligned_instances.jsonl |
| refbank | `reference_bank.py` | `mock_geometry.crop_to_path`、`mock_geometry.position_to_bbox`、`io_utils.read_jsonl`、`io_utils.write_jsonl`、`io_utils.ensure_dir` | image_parse_v2.jsonl、manifest_clean.jsonl | assets.jsonl、reference_bank/\*.jpg |
| priors2 | `priors_enhance.py` | `io_utils.read_jsonl`、`io_utils.ensure_dir` | aligned_instances.jsonl、text_parse_v2.jsonl、assets.jsonl | priors/frequency_tiers.json、priors/subject_pair_distribution.json、priors/multi_reference_priors.json、priors/compatibility_matrix.json |
| recipes | `recipes.py` | `io_utils.read_jsonl`、`io_utils.write_jsonl`、`io_utils.ensure_dir` | aligned_instances.jsonl、assets.jsonl、text_parse_v2.jsonl、manifest_clean.jsonl | candidate_recipes.jsonl |
| audit | `audit.py` | `io_utils.read_jsonl`、`io_utils.ensure_dir` | aligned_instances.jsonl、assets.jsonl、candidate_recipes.jsonl、text_parse_v2.jsonl、image_parse_v2.jsonl | phase1_audit_report.md |

`main.py` 通过 `--step` 参数动态 import 对应入口函数：

```python
# --step patch  → patch_existing_outputs.patch_outputs(config)
# --step align  → align_instances.align_instances(config)
# --step refbank → reference_bank.build_reference_bank(config)
# --step priors2 → priors_enhance.enhance_priors(config)
# --step recipes → recipes.build_recipes(config)
# --step audit  → audit.audit(config)
# --step phase1 → 按上述顺序依次执行
```

---

## 4. 七维度枚举定义

| 维度枚举值 | 评测目标 | 典型场景 |
|-----------|---------|---------|
| `attribute_binding` | 主体外观属性发生指定变化 | 红球变蓝 |
| `action_binding` | 单主体执行指定肢体动作 | 人在挥手 |
| `motion_binding` | 主体产生指定方向/路径的位移 | 球向左运动 |
| `spatial_composition` | 多物体静态空间关系的正确性 | A 在 B 右侧 |
| `background_dynamics` | 背景/场景发生指定变化 | 晴天转为下雨 |
| `view_transformation` | 镜头运动或视角变化 | 推镜头 |
| `interaction_reasoning` | 多主体因果/协作事件 | A 递物品给 B |

判别边界：
- 含位移过程 → `motion_binding`（即使终态涉及空间关系）。`spatial_composition` 仅评判静态帧中关系是否正确。
- 单主体动作 → `action_binding`；多主体协同 → `interaction_reasoning`。

---

## 5. 各模块技术原理

### 5.1 patch_existing_outputs — 字段补齐

**目标**：为 image_parse / text_parse 每行数据补齐 Phase 1 后续模块必需但上游未产出的字段。

#### 5.1.1 图像侧补齐（`patch_image_parse`）

对每行 `image_parse.jsonl`（`parse_success=true`），遍历其 `subjects[]` 数组，逐主体调用 `_patch_image_subject`：

| 补齐字段 | 计算方式 |
|---------|---------|
| `instance_id` | 取原始 `id` 字段；若为空则回退到 `"subj_unknown"` |
| `bbox` | 调用 `mock_geometry.position_to_bbox(position_in_frame)` → 归一化四元组 `[x0, y0, x1, y1]` |
| `segmentation_quality` | P0 阶段固定为 `"low"`（无真实分割） |
| `tracking_feasibility` | 调用 `mock_geometry.estimate_tracking_feasibility(bbox, is_animate)` |

补齐主体后，对整个样本推断 `reference_potential`（函数 `_infer_reference_potential`）：

```
reference_potential = {
    subject:                   有主体 ∧ foreground_background_separability ∈ {high, medium}
    attribute:                 任一主体的 attributes.{color|material_texture|state|wearing} 非空
    object:                    存在 is_animate=false 的主体
    scene_reference_original:  rigid_background_ratio ∈ {high, medium}
    scene_reference_inpainted: 恒为 false（P0 不做 inpainting）
}
```

#### 5.1.2 文本侧补齐（`patch_text_parse`）

对每行 `text_parse.jsonl`（`parse_success=true`），调用 `_patch_text_row`，补齐以下字段：

**维度路由算法**（`_infer_dimension_routing`）

输入：该行的 6 个布尔字段 `involves_attribute_change / involves_action / involves_directed_motion / involves_spatial_relation_change / involves_background_change / involves_camera_movement`，以及 `action_slots[]`、`spatial_relation_slots[]`。

处理流程：
1. 将每个为 `true` 的布尔字段映射为对应维度名，追加到 `candidates[]` 列表。
2. 交互推断：提取 `action_slots` 中所有不同的 `target_subject`（小写去重）。若去重后集合 `|distinct_actors| ≥ 2` 且 `spatial_relation_slots` 非空，将 `"interaction_reasoning"` 插入到 `candidates[0]`。
3. `primary_dimension = candidates[0]`（优先级隐含在插入顺序中）。
4. 查 `LEAKAGE_TABLE[primary]` 获取 `forbidden_dimension_leakage`，并排除 primary 本身。

**优先级序**（隐含于代码顺序）：
```
interaction_reasoning（动态插入到索引 0）
> attribute_binding > action_binding > motion_binding
> spatial_composition > background_dynamics > view_transformation
```

**防泄漏映射表**（`LEAKAGE_TABLE`，硬编码）：

```python
LEAKAGE_TABLE = {
    "attribute_binding":     ["motion_binding", "view_transformation"],
    "action_binding":        ["motion_binding", "view_transformation"],
    "motion_binding":        ["view_transformation", "action_binding"],
    "spatial_composition":   ["motion_binding", "view_transformation"],
    "background_dynamics":   ["view_transformation"],
    "view_transformation":   ["motion_binding"],
    "interaction_reasoning": ["motion_binding", "view_transformation"],
}
```

语义：若主维度为 X，则 `LEAKAGE_TABLE[X]` 中的维度不得出现在生成的 prompt 中，否则会导致评测指标无法隔离归因。

**交互槽位推断**（`_infer_interaction_slots`）

条件：`spatial_relation_slots` 和 `action_slots` 均非空。取第一条 spatial slot 的 `subject_a` / `subject_b`（要求 a ≠ b），构造一条：
```json
{
    "interaction_type": "contact_event",
    "agent_subject": subject_a,
    "patient_subject": subject_b,
    "relation_phrase": target_predicate,
    "expected_outcome": ""
}
```

若条件不满足则 `interaction_slots = []`，`involves_interaction = false`。

**附加补齐字段**：
- `transform_ontology_confidence`：有 primary 时取 0.6，否则 0.0
- `routing_reason`：固定 `"heuristic_from_involves_flags"`

---

### 5.2 mock_geometry — 空间量化工具库

本模块不是独立步骤，而是被 patch 和 refbank 调用的工具集。

#### 5.2.1 九宫格→bbox 映射（`position_to_bbox`）

将画面归一化坐标系（原点左上，x 轴向右，y 轴向下）等分为 3×3 网格，每个位置标签对应一个预定义的归一化矩形：

```
                   x=0.0      x=0.30/0.40   x=0.60/0.70    x=1.0
              y=0.0 ┌──────────┬──────────────┬──────────────┐
                    │upper-left│     top      │ upper-right  │
              y=0.40├──────────┼──────────────┼──────────────┤
                    │center-left│   center    │ center-right │
              y=0.70├──────────┼──────────────┼──────────────┤
                    │lower-left│   bottom     │ lower-right  │
              y=1.0 └──────────┴──────────────┴──────────────┘
```

精确映射值：

| 位置标签 | bbox (x0, y0, x1, y1) | 面积占比 |
|---------|----------------------|---------|
| upper-left | (0.00, 0.00, 0.40, 0.40) | 16% |
| top | (0.30, 0.00, 0.70, 0.40) | 16% |
| upper-right | (0.60, 0.00, 1.00, 0.40) | 16% |
| center-left | (0.00, 0.30, 0.40, 0.70) | 16% |
| center | (0.25, 0.25, 0.75, 0.75) | 25% |
| center-right | (0.60, 0.30, 1.00, 0.70) | 16% |
| lower-left | (0.00, 0.60, 0.40, 1.00) | 16% |
| bottom | (0.30, 0.60, 0.70, 1.00) | 16% |
| lower-right | (0.60, 0.60, 1.00, 1.00) | 16% |
| left（别名） | → center-left | 16% |
| right（别名） | → center-right | 16% |
| 无法识别 / null | (0.20, 0.20, 0.80, 0.80) | 36% |

预处理：`strip().lower().replace("_", "-")`。

#### 5.2.2 跟踪可行性估计（`estimate_tracking_feasibility`）

基于 bbox 面积的分段函数：

```
area = (x1 - x0) × (y1 - y0)

f(area, is_animate) =
    "infeasible"  if area < 0.02
    "hard"        if area < 0.06
    "medium"      if area < 0.20 ∧ is_animate = true
    "easy"        if area < 0.20 ∧ is_animate = false
    "easy"        if area ≥ 0.20
```

设计依据：面积 <2% 的目标在视频中难以稳定跟踪；活体目标（人、动物）存在非刚体形变，同面积下比非活体（家具、建筑）更难跟踪。

#### 5.2.3 裁剪函数（`crop_to_path`）

```python
def crop_to_path(image_path, bbox_norm, output_path, pad_ratio=0.05):
    # 1. 外扩 bbox：每条边向外延伸 bbox宽/高 × pad_ratio
    #    x0' = max(0, x0 - bw×pad_ratio)
    #    y0' = max(0, y0 - bh×pad_ratio)
    #    x1' = min(1, x1 + bw×pad_ratio)
    #    y1' = min(1, y1 + bh×pad_ratio)
    # 2. 归一化→像素：round(coord × dimension)，clamp 到 [0, W] / [0, H]
    # 3. PIL.Image.crop → 保存为 JPEG quality=92
    # 返回裁剪后 (W, H)；图不存在或无 PIL 则返回 None
```

#### 5.2.4 归一化→像素转换（`normalized_to_pixel`）

```
pixel_x0 = max(0, round(x0 × W))
pixel_y0 = max(0, round(y0 × H))
pixel_x1 = min(W, round(x1 × W))
pixel_y1 = min(H, round(y1 × H))
```

---

### 5.3 align_instances — 图文实例对齐与可评测性诊断

**目标**：建立图像主体（VLM 产出）与文本目标主体（LLM 提取的语义槽位中的 target_subject）之间的映射关系，并诊断每个维度在该样本上是否可评测。

#### 5.3.1 文本目标主体收集（`_collect_text_target_subjects`）

从 text_parse_v2 的以下槽位列表中提取所有主体名称（`_normalize_name` = strip + lower）：

| 来源槽位 | 提取字段 |
|---------|---------|
| `attribute_change_slots[]` | target_subject |
| `action_slots[]` | target_subject |
| `motion_slots[]` | target_subject |
| `spatial_relation_slots[]` | subject_a, subject_b |
| `interaction_slots[]` | agent_subject, patient_subject |

去重但保持出现顺序（`seen` 集合 + 列表追加）。

#### 5.3.2 对齐算法（`_align_one_sample`）

对每个图像主体 `s`，遍历文本目标列表 `text_targets`，按以下规则匹配：

```python
for t in text_targets:
    if t == s_name:               # 完全匹配
        confidence = 1.0; break
    elif t in s_desc or s_name in t:  # 子串包含
        confidence = 0.7; break
else:
    confidence = 0.0              # 无匹配
```

匹配策略为贪心（每个图像主体取第一个命中的文本主体），时间复杂度 O(|image_subjects| × |text_targets|)。

`overlap_ratio` = |已匹配文本主体| ÷ |文本目标总数|，反映图文对齐完整度。

#### 5.3.3 七维可评测性诊断（`_diagnose_feasibility`）

对 7 个维度，分别检查两个布尔条件：

| 维度 | text_request 条件 | image_support 条件 |
|------|-----------------|-------------------|
| attribute_binding | `involves_attribute_change` | subjects[] 非空 |
| action_binding | `involves_action` | 存在 is_animate=true 的主体 |
| motion_binding | `involves_directed_motion` | subjects[] 非空 |
| spatial_composition | `involves_spatial_relation_change` | has_multiple_subjects ∧ subjects_clearly_distinguishable |
| background_dynamics | `involves_background_change` | foreground_background_separability ∈ {high, medium} |
| view_transformation | `involves_camera_movement` | has_rigid_reference_structure |
| interaction_reasoning | `involves_interaction` ∨ interaction_slots非空 | has_multiple_subjects ∧ subjects_clearly_distinguishable |

tool_status 推导逻辑：

```
feasible = text_request ∧ image_support

if feasible ∧ ∃aligned_subject.segmentation_quality ∈ {high, medium}:
    tool_status = "valid"
elif feasible:
    tool_status = "low_confidence"   # 仅有 mock 几何
elif text_request ∧ ¬image_support:
    tool_status = "invalid_input"    # 图像不满足条件
elif ¬text_request ∧ image_support:
    tool_status = "tool_uncertain"   # 文本未请求该维度
else:
    tool_status = "invalid_input"    # 两侧均不支持
```

#### 5.3.4 隔离约束构建（`_build_isolation`）

直接继承 text_parse_v2 的 `primary_dimension` 和 `forbidden_dimension_leakage`，不做独立推导。

---

### 5.4 reference_bank — 参考素材提取

**目标**：从首帧图像中抽取多图评测模式所需的参考资产（主体裁剪图、场景全图等）。

#### 5.4.1 抽取判定

遍历 `image_parse_v2` 每个样本，根据 `reference_potential` 字典决定是否抽取以下类型：

| 资产类型 | 抽取条件 | 处理方式 | 适用维度 |
|---------|---------|---------|---------|
| subject | `reference_potential.subject = true` | 对每个主体按 bbox 裁剪（5%外扩） | attribute / action / motion / spatial / interaction |
| attribute | `reference_potential.attribute = true` | 同 subject（语义标签侧重属性值） | attribute |
| object | `reference_potential.object = true` ∧ 该主体 is_animate=false | 仅裁剪非活体物体 | spatial / interaction |
| scene_reference_original | `reference_potential.scene_reference_original = true` | 原图直接复用（不裁剪） | background / view |
| scene_reference_inpainted | — | P0 跳过 | — |

#### 5.4.2 asset_id 命名规则

- 主体类资产：`{sample_id}_{instance_id}_{suffix}`，suffix ∈ {subj, attr, obj}
- 场景类资产：`{sample_id}_scene_orig`

#### 5.4.3 质量评分（quality_score）

```python
bw = bbox[2] - bbox[0]   # bbox 归一化宽度
bh = bbox[3] - bbox[1]   # bbox 归一化高度
quality_score = clamp(0.5 + min(bw, bh), 0.0, 1.0)
```

设计意图：bbox 越大，裁剪后分辨率越高，质量越好。min(bw,bh) 确保窄长条形区域不会获得高分。

#### 5.4.4 provenance 结构

每个资产必须携带来源追溯信息：

```json
{
    "source_sample_id": "tip_000123",
    "source_image_path": "images/tip_000123.jpg",
    "extraction_method": "mock_crop_9grid",      // 或 "original"
    "extraction_params": {"bbox_norm": [...], "pad_ratio": 0.05},
    "created_by": "phase1.reference_bank",
    "created_at": "2025-06-12T10:30:00"
}
```

---

### 5.5 priors_enhance — 统计先验增强

**目标**：为 Phase 2 采样策略提供频率分布、共现信息、多图可行性统计和维度兼容性约束。

#### 5.5.1 frequency_tiers.json — 概念频率三档

**算法**（`_split_tiers`）：

1. 统计 `text_parse_v2` 中所有 `attribute_change_slots[].target_subject`、`action_slots[].target_subject` 的频次 → `subj_counter`
2. 统计 `action_slots[].action_verb` 的频次 → `verb_counter`
3. 统计 `attribute_change_slots[]` 中 `(attribute_type, to_value)` 对的频次 → `attr_to_value_counter`

对每个 Counter，按频次降序排列，计算累计频率 \(\text{cum}_i = \sum_{j=1}^{i} \frac{count_j}{\text{total}}\)：

```
head:      cum ≤ 0.50  （前 50% 累计频率内的高频概念）
torso:     0.50 < cum ≤ 0.85
long_tail: cum > 0.85
```

Phase 2 用途：控制 80/20 的 common/rare 采样比例，避免 benchmark 全集中在高频概念上。

#### 5.5.2 subject_pair_distribution.json — 多主体共现频谱

**算法**（`build_subject_pair_distribution`）：

1. 对 `aligned_instances` 每行，收集所有 `aligned_subjects[].image_subject_name`（去重、排序）。
2. 对排序后的名称列表做 \(\binom{n}{2}\) 组合，每对记一次频次。
3. 输出 top-200 共现对（按频次降序）。

Phase 2 用途：组合多图题时参考实际共现分布，避免拼出训练集中从未出现的组合。

#### 5.5.3 multi_reference_priors.json — 多图模式可行性

统计指标：
- `multi_eligible_samples`：`|aligned_subjects| ≥ 2` 的样本数
- `multi_eligible_ratio`：multi_eligible / total
- `asset_type_distribution`：按 `asset_type` 的 Counter
- `avg_assets_per_sample`：所有有资产的样本的平均资产数
- `samples_with_any_asset`：至少有一个资产的样本数

#### 5.5.4 compatibility_matrix.json — 7×7 维度兼容矩阵

硬编码三级关系标签：

| 标签 | 语义 |
|------|------|
| `"self"` | 对角线 |
| `"compatible"` | 可同时出现于一条 recipe |
| `"leakage_risk"` | 容易混淆主维度归因，需列入 forbidden |
| `"incompatible"` | 绝不允许同时作为主维度+附加维度 |

矩阵对称填充：`matrix[d1][d2] = COMPAT_MATRIX.get(d1,{}).get(d2) or COMPAT_MATRIX.get(d2,{}).get(d1) or "compatible"`。

---

### 5.6 recipes — 候选配方生成

**目标**：将分散在 aligned_instances、assets、text_parse_v2 中的信息整合为结构化的 `CandidateRecipe`，作为 Phase 2 的唯一题目来源。

#### 5.6.1 生成准入条件

```python
# 跳过条件（不生成 recipe）：
if primary_dimension is None:           skip  (计数: skipped_no_primary)
if not feasible:                        skip  (计数: skipped_infeasible)
if tool_status == "invalid_input":      skip
```

仅为 `feasible=true` 且 `tool_status ∈ {valid, low_confidence}` 的样本生成 recipe。

#### 5.6.2 input_mode 与 source_type 决策（`_decide_input_mode_and_source`）

```
令 subj_assets = 该样本的 asset_type="subject" 的资产列表
令 aligned_count = |aligned_subjects|

if primary ∈ {view_transformation, background_dynamics}:
    → single_image + observed_single_image

elif primary ∈ {spatial_composition, interaction_reasoning}:
    if |subj_assets| ≥ 2 ∧ aligned_count ≥ 2:
        → multi_image + derived_multi_reference
    else:
        → single_image + observed_single_image

else:  // attribute / action / motion
    if |subj_assets| ≥ 2:
        → multi_image + derived_single_image
    elif |subj_assets| ≥ 1:
        → single_image + derived_single_image
    else:
        → single_image + observed_single_image
```

#### 5.6.3 preserve_constraints

按 `primary_dimension` 查硬编码表 `PRESERVE_DEFAULTS`，指定该维度评测时必须保持不变的方面：

| 主维度 | 需保持不变的方面 |
|-------|----------------|
| attribute_binding | identity、scene_context、camera_framing |
| action_binding | identity、scene_context、camera_framing |
| motion_binding | identity、scene_context、camera_framing |
| spatial_composition | identity、attribute_color、camera_framing |
| background_dynamics | identity、camera_framing |
| view_transformation | scene_context、spatial_position |
| interaction_reasoning | identity、scene_context、camera_framing |

#### 5.6.4 contrastive_spec

固定 4 种对照 baseline（所有 recipe 相同）：
1. `static_copy`：首帧静态复制
2. `random_motion`：随机运动
3. `global_filter`：全局滤镜（不产生主维度变化）
4. `camera_pan_cheat`：用运镜伪造主体运动

#### 5.6.5 expected_difficulty

```
tool_status = "valid"          → "medium"
tool_status = "low_confidence" → "hard"
其他                           → "hard"
```

#### 5.6.6 base_prompt_draft

格式：`"[dim={primary_dimension}] {clean_prompt_text}"`

---

### 5.7 audit — 审计报告生成

**目标**：Phase 1 → Phase 2 的门禁检查。验证数据完整性、维度覆盖均匀性、provenance 完备性和隔离约束合规性。

#### 5.7.1 报告结构（7 节）

| 章节 | 检查内容 |
|------|---------|
| §1 样本规模 | text/image/aligned/assets/recipes 各多少行 |
| §2 主维度路由分布 | 7 维度 + null 的计数与百分比 |
| §3 七维可评测性 | 每个维度的 feasible率、valid / low_confidence / tool_uncertain / invalid_input 四态计数 |
| §4 资产与 provenance | 各类资产计数 + provenance 非空率（期望 100%） |
| §5 Recipe 分布 | 按 target_dimension / input_mode / source_type 分别统计 |
| §6 隔离有效性 | 检查 recipe 中 `primary_dimension ∈ forbidden_dimensions` 的违规条数（期望 0） |
| §7 验收结论 | 逐条 pass/fail 判定 |

#### 5.7.2 隔离违规检测算法

```python
for recipe in recipes:
    iso = recipe["dimension_isolation"]
    if iso["primary_dimension"] in iso.get("forbidden_dimensions", []):
        isolation_violations += 1
```

任何违规表明 `LEAKAGE_TABLE` 推导逻辑有 bug。

---

## 6. 全部产出文件结构

```
{output_dir}/
├── image_analysis/
│   ├── image_parse.jsonl            # Step 2 原始（不改动）
│   └── image_parse_v2.jsonl         # patch 后
├── text_analysis/
│   ├── text_parse.jsonl             # Step 3 原始（不改动）
│   └── text_parse_v2.jsonl          # patch 后
└── phase1/
    ├── aligned_instances.jsonl      # 图文对齐 + 七维诊断
    ├── assets.jsonl                 # 资产元数据
    ├── reference_bank/              # 资产图像
    │   ├── subject/
    │   ├── attribute/
    │   ├── object/
    │   └── scene_reference_original/
    ├── priors/
    │   ├── frequency_tiers.json
    │   ├── subject_pair_distribution.json
    │   ├── multi_reference_priors.json
    │   └── compatibility_matrix.json
    ├── candidate_recipes.jsonl      # Phase 2 唯一入口
    └── phase1_audit_report.md       # 审计报告
```

---

## 7. 核心数据结构

### AlignedSample（aligned_instances.jsonl 每行）

```python
{
    "sample_id": str,
    "aligned_subjects": [
        {
            "instance_id": str,
            "image_subject_id": str,
            "image_subject_name": str,
            "text_subject_name": Optional[str],
            "alignment_confidence": float,  # 1.0 | 0.7 | 0.0
            "geometry": {
                "instance_id": str,
                "bbox": [x0, y0, x1, y1],
                "mask_path": Optional[str],
                "segmentation_quality": str,
                "tracking_feasibility": str
            }
        }
    ],
    "unmatched_text_subjects": [str],
    "unmatched_image_subjects": [str],
    "overlap_ratio": float,
    "evaluator_feasibility": [
        {
            "dimension": str,
            "feasible": bool,
            "tool_status": str,        # valid | low_confidence | tool_uncertain | invalid_input
            "reasons": [str],
            "required_tools": [str]
        }
    ],  # 共 7 条
    "dimension_isolation": [
        {
            "primary_dimension": str,
            "forbidden_dimensions": [str],
            "leakage_risk_notes": str
        }
    ]
}
```

### ReferenceAsset（assets.jsonl 每行）

```python
{
    "asset_id": str,
    "asset_type": str,              # subject | attribute | object | scene_reference_original
    "asset_path": str,
    "semantic_tags": [str],
    "usable_for": [str],            # 适用维度列表
    "quality": {
        "resolution": [W, H],       # 像素尺寸（scene_orig 为 null）
        "aspect_ratio": float,
        "is_clean_background": bool,
        "has_visible_subject": bool,
        "quality_score": float,
        "quality_flags": [str]
    },
    "provenance": {
        "source_sample_id": str,
        "source_image_path": str,
        "extraction_method": str,   # "mock_crop_9grid" | "original"
        "extraction_params": dict,
        "created_by": str,
        "created_at": str           # ISO 8601
    }
}
```

### CandidateRecipe（candidate_recipes.jsonl 每行）

```python
{
    "recipe_id": str,                   # "recipe_{sample_id}_{dimension}"
    "source_sample_id": str,
    "target_dimension": str,
    "input_mode": str,                  # single_image | multi_image
    "source_type": str,                 # observed_single_image | derived_single_image | derived_multi_reference
    "reference_asset_ids": [str],
    "base_prompt_draft": str,           # "[dim=xxx] {clean_prompt}"
    "preserve_constraints": [{"target": str, "aspect": str, "note": str}],
    "contrastive_spec": [{"contrast_type": str, "description": str}],
    "dimension_isolation": {
        "primary_dimension": str,
        "forbidden_dimensions": [str],
        "leakage_risk_notes": str
    },
    "expected_difficulty": str,         # medium | hard
    "notes": str
}
```

---

## 8. 配置项

`configs/config.yaml` 中 Phase 1 相关参数：

```yaml
paths:
  raw_dir: "E:/I2V-CompBench/raw"
  manifest_dir: "E:/I2V-CompBench/manifest"
  output_dir: "E:/I2V-CompBench/outputs"

phase1:
  geometry_mode: "mock_9grid"       # 未来可切换为 "sam"
  crop_pad_ratio: 0.05             # 裁剪外扩比例
  max_assets_per_sample: 10        # 单样本最大资产数
  recipe:
    enable_multi_image: true       # 允许生成 multi_image recipe
    min_subjects_for_multi: 2      # 多图模式所需最小对齐主体数
```

---

## 9. 运行方式

```bash
# 单步执行
python main.py --step patch    --config configs/config.yaml
python main.py --step align    --config configs/config.yaml
python main.py --step refbank  --config configs/config.yaml
python main.py --step priors2  --config configs/config.yaml
python main.py --step recipes  --config configs/config.yaml
python main.py --step audit    --config configs/config.yaml

# 全量执行
python main.py --step phase1 --config configs/config.yaml
```

---

## 10. 实现状态与扩展计划

**P0 已完成**：六个增强步骤全部实现并集成到 CLI，`--step phase1` 可一键顺序执行。

**后续扩展方向**：
- 接入 SAM / Grounding-DINO 替换 mock_9grid，获取像素级 mask 和精确 bbox
- 实现 `scene_reference_inpainted`（主体抹除后的纯背景图）
- 基于实际运行数据动态更新 compatibility_matrix
- evaluator_feasibility 的 image_support 判定改为对接 Step 4 真实指标
