# Phase 1：先验数据准备

Phase 1 把 TIP-I2V 数据集已有的"首帧图 + prompt"结构化解析结果，加工成 Phase 2 能直接使用的候选配方。全程不调 VLM / LLM，只做规则推导和几何近似。最终交付一份 `candidate_recipes.jsonl`，Phase 2 只从这个文件里出题。

---

## 背景：Phase 1 在接手什么

Step 1–5 已经完成了原始数据的基础解析：

- Step 1 下载首帧、清洗 prompt、生成 `manifest_clean.jsonl`
- Step 2 用 VLM 分析每张首帧，识别出主体列表、空间关系、背景、镜头基线 → `image_parse.jsonl`
- Step 3 用 LLM 分析每条 prompt，判断涉及哪些变化类型并抽取语义槽位 → `text_parse.jsonl`
- Step 4–5 做交叉验证和统计汇总 → `prior_package.json`

这些产物有几个明显缺口：

1. 不知道每条样本该归哪个评测维度（没有维度路由）
2. VLM 只给了"upper-left"这种粗粒度位置，没有数值化的 bbox
3. 文本里说的"红球"和图像里的"subj_0"之间没有显式映射
4. 多图模式需要单独的主体裁剪图或背景图，但现在只有原始首帧
5. Phase 2 需要结构化的 recipe，不是散落在各处的 JSON 字段

Phase 1 六步增强（patch → align → refbank → priors2 → recipes → audit）就是来填这些坑的。

---

## 工程结构

```
tip_i2v_data_analysis/
├── configs/config.yaml
├── main.py                          # CLI 入口
└── src/
    ├── phase1/
    │   ├── patch_existing_outputs.py   # 补齐字段
    │   ├── mock_geometry.py            # 9宫格→bbox
    │   ├── align_instances.py          # 图文对齐
    │   ├── reference_bank.py           # 素材抽取
    │   ├── priors_enhance.py           # 先验增强
    │   ├── recipes.py                  # 配方生成
    │   └── audit.py                    # 审计报告
    └── utils/
        ├── schema.py                   # 基础结构（扩展了 Phase 1 字段）
        └── schema_phase1.py            # Phase 1 专属结构
```

运行方式：

```bash
# 逐步
python main.py --step patch    --config configs/config.yaml
python main.py --step align    --config configs/config.yaml
python main.py --step refbank  --config configs/config.yaml
python main.py --step priors2  --config configs/config.yaml
python main.py --step recipes  --config configs/config.yaml
python main.py --step audit    --config configs/config.yaml

# 一键全跑
python main.py --step phase1 --config configs/config.yaml
```

---

## 七个评测维度

整个 Phase 1 使用统一的七维命名，不论代码还是文档都只认这套枚举：

| 维度 | 测的是什么 | 举例 |
|------|-----------|------|
| attribute_binding | 主体属性发生变化 | 红球变蓝 |
| action_binding | 单个主体做出肢体动作 | 人在挥手 |
| motion_binding | 主体产生位移 | 球向左移动 |
| spatial_composition | 多物体之间的静态空间关系 | A 在 B 右边 |
| background_dynamics | 背景/场景发生变化 | 天晴转下雨 |
| view_transformation | 镜头运动或视角变化 | 推镜头 |
| interaction_reasoning | 多主体之间的因果/协作事件 | A 把杯子递给 B |

容易搞混的边界：
- "红球移动到蓝盒右边"：虽然有终态位置关系，但涉及位移过程，所以是 motion_binding，不是 spatial_composition。spatial_composition 只管"静态关系对不对"。
- "人挥手" vs "人把书递给狗"：前者单人动作 → action_binding，后者多主体协作 → interaction_reasoning。

---

## 逐模块详解

### 1. patch_existing_outputs.py — 在已有解析上补字段

**为什么需要这一步**

Step 2/3 的解析结果已经很丰富了——图像侧有主体列表、属性、位置描述；文本侧有六个 `involves_*` 布尔标记和各类语义槽位。但 Phase 1 后续模块需要几个它们没有的东西：数值化的 bbox、全局唯一的 instance_id、明确的主维度路由。这一步用规则补齐这些字段，完全不重跑模型。

**输入**

- `image_parse.jsonl`（Step 2 产出）：每行一个样本。每个主体有 name、position_in_frame（如 "upper-left"）、attributes、is_animate 等。
- `text_parse.jsonl`（Step 3 产出）：每行一个样本。有六个布尔字段（involves_attribute_change / involves_action / involves_directed_motion / involves_spatial_relation_change / involves_background_change / involves_camera_movement）以及对应的槽位列表。

**图像侧做了什么**

遍历每个样本里的每个主体，补三样东西：

1. **instance_id**：如果原来没有，就用主体本身的 id 字段充当。这个 ID 后续贯穿对齐、资产抽取的全过程。
2. **bbox**：把 position_in_frame 通过 mock_geometry 模块转成归一化坐标 `[x0, y0, x1, y1]`（见下一节）。
3. **质量评估**：segmentation_quality 在 P0 阶段一律标 "low"（因为没有真实分割）；tracking_feasibility 根据 bbox 面积和 is_animate 推断出 easy / medium / hard / infeasible 四档。

另外，还会为整个样本推断 `reference_potential`——判断这张首帧能不能拆出多图素材。判断依据：
- 有主体且前后景分离度 high/medium → subject 可抽取
- 主体属性非空 → attribute 可抽取
- 存在非活体物体 → object 可抽取
- 刚性背景占比 high/medium → scene_reference_original 可抽取
- scene_reference_inpainted 在 P0 一律 False（不做 inpainting）

**文本侧做了什么**

核心任务是**维度路由**——给每条样本决定一个 primary_dimension。

路由规则很简单：看六个 involves_* 布尔字段，加上 interaction 的推断，按固定优先级取第一个命中的：

```
interaction_reasoning（条件：action_slots 中有≥2个不同的 target_subject 且 spatial_relation_slots 非空）
> attribute_binding（involves_attribute_change = true）
> action_binding（involves_action = true）
> motion_binding（involves_directed_motion = true）
> spatial_composition（involves_spatial_relation_change = true）
> background_dynamics（involves_background_change = true）
> view_transformation（involves_camera_movement = true）
```

确定主维度后，通过 `LEAKAGE_TABLE` 自动填充 `forbidden_dimension_leakage`。这张表记录了每个维度容易和哪些维度混淆：

```python
LEAKAGE_TABLE = {
    "attribute_binding":      ["motion_binding", "view_transformation"],
    "action_binding":         ["motion_binding", "view_transformation"],
    "motion_binding":         ["view_transformation", "action_binding"],
    "spatial_composition":    ["motion_binding", "view_transformation"],
    "background_dynamics":    ["view_transformation"],
    "view_transformation":    ["motion_binding"],
    "interaction_reasoning":  ["motion_binding", "view_transformation"],
}
```

比如主维度是 attribute_binding，那 forbidden 就是 motion_binding 和 view_transformation——意思是后续生成 prompt 时不能出现暗示位移或镜头运动的表达。

此外还会推断 `involves_interaction` 和 `interaction_slots`：如果 spatial_relation_slots 非空且 action_slots 中有多个不同主体，就认为存在交互场景，构造一条 contact_event 型的 interaction slot。

**输出**

- `image_parse_v2.jsonl`：在原始数据基础上补齐了 instance_id / bbox / segmentation_quality / tracking_feasibility / reference_potential。
- `text_parse_v2.jsonl`：补齐了 primary_dimension / candidate_dimensions / forbidden_dimension_leakage / involves_interaction / interaction_slots / transform_ontology_confidence / routing_reason。

---

### 2. mock_geometry.py — 把粗粒度位置变成数字

**为什么需要**

VLM 输出的 position_in_frame 是 "upper-left"、"center" 这样的文字。但后面裁剪资产、计算对齐置信度都需要具体的坐标。在 P0 不接 SAM 的前提下，用一张 3×3 映射表近似。

**映射规则**

把画面分成 9 个区域，每个区域对应一个归一化 bbox：

```
upper-left:   (0.00, 0.00, 0.40, 0.40)
top:          (0.30, 0.00, 0.70, 0.40)
upper-right:  (0.60, 0.00, 1.00, 0.40)
center-left:  (0.00, 0.30, 0.40, 0.70)
center:       (0.25, 0.25, 0.75, 0.75)
center-right: (0.60, 0.30, 1.00, 0.70)
lower-left:   (0.00, 0.60, 0.40, 1.00)
bottom:       (0.30, 0.60, 0.70, 1.00)
lower-right:  (0.60, 0.60, 1.00, 1.00)
```

额外兼容 "left" → center-left、"right" → center-right。遇到识别不了的值，返回默认中央区域 `(0.20, 0.20, 0.80, 0.80)`。

**提供的工具函数**

| 函数 | 作用 |
|------|------|
| `position_to_bbox(position_in_frame)` | 9 宫格名 → 归一化 bbox |
| `normalized_to_pixel(bbox_norm, image_size)` | 归一化 → 像素坐标 |
| `crop_to_path(image_path, bbox_norm, output_path, pad_ratio=0.05)` | 按 bbox 裁剪并保存为 JPEG，带 5% 外扩，返回裁剪后尺寸；图不在或没装 PIL 则返回 None |
| `estimate_tracking_feasibility(bbox_norm, is_animate)` | 根据面积判断跟踪难度：<2% → infeasible，<6% → hard，<20% → medium(活体)/easy(非活体)，≥20% → easy |

---

### 3. align_instances.py — 把图里的主体和文本里的主体对上

**为什么需要**

到这一步为止，图像侧有一组主体（"inst_xxx_0: a red ball"），文本侧有一组目标主体（从 "red ball moves left" 中提取出的 "red ball"）。两边各自独立，没有显式关联。这个模块做两件事：把它们对齐起来，然后诊断每个维度在这条样本上能不能评。

**输入**

- `image_parse_v2.jsonl`：有 instance_id 和 bbox 的图像解析
- `text_parse_v2.jsonl`：有 primary_dimension 和 involves_* 的文本解析

**处理过程**

**第一步：主体对齐**

从文本的各种槽位里收集所有目标主体名称——attribute_change_slots 的 target_subject、action_slots 的 target_subject、motion_slots 的 target_subject、spatial_relation_slots 的 subject_a/b、interaction_slots 的 agent/patient。去重后得到一份"文本想要的主体"清单。

然后逐个图像主体和文本主体做名称匹配：
- 完全相等 → alignment_confidence = 1.0
- 一方是另一方的子串 → alignment_confidence = 0.7
- 都不匹配 → confidence = 0.0

匹配成功的记录：instance_id、image_subject_id、image_subject_name、text_subject_name、alignment_confidence、geometry（bbox 等几何信息）。

样本级别算一个 `overlap_ratio` = 成功匹配的文本主体数 ÷ 文本主体总数，反映图文对齐的完整度。

**第二步：七维可评测性诊断**

对七个维度逐一判断"这个样本在该维度上是否可评"。判断涉及两个条件：

- **文本是否请求**：对应的 involves_* 是否为 true
- **图像是否支持**：比如 action_binding 需要画面有活体（has_animate），spatial_composition 需要多个可区分主体

两个条件叠加后给出 `tool_status`：

| 文本请求 | 图像支持 | 几何质量 | → tool_status |
|---------|---------|---------|--------------|
| 是 | 是 | high/medium | valid |
| 是 | 是 | 只有 low | low_confidence |
| 是 | 否 | — | invalid_input |
| 否 | 是 | — | tool_uncertain |
| 否 | 否 | — | invalid_input |

**第三步：构建隔离约束**

从 text_parse_v2 继承 primary_dimension 和 forbidden_dimension_leakage，打包成 `dimension_isolation` 列表。

**输出**

`aligned_instances.jsonl`，每行一个样本：
- `aligned_subjects[]`：对齐结果列表
- `unmatched_text_subjects` / `unmatched_image_subjects`：没配上对的
- `overlap_ratio`：对齐覆盖率
- `evaluator_feasibility[]`：七条诊断记录（dimension / feasible / tool_status / reasons / required_tools）
- `dimension_isolation[]`：隔离约束（primary_dimension / forbidden_dimensions / leakage_risk_notes）

---

### 4. reference_bank.py — 从首帧里拆出多图素材

**为什么需要**

多图评测模式要给模型提供参考图——比如一张单独裁出来的主体图，或者一张原始场景图。这些素材从首帧中获得。

**输入**

- `image_parse_v2.jsonl`：知道每个主体的 bbox 和哪些类型的素材可以抽取（reference_potential）
- `manifest_clean.jsonl`：知道每个样本首帧图的路径

**抽取规则**

根据 reference_potential 中各项为 true/false 来决定抽取什么：

| 资产类型 | 怎么做 | 哪些维度会用到 |
|---------|--------|--------------|
| subject | 按 bbox 裁剪主体区域（5% 外扩） | attribute / action / motion / spatial / interaction |
| attribute | 同 subject（语义标签侧重属性） | attribute |
| object | 只裁剪非活体物体 | spatial / interaction |
| scene_reference_original | 原图直接复用，不裁剪 | background / view |
| scene_reference_inpainted | 去主体后的纯背景（P0 跳过不做） | spatial / interaction |

裁剪调用 mock_geometry 的 `crop_to_path`，带 5% 外扩。

**asset_id 命名**

- 主体类：`{sample_id}_{instance_id}_{suffix}`，suffix = subj / attr / obj
- 场景类：`{sample_id}_scene_orig`

**每个资产必须带 provenance（来源追溯）**

```json
{
  "source_sample_id": "tip_000123",
  "source_image_path": "images/tip_000123.jpg",
  "extraction_method": "mock_crop_9grid",
  "extraction_params": {"bbox_norm": [0.0, 0.0, 0.4, 0.4], "pad_ratio": 0.05},
  "created_by": "phase1.reference_bank",
  "created_at": "2025-06-12T10:30:00"
}
```

场景原图的 extraction_method 是 "original"（因为没有裁剪）。

**质量字段**

每个资产有 quality 子结构：resolution（裁剪后像素尺寸）、aspect_ratio、is_clean_background（mock crop 一律 false）、has_visible_subject、quality_score（根据 bbox 面积粗估，越大越好）、quality_flags（如 `["mock_crop_9grid"]`）。

**输出**

- `assets.jsonl`：所有资产的元数据
- `reference_bank/subject/*.jpg`、`reference_bank/scene_reference_original/*.jpg` 等：实际图像文件

---

### 5. priors_enhance.py — 给 Phase 2 采样准备统计先验

**为什么需要**

Step 5 产出的 prior_package 是六维视角的统计，但 Phase 2 采样还需要知道：概念频率分桶（决定 common/rare 配比）、哪些主体经常一起出现（组合多图题时参考）、多图模式的可行比例、以及哪些维度不能同时出现。

**产出四个文件**

**frequency_tiers.json**

统计所有样本中出现的主体名称、动作动词、属性值的频次，按累计占比分三档：
- head（前 50%）：高频概念，如 "person"、"walk"
- torso（50%–85%）：中频
- long_tail（后 15%）：长尾罕见概念

Phase 2 用此控制 80/20 的 common/rare 比例——确保 benchmark 不全是"人在走路"这种简单题。

**subject_pair_distribution.json**

统计哪些主体类别经常在同一张图里共现（比如 person + dog、car + person），输出 top-200 共现对。Phase 2 组合多图题时参考，避免拼出数据集里从没见过的组合。

**multi_reference_priors.json**

回答"有多少样本能走多图路线"：
- total_samples / multi_eligible_samples（对齐主体 ≥2 的样本数）/ multi_eligible_ratio
- asset_type_distribution：各类资产分别有多少
- avg_assets_per_sample / samples_with_any_asset

**compatibility_matrix.json**

一个 7×7 矩阵，标记任意两个维度之间的关系：
- compatible：可以同时出现
- leakage_risk：容易混淆，需要出现在 forbidden_dimensions 里
- incompatible：绝不允许同时出现

比如 attribute_binding × motion_binding = incompatible（测属性变化的题绝不能有位移），attribute_binding × action_binding = leakage_risk（属性变化和动作容易混淆，需要刻意防范）。

---

### 6. recipes.py — 把散落数据打包成可采样的配方

**为什么需要**

前面几步产出了对齐信息、资产库、先验统计、隔离约束，但这些都是独立文件。Phase 2 需要的是"一条 recipe = 一道题的完整原料清单"。这个模块把所有信息整合成结构化配方。

**生成条件**

不是所有样本都能出题。必须同时满足：
1. 有明确的 primary_dimension
2. 该维度的 evaluator_feasibility 中 feasible = true
3. 该维度的 tool_status 不是 invalid_input

不满足的直接跳过（代码里有计数器记录跳过原因）。

**核心决策：input_mode 和 source_type**

根据主维度和可用资产，决定这道题走单图还是多图：

| 主维度 | 条件 | → input_mode + source_type |
|-------|------|---------------------------|
| view_transformation / background_dynamics | 无条件 | single_image + observed_single_image |
| spatial_composition / interaction_reasoning | 有 ≥2 个 subject 资产且对齐主体 ≥2 | multi_image + derived_multi_reference |
| spatial_composition / interaction_reasoning | 不满足上述 | single_image + observed_single_image |
| attribute / action / motion | 有 ≥2 个 subject 资产 | multi_image + derived_single_image |
| attribute / action / motion | 有 1 个 subject 资产 | single_image + derived_single_image |
| attribute / action / motion | 没有资产 | single_image + observed_single_image |

**其他字段**

- `preserve_constraints`：按维度查表填充默认的"需保持不变"约束。比如 attribute_binding 题要保持 identity（主体身份）、scene_context（场景）、camera_framing（镜头构图）。
- `contrastive_spec`：固定 4 类 baseline 对照组——static_copy / random_motion / global_filter / camera_pan_cheat。
- `dimension_isolation`：从 text_parse_v2 继承 primary_dimension 和 forbidden_dimensions。
- `expected_difficulty`：tool_status = valid → medium，low_confidence → hard。
- `base_prompt_draft`：`[dim=xxx] {clean_prompt}` 格式。

**输出**

`candidate_recipes.jsonl`，每行一条 recipe。这是 **Phase 2 的唯一题目来源**，Phase 2 不会绕过它自己造题。

---

### 7. audit.py — 出门前最后检查一遍

**为什么需要**

Phase 1 → Phase 2 的门禁。如果某个维度的 recipe 数量为零，或者资产缺少 provenance，应该在这里暴露出来而不是让 Phase 2 去踩坑。

**审计内容（7 节报告）**

1. **样本规模**：text_parse_v2 / image_parse_v2 / aligned_instances / assets / recipes 各多少行。
2. **主维度路由分布**：七个维度各分到多少样本，有没有严重不均。
3. **七维可评测性**：每个维度的 tool_status 四态分布（valid / low_confidence / tool_uncertain / invalid_input 各多少）。
4. **资产与 provenance**：各类资产数量 + provenance 非空率（期望 100%）。
5. **Recipe 分布**：按 target_dimension 和 input_mode / source_type 分别统计。
6. **隔离有效性**：检查有没有 recipe 把自己的主维度放进了 forbidden_dimensions（期望 0 个违规）。
7. **验收清单**：逐条 pass/fail 判定（产物是否存在、provenance 是否完整、隔离是否通过）。

**输出**

`phase1_audit_report.md`：Markdown 格式报告。同时在终端打印摘要指标。

---

## Phase 1 全部产出

运行完成后 output_dir 下的文件结构：

```
{output_dir}/
├── image_analysis/
│   ├── image_parse.jsonl          # Step 2 原始输出（不改动）
│   └── image_parse_v2.jsonl       # patch 后
├── text_analysis/
│   ├── text_parse.jsonl           # Step 3 原始输出（不改动）
│   └── text_parse_v2.jsonl        # patch 后
└── phase1/
    ├── aligned_instances.jsonl    # 图文对齐 + 七维诊断
    ├── assets.jsonl               # 资产元数据
    ├── reference_bank/            # 资产图像
    │   ├── subject/
    │   ├── attribute/
    │   ├── object/
    │   └── scene_reference_original/
    ├── priors/
    │   ├── frequency_tiers.json
    │   ├── subject_pair_distribution.json
    │   ├── multi_reference_priors.json
    │   └── compatibility_matrix.json
    ├── candidate_recipes.jsonl    # Phase 2 唯一入口
    └── phase1_audit_report.md     # 审计报告
```

---

## 关键数据结构

### AlignedSample（aligned_instances.jsonl 每行）

```python
class AlignedSample(BaseModel):
    sample_id: str
    aligned_subjects: List[Dict]    # 每条含：
        # instance_id, image_subject_id, image_subject_name,
        # text_subject_name, alignment_confidence, geometry
    unmatched_text_subjects: List[str]
    unmatched_image_subjects: List[str]
    overlap_ratio: float            # 文本主体中被成功对齐的比例
    evaluator_feasibility: List[EvaluatorFeasibility]   # 7 条诊断
    dimension_isolation: List[DimensionIsolation]       # 隔离约束
```

### ReferenceAsset（assets.jsonl 每行）

```python
class ReferenceAsset(BaseModel):
    asset_id: str            # "{sample_id}_{instance_id}_{suffix}" 或 "{sample_id}_scene_orig"
    asset_type: str          # subject / attribute / object / scene_reference_original
    asset_path: str
    semantic_tags: List[str] # 主体名 + 属性值
    usable_for: List[str]   # 适用维度
    quality: AssetQuality   # resolution / aspect_ratio / is_clean_background /
                            # has_visible_subject / quality_score / quality_flags
    provenance: Provenance  # 来源追溯（必填）
```

### CandidateRecipe（candidate_recipes.jsonl 每行）

```python
class CandidateRecipe(BaseModel):
    recipe_id: str                      # "recipe_{sample_id}_{dimension}"
    source_sample_id: str
    target_dimension: str               # 七选一
    input_mode: str                     # single_image / multi_image
    source_type: str                    # observed_single_image / derived_single_image /
                                        # derived_multi_reference
    reference_asset_ids: List[str]      # 关联的资产 ID
    base_prompt_draft: str              # "[dim=xxx] {clean_prompt}"
    preserve_constraints: List[Dict]    # [{target, aspect, note}]
    contrastive_spec: List[Dict]        # 4 类 baseline
    dimension_isolation: Dict           # {primary_dimension, forbidden_dimensions, leakage_risk_notes}
    expected_difficulty: str            # medium / hard
    notes: str
```

---

## 配置

`configs/config.yaml` 中与 Phase 1 相关的部分：

```yaml
paths:
  raw_dir: "E:/I2V-CompBench/raw"
  manifest_dir: "E:/I2V-CompBench/manifest"
  output_dir: "E:/I2V-CompBench/outputs"

phase1:
  geometry_mode: "mock_9grid"       # P0 用 9 宫格，未来可切 "sam"
  crop_pad_ratio: 0.05             # 裁剪外扩 5%
  max_assets_per_sample: 10        # 单样本最多抽取的资产数
  recipe:
    enable_multi_image: true       # 允许生成 multi_image recipe
    min_subjects_for_multi: 2      # 多图模式至少需要 2 个对齐主体
```

---

## 实现状态

**P0 已完成**：六个增强步骤全部实现，CLI 集成完毕，`--step phase1` 可一键跑完。

**后续可扩展**：
- 接入 SAM 替换 mock_9grid，获得真实 bbox 和 mask
- 实现 scene_reference_inpainted（主体抹除后的纯背景）
- 基于真实共现数据动态更新 compatibility_matrix
- evaluator_feasibility 接入 Step 4 真实数据而非纯启发式

---

## 注意事项

1. **位移 ≠ 空间关系**。"球移到盒子右边"是 motion_binding，不是 spatial。
2. **多图不是附属模式**。multi_image 和 single_image 在 recipe 里是平级的。
3. **资产必须有 provenance**。audit 会检查非空率，期望 100%。
4. **单人动作 ≠ 多人交互**。单人挥手 = action_binding，A递给B = interaction_reasoning。
5. **低置信度样本不丢弃**。alignment_confidence 低的样本照样生成 recipe，只是 expected_difficulty 会标为 hard。
6. **不要重跑模型**。Phase 1 的设计核心就是零 API 调用、纯规则推导。
