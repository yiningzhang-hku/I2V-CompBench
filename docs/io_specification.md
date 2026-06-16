# TIP-I2V-Prior-Analysis 输入输出规格说明

> 本文档描述本工程的分析目标、输入数据、输出数据格式，供下游工程（如 LLM Prompt 合成、首帧图像合成）作为上下文参考。

## 一、工程定位

本工程在 I2V-CompBench 整体架构中的位置：

```
TIP-I2V 真实数据集 → [本工程] 分析 + 先验提取 → Prior Package → [下游工程] LLM 合成全维度 prompt + 文生图合成首帧
```

**本工程只做分析，不做合成。** 它的唯一产出是一个结构化的 **Prior Package JSON**，包含了从真实 I2V 数据中提取的所有先验知识。

## 二、分析了什么

### 数据源
- **TIP-I2V** 数据集（HuggingFace: `tipi2v/TIP-I2V`）
- 来源：Pika Discord 社区的 181 万条真实 I2V 生成记录
- 每条样本包含：用户 prompt 文本 + 首帧缩略图（224px）
- 使用的 split：Eval（10K 条），当前处理 1000 条

### 分析的 6 个评测维度

| 维度 | 含义 | 分析内容 |
|------|------|---------|
| **Subject Attribute Binding** | 主体属性变化 | prompt 是否指定属性变化（颜色/纹理/状态/穿着/大小），图像中主体是否有对应属性 |
| **Subject Motion Binding** | 主体运动绑定 | prompt 是否指定绝对方向运动（左/右/上/下等），图像中是否有主体 |
| **Subject Spatial Relation** | 主体空间关系 | prompt 是否涉及空间关系变化（左边/上方/后面等），图像中是否有多个可区分主体 |
| **Subject Action Binding** | 主体动作绑定 | prompt 是否指定语义动作（走/跑/挥手等），图像中是否有活物主体 |
| **Scene / Background Dynamics** | 场景背景变化 | prompt 是否涉及环境变化（天气/光照/状态），图像前景-背景是否可分离 |
| **Camera / View Transformation** | 运镜变换 | prompt 是否包含运镜指令（zoom/pan/tilt 等），图像中是否有刚体参考结构 |

### 三层分析架构

1. **VLM 图像分析**（Step 2）：用视觉语言模型解析首帧图像 → 主体列表、属性、空间关系、背景组成、运镜基线
2. **LLM 文本分析**（Step 3）：用大语言模型解析 prompt 文本 → 意图分类、6 维度语义槽位、词性标注
3. **联合分析 + 先验提取**（Step 4）：对齐文本和图像 → 维度可评测性判定 + 概念分布 + 视觉先验 + 句式模板 + 种子样例

## 三、输入数据格式

### 原始输入（来自 TIP-I2V）

每条样本包含：

```json
{
  "Unique_ID": "uuid-string",
  "Text_Prompt": "a cute cat walking across a snowy field -camera zoom in -motion 2",
  "Image_Prompt": "<PIL.Image 224x224>",
  "Subject_Field": "Animals"
}
```

### 清洗后的 Manifest（Step 1 输出）

```json
{
  "sample_id": "uuid-string",
  "prompt_text": "a cute cat walking across a snowy field -camera zoom in -motion 2",
  "clean_prompt_text": "a cute cat walking across a snowy field",
  "image_path": "E:/I2V-CompBench/manifest/images/uuid-string.jpg",
  "subject_field": "Animals",
  "pika_camera": "zoom in",
  "pika_motion": 2,
  "status": "ok"
}
```

Pika 参数（`-camera`、`-motion`、`-neg`、`-fps`、`-gs`、`-ar`、`-seed`）被提取并从 prompt 中移除。

## 四、输出数据格式

### 核心交付物：`prior_package.json`

路径：`E:\I2V-CompBench\outputs\reports\prior_package.json`

这是一个完整的 JSON 文件，结构如下：

```json
{
  "dataset_name": "TIP-I2V",
  "split": "Eval",
  "total_samples": 1000,
  "clean_samples": 989,
  "analyzed_samples": 950,

  "global_distributions": [
    {
      "category": "primary_intent",
      "entries": [
        {"name": "subject_focused", "count": 650, "pct": 68.4},
        {"name": "mixed", "count": 180, "pct": 18.9},
        ...
      ],
      "total_samples": 950
    },
    {"category": "noun", "entries": [...], "total_samples": 950},
    {"category": "verb", "entries": [...], "total_samples": 950},
    {"category": "adjective", "entries": [...], "total_samples": 950},
    {"category": "dimension_involvement", "entries": [...], "total_samples": 950}
  ],

  "global_visual_prior": {
    "subject_count_distribution": [
      {"value": "1", "count": 500, "pct": 52.6},
      {"value": "2", "count": 250, "pct": 26.3},
      ...
    ],
    "scene_type_distribution": [
      {"value": "daylight/clear/afternoon", "count": 120, "pct": 12.6},
      ...
    ],
    "shot_type_distribution": [...],
    "camera_angle_distribution": [...],
    "background_separability_distribution": [...],
    "typical_subject_categories": [
      {"value": "person", "count": 380, "pct": 40.0},
      {"value": "cat", "count": 85, "pct": 8.9},
      ...
    ]
  },

  "pika_camera_distribution": [
    {"value": "zoom in", "count": 46, "pct": 4.7},
    {"value": "zoom out", "count": 32, "pct": 3.2},
    ...
  ],
  "pika_motion_distribution": [
    {"value": "1", "count": 60, "pct": 6.1},
    {"value": "2", "count": 45, "pct": 4.6},
    ...
  ],

  "dimension_priors": [
    {
      "dimension": "attribute_binding",
      "display_name": "Subject Attribute Binding",
      "sample_count": 52,
      "coverage_pct": 5.5,

      "concept_distributions": [
        {
          "category": "target_subject",
          "entries": [{"name": "person", "count": 21, "pct": 40.4}, ...],
          "total_samples": 52
        },
        {
          "category": "attribute_type",
          "entries": [{"name": "color", "count": 18, "pct": 34.6}, ...]
        }
      ],

      "visual_prior": {
        "subject_count_distribution": [...],
        "scene_type_distribution": [...],
        "shot_type_distribution": [...],
        "camera_angle_distribution": [...],
        "background_separability_distribution": [...],
        "typical_subject_categories": [...]
      },

      "structural_templates": [
        {"pattern": "a {noun} with {adj} {noun} {verb} to {adj}", "count": 5, "pct": 9.6},
        ...
      ],

      "seed_examples": [
        {
          "sample_id": "uuid",
          "original_prompt": "a woman with blonde hair turns brunette in a park",
          "clean_prompt": "a woman with blonde hair turns brunette in a park",
          "dimension": "attribute_binding",
          "text_slots": {
            "attribute_change_slots": [
              {"target_subject": "woman", "attribute_type": "color",
               "from_value": "blonde", "to_value": "brunette"}
            ]
          },
          "image_subjects": [
            {"name": "woman", "attributes": {"color": ["blonde"], "state": ["standing"]},
             "pose_action": "standing in park", "is_animate": true}
          ],
          "image_background": {"lighting": "daylight", "weather": "clear", ...},
          "image_camera": {"shot_type": "medium_shot", "camera_angle": "eye_level", ...},
          "selection_reason": "score=7.50, overlap=1.0",
          "confidence": 0.94
        },
        ...
      ],

      "constraints": {
        "min_subjects": 1,
        "requires_explicit_change": true,
        "valid_attribute_types": ["color", "texture", "state", "wearing", "size"],
        "description": "Prompt must specify an attribute change on a subject visible in the image."
      }
    },
    ... // 其余 5 个维度结构相同
  ],

  "dimension_cooccurrence": {
    "action_binding×scene_dynamics": 15.2,
    "action_binding×camera_transformation": 8.1,
    ...
  }
}
```

### Prior Package 各字段的下游用途

| 字段 | 下游 LLM Prompt 合成时的用途 |
|------|--------------------------|
| `global_distributions` | 了解真实 I2V 用户使用的词汇分布，使合成 prompt 更自然 |
| `global_visual_prior` | 了解真实首帧图像的构成分布，指导文生图 prompt 设计 |
| `pika_camera/motion_distribution` | 运镜维度的真实用户偏好，直接指导运镜类 prompt 合成 |
| `dimension_priors[].concept_distributions` | 每个维度常见的主体/动作/属性/方向等概念分布 |
| `dimension_priors[].visual_prior` | 每个维度适合的首帧图像特征（如 spatial_relation 需要多主体图像） |
| `dimension_priors[].structural_templates` | 真实 prompt 的句式模板，可作为合成模板参考 |
| `dimension_priors[].seed_examples` | 完整标注的高质量示例，可直接用于 LLM few-shot 上下文 |
| `dimension_priors[].constraints` | 每个维度的约束规则（如最少主体数、合法属性类型等） |
| `dimension_cooccurrence` | 维度共现率，帮助设计多维度组合 prompt |

### 其他输出文件

#### Step 2 — VLM 图像分析输出

| 文件 | 格式 | 说明 |
|------|------|------|
| `image_parse.jsonl` | JSONL | 每行一个 ImageAnalysisResult，含 subjects、background、camera_baseline |
| `subject_category_freq.csv` | CSV | 主体类型频率（person, cat, car, ...） |
| `subject_attribute_freq.csv` | CSV | 属性频率（color, state, wearing, ...） |
| `subject_pose_action_freq.csv` | CSV | 姿态/动作频率 |
| `scene_setting_freq.csv` | CSV | 场景设定频率（lighting × weather × time） |
| `shot_type_distribution.csv` | CSV | 镜头类型分布 |
| `camera_angle_distribution.csv` | CSV | 拍摄角度分布 |

#### Step 3 — LLM 文本分析输出

| 文件 | 格式 | 说明 |
|------|------|------|
| `text_parse.jsonl` | JSONL | 每行一个 TextAnalysisResult，含 intent、slots、POS tags |
| `intent_distribution.csv` | CSV | 意图分布（subject_focused / mixed / camera_focused / ...） |
| `target_subject_freq.csv` | CSV | 目标主体频率 |
| `action_verb_freq.csv` | CSV | 动作动词频率 |
| `noun_freq.csv` / `verb_freq.csv` | CSV | 全局名词/动词频率 |

#### Step 4 — 联合分析输出

| 文件 | 格式 | 说明 |
|------|------|------|
| `joint_analysis.jsonl` | JSONL | 逐样本 6 维度可评测性判定 |
| `dimension_coverage.csv` | CSV | 维度覆盖率统计 |
| `dimension_priors.jsonl` | JSONL | 6 个 DimensionPrior 对象（Prior Package 的中间数据） |
| `global_distributions.jsonl` | JSONL | 全局概念分布 |
| `global_visual_prior.json` | JSON | 全局视觉构成先验 |
| `pika_distributions.json` | JSON | Pika 运镜/运动参数分布 |
| `seed_examples/*.jsonl` | JSONL | 每维度种子样例 |
| `mixed_intent_matrix.csv` | CSV | 6×6 维度共现矩阵 |

#### Step 5 — 报告输出

| 文件 | 格式 | 说明 |
|------|------|------|
| **prior_package.json** | JSON | **核心交付物**，上述所有先验的完整组装 |
| summary.md | Markdown | 可读的分析报告，含维度覆盖率、概念分布、种子样例展示 |
| dimension_gap_analysis.csv | CSV | 维度差距分析（text demand vs evaluable rate） |

## 五、语义槽位详细规格

### VLM 图像分析提取的字段

```
subjects[]:
  - id, name, instance_description, count
  - attributes: {color[], size, material_texture[], state[], wearing[]}
  - current_pose_action, position_in_frame, is_animate

subject_relations[]:
  - subject_a, predicate (left_of/right_of/above/below/...), subject_b

background:
  - elements[]: {name, type (rigid/potentially_dynamic), current_state, region}
  - lighting, weather, time_of_day
  - foreground_background_separability (high/medium/low)
  - rigid_background_ratio (high/medium/low)

camera_baseline:
  - shot_type, framing, camera_angle, estimated_depth
  - has_rigid_reference_structure, scene_depth_complexity
```

### LLM 文本分析提取的字段

```
primary_intent: subject_focused / background_focused / camera_focused / mixed / ambiguous

involves_*: 6 个布尔标志（attribute_change, action, directed_motion,
            spatial_relation_change, background_change, camera_movement）

attribute_change_slots[]: {target_subject, attribute_type, from_value, to_value}
action_slots[]:           {target_subject, action_verb, action_detail}
motion_slots[]:           {target_subject, direction, motion_phrase}
spatial_relation_slots[]: {subject_a, target_predicate, subject_b}
background_change_slots[]:{target_region, change_type, from_state, to_state}
camera_movement_slots[]:  {command, target_subject, speed_modifier, framing_constraint}

nouns[], verbs[], adjectives[]    # 词性标注
```

## 六、数据规模参考

| 阶段 | 1000 样本实测 |
|------|-------------|
| Step 1 Manifest | 24 秒，989 clean / 11 bad |
| Step 2 VLM | ~7 小时（25s/样本 × 5 并发） |
| Step 3 LLM | ~9 小时（30s/样本 × 5 并发） |
| Step 4 Joint | < 1 秒 |
| Step 5 Report | < 1 秒 |
| API 成本 | ~¥80-100（SiliconFlow） |
