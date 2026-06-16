# I2V-CompBench：图像到视频组合性评测基准完整技术方案 v2

> 本文档整合先验分析、合成流水线、评测框架三大阶段的全部设计，覆盖从原始数据预处理到最终 Benchmark 构建的完整流程。所有逻辑冲突已按已确认的设计决策统一解决。可直接作为论文撰写的技术底稿。

---

## 已确认核心设计决策（全文约束）

在阅读本文档前，先明确以下已确认且不可违背的设计决策：

| # | 决策 | 内容 | 冲突解决说明 |
|---|------|------|-------------|
| D1 | 控制变量评测 | 每题严格只测一个维度，用该维度专用评测器打分 | dimension_cooccurrence 用于维度隔离，非出跨维度题 |
| D2 | 固定模板 + 槽位填充 | 提示词采用固定模板，LLM 不做创作只做 JSON→自然语言翻译（temperature=0） | 化解了"句式先验未注入 LLM"问题 |
| D3 | 80/20 按排名切分 | 词表按频率降序排列，前 80% **词条** = common，后 20% = rare（按排名位置，非累计百分比） | 明确排除了"累计频率≥80%"的解释 |
| D4 | 兼容性矩阵 | 从 TIP-I2V 真实共现派生，LLM 补缺，不用 ConceptNet | 替代了原方案中的 ConceptNet 依赖 |
| D5 | 先验为主力 | prior_package 定义主力分布，WordNet 仅做罕见度扩展器，COCO 仅做可生成性过滤器 | 解决了"外部词表反客为主"问题(#1) |
| D6 | 对比组强制 | 每题必须有 contrastive_pair_id，200 题 = 100 对 | 用于评测时做模型偏向差分 |
| D7 | Motion v1 边界 | Motion = 单主体绝对方向运动 ONLY；多主体相对关系编辑归 Spatial | 七维度重构方案中 Motion Type B 在 v1 归入 Spatial |
| D8 | E/P/C 评分公式 | S = 0.45×Exec + 0.35×Preserve + 0.20×Coherence | — |

---

## 一、项目定位与研究动机

### 1.1 研究背景

Image-to-Video (I2V) 生成模型在保持首帧锚定的条件下执行组合性变化（compositional change）时，面临严重的绑定失败、变化泄漏和时序不一致问题。现有评测基准存在三大缺口：

| 缺口 | 现有工作 | 不足 |
|------|---------|------|
| 无 I2V 专用组合性基准 | T2V-CompBench, T2I-CompBench++ | 仅面向 T2V/T2I，无首帧锚定约束 |
| 评测维度不完整 | UI2V-Bench | 侧重理解而非生成质量 |
| 无多参考输入覆盖 | 无 | 真实 I2V 使用中大量多图输入 |

### 1.2 I2V-CompBench 的定位

I2V-CompBench 是首个面向 I2V 生成模型的**组合性评测基准**，定位为 T2V-CompBench / T2I-CompBench++ 的 I2V 系列后续工作。

**核心特征**：
- **6 + 1 维度**：6 个主榜维度 + 1 个预留维度（Interaction Reasoning, v2）
- **双范式输入**：单图编辑 + 多参考条件合成
- **数据驱动构建**：从 181 万条真实 I2V 用户数据中提取先验，指导题目合成
- **控制变量评测**：每题严格只测一个维度，用专用评测器打分
- **Preserve–Transform 三轴评分**：Exec / Preserve / Coherence 统一框架

### 1.3 与同类工作的对比

| 项目 | 会议/年份 | 模态 | 维度 | 题目来源 | 输入模式 | 评测方式 |
|------|----------|------|------|---------|---------|---------|
| T2I-CompBench++ | TPAMI 2025 | T2I | 8 | 词表+模板+GPT | 文本 | BLIP-VQA |
| T2V-CompBench | CVPR 2025 | T2V | 7 | VidProM真实分布→GPT | 文本 | MLLM+检测 |
| ConceptMix | Princeton 2024 | T2I | 8类 | GPT-4o组合 | 文本 | k-tuple |
| Generate Any Scene | UW/AI2 2024 | T2I | 场景图 | WordNet+VG程序化 | 文本 | 场景图匹配 |
| UI2V-Bench | 华为 2025 | I2V | 5 | 手工+DALL-E 3 | 图+文 | MLLM |
| **I2V-CompBench** | **Ours** | **I2V** | **6+1** | **TIP-I2V先验+程序化合成** | **图+文(单/多图)** | **专用评测器+E/P/C** |

**方法论借鉴**：
- T2V-CompBench：80/20 common/rare split、对比组设计
- T2I-CompBench++：固定模板 + 槽位填充范式
- Generate Any Scene：程序化合成 + 确定性可复现
- ConceptMix：k-tuple 难度公式化

---

## 二、整体流程架构

### 2.1 三阶段串行流程

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     Phase 1：先验分析（数据预处理）                         │
│                                                                          │
│  TIP-I2V 真实数据集 (181万条) → 采样 1000 条                               │
│      → Step1 清单构建（Pika参数提取 + 分流）                               │
│      → Step2 VLM 图像结构化解析（主体/关系/背景/运镜基线）                  │
│      → Step3 LLM 文本意图解析（意图分类 + 6维度槽位）                       │
│      → Step4 联合分析（可评测性判定 + 概念分布 + 视觉先验 + 共现矩阵）       │
│      → Step5 Prior Package 打包                                           │
│                                                                          │
│  产出：prior_package.json（唯一数据驱动来源）                               │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     Phase 2：Benchmark 合成（题目构建）                     │
│                                                                          │
│  Stage 0: 词表库构建 ← prior 分布(主力) + WordNet(长尾扩展)                │
│  Stage 1: 题目规划   ← 程序化采样 + 固定模板 + 兼容性校验 + 维度隔离        │
│  Stage 2: T2I Prompt ← 纯静态描述（禁止动作/运动/镜头语义）                 │
│  Stage 3: 首帧生成   ← T2I 模型调用（2-3 候选/题）                         │
│  Stage 4: 图像质检   ← VLM 结构化是非问答校验                              │
│  Stage 5: I2V 定稿   ← 基于实际图像调整 prompt + 对比组配对                 │
│  Stage 6: 质量过滤   ← 先验自洽性 + 维度隔离 + 对比组完整性                 │
│                                                                          │
│  产出：1200 道评测题（6维度 × 200题 = 100对 × 6维度）                       │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     Phase 3：评测执行                                      │
│                                                                          │
│  对每个待评 I2V 模型：                                                     │
│  1. 输入：首帧图像(s) + I2V prompt → 模型生成视频                           │
│  2. 评测：专用评测器按维度计算 E / P / C                                    │
│  3. 汇总：S = 0.45·E + 0.35·P + 0.20·C                                    │
│  4. 诊断：按失败模式分类统计                                                │
│                                                                          │
│  产出：模型在 6 维度上的量化得分 + 按难度/罕见度细分 + 失败模式频率           │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 prior_package.json 作为唯一接口

Phase 1 → Phase 2 的唯一数据通道是 `prior_package.json`。下游合成流水线的所有决策必须可追溯到 prior_package 中的具体字段，外部资源（WordNet/COCO）仅做辅助扩展器。

---

## 三、Phase 1：先验分析流水线

### 3.1 数据来源

| 属性 | 值 |
|------|-----|
| 数据集 | TIP-I2V (HuggingFace: `tipi2v/TIP-I2V`) |
| 来源 | Pika Discord 社区 |
| 规模 | 181 万条真实 I2V 生成记录 |
| 样本结构 | 用户 prompt + 首帧缩略图(224px) + 主体类别 + UUID |
| 使用 split | Eval (10K 条)，当前批次 1000 条 |
| 加载方式 | HuggingFace `streaming=True` 流式加载 |

### 3.2 五步流水线详细设计

#### Step 1：Manifest 构建

**输入**：HuggingFace 流式数据流
**输出**：`manifest_clean.jsonl` + `images/<id>.jpg`

核心处理：
1. **Pika 参数提取**：正则识别 `-camera zoom in`, `-motion 2`, `-neg "xxx"`, `-fps`, `-gs`, `-ar`, `-seed`
2. **prompt 清洗**：剥离 Pika 参数，生成 `clean_prompt_text`
3. **首帧落盘**：PIL 图像 → JPEG 转码 → `images/<sample_id>.jpg`
4. **质量分流**：`ok` / `missing_image` / `empty_prompt` / `bad_format`

**Pika 参数的先验价值**：
- `pika_camera`：用户运镜偏好的真实分布（直接服务 View Transformation）
- `pika_motion`：用户对运动强度的偏好（1-4 级，可映射到 Motion 难度）

#### Step 2：VLM 图像结构化解析

**模型**：Qwen3-30B-A3B-Instruct-2507 (VLM 模式)
**输出**：`image_parse.jsonl`，每条为结构化 JSON：

| 解析维度 | 输出字段 | 下游用途 |
|---------|---------|---------|
| 主体(thing类) | subjects[]: name, attributes(color/size/material/state/wearing), pose_action, position_in_frame(九宫格), is_animate | 概念分布、可评测性判定 |
| 主体间关系 | subject_relations[]: subject_a, predicate(11类空间关系), subject_b | Spatial 先验 |
| 背景(stuff类) | background: elements[](rigid/dynamic), lighting, weather, time_of_day, foreground_background_separability | Background 先验 |
| 运镜基线 | camera_baseline: shot_type(6类), camera_angle(5类), depth, has_rigid_reference_structure, scene_depth_complexity(3级) | View 先验、首帧构图指导 |

**技术特点**：异步并发 (batch_size=10) + 断点续传 + 指数退避重试 + 图像自动缩放(长边>2048px)

#### Step 3：LLM 文本意图解析

**模型**：Qwen3-30B-A3B-Instruct-2507 (LLM 模式)
**输出**：`text_parse.jsonl`

**第一层 — 意图分类**：
- `primary_intent` ∈ {subject_focused, background_focused, camera_focused, mixed, ambiguous}
- 三类子意图细分

**第二层 — 6 维度语义槽位提取**：

| 槽位类型 | 提取字段 |
|---------|---------|
| `attribute_change_slots[]` | target_subject, attribute_type(color/texture/state/wearing/size), from_value, to_value |
| `action_slots[]` | target_subject, action_verb, action_detail |
| `motion_slots[]` | target_subject, direction(8方向), motion_phrase |
| `spatial_relation_slots[]` | subject_a, target_predicate(7类), subject_b |
| `background_change_slots[]` | target_region, change_type, from_state, to_state |
| `camera_movement_slots[]` | command(13类运镜), target_subject, speed_modifier, framing_constraint |

**附加输出**：
- 词性标注：`nouns[]`, `verbs[]`, `adjectives[]`
- 维度涉及布尔标记：`involves_attribute_change`, `involves_action`, `involves_directed_motion`, `involves_spatial_relation_change`, `involves_background_change`, `involves_camera_movement`

#### Step 4：联合分析与先验提取

**Part A — 维度可评测性判定**（双约束：text_requests AND image_supports）：

| 维度 | text 条件 | image 条件 |
|------|----------|----------|
| attribute_binding | involves_attribute_change=true | 至少1主体有可见属性 |
| action_binding | involves_action=true | 至少1个 is_animate=true 主体 |
| motion_binding | involves_directed_motion=true | subject_count ≥ 1 |
| spatial_relation | involves_spatial_relation_change=true | 多主体 + 可区分 |
| scene_dynamics | involves_background_change=true | foreground_background_separability ∈ {high, medium} |
| camera_transformation | involves_camera_movement=true | has_rigid_reference_structure=true |

**Part B — 概念分布提取**：按维度聚合所有可评测样本的语义槽位，产出 `ConceptDistribution`（含 count + pct）。

**Part C — 视觉构成先验**：聚合 6 类视觉特征分布：
- subject_count_distribution（→ 难度桶配额）
- scene_type_distribution（lighting × weather × time → T2I prompt 环境参数）
- shot_type_distribution（→ T2I prompt 构图）
- camera_angle_distribution（→ T2I prompt 视角）
- background_separability_distribution
- typical_subject_categories

**Part D — 附加产出**：
- **pika_camera_distribution**：View Transformation 运镜命令权重
- **pika_motion_distribution**：Motion 强度先验（1-4级）
- **dimension_cooccurrence**：6×6 矩阵（→ 维度隔离规则）
- **structural_templates**：高频句式 pattern（→ 固定模板来源）
- **seed_examples**：每维度 Top-10 高质量标注样本

#### Step 5：Prior Package 打包

`prior_package.json` 顶层结构：

```json
{
  "dataset_name": "TIP-I2V",
  "split": "Eval",
  "total_samples": 1000,
  "clean_samples": 989,
  "analyzed_samples": 950,
  "global_distributions": {
    "primary_intent_distribution": [...],
    "noun_frequency": [...],
    "verb_frequency": [...],
    "adjective_frequency": [...]
  },
  "global_visual_prior": {
    "subject_count_distribution": [...],
    "scene_type_distribution": [...],
    "shot_type_distribution": [...],
    "camera_angle_distribution": [...],
    "background_separability_distribution": [...],
    "typical_subject_categories": [...]
  },
  "pika_camera_distribution": [...],
  "pika_motion_distribution": [...],
  "dimension_priors": [
    {
      "dimension": "attribute_binding",
      "evaluable_sample_count": N,
      "concept_distributions": {
        "target_subject": [...],
        "attribute_type": [...],
        "change_pattern": [...]
      },
      "visual_prior": {...},
      "structural_templates": [...],
      "seed_examples": [...],
      "constraints": {...}
    }
    // ... 共 6 个维度
  ],
  "dimension_cooccurrence": {
    "matrix": [[...]],
    "labels": ["attribute", "action", "motion", "spatial", "background", "camera"]
  }
}
```

### 3.3 prior_package 字段与下游 Stage 的映射

| prior_package 字段 | 下游用途 | 对应 Stage |
|---|---|---|
| `dimension_priors[].concept_distributions.target_subject` | 物体采样权重（common 主力来源） | Stage 0/1 |
| `dimension_priors[].concept_distributions.attribute_type` | 属性变化类型采样 | Stage 1 |
| `dimension_priors[].concept_distributions.action_verb` | 动作词采样 | Stage 1 |
| `dimension_priors[].concept_distributions.direction` | Motion 方向分布 | Stage 1 |
| `dimension_priors[].structural_templates` | 固定模板来源（top-N 高频句式） | Stage 1/5 |
| `dimension_priors[].seed_examples` | LLM 翻译的 few-shot 示例 | Stage 5 |
| `dimension_priors[].constraints` | 可评测性约束 | Stage 1 |
| `global_visual_prior.scene_type_distribution` | T2I prompt 环境参数 | Stage 2 |
| `global_visual_prior.shot_type_distribution` | T2I prompt 构图参数 | Stage 2 |
| `pika_camera_distribution` | View 运镜命令权重 | Stage 1 |
| `pika_motion_distribution` | Motion 难度分级参考 | Stage 1 |
| `dimension_cooccurrence` | 维度隔离规则 | Stage 1/5 |

### 3.4 先验补强清单（Phase 1 工程待落地）

以下字段当前 prior_package 中缺失，需在 Step 4/5 中补充实现：

| 编号 | 待补字段 | 含义 | 下游 Stage |
|------|---------|------|-----------|
| P1 | `frequency_tier ∈ {common, rare}` | 对每个词条按排名前80%/后20%预打标签，下游开箱即用 | Stage 1 |
| P2 | `subject_pair_distribution` | (subject_a, subject_b) 双主体类别共现频率 | Stage 1 (S2系列题) |
| P3 | `subject_action_cooccurrence` / `subject_scene_cooccurrence` / `subject_attribute_cooccurrence` | 跨槽位共现（替代 ConceptNet 兼容性矩阵） | Stage 0 |
| P4 | `style_distribution` (photo/anime/3D/cinematic) | T2I prompt 风格选择 | Stage 2 |
| P5 | `change_pattern_pairs` (from→to 配对频率) | Attribute/Background 变化模式 | Stage 1 |
| P6 | `pika_motion_intensity` → motion_binding 难度映射 | Motion 强度梯度 | Stage 1 |
| P7 | `evaluable_combinations` (每维度真实出现的合法组合白名单) | 先验自洽性检查 | Stage 1/6 |
| P8 | `dimension_isolation_rules` (基于 dimension_cooccurrence 的具体压制规则) | 维度隔离执行 | Stage 1/5 |

> **当前缓解策略**：P3 暂用 LLM 批量校验替代（效果次优但可落地），P4 暂统一使用 photorealistic，P7 暂用 LLM 合理性判定替代白名单。

---

## 四、Phase 2：Benchmark 合成流水线

### 4.1 Stage 0：词表库构建

**核心原则**：TIP-I2V prior 是 prompt 分布的真值，WordNet 只是一个"罕见度调节器"。

#### 词表库组件

| 组件 | Common 80% 来源 | Rare 20% 来源 | 输出 |
|------|----------------|--------------|------|
| A: 物体词表 | `concept_distributions.target_subject` 高频项 | WordNet 同 hypernym 子树下位词（如 dog→corgi/husky） | objects.json |
| B: 属性词表 | `concept_distributions.attribute_type` + 高频 from/to 值 | T2I-CompBench++ 33色/23纹理/32形状 中的长尾项 | attributes_*.json |
| C: 动作词表 | `concept_distributions.action_verb` 高频原子动作 | 人工补充低频但可视化的动作 | actions.json |
| D: 运动方向 | 6 个 canonical directions | — | motions.json |
| E: 运镜词表 | `pika_camera_distribution` 高频项 | — | camera_commands.json |
| F: 场景词表 | `visual_prior.scene_type_distribution` | — | scenes.json |
| G: 空间关系 | 6 类基本关系(left/right/above/below/front/behind) | Visual Genome 扩展(on/beside/between) | relations_spatial.json |

#### 兼容性矩阵构建（三层）

```
第一层：TIP-I2V 真实共现统计（P3 字段）
  → subject_action_cooccurrence: {dog: [sit, run, walk, ...]}, {bird: [fly, perch, ...]}
  → subject_scene_cooccurrence: {dog: [park, home, street]}, {fish: [ocean, aquarium]}
  → subject_attribute_cooccurrence: {cup: [empty/full, hot/cold]}, {door: [open/closed]}

第二层：LLM 批量校验扩展
  → 给 LLM 一批未见组合对，判断合理性 + 补充遗漏

第三层：人工抽检校正（可选）
```

**输出**：`compatibility_matrix.json`

#### Common/Rare 切分执行

```python
# 伪代码
sorted_words = sort_by_frequency_descending(concept_list)
cutoff_index = len(sorted_words) * 0.8
for i, word in enumerate(sorted_words):
    word.frequency_tier = "common" if i < cutoff_index else "rare"
```

### 4.2 Stage 1：题目规划（Question Planning）

#### Step 1 — 配额分配

每维度 200 题（= 100 对），按**三维矩阵**分配：

```
                     common(80%=160题)    rare(20%=40题)
Easy(40%=80题)       64 题               16 题
Medium(40%=80题)     64 题               16 题
Hard(20%=40题)       32 题               8 题
```

Rare 中的来源：30 题由 prior 长尾贡献，10 题由 WordNet 下位词扩展。

#### Step 2 — 程序化概念采样（确定性，无 LLM）

```python
for each question_slot:
    1. 确定难度桶 + common/rare
    2. 从 objects.json 按频率加权采样目标物体
    3. 查 compatibility_matrix 确定该物体可用的属性/动作/场景
    4. 从对应词表中按频率加权采样具体槽位值
    5. 查 evaluable_combinations 验证组合先验自洽性
    6. 查 dimension_cooccurrence → 生成维度隔离约束
       (如 Action 题 → 强制 camera_constraint="fixed")
    7. 输出结构化 Question Plan (JSON)
```

#### Step 3 — 模板填充 + LLM 翻译

- 每维度 5-10 个固定模板（来自 `structural_templates` 高频 top-N）
- 槽位值从 Step 2 结果填充
- LLM (temperature=0) 仅做 JSON → 自然语言翻译，无创作自由
- 变化类型限制为几种固定方式（颜色切换、状态开关、方向移动等）

#### Question Plan 输出格式

```json
{
  "question_id": "attr_001",
  "dimension": "attribute_binding",
  "input_mode": "single_image",
  "difficulty_bucket": "E-1b",
  "semantic_rarity": "common",
  "contrastive_pair_id": "attr_pair_01",
  "contrastive_role": "A",
  "first_frame_plan": {
    "subjects": [
      {"id": "s1", "category": "dog", "attributes": {"color": "white"}, "position": "left", "role": "change_target"},
      {"id": "s2", "category": "dog", "attributes": {"color": "brown"}, "position": "right", "role": "distractor"}
    ],
    "scene": {"setting": "park", "lighting": "daylight", "weather": "clear"},
    "shot_type": "medium_shot",
    "camera_angle": "eye_level"
  },
  "change_plan": {
    "target_subject": "s1",
    "change_type": "color",
    "from_value": "white",
    "to_value": "black",
    "preserve_subjects": ["s2"]
  },
  "dimension_isolation": {
    "suppressed_dimensions": ["action", "motion", "camera"],
    "camera_constraint": "fixed",
    "forbidden_words": ["run", "walk", "move", "pan", "zoom"]
  },
  "i2v_prompt_draft": "The white dog turns black while the brown dog remains unchanged.",
  "template_used": "The {color1} {subject} turns {color2} while the {color3} {subject} remains unchanged."
}
```

### 4.3 Stage 2：T2I Prompt 合成

#### 核心原则

**T2I prompt 只描述静态画面，绝不包含任何时间/视频/变化/动作/镜头运动语义。**

#### 通用结构

`[主体描述及布局] + [场景/环境] + [构图/镜头] + [风格/质量修饰]`

#### 各维度 T2I Prompt 设计要点

| 维度 | T2I prompt 关键要求 | 必须包含 | 禁止包含 |
|------|-------------------|---------|---------|
| Attribute | 初态属性清楚、多主体可区分 | 主体颜色/状态明写、位置关系 | 变化词(turns, becomes) |
| Action | neutral pose、动作部位无遮挡 | 全身可见、手臂自然下垂 | 动作词(wave, clap) |
| Motion | 为运动方向预留空间 | 目标位置偏移(right third)、留白 | 运动词(moving, going) |
| Spatial | 起始关系明确、深度线索 | 物体间明确分离、前后有透视 | 运动/变化词 |
| Background | 前景主体 + 可分背景区域 | 初始背景状态清楚 | 天气变化词 |
| View | 刚性背景 + 透视线 + 构图匹配运镜 | 几何结构、走廊/建筑/街道 | 镜头词(camera, pan) |

#### T2I Prompt 质量控制规则

1. **精确控制主体数量**：写 "two dogs"，不用 "some/several"
2. **精确控制空间位置**：用 "on the left" / "in the right third of the frame"
3. **精确控制属性**：颜色、状态、材质明写
4. **禁止时间/运动语言**：不写 walking/changing/turning
5. **禁止镜头运动语言**：不写 camera moves/panning
6. **统一风格修饰**：photorealistic, high quality（v1 统一，待 P4 style_distribution 落地后按分布采样）
7. **构图匹配维度需求**：Motion 留运动空间，View 匹配镜头命令

#### 多图模式 T2I Prompt 策略

每张参考图单独生成 T2I prompt：
- **主体参考图**：干净背景 + 清楚身份 → `"A young woman standing in neutral pose, full body, clean white background, portrait photography."`
- **属性/物体参考图**：产品摄影 + 白底 → `"A red dress on white background, flat lay product photography, high detail."`
- **场景参考图**：无人 + 空间感 → `"An empty cozy café interior with wooden tables, no people, medium shot."`

### 4.4 Stage 3：首帧图像生成

- T2I 模型 API 调用（模型待确认：FLUX.1 Pro / SDXL / DALL-E 3）
- 每个 prompt 生成 2-3 张候选图，供 Stage 4 筛选
- 多图模式：每张参考图独立生成
- 分辨率：待确认（需匹配下游 I2V 模型输入要求）

### 4.5 Stage 4：图像质检

采用 **BLIP-VQA 风格结构化问答**（非 VLM 自由描述），从 Question Plan 自动生成是非题：

```
Attribute E-1b 检查清单:
Q1: "Is there exactly two dogs in the image?" → Yes/No
Q2: "Is the dog on the left white?" → Yes/No
Q3: "Is the dog on the right brown?" → Yes/No
Q4: "Are both dogs standing side by side?" → Yes/No
Q5: "Is the background a park with green grass?" → Yes/No
```

**判定规则**：全 Yes 才通过。

**重试策略**：
1. VLM 判定不合格 → 自动调整 T2I prompt（如放大关键词权重）→ 重新生成
2. 最多重试 3 次
3. 仍不合格 → 标记为 `needs_manual_review`

### 4.6 Stage 5：I2V Prompt 定稿

1. VLM 分析实际首帧，提取实际主体信息（颜色、位置、数量）
2. 对比 Question Plan 与实际图像，调整 I2V prompt 中的指代词
3. 应用维度提示词限制规则：
   - 禁用词检查（维度专属禁词表）
   - 长度检查（8-25 英文词）
   - 维度隔离检查（如 Action 题确认含 "camera remains fixed"）
4. 对比组配对：为每题确认 contrastive pair 存在且互补

### 4.7 Stage 6：质量过滤

| 检查项 | 内容 | 不通过处理 |
|--------|------|-----------|
| 先验自洽性 | 概念组合是否在 evaluable_combinations 中出现 | 标记 out_of_distribution 或丢弃 |
| 维度隔离 | 无跨维度语义泄漏 | 回退 Stage 5 修正 |
| 对比组完整性 | A/B 必须同时存在且质检通过 | 两题同时保留或同时丢弃 |
| 禁词检查 | I2V prompt 无维度禁词 | 回退修正 |
| 长度检查 | 8-25 英文词 | 截断或重写 |

**Human-in-the-loop 三处审查节点**：
- Stage 1 后：抽检 Question Plan 质量（每维度 20 题）
- Stage 4 后：审查 VLM 标记为边界 case 的图像
- Stage 5 后：最终成品 pass/fail 审查

### 4.8 最终输出：一道完整样本长什么样

```json
{
  "question_id": "spatial_042",
  "dimension": "spatial_composition",
  "input_mode": "single_image",
  "difficulty_bucket": "E-rel",
  "semantic_rarity": "common",
  "contrastive_pair_id": "spatial_pair_21",
  "contrastive_role": "A",
  "input_images": [
    {"path": "spatial_composition/spatial_042_frame.png", "role": "first_frame"}
  ],
  "t2i_prompt_used": "A red ball placed on the left side of a blue box on a wooden table. Clear separation between ball and box. Medium shot, slightly elevated angle, indoor soft lighting, photorealistic.",
  "i2v_prompt": "The red ball moves from the left of the blue box to its right side.",
  "metadata": {
    "target_subjects": [
      {"id": "s1", "description": "red ball"},
      {"id": "s2", "description": "blue box", "role": "reference_anchor"}
    ],
    "expected_change": {
      "type": "relation_transition",
      "subject": "s1",
      "reference": "s2",
      "from_relation": "left_of",
      "to_relation": "right_of"
    },
    "preserve_constraints": [
      {"scope": "background", "constraint": "unchanged"},
      {"scope": "s2", "constraint": "position and appearance unchanged"}
    ],
    "camera_constraint": "fixed",
    "dimension_isolation": {
      "suppressed": ["action", "attribute", "camera"],
      "rationale": "spatial+motion cooccurrence=12%, forced camera fixed"
    }
  },
  "evaluation": {
    "E": "target relation reaches right_of (detection + 2D geometry)",
    "P": "blue box position unchanged, background stable",
    "C": "relation transition without oscillation"
  },
  "template_used": "The {subject} moves from the {start_rel} of the {reference} to its {end_rel} side."
}
```

---

## 五、评测框架：Preserve–Transform 三轴评分

### 5.1 双轴分离原则

I2V 评测的核心挑战：模型需同时做到"该变的变对"和"不该变的不变"。

**每道题必须明确定义**：
- **变化轴（Transform）**：什么必须改变、改成什么、由谁执行
- **保留轴（Preserve）**：什么必须保持不变（非目标主体、背景、未指定属性）

### 5.2 三轴定义与权重

| 轴 | 缩写 | 测什么 | 权重 | 判定标准 |
|----|------|--------|------|---------|
| Execution | E | 目标变化是否正确执行 | 0.45 | 变化类型正确 + 施加到正确目标 + 达到目标状态 |
| Preservation | P | 非目标内容是否保持不变 | 0.35 | 非目标主体外观不变 + 背景不变 + 未指定属性不变 |
| Coherence | C | 变化过程是否时序一致 | 0.20 | 无闪烁 + 无回跳 + 过渡平滑 + 无瞬移 |

**样本得分**：`S = 0.45 × E + 0.35 × P + 0.20 × C`  
**维度得分**：该维度 200 题的 S 分平均值  
**总分**：6 维度得分的（待定权重）加权平均

### 5.3 Exec-gated Scoring

当 E = 0（完全未执行目标变化）时，该题整体判定为失败，P 和 C 不贡献有效信息（因为什么都没变时 P 天然满分）。

### 5.4 各维度评测器工具链

| 维度 | E 评测器 | P 评测器 | C 评测器 |
|------|---------|---------|---------|
| Attribute | MLLM 属性问答 + 颜色直方图(Δ HSV) | 跟踪(SAM2) + 非目标外观 SSIM | 逐帧属性一致性(无跳变检测) |
| Action | 姿态估计(DWPose/ViTPose) + 动作分类器 | 非目标主体姿态变化幅度 < 阈值 | 动作完整性(起止帧判定) |
| Motion | 光流(RAFT) + 跟踪(CoTracker) → Δp_rel = Δp_fg - Δp_bg → 方向判定 | 背景光流一致性(无全局拖拽) | 轨迹平滑度(Savitzky-Golay残差) |
| Spatial | GroundingDINO检测 + SAM2分割 + DepthAnything深度 → 几何关系判定 | 非目标检测框 IoU 稳定性 | 逐帧关系判定一致性 |
| Background | SAM2前景分割 + 区域特征变化检测(CLIP embedding Δ) | 前景 mask 内 SSIM/LPIPS | 区域变化渐进性(无突变检测) |
| View | 光流(RAFT) → 单应矩阵估计 → 运镜参数(zoom/pan/tilt分解) | 主体身份(CLIP-I) + 刚性背景内容不改写 | 帧间运动矢量一致性(无jitter) |

### 5.5 关键技术：Motion 的相机补偿

Motion 评测中，前景相对背景运动需抵消相机运动：
```
Δp_rel = Δp_foreground - Δp_background
方向判定基于 Δp_rel 的主方向
```
这避免了"相机 pan right + 主体静止"被误判为"主体向左运动"。

### 5.6 统一失败模式分类

| 失败类型 | 定义 | 跨维度通用 |
|---------|------|-----------|
| Localization Failure | 变化发生在错误区域/主体 | ✓ |
| Binding Failure | 变化绑定到错误目标 | ✓ |
| Leakage | 变化泄漏到保留区域 | ✓ |
| Hallucination | 产生未请求的变化 | ✓ |
| Temporal Inconsistency | 变化不稳定/闪烁/回跳 | ✓ |
| Non-execution | 完全未执行指令 | ✓ |
| Partial Execution | 方向正确但幅度不足 | ✓ |
| Identity Loss | 主体身份在变化过程中丢失 | ✓ |

### 5.7 统一难度分级框架

#### 设计动机

原方案中各维度使用不同的难度命名体系（S×A、S×D、S×C 等），存在三个问题：
1. 同一字母在不同维度含义不同（"S" 既指 subjects 又指 scene），跨维度不可比
2. 部分桶标签模糊（如"Easy-Medium"），无法明确映射到配额矩阵
3. 多图模式独立成轴，与单图系统不融合

#### 三因子通用难度模型

所有维度的难度由以下三个因子的组合决定：

| 因子 | 缩写 | 含义 | 低 | 中 | 高 |
|------|------|------|-----|-----|-----|
| 绑定负载 | BL | 需同时满足的「目标→约束」绑定对数量 | 1 | 2 | ≥3 |
| 区分难度 | DD | 目标主体间（或目标与干扰物间）的视觉可区分程度 | 异类/强对比 | 同类+不同属性 | 同类+相似属性 |
| 感知复杂度 | PC | 场景复杂度 / 方向类型 / 目标大小 / 深度需求 | 简单刚性 | 中等干扰 | 复杂遮挡/小目标/3D |

> **各因子在不同维度的具体含义**（instantiation）各节单独说明。

#### 三级映射规则

| 级别 | 标签 | 配额 | 通用判定条件 |
|------|------|------|-------------|
| L1 | **Easy** | 40% (80题) | BL=1 且 DD≤中 且 PC≤中 |
| L2 | **Medium** | 40% (80题) | BL=2 **或** (BL=1 + DD=高) **或** (BL=1 + PC=高) |
| L3 | **Hard** | 20% (40题) | BL≥3 **或** (BL=2 + DD≥中) **或** (BL=2 + PC=高) |

**关键规则**：
- 每题只属于一个级别，不允许"Easy-Medium"等过渡标签
- 当多个因子同时升高时，取最高映射（如 BL=2 + DD=高 → Hard）
- Easy 桶为"校准桶"：失败率过高说明题目本身有问题而非模型太弱

#### 多图模式的难度归并

多图模式不再独立成轴（如旧的 R2/R3/R3+），而是通过其对 BL 和 PC 的影响自然归入三级：

| 多图场景 | BL 影响 | PC 影响 | 典型级别 |
|---------|---------|---------|----------|
| 2图单属性迁移 | BL=1（单约束） | PC+1（跨图融合） | Medium |
| 3图双属性迁移 | BL=2 | PC+1 | Hard |
| 2图简单空间组装 | BL=1 | PC+1 | Medium |
| 3图多关系组装 | BL≥2 | PC+1 | Hard |
| 主体图+背景图替换 | BL=1 | PC视场景差异 | Easy(相似场景) 或 Medium(跨场景) |

> 注："PC+1" 表示跨图融合本身带来的感知复杂度提升，相当于在单图同等任务上升一级 PC。

#### 各维度因子实例化总览

| 维度 | BL 的具体含义 | DD 的具体含义 | PC 的具体含义 |
|------|-------------|-------------|-------------|
| Attribute | 需绑定的「主体→属性变化」对数 | 目标主体间外观相似度 | 属性变化可视度（颜色>状态>材质） |
| Action | 需绑定的「主体→动作」分配数 | 主体外观相似+布局对称度 | 动作视觉差异度（wave vs clap高；nod vs turn低） |
| Motion | 固定=1（单主体） | 目标与背景对比度 | 场景复杂度×方向类型(平面/深度)×目标尺度 |
| Spatial | 需满足的空间关系数 | 物体间相似度 | 2D平面 vs 3D深度需求 |
| Background | 需变化的背景区域/变化类型数 | 前景-背景分离难度 | 场景层次复杂度 |
| View | 需满足的运镜约束数(命令+构图) | N/A（无歧义目标） | 场景刚性度 + 深度结构复杂度 |

---

## 六、维度 1：Attribute Binding（属性绑定）

### 6.1 评测目标

主体属性/状态变化能否**准确绑定**到指定主体，同时保持非目标主体和未指定属性不变。

核心三问：谁变？变什么？谁不该变？

### 6.2 输入模式与数据配额

| 模式 | 输入 | 测什么 | 配额 |
|------|------|--------|------|
| A: 单图属性变化 | 一张首帧 + prompt | 图中已有主体的属性变化绑定 | 120题(60%) |
| B: 多图属性迁移 | 多张参考图 + prompt | 从参考图提取属性迁移到目标主体 | 80题(40%) |

**多图模式先验支撑说明**：TIP-I2V 数据集为单图+单prompt结构，无多图先验。多图题的设计依据来自真实 I2V 使用场景（用户常提供服装/配饰参考图），属于合理外推。主体词表和属性词表仍从 prior 采样。

### 6.3 属性变化类型分层

| 层级 | 类型 | v1 策略 | 示例 |
|------|------|---------|------|
| 第一层（主力） | 颜色变化 | 大量 | white→black, red→blue |
| 第一层（主力） | 状态变化 | 大量 | empty→full, closed→open, dry→wet |
| 第二层（少量） | 外观附加件 | 少量 | no hat→red hat, no glasses→glasses |
| 第二层（少量） | 属性迁移(多图) | 少量 | 参考服装→穿到主体上 |
| 第三层（v1不主力） | 材质/纹理 | 极少 | smooth→rough, wooden→metallic |

### 6.4 难度分级

#### 因子实例化

| 因子 | Attribute 维度含义 | 低 | 中 | 高 |
|------|-----------------|-----|-----|-----|
| BL | 需绑定的「主体→属性变化」对数 | 1（单主体单变化） | 2（双变化 OR 单主体双属性） | ≥3（三主体多变化） |
| DD | 目标主体间可区分性 | 异类（狗 vs 杯子） | 同类不同色（白狗 vs 棕狗） | 同类相似（两只白狗仅大小不同） |
| PC | 属性变化视觉可辨度 | 颜色变化（强差异） | 状态变化（empty/full） | 材质/纹理变化（微妙） |

#### 三级难度桶定义

| 级别 | 判定条件 | 典型场景 | 首帧示例 | I2V prompt 示例 |
|------|---------|---------|---------|----------------|
| **Easy** | BL=1, DD≤中, PC=低/中 | 单主体单颜色/状态变化；双主体单变化+高区分 | 一个空杯子在桌上；左白狗+右棕狗 | The empty cup becomes full. / The white dog turns black while the brown dog remains unchanged. |
| **Medium** | BL=2 或 (BL=1+DD=高) 或 (BL=1+PC=高) | 单主体双属性同步；双主体各自变化；2图单属性迁移 | 白色关着的门；红+蓝气球；人物图+裙子图 | The white door opens and turns blue. / The red balloon turns green and the blue balloon turns yellow. / The woman puts on the red dress. |
| **Hard** | BL≥3 或 (BL=2+DD≥中) 或 (BL=2+PC=高) | 三主体多变化；双主体双变化+相似外观；3图多属性迁移 | 三只猫（黑/白/橘）；人物图+衣服图+帽子图 | The black cat turns gray and the orange cat turns white. / The man wears the striped shirt and black hat. |

#### 子桶明细（供配额分配用）

| 子桶 | 级别 | 模式 | 定义 | BL | DD | PC |
|------|------|------|------|----|----|----|
| E-1a | Easy | 单图 | 单主体+单颜色/状态变化 | 1 | 低(无干扰) | 低 |
| E-1b | Easy | 单图 | 双主体(异类/明显不同)+单变化 | 1 | 低 | 低 |
| M-2a | Medium | 单图 | 单主体+双属性同步变化 | 2 | 低 | 低 |
| M-2b | Medium | 单图 | 双主体各自单变化(中等区分) | 2 | 中 | 低 |
| M-2c | Medium | 多图 | 2图单属性迁移 | 1 | 低 | 中(+跨图) |
| H-3a | Hard | 单图 | 三主体中1-2个变化 | ≥3 | 中 | 低 |
| H-3b | Hard | 单图 | 双主体双变化+低区分度 | 2 | 高 | 低 |
| H-3c | Hard | 多图 | 3图双属性迁移 | 2 | 低 | 中(+跨图) |

#### 三维配额表（全200题，单图120+多图80）

| | common (160题) | rare (40题) |
|---|---|---|
| **Easy** (40%=80题) | 64 | 16 |
| **Medium** (40%=80题) | 64 | 16 |
| **Hard** (20%=40题) | 32 | 8 |

> 注：多图题自然落入 Medium/Hard（因 PC+1），不单独分配配额。

### 6.5 T2I Prompt 设计

**单图 Easy(E-1b) 示例**：
```
Two dogs standing side by side on green grass in a park.
The dog on the left is white, the dog on the right is brown.
Both are the same breed. Medium shot, eye level, daylight, photorealistic.
```

**多图 Medium(M-2c) 示例**：
```
ref1: A man standing in a neutral pose, full body visible, clean background, photorealistic.
ref2: A striped button-down shirt on white background, product photography.
ref3: A black fedora hat on white background, product photography.
```

### 6.6 提示词规则

- **长度**：8-25 英文词
- **模板**：`The {color1} {subject} turns {color2} while the {color3} {subject} remains unchanged.`
- **禁止**：强动作词(running, jumping)、镜头词(camera, pan, zoom)、风格修饰(cinematic)、绝对方向词(move left)
- **运镜**：默认固定机位（不显式写出，或写 "The camera remains fixed."）

### 6.7 对比组设计

- 正：白狗变黑，棕狗不变
- 反：棕狗变白，白狗不变

### 6.8 E/P/C 评测

| 分数 | 评测内容 | 方法 |
|------|---------|------|
| E_attr | 目标主体达到目标属性 | MLLM 问答("Is the left dog black now?") + 颜色直方图 ΔH |
| P_attr | 非目标主体+背景不变 | SAM2 跟踪+SSIM/LPIPS 比对 |
| C_attr | 无闪烁无回跳 | 逐帧属性一致性检测 |

### 6.9 典型失败模式

| 失败 | 描述 | 示例 |
|------|------|------|
| 绑定错误 | 变化施加到错误主体 | 棕狗变黑而非白狗 |
| 全局泄漏 | 所有主体都变化 | 两只狗都变黑 |
| 属性不完全 | 变化未到位 | 白→灰而非白→黑 |
| 背景污染 | 背景跟着变色 | 草地也变灰 |
| 时序闪烁 | 属性在帧间震荡 | 黑→白→黑→白 |
| 身份丢失(多图) | 迁移后主体外貌改变 | 人脸变了 |

---

## 七、维度 2：Action Binding（动作绑定）

### 7.1 评测目标

指定主体执行指定**肢体动作**，动作不串到其他主体。

核心三问：谁该动？做什么？谁不该做？

### 7.2 输入模式与数据配额

| 模式 | 输入 | 测什么 | 配额 |
|------|------|--------|------|
| A: 单图动作分配 | 首帧(多主体neutral pose) + prompt | 各主体执行各自动作 | 120题(60%) |
| B: 多图场景合成 | 人物参考图+场景图+prompt | 参考人物在场景中执行动作 | 80题(40%) |

### 7.3 动作词表

| 部位 | 推荐动作（原子化、视觉明确） | 禁用动作 |
|------|--------------------------|---------|
| 上肢 | wave, clap, raise_hand, point | dancing, gesturing |
| 头部 | nod, turn_head, tilt_head, look_up | reacting |
| 全身 | bow, sit_down, stand_up, jump | moving energetically |
| 复合(仅Hard) | wave then bow | picks up, pushes(→Interaction) |

### 7.4 难度分级

#### 因子实例化

| 因子 | Action 维度含义 | 低 | 中 | 高 |
|------|-----------------|-----|-----|-----|
| BL | 需绑定的「主体→动作」分配数 | 1（一人动/一动一静） | 2（双人双动作） | ≥3（三人三动作） |
| DD | 主体可区分性 | 异性/异服装/异位置 | 同性不同服装 | 同性相似服装+对称布局 |
| PC | 动作视觉差异度 | 大动作差异(wave vs stand) | 中动作差异(wave vs clap) | 小动作差异(nod vs tilt_head) |

#### 三级难度桶定义

| 级别 | 判定条件 | 典型场景 | 首帧示例 | I2V prompt 示例 |
|------|---------|---------|---------|----------------|
| **Easy** | BL=1, DD≤中 | 单人做动作（校准）；双人一动一静+异服装 | 单人 neutral pose；左白衣右黑衣并排 | The woman waves. Camera fixed. / Woman on left waves, man on right stays still. Camera fixed. |
| **Medium** | BL=2, DD≤中 | 双人双不同动作+可区分服装；2图多人场景合成 | 两人面对面，蓝衣 vs 灰衣 | Man in blue claps, man in gray nods. Camera fixed. |
| **Hard** | BL≥3 或 (BL=2+DD=高) | 三人三动作；双人双动作+相似外观+对称布局 | 三人红/白/蓝衣并排；两同服女性对称站 | Person in red waves, in white claps, in blue nods. Camera fixed. / Woman on left bows, woman on right raises hand. Camera fixed. |

#### 子桶明细

| 子桶 | 级别 | 模式 | 定义 | BL | DD | PC |
|------|------|------|------|----|----|----|
| E-1a | Easy | 单图 | 单人单动作（校准桶） | 1 | N/A | 低 |
| E-1b | Easy | 单图 | 双人一动一静+异服装 | 1 | 低 | 低 |
| M-2a | Medium | 单图 | 双人双不同动作+可区分 | 2 | 中 | 中 |
| M-2b | Medium | 多图 | 2人场景合成+各自动作 | 2 | 低 | 中(+跨图) |
| H-3a | Hard | 单图 | 双人双动作+相似外观+对称布局 | 2 | 高 | 中 |
| H-3b | Hard | 单/多图 | 三人三不同动作 | ≥3 | 中 | 中 |

#### 三维配额表（全200题）

| | common (160题) | rare (40题) |
|---|---|---|
| **Easy** (40%=80题) | 64 | 16 |
| **Medium** (40%=80题) | 64 | 16 |
| **Hard** (20%=40题) | 32 | 8 |

> 注：Easy 桶中单人动作题(校准桶)不超过 20 题，其余为双人一动一静。

### 7.5 首帧设计原则

1. 动作部位不遮挡、不出框（手、头、躯干完整可见）
2. 初始姿态为 neutral pre-action pose
3. 多主体用**位置+服装颜色**区分（禁止代词链）
4. Hard(H-3a) 用镜像布局+相似服装制造高混淆

### 7.6 T2I Prompt 示例

```
Medium(M-2a):
Two men standing face to face in an office.
The man on the left wears a blue shirt, the man on the right wears a gray jacket.
Both in neutral standing pose with arms at sides.
Medium-wide shot, eye level, indoor lighting, photorealistic.
```

### 7.7 提示词规则

- **四段式**：`场景壳 + 主体锚定 + 动作分配 + 运镜约束`
- **强制**：`The camera remains fixed.`（维度隔离：action+camera 共现率 8%）
- **动词必须原子化**：wave/clap/nod ✓；dance/gesture/interact ✗
- **禁止代词链**：❌ "she then turns to her" → ✓ "the woman on the left...the woman on the right..."

### 7.8 对比组

左边人挥手右边不动 ↔ 右边人挥手左边不动

### 7.9 E/P/C 评测

| 分数 | 评测内容 | 方法 |
|------|---------|------|
| E_act | 指定主体执行正确动作 | DWPose姿态估计 + 动作分类器(wave/clap/nod分类头) |
| P_act | 非目标主体未误执行 | 非目标主体姿态变化幅度 < θ |
| C_act | 动作完整连续 | 动作起止帧检测（非单帧伪成功） |

### 7.10 典型失败模式

| 失败 | 描述 |
|------|------|
| 动作互换 | A做了B的动作，B做了A的 |
| 动作泄漏 | 所有主体都做了相同动作 |
| 动作不完整 | 仅一帧看似成功（pose但非motion） |
| 幽灵动作 | 产生未请求的额外动作 |
| 主体消失 | 动作过程中主体身份丢失 |

---

## 八、维度 3：Motion Binding（运动绑定）

### 8.1 评测目标

指定主体沿正确的**整体轨迹方向**运动。

**v1 主榜范围**（已确认决策 D7）：
- ✅ 单主体、六方向绝对方向运动
- ❌ 多主体相对位移（→ 归入 Spatial Composition）
- ❌ 多图组合运动（→ stress subset / v2）

### 8.2 输入模式

v1 全部为**单图输入**：首帧含一个可追踪目标主体 + 预留运动空间方向。

### 8.3 六方向定义

| 方向类型 | 方向 | 首帧要求 | 视觉判定依据 |
|---------|------|---------|------------|
| 平面方向(D1) | left, right, up, down | 2D 空间预留 | 像素级位移 |
| 深度方向(D2) | toward_camera, away_from_camera | 透视线索 + 深度感 | 尺度变化 + 透视位移 |

### 8.4 难度分级

#### 因子实例化

Motion 维度的 BL 固定为 1（v1 只测单主体），难度完全由 DD 和 PC 决定：

| 因子 | Motion 维度含义 | 低 | 中 | 高 |
|------|-----------------|-----|-----|-----|
| BL | 固定=1 | — | — | — |
| DD | 目标与背景的对比度/可追踪性 | 大目标+简单背景(强对比) | 中目标+中等背景(有干扰物) | 小目标+复杂背景(低对比) |
| PC | 方向类型 + 场景结构 | 平面方向+简单场景 | 平面方向+中等场景 或 深度方向+简单场景 | 深度方向+复杂场景 |

#### 三级难度桶定义

| 级别 | 判定条件 | 典型场景 | 首帧示例 | I2V prompt 示例 |
|------|---------|---------|---------|----------------|
| **Easy** | DD=低, PC=低 | 大目标+干净背景+平面方向 | 蓝天中红气球在画面右侧 | The red balloon moves to the left. |
| **Medium** | DD=中 或 PC=中 | 中目标+有干扰+平面；或 中目标+透视场景+深度 | 草地+树丛中棕狗；透视走廊中红玩具车 | The small brown dog moves to the left. / The red toy car moves toward the camera. |
| **Hard** | DD=高 或 PC=高 或 两者皆高 | 小目标+复杂背景+平面；或 小目标+深度方向 | 杂乱房间角落白纸飞机；多棵树中远处小鸟 | The white paper airplane moves to the right. / The small bird moves away from the camera. |

#### 子桶明细

| 子桶 | 级别 | 方向类型 | 定义 | DD | PC |
|------|------|---------|------|----|----|----|
| E-p | Easy | 平面(L/R/U/D) | 大目标+干净背景 | 低 | 低 |
| M-p | Medium | 平面(L/R/U/D) | 中目标+有干扰物 | 中 | 低-中 |
| M-d | Medium | 深度(T/A) | 中目标+透视场景 | 低-中 | 中 |
| H-p | Hard | 平面(L/R/U/D) | 小目标+复杂场景 | 高 | 中 |
| H-d | Hard | 深度(T/A) | 小/中目标+复杂深度场景 | 中-高 | 高 |

#### 难度与 pika_motion 的映射（P6 待落地）

| pika_motion 值 | 含义 | 对应用途 |
|---------------|------|----------|
| 1 | subtle | 定义"Easy 桶的最低期望位移量" |
| 2-3 | moderate | Medium 桶的参考位移量 |
| 4 | strong | 定义"E_mot=1 的足够位移阈值" |

### 8.5 数据配额

200 题全部为 Type A（单主体绝对方向），分布：
- 平面方向(left/right/up/down)：140 题 (70%)
- 深度方向(toward/away)：60 题 (30%)
- 方向内平衡：left≈right, up≈down, toward≈away（对比组自然保证）

**三维配额表**：

| | common (160题) | rare (40题) |
|---|---|---|
| **Easy** (40%=80题) | 64 | 16（目标为罕见物体） |
| **Medium** (40%=80题) | 64 | 16 |
| **Hard** (20%=40题) | 32 | 8（罕见物体+高场景复杂度） |

> 方向平衡为正交约束：各难度桶内 L≈R≈U≈D≈T≈A，通过对比组配对自然实现。平面方向(70%)与深度方向(30%)的比例反映在子桶分布中（E-p/M-p 多于 M-d/H-d）。

### 8.6 首帧设计原则

1. **必须预留运动方向空间**：向左题目标不能在最左边
2. **目标用外观属性唯一指代**：the red balloon, the small brown dog（非位置指代）
3. **toward/away 必须有透视线索**：走廊透视线、地面消失点、前中后景层次
4. **避免强动态背景**：避免海浪/火焰/大面积树叶摆动
5. **目标与背景有足够对比**：不要白色目标在白色背景中

### 8.7 T2I Prompt 示例

```
Medium(M-d) (toward camera):
A red toy car positioned at the far end of a hallway with clear perspective lines.
The car is facing the camera. The hallway walls are white with minimal decoration.
Medium shot, eye level, indoor lighting, photorealistic.
```

### 8.8 提示词规则

- **模板**：`The [target] moves to the [direction].` 或 `The [target] moves [direction_phrase].`
- **中性动词**：move, shift, drift ✓
- **禁止**：局部动作词(wave/nod)、语义动作词(run/walk/fly)、多阶段词(first then)、运镜词(pan/zoom)、属性变化词(turns red)

### 8.9 对比组

- left ↔ right（相同物体，反方向）
- up ↔ down
- toward ↔ away

### 8.10 E/P/C 评测

| 分数 | 评测内容 | 方法 |
|------|---------|------|
| E_mot | 主体沿正确方向发生足够位移 | RAFT光流+CoTracker跟踪→Δp_rel=Δp_fg-Δp_bg→主方向判定 |
| P_mot | 背景/非目标未被误带动 | 背景光流一致性检测（排除全局拖拽） |
| C_mot | 路径平滑无瞬移 | 轨迹Savitzky-Golay拟合残差 |

**关键**：`Δp_rel = Δp_foreground - Δp_background` 抵消相机运动干扰。

### 8.11 典型失败模式

| 失败 | 描述 |
|------|------|
| 方向错误 | 向右而非向左 |
| 全局拖拽 | 整个画面移动（相机动而非主体动） |
| 主体静止 | 背景动了但主体没动 |
| 位移不足 | 有趋势但幅度太小 |
| 主体消失 | 运动过程中目标丢失/融入背景 |

---

## 九、维度 4：Spatial Composition（空间组合）

### 9.1 评测目标

多个主体/物体能否被**正确放置到指定空间关系**中，或已有空间关系能否被正确编辑。

**v1 范围**（边界重构后）：
- ✅ 多图空间组装：将多个来源不同的物体放置到指定空间关系（静态放置）
- ✅ 单图关系转移：多主体间的相对关系从 A 编辑为 B（含位移，原七维度方案中的 Motion Type B）
- **与 Motion 的边界**：Motion = 单主体绝对方向(left/right)；Spatial = 多主体相对关系(左边→右边 of 参照物)

### 9.2 输入模式与数据配额

| 模式 | 输入 | 测什么 | 配额 |
|------|------|--------|------|
| A: 单图关系转移 | 首帧(多主体+起始关系) + prompt | 关系正确编辑（含位移） | 40题(20%) |
| B: 多图空间组装(核心) | 物体参考图 + 场景图 + prompt | 多物体正确放置 | 160题(80%) |

### 9.3 空间关系类型

| 类型 | 关系 | 判定方式 |
|------|------|---------|
| 2D 平面 | left_of, right_of, above, below | 检测框中心 x/y 坐标比较 |
| 3D 深度 | in_front_of, behind | DepthAnything 深度值比较 |
| 接触性 | on(在……上面) | 检测框底边重合 + 深度相近 |
| 邻近性 | beside, next_to, between | 检测框距离 + 相对位置 |

### 9.4 难度分级

#### 因子实例化

| 因子 | Spatial 维度含义 | 低 | 中 | 高 |
|------|-----------------|-----|-----|-----|
| BL | 需同时满足的空间关系数 | 1 | 2 | ≥3 |
| DD | 物体间相似度 | 异类物体(猫 vs 沙发) | 同类不同属性(三不同色椅子) | 同类近似(三把相似椅子) |
| PC | 空间关系类型 | 2D平面(左右/上下) | 深度关系(前后) | 深度+接触性+多层复合 |

#### 三级难度桶定义

| 级别 | 判定条件 | 典型场景 | 输入示例 | I2V prompt 示例 |
|------|---------|---------|---------|----------------|
| **Easy** | BL=1, PC=低(2D) | 2物体+左右/上下关系 | 红花瓶图+蓝书架图+客厅场景 | Place the red vase to the left of the blue bookshelf. |
| **Medium** | BL=2 或 PC=中(3D) | 2物体+前后关系；或 3物体+纯2D | 白杯+笔记本+桌面；猫+沙发+绿植+客厅 | Place the white cup in front of the laptop. / Place the cat on the sofa and the plant beside the sofa. |
| **Hard** | BL≥3 或 (BL=2+PC=高) | 3物体+含深度；或 4+物体复杂布局 | 椅子+桌子+花瓶；床+床头柜+书桌+椅子 | Place the chair in front of the table, with the vase on the table. / Arrange the bed against the back wall, nightstand to the right of bed... |

#### 子桶明细

| 子桶 | 级别 | 模式 | 定义 | BL | DD | PC |
|------|------|------|------|----|----|----|
| E-2d | Easy | 多图 | 2物体+2D关系(左右/上下) | 1 | 低 | 低 |
| E-rel | Easy | 单图 | 2主体关系转移(2D) | 1 | 低 | 低 |
| M-3d | Medium | 多图 | 2物体+3D关系(前后) | 1 | 低 | 中 |
| M-m2d | Medium | 多图 | 3物体+多个2D关系 | 2 | 低 | 低 |
| M-rel | Medium | 单图 | 2-3主体关系转移(3D) | 1-2 | 中 | 中 |
| H-m3d | Hard | 多图 | 3物体+含3D+接触性关系 | 2 | 低-中 | 高 |
| H-c4 | Hard | 多图 | 4+物体复杂布局 | ≥3 | 中 | 中-高 |

**单图关系转移示例**（原 Motion Type B 归入此处）：

| 首帧 | I2V prompt | 级别 |
|------|------------|------|
| 红球在蓝盒左边 | The red ball moves from the left of the blue box to its right side. | Easy |
| 白车在黑卡车后方 | The white car is repositioned in front of the black truck. | Medium |
| 猫在狗左侧，椅子在右 | The cat moves to the middle between the dog and the chair. | Medium |

#### 三维配额表（全200题，单图40+多图160）

| | common (160题) | rare (40题) |
|---|---|---|
| **Easy** (40%=80题) | 64 | 16 |
| **Medium** (40%=80题) | 64 | 16 |
| **Hard** (20%=40题) | 32 | 8 |

### 9.5 首帧/参考图设计原则

**多图模式**：
1. 每张参考图只含一个目标物体，背景干净
2. 物体尺度/视角/光照与场景协调
3. 场景图有足够空间放置所有物体
4. 前后关系的场景图需深度线索
5. 场景图中不要有与参考物体相似的已有物体

**单图模式**：
1. 起始关系一眼可判
2. 目标关系可达（留够空间）
3. 主体用颜色/大小/纹理可区分

### 9.6 提示词规则

- **多图结构**：`Place [A] [relation] [B] + [可选场景说明]`
- **单图结构**：`The [subject] moves from [start_rel] of [reference] to its [end_rel] side.`
- **禁止**：运镜词、风格词、绝对方向词（"move left" 属于 Motion）

### 9.7 对比组

- 猫在沙发左边 ↔ 猫在沙发右边
- 花瓶在桌上 ↔ 花瓶在桌下
- 椅子在桌前 ↔ 椅子在桌后

### 9.8 E/P/C 评测

| 分数 | 评测内容 | 方法 |
|------|---------|------|
| E_rel | 目标空间关系最终成立 | GroundingDINO检测+SAM2分割+DepthAnything→几何关系判定 |
| P_rel | 非目标物体/背景布局保持 | 非目标检测框IoU稳定性+背景SSIM |
| C_rel | 关系稳定不振荡 | 逐帧关系判定一致性（最后N帧一致） |

### 9.9 典型失败模式

| 失败 | 描述 |
|------|------|
| 关系错误 | 放到右边而非左边 |
| 物体缺失 | 某个参考物体未出现 |
| 尺度不匹配 | 物体大小与场景不协调 |
| 遮挡过强 | 前后关系中物体完全重叠 |
| 关系振荡 | 关系在帧间来回翻转 |
| 身份混淆 | 相似物体摆放位置互换 |

---

## 十、维度 5：Background Dynamics（背景动态）

### 10.1 评测目标

背景区域发生指定变化，同时前景主体和非目标区域保持稳定。

核心三问：背景变了吗？变对区域了吗？前景被污染了吗？

### 10.2 输入模式与数据配额

| 模式 | 输入 | 测什么 | 配额 |
|------|------|--------|------|
| A: 环境状态变化 | 首帧(前景+背景) + prompt | 背景区域动态变化 | 120题(60%) |
| B: 背景迁移/替换 | 主体图 + 目标背景图 + prompt | 保持主体，替换背景 | 80题(40%) |

### 10.3 Type A 子类型

- **动态纹理**：云流动、海浪变大、树叶摇晃、烟雾扩散
- **场景状态**：天空变暗、远处起雾、开始下雨、灯光亮起

### 10.4 难度分级

#### 因子实例化

| 因子 | Background 维度含义 | 低 | 中 | 高 |
|------|-----------------|-----|-----|-----|
| BL | 需变化的背景区域/变化类型数 | 1(单区域单变化) | 1-2(单区域强变化 或 双弱变化) | ≥2(多区域复合变化) |
| DD | 前景-背景分离难度 | 前景明显+背景开阔 | 前景有交叉+中等分离 | 前景混入背景+遮挡多 |
| PC | 场景层次复杂度 | 单层(天空+地面) | 双层(前景+中景+背景) | 多层(多前景+叠层背景) |

#### 三级难度桶定义

**Type A（环境状态变化）**：

| 级别 | 判定条件 | 首帧示例 | I2V prompt 示例 | 核心挑战 |
|------|---------|---------|----------------|----------|
| **Easy** | BL=1, DD=低, PC=低 | 人+海边+晴天(前景少，天空开阔) | Clouds slowly drift while the person remains still. Camera fixed. | 基础区域变化 |
| **Medium** | BL=1+DD=中 或 BL=2+DD=低 | 人+公园+树林(前景与背景有交叉) | Leaves on the trees sway and some fall. Person remains still. Camera fixed. | 区域隔离+局部变化 |
| **Hard** | BL≥2+DD≥中 或 PC=高 | 车+城市街道(多层场景) | Dark clouds gather, streetlights flicker on, puddles begin to form. Car unchanged. Camera fixed. | 多区域+因果链 |

**Type B（背景迁移/替换）**：

| 级别 | 判定条件 | 输入示例 | I2V prompt 示例 | 核心挑战 |
|------|---------|---------|----------------|----------|
| **Easy** | BL=1, DD=低(主体清晰+场景类型相似) | 女人+相似室内 | The woman is now in the new room. Camera fixed. | 简单替换 |
| **Medium** | BL=1, PC=中(跨场景类型) | 男人+跨城市 | The man is now on a Parisian street with the Eiffel Tower. Camera fixed. | 跨场景+光照调和 |
| **Hard** | DD=中(多主体) 或 PC=高(光照差异大) | 两人+光照差异大 | The two people are now walking on the sunny beach. Camera fixed. | 多主体保持+光照一致性 |

#### 三维配额表（全200题，Type A 120+Type B 80）

| | common (160题) | rare (40题) |
|---|---|---|
| **Easy** (40%=80题) | 64 | 16 |
| **Medium** (40%=80题) | 64 | 16 |
| **Hard** (20%=40题) | 32 | 8 |

> 注：Type B 因跨图融合带来 PC 提升，自然偏向 Medium/Hard。Easy 桶主要由 Type A 简单题填充。

### 10.5 首帧设计原则

- **Type A**：目标背景区域占 15%-60%、有前景 preserve 区、初态不接近终态
- **Type B**：主体轮廓清楚(便于分离)、背景图空间可容纳主体、光照差异合理

### 10.6 提示词规则

- **Type A**：`[背景区域] + [变化操作] + [preserve clause] + camera fixed`
- **Type B**：`[主体保持说明] + [目标背景描述] + camera fixed`
- **强制固定机位**（v1）：避免背景运动与镜头运动混淆
- **受控变化词表**：drift, sway, ripple, spread, darken, brighten, become foggy, gradually, slowly

### 10.7 对比组

- Type A：天空变暗 ↔ 天空变亮
- Type B：伦敦→巴黎 ↔ 巴黎→伦敦

### 10.8 E/P/C 评测

| 分数 | 评测内容 | 方法 |
|------|---------|------|
| E_bg | 目标背景区域发生指定变化 | SAM2 前景分割→背景区域 CLIP embedding 变化检测 |
| P_bg | 前景主体+非目标背景保持 | 前景 mask 内 SSIM/LPIPS + 非目标区域稳定性 |
| C_bg | 变化渐进无闪烁 | 时序一致性(逐帧差分无突变) |

### 10.9 典型失败模式

| 失败 | 描述 |
|------|------|
| 全图滤镜 | 整张图变暗而非局部天空(伪成功) |
| 前景污染 | 人物也跟着变暗/变色 |
| 背景不变 | 完全忽略变化指令 |
| 伪成功 | 一帧闪变然后恢复 |
| 身份丢失(Type B) | 背景替换后人物外观改变 |
| 边界伪影(Type B) | 主体与新背景交界处明显不自然 |

---

## 十一、维度 6：View Transformation（视角变换）

### 11.1 评测目标

模型正确执行**镜头运动/视角变化**，同时保持主体身份、属性和场景内容不被额外改写。

本质测试："给你这张图，你接下来应该怎么拍它。" 核心变量是 camera/view change，非内容语义编辑。

### 11.2 输入模式与数据配额

以**单图输入为主**(95%)。少量多图(5%)作为 stress subset。

任务子类型：
- A: 单图单镜头命令（主力）
- B: 镜头 + 构图约束（medium/hard）
- C: static camera 负控制（必须包含，用于对比组）

### 11.3 运镜词表

| 类型 | 允许 | 禁止 |
|------|------|------|
| 基础运镜 | zoom in, zoom out, pan left, pan right, tilt up, tilt down | cinematic, dramatic, orbit |
| 景别 | push in, pull out | crane shot, dolly |
| 稳定 | keep the camera static, camera remains fixed | handheld, shaky |
| 构图约束 | keep the subject roughly centered | over-the-shoulder, Dutch angle, POV |

### 11.4 难度分级

#### 因子实例化

| 因子 | View 维度含义 | 低 | 中 | 高 |
|------|-----------------|-----|-----|-----|
| BL | 需满足的运镜约束数(命令+构图) | 1(单命令) | 2(双约束或命令+构图) | ≥3(复合控制) |
| DD | N/A | — | — | — |
| PC | 场景刚性度 + 深度结构 | 刚性背景(走廊/建筑)+单主体 | 中等场景+少量动态元素 | 复杂多层+可变形背景 |

#### 三级难度桶定义

| 级别 | 判定条件 | 典型场景 | 首帧示例 | I2V prompt 示例 |
|------|---------|---------|---------|----------------|
| **Easy** | BL=1, PC=低 | 单命令+简单刚性场景 | 走廊+女人；草地+狗 | The camera slowly zooms in. / The camera slowly pans right. |
| **Medium** | BL=2 或 (BL=1+PC=中) | 双约束+简单场景；或 单命令+中等场景 | 街边+女人；咖啡厅+两人 | The camera pans right while keeping the woman centered. / The camera pushes in while tilting slightly downward. |
| **Hard** | BL≥3 或 (BL=2+PC≥中) | 复合控制+复杂场景 | 拥挤集市 | Camera pans left, then zooms in keeping fruit stand centered. |

#### 子桶明细

| 子桶 | 级别 | 定义 | BL | PC | 说明 |
|------|------|------|----|----|------|
| E-z | Easy | zoom in/out+刚性场景 | 1 | 低 | 基础景别控制 |
| E-p | Easy | pan L/R+简单场景 | 1 | 低 | 基础横向控制 |
| E-t | Easy | tilt U/D+简单场景 | 1 | 低 | 基础纵向控制 |
| M-cf | Medium | 单命令+构图约束 | 2 | 低 | 运镜+保持居中 |
| M-sc | Medium | 单命令+中等场景 | 1 | 中 | 复杂场景下基础运镜 |
| H-cc | Hard | 双命令+构图约束+中等场景 | ≥3 | 中 | 复合控制(v1少量) |
| static | 对照 | camera remains static | 0 | 任意 | 负控制（对比组必备） |

> static 负控制题不单独计入难度分布，而是作为对比组的另一半分散在各级别中。

#### 三维配额表（全200题，单图95%+多图5%）

| | common (160题) | rare (40题) |
|---|---|---|
| **Easy** (40%=80题) | 64 | 16 |
| **Medium** (40%=80题) | 64 | 16 |
| **Hard** (20%=40题) | 32 | 8 |

> 注：每级别内 static 负控制占 ~25%（与对应运镜题配对）。

### 11.5 首帧设计原则

1. **背景以刚性结构为主**：走廊、建筑、街道（非水面/树叶/烟雾）
2. **有明确透视线或几何结构**
3. **初始构图匹配镜头命令**：
   - zoom in：主体不能已占满画面
   - pan left/right：画面两侧有可见空间
   - tilt up/down：画面上下有信息
   - static：任何合理场景
4. **避免动态元素**：大面积水面、火焰、镜面强反射

### 11.6 T2I Prompt 示例

```
Medium(M-cf) (pan + framing constraint):
A woman standing on a sidewalk with a café and parked cars in the background.
The woman is slightly left of center. Street visible on both sides.
Medium shot, eye level, daylight, photorealistic.
```

### 11.7 提示词规则

- **必须含至少一个 camera verb**
- **v1 每题最多两个镜头约束**
- **构图约束最多一个**
- **禁止抽象描述**：❌ "cinematic feel" → ✓ "The camera slowly zooms in."

### 11.8 对比组

- zoom in ↔ zoom out
- pan left ↔ pan right
- tilt up ↔ tilt down
- 任意运镜 ↔ static camera（相同首帧）

### 11.9 E/P/C 评测

| 分数 | 评测内容 | 方法 |
|------|---------|------|
| E_cam | 运镜类型/方向/趋势正确 | RAFT光流→全局单应矩阵→分解为zoom/pan/tilt参数→与指令比对 |
| P_cam | 主体身份+属性+刚性背景不改写 | CLIP-I 主体一致性 + 背景内容改写检测 |
| C_cam | 镜头平滑无jitter | 帧间运动矢量一致性(标准差<阈值) |

### 11.10 典型失败模式

| 失败 | 描述 |
|------|------|
| 方向错误 | pan right 执行为 pan left |
| 内容改写 | zoom in 过程中背景凭空出现新物体 |
| 运镜不执行 | 画面完全静止 |
| 虚假运镜 | 主体移动伪装成镜头运动 |
| 抖动/跳变 | 镜头不平滑有突变 |
| 构图约束违反 | 要求居中但主体偏离画面 |

---

## 十二、维度 7：Interaction Reasoning — v2 预留

### 12.1 评测目标

多物体/主体间的交互行为是否符合**物理常识和因果推理**。

### 12.2 与其他维度的区别

| 对比 | 判定规则 |
|------|---------|
| vs Action | Action = 单主体自身动作；Interaction = 多主体因果交互 |
| vs Motion | Motion = 纯位移无因果；Interaction = 有因果链+状态变化 |

### 12.3 三类交互

- **物理交互**：球撞积木→倒塌, 水倒入杯子, 锤子砸玻璃
- **社会交互**：握手, 递东西, 追逐
- **功能交互**：用钥匙开门, 按开关灯亮

### 12.4 难度分桶（预留）

| 桶 | 定义 |
|----|------|
| I1 | 双物体简单物理（球碰杯→倒） |
| I2 | 较复杂物理/社会交互 |
| I3 | 链式因果（A→B→C） |
| I4 | 复合交互+多状态变化 |

### 12.5 暂缓原因

1. prior_package 中无对应先验数据
2. 需额外构建 Visual Genome 关系库 + 交互词表
3. 评测器设计复杂度高（需因果链检测 + 物理合理性判定）

---

## 十三、维度间边界隔离规则

### 13.1 判定规则表

| 容易混淆的对 | 判定规则 | 典型歧义示例 → 归属 |
|-------------|----------|-------------------|
| Action vs Motion | 肢体动作→Action；整体位移→Motion | "dog runs left" → Motion(位移为主) |
| Action vs Interaction | 单主体自身→Action；多主体因果→Interaction | "picks up cup" → Interaction |
| Motion vs Spatial | 单主体绝对方向→Motion；多主体相对关系→Spatial | "ball moves left"→Motion; "ball moves left of box"→Spatial |
| Motion vs View | 主体在动→Motion；镜头在动→View | "man walks toward viewer" 无 camera 词→Motion |
| Attribute vs Action | 外观/状态变化→Attribute；肢体动作→Action | "dog turns black"→Attribute; "dog turns head"→Action |
| Background vs View | 环境自身变化→Background；镜头运动→View | "sky darkens"→Background; "camera tilts up"→View |
| Spatial vs Background | 可数主体空间布局→Spatial；不可数环境区域→Background | "vase on table"→Spatial; "fog spreads"→Background |

### 13.2 维度隔离执行机制

基于 `dimension_cooccurrence` 矩阵，高共现维度对在出题时**强制压制**另一维度：

| 维度 | 隔离策略 |
|------|---------|
| Attribute | 禁动作词+方向词；camera默认fixed |
| Action | 强制 "camera remains fixed"；禁方向词 |
| Motion | 强制 "camera remains fixed"；禁动作词/属性变化词 |
| Spatial | 禁运镜词；可含轻微运动(因关系转移需位移) |
| Background | 强制 "camera remains fixed"；前景主体"remains still" |
| View | 禁内容语义编辑词；允许主体有轻微自然动作(如呼吸) |

---

## 十四、质量保障与对齐机制

### 14.1 先验对齐原则

1. **先验是唯一真值来源**：所有比例/权重/选择追溯到 prior_package
2. **外部资源只做扩展**：WordNet=罕见度调节器，COCO=可生成性过滤器
3. **句式还原真实用户**：模板来自 structural_templates，不是学术论文体
4. **难度分布反映真实**：subject_count_distribution → 难度桶配额

### 14.2 多图模式无先验支撑的缓解策略

TIP-I2V 为单图+单prompt结构，~40%多图题目无直接真实分布支撑。缓解：
1. 多图题的**主体词表和属性词表**仍从 prior 采样（保持概念分布对齐）
2. 多图比例设计基于**真实 I2V 使用场景调研**（Pika/Runway 多图功能使用频率）
3. 在论文中**显式声明**多图比例为合理外推而非数据驱动

### 14.3 数据质量保障链

| 环节 | 保障机制 |
|------|---------|
| 概念采样(Stage 1) | 兼容性矩阵校验 + evaluable_combinations(P7) |
| T2I 生成(Stage 3-4) | BLIP-VQA 风格结构化问答质检 |
| I2V 定稿(Stage 5) | 禁词检查 + 长度检查 + 维度隔离检查 |
| 落库前(Stage 6) | 先验自洽性 + 对比组完整性 |
| 全流程 | Human-in-the-loop 三处审查节点 |

### 14.4 可复现性保障

- Stage 1 Step 2 为确定性程序化采样（给定种子可完全重现）
- LLM 翻译使用 temperature=0
- Question Plan 全程保留，可回溯任何题目的生成逻辑
- 所有中间产出（vocabulary_bank/, question_plans/, t2i_prompts/）均持久化

---

## 十五、论文级统计规划

### 15.1 Benchmark 规模

| 维度 | 题数 | 对比组 | 单图占比 | 多图占比 | v1 主榜/stress |
|------|------|--------|---------|---------|---------------|
| Attribute Binding | 200 | 100对 | 60% | 40% | 全主榜 |
| Action Binding | 200 | 100对 | 60% | 40% | 全主榜 |
| Motion Binding | 200 | 100对 | ~100% | — | 全主榜(仅Type A) |
| Spatial Composition | 200 | 100对 | 20% | 80% | 全主榜 |
| Background Dynamics | 200 | 100对 | 60% | 40% | 全主榜 |
| View Transformation | 200 | 100对 | 95% | 5% | 全主榜 |
| **合计** | **1200** | **600对** | — | — | — |

### 15.2 待评测模型列表（建议）

| 类别 | 模型 |
|------|------|
| 开源 I2V | AnimateDiff, I2VGen-XL, DynamiCrafter, SVD, CogVideoX |
| 商业 I2V | Pika, Runway Gen-3, Kling, Hailuo MiniMax |
| Baseline | 静态复制(首帧不变), 随机运动 |

### 15.3 汇报指标

- 每维度独立 S 分 (= 0.45E + 0.35P + 0.20C)
- 总分 = 6 维度 S 分（待定权重）加权平均
- **细分分析**：按难度桶(Easy/Medium/Hard)、按罕见度(common/rare)
- **对比组偏向差分**：pair_A_score - pair_B_score 排除模型偏好
- **失败模式频率分布**：每维度 Top-3 失败模式
- **Radar chart**：6 维度 + 总分 可视化

### 15.4 论文消融实验建议

| 消融项 | 对比 | 验证什么 |
|--------|------|---------|
| common vs rare | 160题 vs 40题分别评分 | 模型对罕见概念的泛化能力 |
| Easy vs Hard | 按难度桶分别评分 | 复杂场景下的鲁棒性 |
| 单图 vs 多图 | 同维度两种模式分别评分 | 多参考条件下的组合能力 |
| E/P/C 分解 | 三轴独立报告 | 模型薄弱点诊断 |
| 对比组差分 | 正反对方向分别统计 | 方向/位置偏向 |

---

## 十六、待确认/待解决问题

| # | 问题 | 当前状态 | 影响范围 |
|---|------|---------|---------|
| Q1 | T2I 模型选择 | 待确认(FLUX.1 Pro / SDXL / DALL-E 3) | Stage 3 |
| Q2 | 图像分辨率 | 待确认(需匹配I2V模型输入) | Stage 3 |
| Q3 | 多图模式比例是否最终版 | 基本确认但可微调 | Stage 1 配额 |
| Q4 | Interaction Reasoning 恢复时间 | v2 | 维度7 |
| Q5 | 6 维度总分加权方式 | 待确认(等权 or 按难度) | 论文汇报 |
| Q6 | 先验补强 P1-P8 落地优先级 | P1/P3/P7 高优 | Phase 1 代码 |
| Q7 | 风格分布(P4) 落地前 T2I 统一 photorealistic 是否可接受 | 暂可接受 | Stage 2 |
| Q8 | Human-in-the-loop 审查人数和一致性要求 | 待确认 | Stage 1/4/5 |
