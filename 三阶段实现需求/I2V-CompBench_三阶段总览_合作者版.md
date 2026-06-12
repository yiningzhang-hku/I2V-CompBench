# I2V-CompBench Phase 1/2/3 三阶段总览（合作者版）

> 本文档面向项目合作者，说明 I2V-CompBench 的三阶段流程：先从真实 I2V 数据中提取题目构建依据，再生成标准化评测题，最后对不同模型生成的视频进行自动评测。文档保留必要的专业定义，同时用较直接的说明解释关键字段和处理流程，方便新加入项目的合作者理解整体设计。

---

## 0. 项目目标概述

I2V-CompBench 是一个面向图像到视频生成模型的结构化评测基准。它的目标不是笼统判断视频是否清晰、流畅，而是检查模型是否能在图像条件约束下完成指定变化。

- **Phase 1：数据分析与资产构建**。分析真实用户 prompt 和首帧图像，提取可用于构题的主体、属性、动作、关系、背景和参考图资产。
- **Phase 2：评测题生成**。根据 Phase 1 的分析结果生成标准化题目，包括输入图像、I2V prompt、目标变化和评测元数据。
- **Phase 3：模型生成与评测执行**。让待评模型生成视频，并检查输出是否满足题目要求。

该 benchmark 重点回答四类问题：

1. 目标内容是否按要求发生变化。
2. 变化是否绑定到正确主体。
3. 非目标主体、属性、背景或关系是否保持稳定。
4. 多张参考图中的主体、属性和场景是否能够被正确组合到同一视频中。

### 0.1 评测范围与设计原则

| 原则 | 说明 |
|---|---|
| 多图/多参考模式进入主评测 | 评测不仅覆盖单张首帧输入，也覆盖多张参考图输入。例如人物参考图、服装参考图和场景参考图共同作为输入。 |
| 七个维度共同组成评测主体 | 主评测覆盖属性、动作、运动、空间、背景、视角和交互七类能力。 |
| Phase 1 输出可构题资产 | Phase 1 不只输出统计表，还需要输出 `reference_bank`、`candidate_recipes`、兼容性矩阵和评测元数据。 |
| Phase 3 使用执行门控评分 | 样本得分为 `S = E * (0.6P + 0.4C)`。如果模型没有执行目标变化，即使画面稳定，总分也应较低。 |

### 0.2 关键术语说明

| 术语 | 简单解释 | 作用说明 |
|---|---|---|
| `prior_package` | 从真实数据中统计出的分布，例如常见主体、常见动作和常见场景 | 提供题目采样和论文分析的统计依据 |
| `reference_bank` | 从首帧中提取出的主体图、属性图、场景图等素材 | 提供多图题目的可复用参考资产 |
| `candidate_recipes` | 可转成题目的结构化候选方案 | 规定题目所需素材、目标变化和保持约束 |
| `Question Plan` | 一道题的结构化设计，包括输入图、目标变化、保持约束和评测方法 | 连接题目生成和后续评测 |
| `metadata` | 给评测器使用的目标定义和评分提示 | 明确评测目标、保持约束和工具选择 |
| E/P/C | Execution、Preservation、Coherence 三个分数 | 分别衡量目标执行、内容保持和时序连贯 |
| VLM | 能看图并回答问题的视觉语言模型 | 用于图像质检和部分视觉判断 |
| LLM | 负责读文本、抽取意图和槽位的语言模型 | 用于 prompt 解析和结构化字段抽取 |
| grounding | 在图像或视频中定位目标主体 | 确定评测对象的位置 |
| tracking | 在视频帧之间持续跟踪同一个主体 | 记录目标主体的时序变化 |
| baseline | 用来测试评测器可靠性的简单对照方法 | 检查评测器是否会奖励退化解或捷径解 |
| `contrastive pair` | 对比题对：一道正向题 + 一道反向/对照题 | 用于检查模型是否真听懂指令，而不是默认输出某一种结果。例：A "红球移到盒子左边" / B "红球移到盒子右边" |

### 0.3 难度桶定义（Easy / Medium / Hard）

文档全文使用 `difficulty ∈ {easy, medium, hard}` 与 40/40/20 比例分桶。判定规则如下：

| 难度 | 典型特征 |
|---|---|
| Easy | 单主体 + 常见概念 + 大目标（占首帧 ≥10%） + 单一变化 + 无 distractor |
| Medium | 双主体 + 常见概念 / 单主体 + 罕见概念；含 1 个 distractor；或多图但只有 2 张参考 |
| Hard | ≥3 主体或 ≥3 张参考图 / 罕见组合（rare semantic）/ 小目标（<3% 面积）/ 易混淆 distractor / 多变化共存 |

每条 `candidate_recipe` 在 Phase 1 由 `difficulty_factors`（binding_load + discrimination_difficulty + perceptual_complexity）综合算出难度桶；Phase 2 配额按 40/40/20 分别从对应桶采样。

---

## 1. 七个主维度与边界

| 编号 | 维度 | 核心问题 | 单图模式 | 多图/多参考模式 |
|---|---|---|---|---|
| D1 | Attribute Binding | 指定属性/状态是否绑定到正确主体 | 首帧内主体变色、开关、湿/干等状态变化 | 把参考服装、纹理、颜色或配饰迁移到目标主体 |
| D2 | Action Binding | 指定动作是否由正确主体执行 | 首帧内一个或多个主体执行不同肢体动作 | 多个参考人物/动物进入同一场景并执行各自动作 |
| D3 | Motion Binding | 轨迹级位移是否正确 | 单主体绝对方向运动；多主体相对位移变化 | 参考主体/物体进入场景后沿指定轨迹运动 |
| D4 | Spatial Composition | 多个主体是否被静态放置到正确关系 | 单图中空间布局保持或验证 | 多参考主体/物体按指定空间关系组装到场景 |
| D5 | Background Dynamics | 背景/环境是否正确变化且不污染前景 | 天气、光照、云、海浪、雾等背景区域变化 | 主体参考图 + 背景参考图的背景迁移/替换 |
| D6 | View Transformation | 镜头/视角运动是否符合指令 | zoom、pan、tilt、static 等运镜控制 | 少量多参考场景组合后的镜头控制 |
| D7 | Interaction Reasoning | 多主体/物体交互是否符合因果与常识 | 首帧中物体发生碰撞、递交、使用工具等交互 | 多个参考主体/物体被组合后发生物理/社会/功能交互 |

为便于理解，下面给出七个维度的典型任务示例：

- Attribute：白狗变黑，但棕狗不变。
- Action：左边的人挥手，右边的人保持站立。
- Motion：红球移动到蓝盒右边。
- Spatial：把猫放在沙发上，把绿植放在沙发旁边。
- Background：天空变阴，但前景人物不变。
- View：镜头慢慢拉近，但画面内容不被改写。
- Interaction：球撞倒积木，且倒下发生在碰撞之后。

**Motion vs Spatial 的判定规则**：

- 有主体位移轨迹：归入 **Motion Binding**。包括单主体向左/向右、A 从 B 左边移动到 B 右边、多主体重排。
- 无主体位移，只要求生成后的静态关系正确：归入 **Spatial Composition**。例如把猫放在沙发上、把花瓶放到书架左侧。

**Action vs Interaction 的判定规则**：

- 单主体自身肢体动作：Action Binding。
- 多主体/物体之间有因果、接触、传递或功能使用：Interaction Reasoning。

---

## 2. 三阶段总流程

三阶段流程可以概括为“数据分析、题目生成、模型评测”：

- **数据分析**：从真实 I2V 数据中提取用户需求、图像结构和可评测变化。
- **题目生成**：将真实需求转化为标准化题目，例如“红球移动到蓝盒右边”。
- **模型评测**：模型生成视频后，检查输出是否满足题目要求。

```text
Phase 1: 先验分析与可出题资产构建
  TIP-I2V prompt + 首帧
    -> 清洗 manifest
    -> 图像结构化解析
    -> 文本变化意图解析
    -> 图文对齐与可评测性判定
    -> 首帧多主体分解，构建 reference_bank
    -> 兼容性矩阵、先验统计、candidate_recipes
    -> phase1_bundle

Phase 2: Benchmark 题目生成
  phase1_bundle
    -> 七维度配额与输入模式采样
    -> 从 candidate_recipes 采样，不直接随机拼词
    -> 生成 Question Plan
    -> 生成/抽取首帧与参考图
    -> VQA/QC 质检
    -> 定稿 I2V prompt
    -> 导出 benchmark_samples 与 phase3_manifest

Phase 3: 模型生成与评测执行
  benchmark_samples + 待评模型
    -> 生成视频
    -> 视频标准化
    -> 输入图像 grounding 与视频 tracking
    -> 维度专属 E/P/C 评测
    -> 执行门控评分
    -> 失败模式、baseline、人类一致性验证
    -> 模型报告与论文图表
```

这里有一个重要原则：Phase 2 不应脱离 Phase 1 随机构题。每一道题都应该能追溯到 Phase 1 的统计、素材或 recipe。这样在论文中可以清楚说明，题目不是人工随意构造的，而是从真实 I2V 使用场景中整理出来的。

---

## 3. Phase 1：先验分析与可出题资产构建

### 3.1 Phase 1 的目标

Phase 1 不是只统计词频，而是把真实 I2V 数据转化成可以被 Phase 2 直接采样的题目级构建资产。

直观地说，Phase 1 是对原始数据进行结构化整理：原始数据中包含用户 prompt、首帧图片和平台参数，我们需要把它们转化为清楚的标签、主体实例、空间关系、图像切片和可构题方案。

它要回答四个问题：

1. 真实 I2V 用户常要求哪些主体、属性、动作、运动、背景和运镜？
2. 首帧中有哪些可识别主体、关系、背景、视角和可分割区域？
3. 哪些图文组合可以变成可评测题目？
4. 如何从单首帧中派生出多图/多参考任务，并让这些任务有可追溯的先验来源？

### 3.2 Phase 1 输入

主输入：

- TIP-I2V 样本：用户 prompt、首帧缩略图、UUID、Pika 参数、主体类别等。
- 实践上可以先跑 pilot：100 到 300 条；正式先验最好至少覆盖 10K 条。

辅助输入：

- WordNet/COCO/OpenImages：用于补充可视化物体词表。
- Visual Genome/Places365：用于补充空间关系和场景词表。
- 人工规则表：维度边界、禁词、可评测动作/属性/背景变化表。

这些辅助资源的作用不是替代 TIP-I2V，而是补洞。例如真实数据里可能很少出现某些罕见物体，但 benchmark 需要少量长尾样本，就可以用 WordNet 或 COCO 补充；真实数据里有关系，但关系类别不够标准，就可以用 Visual Genome 的关系表统一命名。

### 3.3 Phase 1 数据处理流程

#### Step 1：Manifest 清洗与参数解析

输入 TIP-I2V 原始流，输出 `manifest_clean.jsonl` 与落盘首帧。

`manifest` 可以理解成“样本清单”。它记录每条原始数据的编号、清洗后的 prompt、首帧图片路径、平台参数和当前状态。后面的所有步骤都靠这个清单找到样本。

处理内容：

- 解析 Pika 参数：camera、motion、negative prompt、seed、fps、aspect ratio 等。
- 清洗 prompt：去除平台参数，保留自然语言意图。
- 首帧落盘：统一格式、尺寸、路径。
- 质量分流：`ok`、`missing_image`、`empty_prompt`、`bad_format`。

产出字段示例：

```json
{
  "sample_id": "tip_000001",
  "clean_prompt": "a dog runs across a grassy field",
  "image_path": "data/phase1/images/tip_000001.jpg",
  "pika_camera": "static",
  "pika_motion": 2,
  "status": "ok"
}
```

字段解释：

- `sample_id`：我们给样本重新编号，方便后续追踪。
- `clean_prompt`：去掉平台参数后的自然语言描述。
- `pika_camera`：用户原来选择的镜头设置，可以帮助我们了解真实用户喜欢什么运镜。
- `pika_motion`：用户原来选择的运动强度，可以辅助 Motion 难度设计。
- `status`：这条样本是否能继续进入后续流程。

#### Step 2：图像结构化解析

对每张首帧进行 VLM + detection/segmentation 解析，输出 `image_parse.jsonl`。

核心字段：

- `subjects[]`：主体类别、bbox、mask、颜色/材质/状态、姿态、画面位置、是否 animate、可分割质量。
- `subject_relations[]`：left/right/above/below/in_front_of/behind/on/beside/between。
- `background`：场景类别、动态区域、天气、光照、前景背景可分离性。
- `camera_baseline`：shot type、camera angle、depth cues、是否有刚性参照结构。
- `reference_potential`：该首帧是否适合拆成多参考图。

简单理解：

- `bbox` 是物体外面的矩形框，用来告诉评测器“这个主体大概在哪里”。
- `mask` 是更精细的主体轮廓，用来区分主体和背景。
- `subject_relations` 记录物体之间的位置关系，例如狗在球左边。
- `foreground_background_separability` 表示前景和背景容不容易分开。它对 Background 维度很重要，因为如果前景背景混在一起，后面很难判断模型是不是只改了背景。
- `reference_potential` 判断这张首帧能不能拆成多张参考图。比如一张图里有清楚的人、衣服和背景，就比较适合派生多图题。

#### Step 3：文本变化意图解析

对清洗后的 prompt 进行 LLM 结构化解析，输出 `text_parse.jsonl`。

必须覆盖七个维度槽位：

| 槽位 | 示例字段 |
|---|---|
| Attribute | target_subject, attribute_type, from_value, to_value |
| Action | target_subject, action_verb, action_detail |
| Motion | target_subject, direction, target_relation, reference_subject |
| Spatial | subject_list, target_static_relations |
| Background | target_region, change_type, from_state, to_state |
| View | camera_command, direction, framing_constraint |
| Interaction | actor, object, interaction_type, expected_effect |

文本解析还要输出：

- `primary_dimension` 与 `candidate_dimensions`
- `transform_ontology`
- `contrastive_transform`
- `forbidden_dimension_leakage`
- 解析置信度与歧义标记

这些字段的意思：

- `primary_dimension`：这条 prompt 最主要在测哪个维度。
- `candidate_dimensions`：如果有歧义，记录可能涉及的维度。
- `transform_ontology`：把各种自然语言说法归一成标准变化类型。例如 “move left”“go to the left”“shift leftward” 都可以归成 `move_left`。
- `contrastive_transform`：这道题的反向对照，例如 `move_left` 的对照可以是 `move_right`。
- `forbidden_dimension_leakage`：出题时要避免混入的其他能力。例如 Attribute 题里不要出现明显运镜词，否则题目就不纯了。

#### Step 4：图文对齐与可评测性判定

把文本里的主体、关系和变化要求对齐到图像实例。

判定逻辑：

- 文本要求的目标是否在图像中可见？
- 目标主体是否可检测、可分割、可追踪？
- 该变化是否能被 Phase 3 自动或半自动评测？
- 是否混入其他维度，导致题目边界不干净？
- 是否能构造对比组？

输出 `evaluator_feasibility`：

```json
{
  "dimension": "motion_binding",
  "is_evaluable": true,
  "required_tools": ["GroundingDINO", "SAM2", "CoTracker", "RAFT"],
  "target_instance_ids": ["s1"],
  "risk_flags": ["small_target"],
  "confidence": 0.82
}
```

这里的 `evaluator_feasibility` 是在提前问：“这道题未来能不能被机器批改？”例如目标太小、遮挡太严重、主体根本检测不到，即使题目语义合理，也不适合作为自动 benchmark 样本。

#### Step 5：多参考资产构建 `reference_bank`

这是多图进入主维度后的关键新增步骤。

`reference_bank` 是可复用参考素材库。我们把首帧里有用的主体、属性和场景拆分出来，供 Phase 2 构建多图题目使用。这样，多图题的参考素材来自真实 I2V 首帧，而不是由无关图片临时拼接。

从 TIP-I2V 单首帧中提取：

- 主体参考图：每个主体输出 tight crop、padded crop、masked RGBA 三种版本。
- 属性参考图：服装、配饰、局部纹理等可迁移属性区域。
- 场景参考图：原始场景、主体移除后的 inpainted scene、可选合成 clean scene。
- 关系参考：该首帧中的真实主体共现、尺度、光照、空间关系。

每个 reference asset 必须保存质量字段：

```json
{
  "ref_id": "tip_000001_s1_masked",
  "source_sample_id": "tip_000001",
  "role": "subject_reference",
  "category": "dog",
  "path": "reference_bank/tip_000001/s1_masked.png",
  "bbox": [120, 80, 340, 420],
  "mask_quality": 0.88,
  "background_leakage_risk": "low",
  "identity_visibility": "high",
  "usable_for": ["attribute_binding", "action_binding", "motion_binding", "spatial_composition", "interaction_reasoning"]
}
```

这一步解决“多图模式没有先验”的问题：多图题不是凭空拼出来的，而是从真实 I2V 首帧中的主体共现、场景关系和视觉属性中派生出来。

质量字段很重要：

- `mask_quality`：主体轮廓切得好不好。
- `background_leakage_risk`：裁出来的主体图里是否带了太多原背景。风险高时，模型可能偷看背景信息，影响多图评测公平性。
- `identity_visibility`：主体身份是否清楚。比如人脸太小、动物只露一半，身份保持就不好评。
- `usable_for`：这个参考图适合用在哪些维度。

#### Step 6：兼容性矩阵与先验统计

Phase 1 要输出三类兼容性信息：

1. **真实共现矩阵**：来自 TIP-I2V 首帧和 prompt 的 subject-subject、subject-scene、subject-action、subject-attribute 共现。
2. **视觉兼容矩阵**：来自 reference_bank 的尺度、视角、光照、分割质量、背景污染风险。
3. **评测兼容矩阵**：是否能被 Phase 3 工具稳定评测。

示例：

```json
{
  "subject_scene_compatibility": {
    "dog|park": {"count": 84, "score": 0.93},
    "fish|street": {"count": 0, "score": 0.05}
  },
  "multi_reference_compatibility": {
    "subject_ref|scene_ref": {
      "scale_match": 0.81,
      "lighting_match": 0.74,
      "scene_capacity": "enough",
      "composition_score": 0.79
    }
  }
}
```

兼容性矩阵的作用是避免语义不合理或难以评测的题目。例如，“鱼在街上跑步”虽然语法成立，但不适合作为常规 I2V benchmark 题；“狗在公园里奔跑”更符合常见视觉语境，也更容易评测。多图模式还需要检查参考图之间是否兼容，例如人物光照和背景光照差异过大时，后续生成和评测都会受到影响。

#### Step 7：生成 `candidate_recipes`

`candidate_recipes.jsonl` 是 Phase 2 的主要采样入口。每条 recipe 是一个可转化成题目的结构化候选项。

这里的 `recipe` 是结构化题目候选方案。它不一定已经是一道最终题，但会说明所需主体、目标变化、参考图要求、保持约束以及后续评测方式。

recipe 来源分四类（与 Phase 1/2/3 实现需求文档保持一致）：

| 来源 `source_type` | 含义 | 是否进入主维度 |
|---|---|---|
| `observed_single_image` | 直接来自 TIP-I2V 图文意图的单图题 | 是 |
| `derived_single_image` | 从真实先验组合中派生出的单图题 | 是 |
| `derived_multi_reference` | 从 TIP 首帧拆解出的多参考题 | 是 |
| `external_multi_reference_optional` | 外部公开多参考资源补充（pilot 不使用，正式版本可选启用，必须保留 provenance） | 可选补充 |

### 3.4 Phase 1 生成结果

Phase 1 最终输出一个 `phase1_bundle/`：

```text
phase1_bundle/
  manifest_clean.jsonl
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

其中：

- `prior_package.json`：统计先验，用于论文描述真实分布与配额。
- `reference_bank/`：多图题目的图像资产来源。
- `candidate_recipes.jsonl`：Phase 2 的唯一题目候选来源。
- `compatibility_matrix.json`：防止随机、不合理、不可评测组合。

这四类输出的关系可以这样理解：`prior_package` 告诉我们“真实世界里什么常见”；`reference_bank` 提供“可以拿来出题的素材”；`compatibility_matrix` 负责“检查组合是否合理”；`candidate_recipes` 则把这些信息合成“可出题方案”。

---

## 4. Phase 2：Benchmark 题目生成流程

### 4.1 Phase 2 的目标

Phase 2 要把 Phase 1 的 recipe 和参考资产转成最终 benchmark 样本。每道题不仅要有输入图像和 prompt，还要有 Phase 3 可直接使用的评测元数据。

Phase 2 的作用是将 Phase 1 的候选方案转化为正式评测样本。最终样本需要包含输入图像、文本指令、目标变化、保持约束，以及 Phase 3 评测时使用的元数据。

正式目标规模：

```text
7 个维度 × 200 题 = 1400 道题
每题尽量有 contrastive pair
common/rare = 80/20
difficulty = Easy/Medium/Hard = 40/40/20
```

如果算力或人力不足，可以先跑 pilot：

```text
7 个维度 × 20 题 = 140 道题
每维度至少覆盖 single_image 与 multi_image
```

### 4.2 主维度中的单图/多图配额

本项目采用以下单图/多图比例。这里的比例不是为了追求形式平均，而是根据不同维度的任务性质来定：Spatial Composition 本来就更适合多参考空间组装，所以多图占比高；View Transformation 主要测镜头控制，多图意义较小，所以多图占比低。

| 维度 | 单图/多图比例 |
|---|---|
| Attribute Binding | 60% 单图属性变化 + 40% 多图属性迁移 |
| Action Binding | 60% 单图动作绑定 + 40% 多图动作场景合成 |
| Motion Binding | 50% 类型A单主体绝对运动 + 35% 类型B多主体相对位移 + 15% 类型C多图组合运动 |
| Spatial Composition | 20% 单图布局保持/验证 + 80% 多图空间组装 |
| Background Dynamics | 60% 单图背景状态变化 + 40% 多图背景迁移/替换 |
| View Transformation | 95% 单图运镜控制 + 5% 多图场景组合后运镜 |
| Interaction Reasoning | 50% 单图交互 + 50% 多图交互 |

这张表是论文里解释多图模式的关键：多图不是独立维度，而是每个组合能力维度下的一种输入模式。

### 4.3 Phase 2 详细流程

#### Stage 0：配额表与模板注册

输入：

- `candidate_recipes.jsonl`
- `prior_package.json`
- `compatibility_matrix.json`
- 七维度模板库

输出：

- `quota_plan.json`
- `template_registry.json`

模板必须声明适用条件：

- dimension
- input_mode
- required slots
- forbidden dimensions
- evaluator requirements
- contrastive transform

这一阶段像是在决定“试卷结构”：每个维度出多少题，单图和多图各多少，简单题和困难题各多少。模板则像题型格式，例如 Attribute 维度可以有 “The [target] turns [color] while [distractor] remains unchanged.” 这种固定句式。

#### Stage 1：Recipe 采样

Phase 2 不允许直接从词表随机拼 prompt。它必须从 `candidate_recipes` 中采样。

采样顺序：

1. 按维度和 input_mode 读取配额。
2. 按 difficulty 与 common/rare 分桶。
3. 从 matching recipes 中采样。
4. 检查 compatibility 与 evaluator_feasibility。
5. 为可对照任务生成 contrastive pair。

`contrastive pair` 是对比题。比如一题是“红球移动到盒子左边”，另一题是“红球移动到盒子右边”。这样可以检查模型是不是真的听懂方向，而不是总偏向某一种默认结果。

#### Stage 2：生成 Question Plan

每道题先生成结构化 Question Plan，而不是直接生成自然语言 prompt。

Question Plan 必须包含：

- 输入图像计划：single first frame 或多个 reference images。
- 目标变化：谁应该变、变成什么、在哪里发生。
- 保持约束：谁不应该变、哪些属性/区域不能变。
- 维度隔离约束：禁止动作、运动、镜头、属性等混入。
- Phase 3 evaluator metadata：目标 bbox/mask/ref_id、预期终态、工具提示。

`Question Plan` 是最终题目前的“设计图”。它比 prompt 更重要，因为 prompt 是给模型看的自然语言，而 Question Plan 是给出题系统和评测系统看的结构化说明。没有它，Phase 3 就只能重新猜 prompt 的意思，评测会不稳定。

#### Stage 3：首帧与参考图构建

三种图像来源按优先级使用：

1. **TIP-derived real assets**：如果原始首帧或 reference_bank 资产质量足够，优先使用，增强真实分布可信度。
2. **T2I-generated clean assets**：用于补足难度桶、罕见组合和对比组。
3. **external optional assets**：用于补充真实多参考场景，但必须记录 provenance。

单图模式：

- 可复用 TIP 首帧。
- 可由 T2I 根据 Question Plan 生成 clean first frame。

多图模式：

- 主体图、属性图、场景图分别来自 reference_bank 或 T2I 生成。
- 每张参考图尽量承载一个清晰元素。
- 参考图必须记录来源、crop 污染风险、mask 质量和与其他参考图的兼容性。

多图题里，每张参考图最好“职责单一”。例如图 1 只负责人物身份，图 2 只负责红色外套，图 3 只负责咖啡馆场景。这样模型失败时，我们才能判断它到底是没保住人物、没迁移外套，还是没合成场景。

#### Stage 4：图像与参考资产质检

采用结构化 VQA，而不是自由描述。

质检内容：

- 主体数量是否正确。
- 目标属性/状态是否清楚。
- 目标主体是否可分割、可追踪。
- 多图参考是否单一、干净、无严重背景泄漏。
- 场景参考图是否有足够空间。
- 关系是否无歧义。
- 首帧是否没有提前泄露终态。

不通过处理：

- 自动重写 prompt 或重新选择 reference。
- 最多重试 3 次。
- 仍失败则进入 `needs_manual_review` 或丢弃。

结构化 VQA 的意思是把检查拆成一串是非题或选择题，而不是让模型自由评价“这张图好不好”。例如：

```text
Q1: Is there exactly one red ball?
Q2: Is the blue box visible?
Q3: Is the red ball initially left of the blue box?
Q4: Is there enough empty space on the right side?
```

这种做法更稳定，也更容易复现。

#### Stage 5：I2V Prompt 定稿

基于实际通过质检的图像，定稿 I2V prompt。

规则：

- prompt 只表达目标维度。
- 每个主体指代唯一。
- 长度以 8 到 25 个英文词为主。
- 禁用其他维度的触发词。
- 多图 prompt 必须明确 reference 的作用，例如 “the woman from image 1”，“the red jacket from image 2”，“the empty cafe from image 3”。

这里要注意，I2V prompt 是给模型看的，不是给评测器看的。它应该简洁、明确、只说本题要测的能力。评测器真正依赖的是后面的 `metadata`。

#### Stage 6：导出 benchmark sample

每道题输出 `BenchmarkSample`：

```json
{
  "question_id": "motion_B_0042",
  "dimension": "motion_binding",
  "input_mode": "single_image",
  "subtype": "relative_displacement",
  "difficulty": "medium",
  "semantic_rarity": "common",
  "contrastive_pair_id": "motion_pair_021",
  "input_images": [
    {
      "path": "benchmark_dataset/images/motion_B_0042.png",
      "role": "first_frame"
    }
  ],
  "i2v_prompt": "The red ball moves to the right side of the blue box.",
  "metadata": {
    "target_subjects": ["s1"],
    "reference_subjects": ["s2"],
    "expected_change": {
      "type": "relative_displacement",
      "subject": "s1",
      "reference": "s2",
      "target_relation": "right_of"
    },
    "preserve_constraints": [
      {"scope": "s2", "constraint": "appearance_and_position"},
      {"scope": "background", "constraint": "stable"}
    ],
    "evaluator_hints": {
      "tools": ["GroundingDINO", "SAM2", "CoTracker", "RAFT"],
      "primary_metric": "relative_displacement"
    }
  },
  "source_trace": {
    "recipe_id": "recipe_motion_0091",
    "source_type": "derived_single_image",
    "phase1_sample_ids": ["tip_000918"]
  }
}
```

这个样本里几个关键字段的含义：

- `dimension`：这题测哪个能力。
- `input_mode`：是单图输入还是多图输入。
- `subtype`：该维度下更细的题型，例如 Motion 里的相对位移。
- `difficulty`：难度桶。
- `semantic_rarity`：常见/罕见组合。
- `metadata.expected_change`：这道题要求模型完成的目标变化，相当于隐藏答案。
- `metadata.preserve_constraints`：不该被改变的内容。
- `metadata.evaluator_hints`：评测器应该优先使用哪些工具和指标。
- `source_trace`：这道题从哪条 recipe、哪批 Phase 1 样本来，保证可追溯。

一个例子：如果 prompt 是 “The red ball moves to the right side of the blue box.”，那么 `expected_change` 会告诉评测器“红球最终应该在蓝盒右边”，`preserve_constraints` 会告诉评测器“蓝盒外观和位置要保持，背景要稳定”。

多图样本必须额外包含：

```json
{
  "input_mode": "multi_image",
  "reference_assets": [
    {"ref_id": "ref_person_001", "role": "target_subject"},
    {"ref_id": "ref_jacket_002", "role": "attribute_reference"},
    {"ref_id": "ref_scene_003", "role": "scene_reference"}
  ],
  "multi_reference_quality": {
    "crop_leakage_risk": "low",
    "scene_leakage_risk": "medium",
    "identity_visibility": "high",
    "scale_compatibility": 0.82
  }
}
```

这些多图质量字段决定样本是否公平：

- `crop_leakage_risk`：主体参考图是否带了太多原场景信息。
- `scene_leakage_risk`：场景参考图里是否残留了不该出现的主体。
- `identity_visibility`：参考主体是否足够清楚，便于之后判断 identity 是否保持。
- `scale_compatibility`：不同参考图之间尺度是否合理，例如不能让一把椅子看起来比房间还大。

### 4.4 Phase 2 生成结果

Phase 2 最终输出：

```text
benchmark_dataset/
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

其中 `phase3_manifest.jsonl` 是评测执行的主入口。它必须保证 Phase 3 不需要重新解析 prompt 才能知道目标、保持约束和评测方式。

`phase3_manifest.jsonl` 是评测执行清单。它列出每道题的输入、prompt、目标变化和评分规则，使 Phase 3 不需要重新解释自然语言 prompt。

---

## 5. Phase 3：评测执行、题目定义与评测器

### 5.1 Phase 3 的目标

Phase 3 对每个待评 I2V 模型执行：

1. 输入 benchmark sample 中的图像和 prompt，生成视频。
2. 使用输入图像和 reference metadata 锚定目标实例。
3. 按维度计算 Execution、Preservation、Coherence。
4. 用执行门控总分防止静态复制退化解。
5. 输出失败模式、工具置信度、人类一致性验证和模型报告。

直观地说，Phase 3 是对模型输出视频进行结构化评分。评测器需要判断视频是否按题目要求完成任务。这里最重要的是不能只看视频是否流畅或清晰，而要检查它是否完成了指定变化。

### 5.2 统一评分定义

每道题都有三个分数：

- **Execution, E**：指定目标变化是否发生在正确目标上，并达到题目要求。
- **Preservation, P**：非目标主体、属性、背景、身份、空间关系是否被错误污染。
- **Coherence, C**：变化过程是否稳定、连续、无闪烁、无回跳、无突变。

可以用一个简单例子理解 E/P/C：

题目是“让白狗变黑，同时棕狗保持不变”。

- E 看白狗有没有真的变黑。
- P 看棕狗、背景、白狗的其他身份特征有没有被误改。
- C 看变黑的过程是否稳定，不要一会儿黑、一会儿白，或者狗突然变形消失。

样本分数：

```text
S = E * (0.6P + 0.4C)
```

含义：

- 如果模型没有执行目标变化，`E=0`，总分为 0。
- P/C 不奖励“什么都不做”。
- E/P/C 仍需独立报告，因为它们对应不同失败原因。

为什么要用乘法门控？因为 I2V 模型很容易出现一种退化解：直接把首帧复制成一个静态视频。这个视频通常非常稳定，P 和 C 看起来会很高，但它根本没有执行目标变化。如果用普通加权平均，这类输出可能获得不合理的高分。用 `E * (...)` 后，只要目标变化没有执行，总分就会被压低。

维度分数：

```text
S_dim = mean(S_i)
E_dim = mean(E_i)
P_dim = mean(P_i)
C_dim = mean(C_i)
```

总分（默认混合 input_mode）：

```text
S_overall = mean(S_dim across 7 dimensions)
S_dim     = mean(S_i for all samples in this dimension, regardless of input_mode)
```

为避免不支持多图的模型被显著低估，同时**单独**报告 input_mode 切片下的分数：

```text
S_dim_single = mean(S_i for samples with input_mode = single_image)
S_dim_multi  = mean(S_i for samples with input_mode = multi_image)
S_overall_single = mean(S_dim_single across 7 dimensions)
S_overall_multi  = mean(S_dim_multi across 7 dimensions)
```

报告口径：

- 主表使用混合 `S_overall` 进行模型横向对比。
- 附加表分别报告 single-only 与 multi-only，便于揭示模型在多参考组合任务上的能力短板。
- 不支持多图的模型在 multi-only 表中将自然得到较低分数，但其 single-only 分数不受影响。

### 5.3 七维度评测题目定义与评测器

#### D1 Attribute Binding

题目定义：

- 单图：首帧中指定主体发生属性/状态变化，其他主体保持。
- 多图：目标主体吸收参考图中的属性、服装、配饰或纹理。

E：

- 指定主体达到目标属性/状态。
- 多图中参考属性被迁移到正确主体。

P：

- 非目标主体属性不被污染。
- 目标主体未指定属性与身份保持。
- 背景不过度变化。

C：

- 属性变化过程无闪烁、无回跳、无错误扩散。

评测器：

- 颜色/亮度/状态：HSV/Lab、区域统计、结构化 VQA。
- 高层属性：MLLM 多选题，不用自由问答。
- 目标与非目标区域：GroundingDINO + SAM2 + tracker。

示例：白狗变黑。评测器先找到白狗区域，再判断该区域颜色是否从白色变成黑色；同时检查旁边棕狗的颜色是否没有被一起改掉。

#### D2 Action Binding

题目定义：

- 单图：指定主体执行目标动作，干扰主体不执行或执行不同动作。
- 多图：多个参考人物/动物进入同一场景，并执行各自被分配的动作。

E：

- 目标主体在 tube-level 上执行目标动作，而不是单帧伪动作。

P：

- 非目标主体没有动作泄漏。
- 多主体动作没有互换。
- 身份保持。

C：

- 动作完整、持续、时序自然。

评测器：

- DWPose/ViTPose 姿态序列。
- 原子动作分类器或规则证据。
- 非目标主体姿态变化幅度与动作泄漏检测。

示例：左边的人挥手，右边的人不动。评测器不只看某一帧是否像挥手，而要看一段时间里的手臂轨迹是否构成挥手动作，同时检查右边的人是否也出现了挥手动作。

#### D3 Motion Binding

题目定义：

- 类型 A：单主体沿 canonical direction 移动。
- 类型 B：主体移动到参照主体的目标关系位置。
- 类型 C：多图参考主体进入场景后沿指定轨迹运动。

E：

- 方向正确。
- 位移超过最小阈值。
- 类型 B 的终态关系正确。
- toward/away 需要尺度或深度变化支持。

P：

- 背景没有被整体拖动。
- 非目标主体没有被误带动。
- 没有用 camera pan 冒充主体运动。

C：

- 轨迹平滑，无瞬移、无大幅 tracking 跳变。

评测器：

- GroundingDINO/SAM2 定位目标。
- CoTracker 跟踪主体中心或关键点。
- RAFT/光流估计背景运动。
- 相机补偿：

```text
delta_p_rel = delta_p_foreground - delta_p_background
```

这个公式的意思是把相机运动扣掉。比如镜头向右平移时，画面里的所有东西都会看起来向左移动。如果不做相机补偿，评测器可能把“镜头在动”误判成“主体在动”。`delta_p_rel` 看的是主体相对于背景的真实位移。

#### D4 Spatial Composition

题目定义：

- 单图：已有空间布局在视频中保持正确。
- 多图：把多个参考主体/物体静态放置到指定空间关系中。

注意：如果主体发生位置移动以达到关系，属于 Motion Binding；Spatial 关注生成后布局是否正确且稳定。

E：

- 目标静态关系成立：left/right/above/below/in_front_of/behind/on/beside/between。

P：

- 参考主体身份、外观、数量保持。
- 非目标关系和背景不被破坏。

C：

- 关系在视频中稳定，不出现主体消失、穿插、尺度崩坏。

评测器：

- GroundingDINO + SAM2 检测与分割。
- DepthAnything 辅助前后关系。
- bbox/mask 几何关系判定。
- 多图 identity preservation 使用 CLIP-I、DINO 特征或人脸/物体特征匹配。

示例：把猫放在沙发上，绿植放在沙发旁边。评测器需要找出猫、沙发和绿植，再判断 “on” 和 “beside” 这两个关系是否成立。这里不要求猫从某处移动到沙发上；如果题目要求移动过程，则应归入 Motion。

#### D5 Background Dynamics

题目定义：

- 单图：指定背景区域发生天气、光照、动态纹理或场景状态变化。
- 多图：主体保留，背景迁移到参考场景。

E：

- 背景目标变化可见、局部发生、语义符合 prompt。

P：

- 前景主体身份、外观、位置保持。
- 非目标区域不被全局滤镜污染。

C：

- 背景变化渐进、稳定，无突变闪烁。

评测器：

- SAM2 前景/背景分割。
- CLIP embedding / LPIPS / SSIM 区域变化。
- anti-shortcut：比较目标背景区域变化量与前景/非目标区域变化量。

示例：让天空变阴，但人物保持不变。合格输出应主要改变天空区域；如果模型对整张图施加灰色滤镜，导致人物也变暗，则应扣 P。

#### D6 View Transformation

题目定义：

- 评测相机/视角运动是否符合 prompt，而不是内容语义变化。
- 多图只占少量，用于组合场景后的镜头控制。

E：

- camera motion 类型、方向、幅度正确：zoom/pan/tilt/static。

P：

- 主体身份与属性保持。
- 刚性背景内容不被重写。

C：

- 运镜平滑，无 jitter、无突然裁切。

评测器：

- RAFT 光流。
- 单应矩阵估计。
- zoom/pan/tilt 参数分解。
- 主体 identity 与背景一致性检测。

示例：题目要求 camera slowly zooms in。评测器需要判断画面是否呈现整体放大，同时检查主体和背景内容是否没有被模型重写。

#### D7 Interaction Reasoning

题目定义：

- 物理交互：碰撞、推动、倒下、倒水、破坏、接取。
- 社会交互：握手、拥抱、递物、追逐、跟随。
- 功能交互：钥匙开门、按开关、工具使用。

E：

- 交互双方正确。
- 交互行为发生。
- 预期结果或因果效果成立。

P：

- 非交互主体不被卷入。
- 主体身份和关键属性保持。
- 背景不被无关改写。

C：

- 事件顺序合理，因果链连续，无结果提前出现或突然跳变。

评测器：

- 结构化 MLLM 问答作为主评测器。
- detection/tracking 验证参与主体、接触、距离变化和结果状态。
- 对物理任务使用状态规则：是否倒下、是否接触、是否进入容器、是否破碎等。
- Interaction 维度必须配较高比例人工验证，因为自动评测器难度最高。

示例：球滚下斜坡撞倒积木。评测器需要检查球是否接近并接触积木，积木是否在接触后倒下；如果积木一开始就倒下，或球没有接触积木而积木直接倒下，都属于不合理输出。

### 5.4 工具置信度与 uncertain case

每个样本必须输出工具置信度：

```json
{
  "tool_confidence": {
    "grounding": 0.91,
    "tracking": 0.83,
    "vlm": 0.78,
    "flow": 0.81
  },
  "tool_status": "valid"
}
```

状态：

| 状态 | 含义 | 处理 |
|---|---|---|
| `valid` | 工具读数可靠 | 纳入主分 |
| `low_confidence` | 工具可读但不稳定 | 纳入主分并单独报告 |
| `tool_uncertain` | 关键工具无法可靠读取 | 不纳入主分，进入人工复核池 |
| `invalid_input` | benchmark 样本自身不合格（极少见，应在 Phase 2 拦截） | **保留在 manifest 中但不参与 mean 聚合**，并在 `failure_diagnostics.json` 中显式记录 question_id 与排除原因，保证不同模型的分母一致 |

为什么要有 `tool_uncertain`？因为生成视频可能会有严重形变，导致检测器或跟踪器读不准。此时不能粗暴地说模型 E=0，因为那可能是评测工具失灵，而不是模型没有执行。正确做法是把这类样本标出来，单独统计或交给人工复核。

### 5.5 必跑 baseline

为了证明评测器不会奖励退化解，Phase 3 必须包含：

- **Static Copy**：首帧重复成视频。预期 `S≈0`。
- **Random Motion**：随机局部移动或随机光流。预期 E 低、C 低。
- **Global Filter**：全局变色/变亮。用于 Background 和 Attribute anti-shortcut。
- **Camera Pan Cheat**：整体平移画面。用于 Motion vs View 区分。

这些 baseline 用于检测评测器是否会奖励退化解或捷径解。如果评测器无法识别这些简单对照，benchmark 的可信度就会很弱。例如，Static Copy 如果还能拿高分，就说明评分公式存在问题；Camera Pan Cheat 如果在 Motion 维度拿高分，就说明评测器没有区分主体运动和镜头运动。

### 5.6 人类一致性验证

Benchmark 论文必须验证自动评测器与人类判断相关。

推荐的人工验证设置：

- 每个维度抽 50 到 100 道题。
- 每题选 3 到 4 个模型输出。
- 人工分别标注 E/P/C，而不是只给整体分。
- 报告 Spearman/Kendall 相关系数与标注者间一致性。
- 按 artifact severity 分桶报告相关性，证明工具在生成视频上的有效边界。

这一步的作用是证明自动评测结果与人工判断大体一致。如果自动分数和人类判断完全不相关，评测器再复杂也没有意义。对 benchmark 论文来说，这通常是最关键的可信度证据之一。

### 5.7 Phase 3 生成结果

每个模型输出：

```text
runs/eval/{model_name}/
  generations/
  per_sample_scores.jsonl
  dimension_scores.json
  input_mode_scores.json
  overall_report.json
  failure_diagnostics.json
  tool_uncertainty_report.json
  human_subset_manifest.jsonl
  human_correlation_report.json
  visualizations/
```

论文主要图表：

- 七维度雷达图。
- E/P/C 分解柱状图。
- single-image vs multi-reference 对比。
- 与 VBench-I2V 排名差异。
- 静态复制 baseline 表。
- 人类一致性相关性表。
- 典型失败案例：binding failure、leakage、non-execution、identity loss、global filter shortcut、camera-motion cheating。

---

## 6. 论文可以怎么讲

### 6.1 方法贡献

论文里可以把方法贡献集中在三点：

1. **Image-grounded selective compositionality**：在首帧或多参考图锚定下，评测目标变化是否正确绑定。
2. **TIP-derived multi-reference construction**：从真实 I2V 首帧拆解多参考资产，解决多图题完全无先验的问题。
3. **Exec-gated E/P/C evaluation**：把执行、保持和连贯性拆开，避免静态复制退化解。

### 6.2 与 T2I-CompBench/T2V-CompBench 的关系

继承：

- 真实分布驱动的题目构造。
- 维度化评测。
- common/rare split。
- 对比组设计。
- 自动评测器与人类判断相关性验证。

扩展：

- 从 T2I/T2V 的文本组合，转向 I2V 的图像锚定选择性变化。
- 加入 Preservation/leakage 作为核心轴。
- 加入多参考组合能力。

### 6.3 与 VBench-I2V 等 I2V benchmark 的区别

现有 I2V benchmark 更关注：

- 首帧一致性。
- 视频质量。
- 动态程度。
- 相机运动。
- 主体/背景稳定。

I2V-CompBench 关注：

- 指令变化是否执行。
- 执行是否绑定到正确主体。
- 非目标内容是否被误改。
- 多参考元素是否能正确组合。
- failure mode 是否可诊断。

---

## 7. 最小可跑版本

第一轮不要直接做 1400 题。更稳的方式是先跑一个小闭环：

```text
Phase 1 pilot:
  TIP-I2V 100-300 条
  产出 phase1_bundle

Phase 2 pilot:
  7 维度 × 20 题 = 140 题
  多图占比按 4.2 节比例分摊（Spatial 16 道、Action/Attribute/Background 各 8 道、Interaction 10 道、Motion Type C 3 道、View 1 道）
  View 维度多图样本极少，可在 pilot 中合并入下一轮观察，不强求 contrastive pair

Phase 3 pilot:
  2-3 个模型
  Static Copy + Global Filter + Camera Pan Cheat baseline
  每维度人工 spot-check 20 条
```

pilot 通过标准：

- 每道题都有完整 metadata。
- 多图样本都有 reference provenance 与质量字段。
- Phase 3 能输出 E/P/C、tool_status 和 failure modes。
- Static Copy 总分接近 0。
- 人工 spot-check 中明显错误评测少于 25%。

---

## 8. 一句话总结

I2V-CompBench 的三阶段闭环是：Phase 1 从真实 I2V 数据中提取可追溯的单图与多参考 recipe；Phase 2 按七维度边界生成带有评测元数据的题目；Phase 3 用执行门控的 E/P/C 评测器检查模型是否真正把指定变化绑定到正确对象，同时保住不该变的内容。这条线既能支撑工程实现，也能支撑论文里的创新性和可信度。
