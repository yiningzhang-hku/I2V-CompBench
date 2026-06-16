# I2V-CompBench Benchmark 合成流水线详细方案

> 本文档整合了七维度重构方案、Prior Package 先验分析成果、相关论文调研，以及多轮讨论中形成的所有决策，完整描述从"词表库构建"到"最终题目输出"的全流程。

---

## 一、研究调研总结

### 1.1 已调研的相关项目与方法论

| 项目 | 会议/年份 | 题目生成方法 | 词表来源 | 关键技术 |
|------|-----------|------------|----------|----------|
| **T2V-CompBench** | CVPR 2025 | VidProM 真实分布 → 高频词表 → ChatGPT 模板生成 → 人工筛查 | VidProM 1.67M 真实 prompt | 80/20 常见/罕见 split、对比组 |
| **T2I-CompBench++** | TPAMI 2025 | 固定词表 + 模板 + ChatGPT 扩展 | 手工整理：2470 nouns, 33 colors, 32 shapes, 23 textures | BLIP-VQA 评测、UniDet 空间评测 |
| **ConceptMix** | Princeton 2024 | 8 类视觉概念随机采样 → GPT-4o 自由组合 → 验证 | 文献 + GPT-4 扩展 | 可控难度 k-tuple |
| **Generate Any Scene** | UW/AI2 2024 | 场景图结构化枚举 → 元数据填充 → 程序化翻译 | **WordNet 28787 objects** + **Visual Genome 10492 relations** + Wikipedia 1494 attrs | 场景图驱动、可复现 |
| **UI2V-Bench** | 华为 2025 | 手工设计 + DALL-E 3 生成 + 真实图检索 | COCO/手工 | 首个 I2V 理解基准、reasoning 维度 |

### 1.2 核心发现与我们的借鉴

| 方法论 | 来源 | 对 I2V-CompBench 的适用性 |
|--------|------|-------------------------|
| 真实分布驱动的词表 | T2V-CompBench (VidProM) | 我们已有 TIP-I2V prior_package，可直接用 |
| WordNet 层级物体词库 | Generate Any Scene | 用于扩展罕见物体采样、构建 common/uncommon split |
| Visual Genome 关系库 | Generate Any Scene | 用于 Spatial Composition 和 Interaction Reasoning |
| GPT-4o 自由组合 + 验证 | ConceptMix | 用于从结构化规划生成自然语言 prompt |
| 80/20 常见/罕见 split | T2V-CompBench | 直接采用 |
| 对比组设计 | T2V-CompBench | 直接采用 |
| 混合图像来源 | UI2V-Bench | 大部分 T2I 生成，少量真实图检索 |

---

## 二、整体流水线架构

```
[Stage 0] 词表库构建 (Vocabulary Bank Construction)
WordNet + Visual Genome + TIP-I2V prior_package + ConceptNet → 结构化词表库
                ↓
[Stage 1] 题目规划 (Question Planning)
词表库 + 六维度规格 + LLM → 结构化 Question Plan
                ↓
[Stage 2] T2I Prompt 合成 (First-frame Prompt Synthesis)
Question Plan → LLM → 文生图 prompt（单图1个 / 多图N个）
                ↓
[Stage 3] 首帧图像生成 (Image Generation)
调用 T2I 模型 API → 生成首帧 / 参考图
                ↓
[Stage 4] 图像质检 (Image Verification)
VLM 检查生成图像是否符合 Question Plan → 不合格则回退 Stage 2 重试
                ↓
[Stage 5] I2V Prompt 定稿 (I2V Prompt Finalization)
基于实际首帧 + Question Plan → 生成最终 I2V 评测提示词
                ↓
[Stage 6] 质量过滤 (Quality Filtering) — 待后续讨论评测公式后再定
                ↓
[Output] benchmark_dataset/
每道题：{首帧图像(s) + I2V prompt + 结构化元数据}
```

**v1 范围**：6 个维度 × 200 题 = 1200 道题（Interaction Reasoning 暂缓）。

**Human-in-the-loop 节点**：
- Stage 1 后：抽检 Question Plan 质量（每维度抽 20 题）
- Stage 4 后：审查 VLM 标记为"边界case"的图像
- Stage 5 后：最终成品的人工 pass/fail 审查

---

## 三、Stage 0 — 词表库构建

### 3.1 为什么需要词表库

T2V-CompBench 和 T2I-CompBench++ 的经验表明：prompt 的质量和多样性取决于底层词表的丰富度和结构化程度。仅靠 LLM 随机生成会导致概念分布不均、罕见组合覆盖不足。Generate Any Scene 证明了结构化词表（WordNet + Visual Genome）可以系统性地提升覆盖率和组合多样性。

### 3.2 词表库的五个组件

#### 组件 A：物体词表 (Object Vocabulary)

**来源 1：TIP-I2V prior_package**
- `concept_distributions.target_subject` → 高频主体（person, dog, cat, car, ...）
- `global_visual_prior.typical_subject_categories` → 主体类别分布
- 标记为 `source: tip_i2v`，权重高（真实 I2V 用户偏好）

**来源 2：WordNet (NLTK)**

WordNet 是一个英语词汇语义数据库，物体概念以树形层级组织：
```
entity → physical_entity → object → artifact → furniture → chair
entity → physical_entity → organism → animal → dog → corgi
```

Generate Any Scene 从中提取了 28787 个物体概念，但其中大量不可视化、太抽象或 T2I 模型画不出来。

**我们的筛选策略**：
- 重点提取：`animal.n.01`、`artifact.n.01`、`vehicle.n.01`、`furniture.n.01`、`food.n.01`、`plant.n.01` 子树
- 过滤条件：物体必须可视化 + T2I 模型有能力生成
- 目标规模：**500-1000 个高质量可视化物体**
- 用 hypernym 层级做 common/uncommon 分类：上位词（dog, cat, car）= common；下位词（corgi, tabby, sedan）= uncommon
- 工具：`nltk.corpus.wordnet`

**来源 3：COCO 80 类**
- 作为 T2I 模型已知能生成好的"安全物体"白名单
- 主力桶优先从 COCO 类别采样

**输出格式**：
```json
{
  "object_id": "dog.n.01",
  "name": "dog",
  "hypernyms": ["canine", "domestic_animal", "animal"],
  "frequency_tier": "common",
  "source": ["tip_i2v", "wordnet", "coco"],
  "is_animate": true,
  "typical_attributes": ["color", "size", "breed"],
  "compatible_scenes": ["park", "home", "street"],
  "compatible_actions": ["sit", "stand", "run", "walk"]
}
```

#### 组件 B：属性词表 (Attribute Vocabulary)

| 类别 | 来源 | Common 示例 | Uncommon 示例 |
|------|------|------------|--------------|
| 颜色 | T2I-CompBench++ 33色 + TIP-I2V | red, blue, white, black, green | turquoise, maroon, lavender, chartreuse |
| 状态 | 七维度重构方案 | open/closed, empty/full, dry/wet, dark/bright | intact/broken, frozen/melted |
| 材质/纹理 | T2I-CompBench++ 23纹理 | smooth, rough, metallic, wooden | glass, leather, stone |
| 形状 | T2I-CompBench++ 32形状 | round, square, triangular | cylindrical, hexagonal |
| 服装（多图用） | 手工编写 | shirt, pants, sneakers, hat | vest, boots, scarf |

#### 组件 C：动作/运动词表 (Action/Motion Vocabulary)

**动作词表（Action Binding 用）**：
- 上肢：wave, clap, raise_hand, point
- 头部：nod, turn_head, tilt_head, look_up
- 全身：bow, sit_down, stand_up, jump

**运动方向词表（Motion Binding 用）**：
- 绝对方向：left, right, up, down, toward_camera, away_from_camera
- 中性动词：move, shift, drift

**运镜词表（View Transformation 用）**：
- 基础运镜：zoom_in, zoom_out, pan_left, pan_right, tilt_up, tilt_down, static
- 景别：push_in, pull_out
- 来源：TIP-I2V prior_package 的 `pika_camera_distribution`

#### 组件 D：关系词表 (Relation Vocabulary)

**空间关系**：left_of, right_of, above, below, in_front_of, behind, on, beside, next_to, between
来源：Visual Genome spatial relations 子集

**交互关系（Interaction Reasoning 用，暂缓构建）**：
- 物理交互：hit, push, knock_over, pour_into, break, catch, throw
- 社会交互：handshake, hug, high_five, hand_over, follow, chase
- 功能交互：open_with_key, hammer_nail, press_switch

#### 组件 E：场景词表 (Scene Vocabulary)

**来源 1：TIP-I2V prior_package** → `visual_prior.scene_type_distribution`
**来源 2：Places365** → 室内（living_room, kitchen, cafe, ...）和室外（park, street, beach, ...）
**来源 3：背景迁移场景对** → London_street / Paris_street, beach / mountain, city / countryside

### 3.3 Common/Uncommon 分类策略

借鉴 T2V-CompBench 的 80/20 split：

| 判定方式 | 规则 |
|----------|------|
| TIP-I2V 频率 | 在 prior_package 中出现频率 top 70% → common |
| WordNet 深度 | hypernym 路径短（≤3层）→ common；深层具体概念 → uncommon |
| COCO 覆盖 | COCO 80 类内 → common；COCO 外但 WordNet 内 → uncommon |
| 组合罕见度 | 单个概念 common 但组合罕见（blue apple, metallic cat）→ uncommon |

### 3.4 兼容性矩阵

**问题**：不是所有物体-属性-动作-场景组合都合理。随机组合会产生不合理题目。

**示例**：

| 组合 | 合理？ | 原因 |
|------|--------|------|
| dog + sit + park | ✓ | 狗可以坐，公园有狗 |
| dog + fly + ocean | ✗ | 狗不能飞 |
| cup + open/closed | ✗ | 杯子没有开关状态 |
| door + open/closed | ✓ | 门有开关状态 |
| fish + street | ✗ | 鱼不会出现在街上 |

**兼容性矩阵内容**：
- **物体-动作兼容性**：dog → sit/run/walk ✓, fly ✗; bird → fly ✓, drive ✗
- **物体-属性兼容性**：cup → empty/full ✓, open/closed ✗; door → open/closed ✓
- **物体-场景兼容性**：fish → aquarium/ocean ✓, street ✗; car → street/parking_lot ✓

**三层构建方式**：
1. **ConceptNet API 自动初始化**：利用 ConceptNet 的 `CapableOf`（狗 CapableOf 跑）、`HasProperty`（杯子 HasProperty 圆的）、`AtLocation`（鱼 AtLocation 海里）关系
2. **LLM 批量校验扩展**：给 LLM 一批组合对，让它判断合不合理并补充遗漏
3. **人工抽检校正**：最终由人工确认准确性

### 3.5 词表库工程输出

```
vocabulary_bank/
├── objects.json              # 物体词表（含 WordNet ID、层级、频率）
├── attributes_color.json
├── attributes_state.json
├── attributes_texture.json
├── attributes_shape.json
├── attributes_clothing.json
├── actions.json              # 动作词表
├── motions.json              # 运动方向词表
├── camera_commands.json      # 运镜词表
├── relations_spatial.json
├── scenes.json               # 场景词表
├── scene_pairs.json          # 背景迁移场景对
└── compatibility_matrix.json # 物体-属性-场景-动作兼容性矩阵
```

---

## 四、Stage 1 — 题目规划 (Question Planning)

### 4.1 输入

- **词表库** (Stage 0 输出)
- `prior_package.json`：真实用户分布先验
- **六维度规格**：每个维度的难度分桶、题型定义、首帧设计原则、提示词限制规则

### 4.2 三步走规划逻辑

**Step 1：配额分配**

根据维度方案中的比例，将 200 题分配到各难度桶和输入模式。

**Step 2：结构化概念采样（程序化，不依赖 LLM）**

从词表库中按规则采样：
- 从 `objects.json` 中按 80/20 common/uncommon 比例采样目标物体
- 查 `compatibility_matrix.json` 确定该物体可用的属性/动作/场景
- 从对应词表中采样具体值
- 构造一个类似场景图的结构化 Question Plan

这一步保证概念组合的合理性和覆盖率。

**Step 3：LLM 润色 + 自然语言化**

将结构化 Question Plan 交给 LLM，生成：
- I2V prompt 草稿（自然语言）
- T2I prompt（静态描述）
- 首帧设计说明

### 4.3 Prior Package 的使用方式

| Prior Package 字段 | 在 Stage 1 中的用途 |
|---|---|
| `concept_distributions.target_subject` | 物体采样的真实用户偏好权重 |
| `concept_distributions.attribute_type` | 属性变化类型采样权重 |
| `visual_prior.scene_type_distribution` | 场景类型采样 |
| `visual_prior.shot_type_distribution` | 首帧构图采样 |
| `structural_templates` | I2V prompt 句式参考 |
| `seed_examples` | LLM few-shot 上下文 |
| `pika_camera_distribution` | View Transformation 运镜命令权重 |

### 4.4 Question Plan 输出格式

**单图模式**：
```json
{
  "question_id": "attr_001",
  "dimension": "attribute_binding",
  "input_mode": "single_image",
  "difficulty_bucket": "S2A1",
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
    "camera_angle": "eye_level",
    "design_notes": "Same breed, clearly different colors, side by side"
  },
  "change_plan": {
    "target_subject": "s1",
    "change_type": "color",
    "from_value": "white",
    "to_value": "black",
    "preserve_subjects": ["s2"]
  },
  "i2v_prompt_draft": "The white dog turns black while the brown dog remains unchanged."
}
```

**多图模式**：
```json
{
  "question_id": "attr_150",
  "dimension": "attribute_binding",
  "input_mode": "multi_image",
  "difficulty_bucket": "R3",
  "reference_images_plan": [
    {"image_id": "ref1", "role": "target_subject", "content": "A young woman standing in neutral pose", "requirements": "full body, clean background"},
    {"image_id": "ref2", "role": "attribute_source", "content": "A red dress", "requirements": "product photo, white background"},
    {"image_id": "ref3", "role": "attribute_source", "content": "White sneakers", "requirements": "product photo, white background"}
  ],
  "composition_plan": {
    "target_subject": "ref1",
    "attribute_transfers": [
      {"source_ref": "ref2", "attribute_type": "wearing", "target_slot": "dress"},
      {"source_ref": "ref3", "attribute_type": "wearing", "target_slot": "shoes"}
    ]
  },
  "i2v_prompt_draft": "The woman wears the red dress and white sneakers."
}
```

---

## 五、Stage 2 — T2I Prompt 合成

### 5.1 核心原则

**T2I prompt 只描述静态画面，绝不包含任何时间/视频/变化语义。**

通用结构：`[主体描述及布局] + [场景/环境] + [构图/镜头] + [风格/质量修饰]`

### 5.2 各维度 T2I Prompt 示例

**Attribute Binding (S2A1)**
```
Question Plan: 两只狗，左白右棕，公园，白狗将变黑
T2I Prompt: "Two dogs standing side by side on green grass in a park. The dog on the left is white, the dog on the right is brown. Both are the same breed. Medium shot, eye level, daylight, photorealistic."
```

**Action Binding (S2D2)**
```
Question Plan: 两个女人并排站立，即将分别做 wave 和 clap
T2I Prompt: "Two women standing side by side in a park, neutral standing pose, arms at their sides. The woman on the left wears a white shirt, the woman on the right wears a black shirt. Medium-wide shot, eye level, daylight, photorealistic."
```

**Motion Binding Type A (S1-D1)**
```
Question Plan: 棕色小狗在画面偏右，将向左移动
T2I Prompt: "A small brown dog sitting on grass, positioned in the right third of the frame. The left side shows open grass. Medium shot, eye level, daylight, photorealistic."
```

**Motion Binding Type B (M2R1)**
```
Question Plan: 红球在蓝盒左边，将移到右边
T2I Prompt: "A red ball on the left side of a blue box on a wooden table. Clear separation between ball and box. Medium shot, slightly elevated angle, indoor soft lighting, photorealistic."
```

**Background Dynamics Type A**
```
Question Plan: 人站在海边，天空晴朗，将变为多云
T2I Prompt: "A person standing on a beach facing the ocean. Clear blue sky with no clouds. The person is in the lower third. Medium shot, eye level, daylight, photorealistic."
```

**View Transformation (S1C1)**
```
Question Plan: 走廊中的女人，将 zoom in
T2I Prompt: "A woman standing in a long indoor corridor with rigid walls and clear perspective lines. The woman is centered, shown at medium distance. Medium shot, eye level, indoor lighting, photorealistic."
```

### 5.3 多图模式 T2I Prompt 策略

每张参考图单独生成 T2I prompt：

- **主体参考图**（干净背景 + 清楚身份）：`"A young woman with short black hair, standing in a neutral pose, full body visible, clean white background, portrait photography style."`
- **属性/物体参考图**（产品摄影风格 + 白底）：`"A red floral dress displayed on a white background, flat lay product photography, high detail."`
- **场景参考图**（无人场景 + 空间感）：`"An empty cozy café interior with wooden tables and warm lighting, no people, medium shot, eye level."`

### 5.4 T2I Prompt 质量控制规则

1. **必须精确控制主体数量**：写 "two dogs"，不用 "some" / "several"
2. **必须精确控制空间位置**：用 "on the left" / "in the right third"
3. **必须精确控制属性**：颜色、状态明写
4. **禁止时间/运动语言**：不写 "walking" / "changing" / "turning"
5. **禁止镜头运动语言**：不写 "camera moves" / "panning"
6. **统一追加质量修饰**：photorealistic / high quality
7. **构图匹配维度需求**：Motion 题留出运动空间，View 题匹配镜头命令

---

## 六、Stage 3 — 首帧图像生成

### 6.1 T2I 模型调用

- 对每个 T2I prompt 调用文生图 API
- 单图模式：1 次调用
- 多图模式：N 次调用（每张参考图独立生成）
- 每个 prompt 生成 2-3 张候选图，供 Stage 4 筛选

### 6.2 图像命名规则

```
benchmark_images/
├── attribute_binding/
│   ├── attr_001_frame.png          # 单图模式首帧
│   ├── attr_150_ref1_subject.png   # 多图模式参考图1
│   ├── attr_150_ref2_dress.png     # 多图模式参考图2
│   └── attr_150_ref3_shoes.png     # 多图模式参考图3
├── action_binding/
│   └── ...
├── motion_binding/
│   └── ...
├── spatial_composition/
│   └── ...
├── background_dynamics/
│   └── ...
└── view_transformation/
    └── ...
```

---

## 七、Stage 4 — 图像质检

### 7.1 VLM 自动验证

**单图模式检查清单**：
- [ ] 主体数量正确？
- [ ] 主体属性正确（颜色、状态、类别）？
- [ ] 主体空间位置正确？
- [ ] 足够的运动/变化空间？
- [ ] 构图合理？

**多图模式检查清单**：
- [ ] 参考图中目标物体清楚可见？
- [ ] 背景足够干净（主体/物体参考图）？
- [ ] 场景参考图有足够空间？

### 7.2 重试策略

1. VLM 判定不合格 → 自动调整 T2I prompt → 重新生成
2. 最多重试 3 次
3. 仍不合格 → 标记为 `needs_manual_review`

---

## 八、Stage 5 — I2V Prompt 定稿

### 8.1 为什么不在 Stage 1 就定稿

T2I 模型生成的图像可能与 Question Plan 有偏差。I2V prompt 必须与**实际生成的首帧图像**保持一致。

### 8.2 定稿流程

1. VLM 分析实际首帧，提取实际主体信息
2. 对比 Question Plan 和实际图像，调整 I2V prompt 中的指代词
3. 应用各维度的提示词限制规则（禁用词检查、长度检查、运镜约束）
4. 输出最终 I2V prompt

### 8.3 各维度 I2V Prompt 模板

| 维度 | 模板结构 | 示例 |
|------|----------|------|
| Attribute Binding (单图) | `[主体锚定] + [变化要求] + [轻度背景]` | The white dog turns black while the brown dog remains unchanged. |
| Attribute Binding (多图) | `[主体指代] + [属性迁移指令]` | The woman wears the red dress and white sneakers. |
| Action Binding | `场景壳 + 主体锚定 + 动作分配 + 运镜约束` | The woman on the left waves while the woman on the right claps. The camera remains fixed. |
| Motion Binding A | `The [target] moves [direction].` | The small brown dog moves to the left. |
| Motion Binding B | `[target] + 关系结果词 + [reference]` | The red ball moves to the right side of the blue box. |
| Spatial Composition | `Place [A] [relation] [B] ...` | Place the cat on the sofa and the plant beside the sofa. |
| Background Dynamics A | `[背景区域] + [变化操作] + [preserve]` | The clouds slowly drift while the person remains still. The camera remains fixed. |
| Background Dynamics B | `[主体保持] + [目标背景]` | The man is now on a Parisian street with the Eiffel Tower in the background. |
| View Transformation | `The camera [verb] ...` | The camera slowly zooms in. |

---

## 九、Stage 6 — 质量过滤

> **待定**：此阶段的详细设计（自动过滤规则、评测指标、评分公式）将在确定评测框架后再补充。

---

## 十、各维度详细规格

### 维度 1：Attribute Binding（属性绑定）

**核心测试**：主体属性/状态变化能否准确绑定到指定主体。

**输入模式**：
- 模式 A（单图属性变化）：首帧中已有主体发生属性变化
- 模式 B（多图属性迁移）：参考图中的属性（服装、颜色等）迁移到目标主体

**属性变化类型分层**：
| 层级 | 类型 | v1 策略 |
|------|------|---------|
| 第一层（主力） | 颜色变化、状态变化 | 大量使用 |
| 第二层（少量） | 外观附加件/局部细节、属性迁移 | 少量使用 |
| 第三层（谨慎） | 材质/纹理 | v1 不作主力 |

**难度分桶**：

单图（S×A 模型）：S1A1, S1A2, S2A1, S2A2, S3A1, S3A2
多图（参考图数量）：R2, R3, R3+

语义罕见度：80% 常见 + 20% 罕见

**各难度桶示例**：

| 桶 | 首帧描述 | I2V Prompt | 难度说明 |
|----|----------|------------|----------|
| **S1A1** (Easy) | 一个空的白色杯子放在桌上 | The empty cup slowly becomes full of water. | 单主体、单属性（状态：空→满），最简单 |
| **S1A2** (Easy-Medium) | 一扇关着的白色木门 | The white door opens and turns blue. | 单主体、双属性（状态：关→开 + 颜色：白→蓝） |
| **S2A1** (Medium) | 两只狗并排站在草地上，左边白色右边棕色 | The white dog turns black while the brown dog remains unchanged. | 双主体、单属性变化，核心测绑定准确性 |
| **S2A2** (Medium-Hard) | 两只气球并排漂浮，左红右蓝 | The red balloon turns green and the blue balloon turns yellow. | 双主体各自一个颜色变化，双绑定 |
| **S3A1** (Hard) | 三个杯子一排：左白、中红、右蓝 | The red cup in the middle becomes empty while the white and blue cups remain unchanged. | 三主体中指定一个变化，其余两个保持 |
| **S3A2** (Hard) | 三只猫坐成一排：左黑、中白、右橘 | The black cat turns gray and the orange cat turns white, while the white cat in the middle remains unchanged. | 三主体中两个各发生颜色变化 |
| **R2** (多图 Medium) | 输入图1：站立的年轻女性；输入图2：一件红色连衣裙 | The woman puts on the red dress. | 2张参考图，单属性迁移 |
| **R3** (多图 Hard) | 输入图1：站立男性；输入图2：条纹衬衫；输入图3：黑色礼帽 | The man wears the striped shirt and the black hat. | 3张参考图，双属性迁移 |
| **R3+** (多图 Hard) | 输入图1：女性；图2：白色运动鞋；图3：蓝色牛仔裤；图4：灰色卫衣 | The woman wears the gray hoodie, blue jeans, and white sneakers. | 4张参考图，三属性迁移 |

**数据配额**：200 题 = 120 单图(60%) + 80 多图(40%)

**首帧设计原则**：
- 单图：初态清楚、多主体可区分、不泄露终态
- 多图：每张参考图只承载一个属性、参考属性视觉可区分

**提示词规则**：
- 单图三段式：`[主体锚定] + [变化要求] + [轻度背景]`
- 多图三段式：`[目标主体指代] + [参考属性描述] + [融合说明]`
- 长度：8-25 英文词
- 禁止：强动作词、镜头词、风格修饰、绝对方向词

**对比组**：白狗变黑/棕狗不变 ↔ 棕狗变白/白狗不变

---

### 维度 2：Action Binding（动作绑定）

**核心测试**：动作能否正确绑定到指定主体，不串到别的主体。

**输入模式**：
- 模式 A（单图动作分配）：首帧中多主体执行各自动作
- 模式 B（多图场景动作合成）：参考人物在指定场景中执行动作

**推荐动作词**：wave, clap, nod, bow, raise_hand, point, sit_down, stand_up, jump, turn_head

**难度分桶（S×D 模型）**：S1D1, S2D1, S2D2, S2D3, S3D2

**各难度桶示例**：

| 桶 | 首帧描述 | I2V Prompt | 难度说明 |
|----|----------|------------|----------|
| **S1D1** (Easy) | 一个女人站在公园里，自然站立姿态 | The woman waves her hand. The camera remains fixed. | 单主体单动作，校准桶 |
| **S2D1** (Easy-Medium) | 两个人站在广场上，左边穿白衣右边穿黑衣 | The woman on the left waves while the man on the right stays still. The camera remains fixed. | 双主体，一动一静，最基础绑定 |
| **S2D2** (Medium) | 两个男人面对面站在办公室里，左穿蓝衬衫右穿灰夹克 | The man in blue claps while the man in gray nods. The camera remains fixed. | 双主体双不同动作 |
| **S2D3** (Hard) | 两个穿相似白衬衫的女性左右对称站立 | The woman on the left bows while the woman on the right raises her hand. The camera remains fixed. | 双主体高混淆：同类主体、相似外观、对称布局 |
| **S3D2** (Hard) | 三个人并排站在走廊中：左穿红衣、中穿白衣、右穿蓝衣 | The person in red waves, the person in white claps, and the person in blue nods. The camera remains fixed. | 三路动作分配 |
| **多图** | 输入图1：男性A半身照；图2：女性B半身照；图3：咖啡厅内景 | In the café, the man waves while the woman nods. The camera remains fixed. | 参考人物在指定场景中执行各自动作 |

**数据配额**：200 题 = 120 单图(60%) + 80 多图(40%)

**首帧设计原则**：
- 动作部位不遮挡、不出框
- 初始姿态为 neutral pre-action pose
- 多主体布局控制难度（S2D3 用镜像布局 + 相似服装）

**提示词规则**：
- 四段式：`场景壳 + 主体锚定 + 动作分配 + 运镜约束`
- 运镜：single continuous shot + camera remains fixed + no cuts
- 禁止代词链

**对比组**：左边人挥手右边不动 ↔ 右边人挥手左边不动

---

### 维度 3：Motion Binding（运动绑定）

**核心测试**：主体轨迹级运动是否正确。

**三种题型**：
- 类型 A：单主体绝对方向运动（left/right/up/down/toward/away）
- 类型 B：多主体相对位移变化（球从盒子左边移到右边）
- 类型 C：多图组合运动（飞机飞过城市上空）

**难度分桶**：
- 类型 A（S×D）：S1-D1, S2-D1, S2-D2, S3-D1, S3-D2
- 类型 B（M×R）：M2R1, M2R1-D, M3R1, M3R2, M4R2

**类型 A 各难度桶示例**：

| 桶 | 首帧描述 | I2V Prompt | 难度说明 |
|----|----------|------------|----------|
| **S1-D1** (Easy) | 红色气球在画面中央偏右，背景干净蓝天 | The red balloon moves to the left. | 简单场景 + 平面方向，目标显著 |
| **S2-D1** (Medium) | 棕色小狗在画面右侧草地上，背景有树木和栅栏 | The small brown dog moves to the left. | 中等场景有干扰物，靠属性指代目标 |
| **S2-D2** (Medium-Hard) | 一辆红色玩具车在有透视感的走廊中央 | The red toy car moves toward the camera. | 中等场景 + 深度方向，需透视线索 |
| **S3-D1** (Hard) | 一只白色纸飞机在拥挤的房间角落，周围有书本、杯子等杂物 | The white paper airplane moves to the right. | 小目标 + 复杂背景 |
| **S3-D2** (Hard) | 一只小鸟站在远处树枝上，场景有多棵树和建筑 | The small bird moves away from the camera. | 复杂场景 + 深度方向 + 小目标，最难 |

**类型 B 各难度桶示例**：

| 桶 | 首帧描述 | I2V Prompt | 难度说明 |
|----|----------|------------|----------|
| **M2R1** (Easy) | 红球在蓝盒子左边，桌面上 | The red ball moves to the right side of the blue box. | 2主体，1个2D位移关系变化 |
| **M2R1-D** (Medium) | 白色茶杯在花瓶后方，有深度感的桌面 | The white teacup moves in front of the vase. | 2主体，1个前后关系变化 |
| **M3R1** (Medium) | 三个玩具：红车在左、绿球在中、蓝积木在右 | The green ball moves to the left of the red car. | 3主体，1个核心位移 |
| **M3R2** (Hard) | 猫在狗左侧，椅子在最右侧 | The cat moves between the dog and the chair, and the dog moves to the right of the chair. | 3主体，2个关系同时变化 |
| **M4R2** (Hard) | 四件物品一字排开：书、杯、笔、尺 | The cup moves to the left end and the ruler moves next to the book. | 4主体多物重排 |

**类型 C 示例**：

| 首帧描述 | I2V Prompt | 说明 |
|----------|------------|------|
| 输入图1：城市天际线；输入图2：客机侧面照 | The airplane flies across the city skyline from left to right. | 多图组合 + 水平运动 |
| 输入图1：蓝天白云；输入图2：一个热气球 | The hot air balloon drifts upward into the sky. | 多图组合 + 垂直运动 |

**数据配额**：200 题 = 100 类型A(50%) + 70 类型B(35%) + 30 类型C(15%)

**首帧设计原则**：
- 必须预留运动空间
- 类型 A 用外观属性唯一指代目标
- 类型 B 起始关系必须一眼可判
- toward/away 题必须有深度线索

**提示词规则**：
- 类型 A：`The [target] moves [direction].`（中性动词：move, shift, drift）
- 类型 B：含关系结果词，禁止绝对方向词
- 通用禁止：局部动作词、语义动作词、多阶段词、运镜词、属性变化词、交互词

**对比组**：left/right 对照、up/down 对照、toward/away 对照

---

### 维度 4：Spatial Composition（空间组合）

**核心测试**：多个参考主体/物体能否正确放置到指定空间关系中。

**关键判定**：主体不发生位移，只是组合放置 → Spatial；有位移 → Motion。

**输入模式**：
- 模式 A（单图布局验证）：验证场景中物体空间关系保持
- 模式 B（多图空间组装，核心题型）：将参考图中的物体放入场景指定位置

**空间关系**：left_of, right_of, above, below, in_front_of, behind, on, beside, between

**难度分桶**：C2-2D, C2-3D, C3-2D, C3-3D, C4+

**各难度桶示例**：

| 桶 | 输入描述 | I2V Prompt | 难度说明 |
|----|----------|------------|----------|
| **C2-2D** (Easy) | 输入图1：红色花瓶；图2：蓝色书架；场景：客厅一角 | Place the red vase to the left of the blue bookshelf. | 2物体，纯2D左右关系 |
| **C2-3D** (Medium) | 输入图1：白色茶杯；图2：黑色笔记本电脑；场景：办公桌 | Place the white cup in front of the laptop on the desk. | 2物体，3D前后关系，需深度感 |
| **C3-2D** (Medium) | 输入图1：橘猫；图2：灰色沙发；图3：绿植；场景：客厅 | Place the cat on the sofa and the plant to the right of the sofa. | 3物体，2D混合关系（on + right_of） |
| **C3-3D** (Hard) | 输入图1：红椅子；图2：蓝桌子；图3：白色花瓶；场景：餐厅 | Place the red chair in front of the blue table, with the white vase on the table. | 3物体，含3D前后关系 + on关系 |
| **C4+** (Hard) | 输入图1：双人床；图2：床头柜；图3：书桌；图4：椅子；场景：空卧室 | Arrange the bed against the back wall, the nightstand to the right of the bed, the desk on the left wall, and the chair in front of the desk. | 4物体复杂布局 |
| **单图模式** | 首帧已含猫在沙发上、绿植在沙发旁 | The cat stays on the sofa and the plant remains beside the sofa. The camera remains fixed. | 验证已有布局的保持能力 |

**数据配额**：200 题 = 40 单图(20%) + 160 多图(80%)

**参考图设计原则**：
- 每张参考图只含一个目标物体、背景干净
- 场景参考图有足够空间
- 前后关系需深度线索

**提示词规则**：
- 结构：`Place [A] [relation] [B] + [可选场景说明]`
- 禁止：运动词(moves, shifts)、动作词、镜头词

**对比组**：猫在沙发左边 ↔ 猫在沙发右边

---

### 维度 5：Background Dynamics（背景动态）

**核心测试**：对背景区域施加变化，同时前景不被破坏。

**两种题型**：
- 类型 A（环境状态变化）：天气、光照、纹理动态
- 类型 B（背景迁移/替换）：将主体背景替换为参考图背景

**难度分桶**：
- 类型 A：三轴控制（场景复杂度 × 变化复杂度 × 语义推理负荷）
- 类型 B：B1(简单替换), B2(跨场景类型), B3(多主体+光照一致性)

**类型 A 各难度示例**：

| 难度 | 首帧描述 | I2V Prompt | 三轴评级 |
|------|----------|------------|----------|
| **Easy** | 人站在海边，天空晴朗开阔，前景只有人 | The clouds in the sky slowly drift while the person remains still. The camera remains fixed. | 场景低 + 变化低 + 推理低 |
| **Medium（变化强）** | 公园长椅上坐着一个人，背景有大面积树林 | The leaves on the trees gradually sway and some fall. The person remains still. The camera remains fixed. | 场景中 + 变化中（动态纹理）+ 推理低 |
| **Medium（推理高）** | 人站在路边，远处有山和天空 | The sky gradually darkens as clouds gather in the distance. The car remains unchanged. The camera remains fixed. | 场景中 + 变化中 + 推理中（需区分局部天空 vs 整图滤镜） |
| **Hard** | 一辆车停在城市街道旁，有建筑、路灯、行道树 | Dark clouds gather, the streetlights flicker on, and puddles begin to form on the ground. The car remains unchanged. The camera remains fixed. | 场景高 + 变化高（多区域复合）+ 推理高（因果链：乌云→变暗→灯亮→积水） |

**类型 B 各难度示例**：

| 桶 | 输入描述 | I2V Prompt | 难度说明 |
|----|----------|------------|----------|
| **B1** (Easy) | 输入图1：女人站在室内白墙前；图2：另一面白墙的室内（风格相似） | The woman is now in the new room. The camera remains fixed. | 简单主体 + 相似场景类型替换 |
| **B2** (Medium) | 输入图1：男人站在伦敦街头；图2：巴黎埃菲尔铁塔街景 | The man is now walking on a Parisian street with the Eiffel Tower in the background. | 跨城市/跨场景类型替换 |
| **B3** (Hard) | 输入图1：两个人在阴天公园散步；图2：阳光沙滩 | The two people are now walking on the sunny beach. The camera remains fixed. | 多主体 + 光照差异大（阴天→阳光）+ 需保持人物一致性 |

**数据配额**：200 题 = 120 类型A(60%) + 80 类型B(40%)

**首帧设计原则**：
- 类型 A：目标背景区域 15%-60%、有前景 preserve 区、初态不接近终态
- 类型 B：主体轮廓清楚、背景参考图空间能容纳主体、光照差异合理

**提示词规则**：
- 类型 A：`目标区域 + 变化操作 + preserve clause + 运镜约束`（v1 强制固定机位）
- 类型 B：`主体保持 + 目标背景描述 + 运镜约束`

**对比组**：天空变暗 ↔ 天空变亮；伦敦→巴黎 ↔ 巴黎→伦敦

---

### 维度 6：View Transformation（视角变换）

**核心测试**：镜头/视角控制能力。

**输入模式**：以单图输入为主。

**难度分桶（S×C 模型）**：
- 场景复杂度：S1(简单), S2(中等), S3(复杂)
- 运镜复杂度：C1(单命令), C2(双约束), C3(复合控制)
- 推荐起步桶：S1C1, S2C1, S1C2, S2C2

**各难度桶示例**：

| 桶 | 首帧描述 | I2V Prompt | 难度说明 |
|----|----------|------------|----------|
| **S1C1** (Easy) | 走廊里站着一个女人，背景是刚性墙壁和地板 | The camera slowly zooms in. | 简单场景 + 单命令 |
| **S2C1** (Easy-Medium) | 草地上有一条狗，背景有树木和栅栏 | The camera slowly pans right. | 中等场景 + 单命令 |
| **S1C2** (Medium) | 街边站立的女人，周围环境简洁 | The camera slowly pans right while keeping the woman near the center of the frame. | 简单场景 + 双约束（平移 + 构图约束） |
| **S2C2** (Medium-Hard) | 咖啡厅里坐着两个人，桌上有咖啡杯，窗外有街景 | The camera pushes in while tilting slightly downward. | 中等场景 + 双约束（推进 + 俯仰） |
| **S3C3** (Hard) | 拥挤的集市，多个摊位、行人、遮阳棚 | The camera pans left, then zooms in while keeping the fruit stand roughly centered. | 复杂场景 + 复合控制（v1 少量） |
| **static 对照** | 草地上的狗（同 S2C1 首帧） | The camera remains completely static. | 与 S2C1 构成对比组 |

**运镜词表**：
- 允许：zoom in/out, pan left/right, tilt up/down, keep static, push in, pull out
- 禁止：cinematic, dramatic, orbit, crane, handheld, whip pan, POV, Dutch angle

**数据配额**：200 题 = 190 单图(95%) + 10 多图(5%)

**首帧设计原则**：
- 主体清楚、背景刚性可分层
- 初始构图匹配镜头命令
- 避免动态元素背景

**提示词规则**：
- 必须含至少一个 camera verb
- v1 每题最多两个镜头约束

**对比组**：zoom in ↔ zoom out；pan left ↔ pan right；运镜 ↔ static camera

---

### 维度 7：Interaction Reasoning（交互推理）— v1 暂缓

> 本维度在 v1 中暂缓实施。框架设计保留，待后续补充。prior_package 中无对应先验数据，需要额外构建 Visual Genome 关系库 + ConceptNet 交互关系词表。

**核心测试**：多物体/主体间的交互行为是否符合物理常识和因果推理。

**三种交互类型**：物理交互、社会交互、功能交互

**难度分桶**：I1(双物体简单物理), I2(较复杂物理/社会), I3(链式因果), I4(复合交互)

---

## 十一、维度间边界隔离规则

| 容易混淆的对 | 判定规则 |
|-------------|----------|
| Action vs Motion | 肢体动作 → Action；整体位移 → Motion |
| Action vs Interaction | 单主体自身动作 → Action；多主体因果交互 → Interaction |
| Motion vs Spatial | 有位移运动 → Motion；静态放置组合 → Spatial |
| Motion vs View | 主体在动 → Motion；镜头在动 → View |
| Attribute vs Action | 外观/状态变化 → Attribute；肢体动作 → Action |
| Background vs View | 环境自身变化 → Background；镜头运动 → View |
| Spatial vs Background | 可数主体空间布局 → Spatial；不可数环境区域 → Background |
| Interaction vs Motion | 有因果链和状态变化 → Interaction；纯位移无因果 → Motion |

---

## 十二、通用设计原则

### 对比组设计（所有维度通用）

每个维度中必须包含对比组样本，用于排除模型偏向。

### 语义罕见度（所有维度通用）

每个维度 200 题中，80%（160题）使用常见组合，20%（40题）使用罕见但合理的组合。

### 结构化元数据（所有维度通用）

每道题必须输出完整的结构化元数据：
- `dimension`：维度名称
- `input_mode`：单图 / 多图
- `input_images`：输入图像列表及角色说明
- `prompt`：文本提示词
- `difficulty_bucket`：难度桶标签
- `target_subjects`：目标主体列表
- `expected_change`：预期变化类型
- `preserve_constraints`：保持约束
- `semantic_rarity`：common / uncommon
- `contrastive_pair_id`：对比组配对 ID

### Generative Numeracy 不独立成维度

在多图输入范式下，输入几张参考图就天然定义了物体数量。数量正确性内嵌在各维度评测中。单图模式下的数量保持能力作为评分框架中的通用评分项。

---

## 十三、最终输出格式

### 单条题目（单图模式）

```json
{
  "question_id": "attr_001",
  "dimension": "attribute_binding",
  "input_mode": "single_image",
  "difficulty_bucket": "S2A1",
  "semantic_rarity": "common",
  "contrastive_pair_id": "attr_pair_01",
  "contrastive_role": "A",
  "input_images": [
    {"path": "attribute_binding/attr_001_frame.png", "role": "first_frame"}
  ],
  "i2v_prompt": "The white dog turns black while the brown dog remains unchanged.",
  "metadata": {
    "target_subjects": [{"id": "s1", "description": "white dog on the left"}],
    "expected_change": {"type": "color", "from": "white", "to": "black"},
    "preserve_constraints": [{"id": "s2", "description": "brown dog on the right", "constraint": "unchanged"}],
    "camera_constraint": "fixed"
  },
  "t2i_prompt_used": "Two dogs standing side by side on green grass in a park...",
  "quality_score": 0.92
}
```

### 单条题目（多图模式）

```json
{
  "question_id": "attr_150",
  "dimension": "attribute_binding",
  "input_mode": "multi_image",
  "difficulty_bucket": "R3",
  "input_images": [
    {"path": "attribute_binding/attr_150_ref1_woman.png", "role": "target_subject"},
    {"path": "attribute_binding/attr_150_ref2_dress.png", "role": "attribute_source"},
    {"path": "attribute_binding/attr_150_ref3_shoes.png", "role": "attribute_source"}
  ],
  "i2v_prompt": "The woman wears the red dress and white sneakers.",
  "metadata": {
    "target_subjects": [{"id": "ref1", "description": "young woman"}],
    "attribute_transfers": [
      {"source": "ref2", "type": "wearing", "item": "red dress"},
      {"source": "ref3", "type": "wearing", "item": "white sneakers"}
    ],
    "camera_constraint": "fixed"
  }
}
```

---

## 十四、待确认问题

| 编号 | 问题 | 当前状态 |
|------|------|----------|
| Q1 | **T2I 模型选择**：FLUX.1 Pro / FLUX.1 Schnell / SDXL / DALL-E 3 / 其他 API？ | 待确认 |
| Q2 | **图像分辨率**：1024×1024 / 1280×720 / 1024×576？需要匹配下游 I2V 模型输入 | 待确认 |
| Q3 | **多图模式的比例是否为最终版**：Attr 60/40, Action 60/40, Motion 50/35/15, Spatial 20/80, Background 60/40, View 95/5 | 待确认 |
| Q4 | **Interaction Reasoning 恢复时间**：v2 再做？还是 v1 后期补充？ | v1 暂缓 |
| Q5 | **Stage 6 质量过滤规则**：待讨论评测模型和评分公式后再定 | 待后续讨论 |
