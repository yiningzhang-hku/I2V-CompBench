```
# Phase 2 首帧来源与 Prompt 定稿策略 · 讨论纪要

> 本文档汇总围绕 Phase 2 "首帧图像 + 变化提示词" 来源策略的三轮讨论，整理用户提出的核心观点与对应的方法论分析，作为 Phase 2 设计决策的依据档案。
>
> 配套文档：
> - [Phase2_Benchmark数据集合成.md](./Phase2_Benchmark数据集合成.md)
> - [各阶段产物与用途说明.md](./各阶段产物与用途说明.md)

---

## 0. 背景与缘起

Phase 2 需要为每条 recipe 同时产出 **首帧图像（含多图参考）** 与 **针对该首帧的 I2V 变化 prompt**，二者绑定后导出为 `BenchmarkSample`。围绕"首帧从哪里来"，讨论中提出了三个递进的问题：

1. Phase 2 是不是要合成配套的首帧图像和变化提示词？
2. 为什么是"挑选"首帧而不是全部用 T2I 合成？
3. 既然 Phase 1 已经解析过首帧内容，根据先验全合成不行吗？

最终结论：**Phase 2 采用 "真实首帧为主、T2I 合成为补足" 的分层策略**，下面逐点说明。

---

## 1. 议题一：Phase 2 是否同时承担首帧合成与 prompt 定稿

### 1.1 用户观点

> Phase 2 应该需要合成配套的首帧图像和针对首帧图像的变化提示词。

### 1.2 文档依据

#### 首帧合成职责（[Phase2 §4.4 `construct_inputs.py`](./Phase2_Benchmark数据集合成.md)）

- single_image：可使用 TIP 首帧、T2I 生成首帧或外部真实图。
- multi_image：使用 `reference_bank` 或 T2I 补足。
- 输出位置：

  ```text
  data/benchmark_dataset/
    images/        ← 首帧图像
    references/    ← 多图参考资产
  ```

- 每张图都要落 `input_assets_manifest.jsonl`（含 `role` / `source_type` / `quality`）。
- 紧接 §4.5 `verify_inputs.py` 做结构化 VQA/QC，不通过则**重写或重采样，最多 3 次重试**。

#### Prompt 定稿职责（[Phase2 §4.6 `finalize_prompts.py`](./Phase2_Benchmark数据集合成.md)）

prompt 在 Phase 2 内部经历两次形态：

1. **prompt_draft**（§4.3 `build_question_plan.py`）：依据模板 + recipe 生成的初稿。
2. **i2v_prompt**（§4.6 `finalize_prompts.py`）：结合实际产出的首帧 + QC 结果定稿，过禁词与长度检查后写入 `prompts/final_prompts.jsonl`。

定稿要求：

- 英文，8–25 词为主。
- 多图 reference 指代稳定（image 1 / reference jacket）。
- 不混入非目标维度（dimension_isolation 强约束）。

### 1.3 结论

**Phase 2 = "为每条 recipe 配套合成/挑选首帧（含多图参考），并基于真实产出的首帧定稿一条只描述目标变化、不泄漏其它维度的英文 I2V prompt"**。

首帧与 prompt 的绑定顺序是 **"先按 question_plan 出图 → QC pass → 再按真实出图结果定 prompt"**，保证：

- prompt 中提到的实体在首帧里真的存在且唯一可定位；
- 多图 prompt 指代的 *image 1 / reference X* 与 `input_assets_manifest` 的 role 一一对应；
- prompt 不会泄漏其它维度。

---

## 2. 议题二：为什么不全部 T2I 合成

### 2.1 用户观点

> 不应该全部都是合成吗？

### 2.2 立项决定：benchmark 的真实性必须可追溯到 TIP-I2V

I2V-CompBench 是**数据驱动**的 benchmark，其论文叙事的核心差异化卖点是"基于 TIP-I2V 真实先验"。如果 Phase 2 把首帧全部改成 T2I 合成，会同时摧毁三条 benchmark 可信度链路：

| 链路 | 全合成会发生什么 |
|---|---|
| **分布真实性**：题目分布对齐真实 I2V 用户行为（prior_package 来自 TIP）| T2I 生成图遵循的是 T2I 训练分布，不再是 I2V 用户首帧分布 |
| **provenance 可审计**：`source_trace` 要能追到 `tip_xxx_sample_id` | 全合成后 reviewer 无法验证"题目是否来自真实场景" |
| **评测公平性**：模型在真实首帧上的能力才是论文要测的 | 合成首帧上模型可能拿到虚高分（同分布偏置）|

**与 T2I-CompBench / VBench 的对照**：那些 benchmark 可以全合成，是因为它们评测的是 T2I / T2V 模型，输入本来就是文字。**I2V 的输入是图像，benchmark 的图像必须贴近真实 I2V 用户场景**——这是本项目区别于 VBench-I2V 的差异化卖点之一。

### 2.3 文档显式分级

[Phase2 §4.3](./Phase2_Benchmark数据集合成.md) `QuestionPlan` schema 中：

```json
"input_plan": {
  "source_preference": ["tip_derived_reference", "t2i_generated"]
}
```

**`source_preference` 是有序数组**——优先 `tip_derived_reference`，T2I 只是兜底。这是设计层面就钉死的优先级，不是工程偷懒。

### 2.4 T2I 不可或缺的三类场景（保留 T2I 通道的原因）

| 场景 | 真实首帧能做到吗 | T2I 的必要性 |
|---|---|---|
| Contrastive 反向题（"red→blue" 与 "blue→red" 配对）| 真实分布通常只有正向 | T2I 提供反向首帧 |
| 多图样本中"干净 attribute patch / object crop" | TIP 首帧抠出来的 crop 可能背景泄漏 | T2I 直接画一张干净背景的 patch |
| dimension_isolation 要求"维度纯净"的首帧 | 真实首帧通常多维度耦合 | T2I 按禁词清单画"只含目标维度信号"的图 |
| Hard 难度桶（违反常见共现）| 真实数据里几乎不存在该组合 | T2I 兜底覆盖率 |

### 2.5 结论

**"挑选"不是与"合成"对立，而是"先从真实候选采样、采不到再请 T2I 合成"的分层策略**。

---

## 3. 议题三：Phase 1 已解析首帧，能否根据先验全合成

### 3.1 用户观点

> 我在 Phase 1 已经描述、分析了首帧图像里有什么了呀，根据这个先验去合成不行吗？

这是讨论中最深的问题，触及"图像信息" vs "图像标签"的本质区别。

### 3.2 反驳一：Phase 1 的解析是"有损压缩"，T2I 合成是"基于压缩结果的二次解码"

[Phase 1 Step 2 `parse_images.py`](./各阶段产物与用途说明.md) 的输出字段：

> `subjects[]`（含 bbox / mask_path / attributes / segmentation_quality / tracking_feasibility）/ `subject_relations[]` / `background` / `camera_baseline` / `reference_potential`

但这些字段是为了"出题需要"抽取的**离散标签**，不是首帧的完整描述。一张真实首帧蕴含的视觉信息至少包括：

| 信息类别 | Phase 1 是否记录 | 对 I2V 评测是否重要 |
|---|---|---|
| 主体类别、bbox、mask | ✅ | 重要 |
| 主体属性（颜色/服饰）| ✅（粗粒度）| 重要 |
| 光照方向 / 色温 / 阴影硬度 | ❌ | **重要**（影响 motion 评测的 flow） |
| 纹理 / 材质细节 | ❌ | **重要**（影响 identity preservation） |
| 透视 / 焦距 / 景深 | ✅（仅 shot type 粗粒度）| **重要**（影响 view 维度判定） |
| 主体姿态 / 关节细节 | ❌ | **重要**（影响 action 维度评测）|
| 噪声 / 压缩 artifact / 摄像机风格 | ❌ | 重要（决定生成视频是否"看起来像视频"）|

把"a woman in a red jacket on a beach, medium shot"丢给 T2I 重画，得到的图像在 **80% 维度上和原图无关**。这意味着用 T2I 合成 = 把 Phase 1 的标签当瓶颈，**评测的实际上不是模型对"真实首帧"的处理能力**。

### 3.3 反驳二：分布偏移（distribution shift）

I2V 模型在生产环境的实际输入分布是：

```
真实拍摄 + 互联网图 + 用户自截图  ≈ TIP-I2V 分布
```

T2I 合成图分布是：

```
T2I 训练集分布 + T2I 生成偏置（过曝光、过对称、AI 平滑感、手指 artifact 等）
```

如果 benchmark 主线全部用 T2I 合成首帧，会出现一个**反讽性结果**：

> 论文测出的"模型 X 在 motion 维度得分 0.72"，其实是"模型 X 在 T2I 合成图上做 motion 的得分 0.72"——和模型在真实场景下的 motion 能力可能差很大。

这违反了 benchmark 的 **ecological validity（生态效度）** 原则——评测条件应贴近真实部署条件。

### 3.4 反驳三：解析误差会被 T2I 放大，不会被消除

[Phase 1 Step 4 `align_instances.py`](./各阶段产物与用途说明.md) 自己已经预见解析有误差：

> `alignment_confidence`：图文对齐的整体置信度。低于阈值的样本不进入 candidate_recipes。

也就是说 Phase 1 承认：**解析结果是带置信度的、概率性的**。如果用这个有损、带噪声的解析去驱动 T2I：

```
真实图  ──VLM解析──►  结构化标签（损失 + 噪声）  ──T2I合成──►  合成图
                                                    ↑
                                          T2I 又会基于自己的偏置再"脑补"一次
```

最后得到的合成图与真实首帧之间隔了**两层有损变换**，错误被复合放大。而直接用真实首帧 + Phase 1 的标签出题，标签只用于"知道要测什么"，不参与"重建图像"，**误差不会作用到图像本身**。

### 3.5 关键澄清：Phase 1 解析的角色

> **Phase 1 的解析是"出题说明书"，不是"图像压缩文件"。**

- 解析告诉 Phase 2 "这张图里有什么、可以测什么、要保持什么"——用于**出题决策**。
- 解析**不**用来"重建图像"，因为重建意味着用一份有损标签替换一份高保真原图。

---

## 4. 最终策略汇总

### 4.1 Phase 2 首帧来源分层策略

| 题目类型 | 首帧来源主选 | 何时用 T2I |
|---|---|---|
| Single-image 主流题（attribute / motion / action / view 等）| TIP 首帧（`tip_derived_reference`）| TIP 找不到合规候选时 |
| Multi-image 主流题（spatial / interaction 等）| `reference_bank` 真实 crop + 真实 inpainted scene | 缺少干净 attribute patch / object crop 时 |
| Contrastive 反向题 | —— | 反向组合在 TIP 里稀缺时直接 T2I |
| Hard 难度桶（违反常见共现）| —— | 真实数据里几乎不存在该组合时 |
| dimension_isolation 要求维度纯净 | 真实首帧大概率多维度耦合 | T2I 按禁词清单生成"只含目标维度信号"的图 |

### 4.2 Phase 2 首帧 + Prompt 配套流程

```text
Phase 1 解析（出题说明书）
        │
        ▼
QuestionPlan（input_plan + target_plan + dimension_isolation + evaluator_plan）
        │
        ▼
construct_inputs：source_preference = [tip_derived_reference, t2i_generated]
        │
        ▼
verify_inputs：结构化 VQA/QC（最多 3 次重试）
        │
        ▼  pass
finalize_prompts：基于真实产出的首帧定稿 i2v_prompt（禁词 + 长度 + 角色指代）
        │
        ▼
export_dataset：BenchmarkSample（input_images + i2v_prompt + metadata + source_trace）
        │
        ▼
phase3_manifest.jsonl
```

### 4.3 论文叙事影响

| 维度 | "真实优先 + 合成补足" | "全部 T2I 合成" |
|---|---|---|
| 数据真实性 | 大部分题来自 TIP 真实首帧，可辩护 | 全部为合成图，与真实 I2V 输入分布脱节 |
| Provenance | 每张图都有 `source_type` 与 `phase1_sample_ids` 可追溯 | 失去 TIP 立项叙事 |
| 与 VBench-I2V 的差异化 | 显著差异化（数据驱动）| 退化为另一个合成 benchmark |
| Reviewer 关切 | 可分别报告"真实题 vs 合成题"上的模型差异 | 难以回答"模型在真实场景下表现如何" |
| Ecological validity | 高 | 低 |

---

## 5. 一句话纪要

- **Phase 2 必须同时产出首帧图像与变化提示词**，二者通过"先出图 → QC → 再定 prompt"的顺序绑定。
- **Phase 2 不全合成首帧**，因为"基于 TIP-I2V 真实先验"是 I2V-CompBench 的核心立项叙事，全合成会摧毁分布真实性、provenance 可审计性、ecological validity 三条可信度链路。
- **Phase 1 解析是出题说明书，不是图像压缩文件**——用它驱动 T2I 重建会引入有损压缩 + 分布偏移 + 误差放大三重伤害。
- **T2I 在 Phase 2 的角色是"补足通道"**，专攻 contrastive 反向、干净 patch、维度纯净、Hard 桶四类真实数据无法覆盖的场景，并通过 `source_type` 字段在评测时分别报告。

最终设计：**真实首帧为主，T2I 合成为补足；先验是出题说明书，不是图像生成器**。
```
