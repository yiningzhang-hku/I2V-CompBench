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

### 0.1 vs T2V-CompBench / VBench-I2V 的差异化定位

本阶段产出的 benchmark 在以下三点上与现有 SOTA 做出区分，这些区别是 Phase 2 设计选择的均衡点不是图省事：

| 维度 | T2V-CompBench (NeurIPS’24) | VBench-I2V | 本框架 (Phase 2 产出) |
|------|------|------|------|
| **题目来源** | VidProM 167 万 prompt + WordNet 分类 + GPT-4 生成 | 人工设计主题词 | TIP-I2V 真实分布采样 + Phase 1 recipe，零拼词表 |
| **输入形态** | T2V（只有文本） | I2V 单图 | I2V 单图 + **多图作为主维度**（非 stress） |
| **评测维度** | 7 类（含 Numeracy） | 16 类但混合质量+指令遵循 | 7 维纯指令遵循，明确不包 Numeracy（详见 Phase 1 §5） |
| **对照设计** | static_copy 一项 | 无 | static_copy / random_motion / global_filter / camera_pan_cheat / **subject_swap_inverse** 五项 |
| **评分公式** | 加权平均 | 维度指标拼接 | **执行门控乘法** `S = E·(0.6P + 0.4C)` |

**本框架不跟随何者**：不复制 T2V-CompBench 的词表拼接路径（理由见 Phase 1 §10 决策记录），也不堆维度凑数量（VBench 的 16 类有多个与质量耦合不可独立归因）。

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

**关键设计与底层原理**

- **为什么用最大余数法而不是四舍五入**：整数配额按比例切分时，简单四舍五入会让总和漂移（例如把 20 按 0.95/0.05 切，单纯舍入得到 19+1=20 看似没问题，但 20 按 0.6/0.3/0.1 切就会变成 12+6+2=20 与 12+6+2=20 时还行，遇到 0.33/0.33/0.34 就会出现 7+7+7=21 的越界）。最大余数法在每一级都先取整再把误差按"余数排序"分配，保证 `target_count` 求和精确等于 `num_per_dimension × 7`，下游 audit 才能用减法直接得到缺口。
- **为什么把 bucket_id 提前固化到 quota_plan**：评测的真实分组键就是这五个维度，预先生成稳定主键让 sample / export / audit 都能用 join 串起来，不必每步重新算配额。下游模块只需理解"桶"这个概念，不必再读 yaml。
- **为什么七维度共用 difficulty/rarity 但 input_mode_ratio 维度独立**：难度与稀有度是评测体系横向可比的统一概念，应当保持一致；而输入模式分布与维度本身的真实先验强耦合（view 95% 单图来自"运镜数据天然只有一张起始帧"，spatial 80% 多图来自"空间组合天然需要多个参照"），共用一份比例反而失真。
- **为什么 motion 维度退化为 subtype 分桶**：motion 在 TIP-I2V 真实分布里的分类轴是 `type_a_absolute_single / type_b_relative_single / type_c_multi_motion`，并非 single/multi 二元。强行套 input_mode 会把"绝对位移单图"和"相对位移单图"挤进同一桶，违反 motion 边界规则（§2.3）。

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

**关键设计与底层原理**

- **为什么绝不允许 Phase 2 自己拼词表**：benchmark 的可信度建立在"题目源自真实 I2V 分布"——一旦 Phase 2 用启发式拼出新组合，reviewer 的第一个质疑就是"你怎么证明这条题目对应真实用户场景"。让 candidate_recipes 成为唯一题源，每条样本都能反向追到一条 TIP 视频，省去整本论文里反复辩护"分布合理性"的成本。
- **为什么阈值定在 quality_score ≥ 0.4 且 is_clean_background**：低质量素材会在 Phase 3 变成捷径——背景里残留杂物会让"看背景反推空间关系"成为模型的偷懒解法，模型即使没真做空间组合也能拿高分。把这层闸门放在抽样阶段（而不是 QC 阶段）是为了用最便宜的方式先剔除大批不可用 recipe，节省后续 T2I 与 VQA 调用。
- **为什么多图至少 2 张达标**：multi_image 任务的本质是"组合多个独立线索"，1 张 reference 退化成单图任务，违背维度设计。这里硬卡比下游再降级更早、更便宜——construct_inputs 才发现少图时已经浪费了一次 question_plan 构建。
- **为什么 A/B 必须原子化配对**：contrastive consistency 是论文里 anti-shortcut 的关键证据，需要"同一题型正反方向各跑一次"。如果只采到 A 没采到 B，这条数据在 contrastive 表里直接作废，比起多采几道无关题，"配对完整性"对评测意义更大。
- **为什么缺口要分四类原因而不是只给一个 gap 数字**：reviewer 关心的不是"差几条"，而是"差在哪"——`no_candidate` 提示数据扩充方向，`no_reference_asset` 提示要补 Phase 1 资产，`quality_below_threshold` 提示要调阈值，`blocking_flag` 提示 Phase 1 patch 漏标。粒度化分类让缺口直接转成 actionable 的工单，而不是一句"补数据"。

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

**关键设计与底层原理**

- **为什么三层叠加且后者覆盖前者**：三个数据源各自承载不同语义——`image_parse` 给"图里实际有什么"（视觉真值），`aligned_instances` 给"文本指代到底绑到了哪个实例"（图文对齐），`text_parse` 给"用户到底想要什么变化"（意图）。后者覆盖前者体现一个原则："意图 > 对齐 > 现状"，最终 prompt 是为意图服务的，所以让 text 解析的字段拥有最高优先级。
- **为什么 subtype 走三级回退而不是直接抛错**：模板库不可能枚举所有真实分布的 subtype 组合（例如某些罕见 subtype 在 pilot 里恰好没被 Phase 1 标出来），让 input_mode 一致即可，再不行退到该维度第一个可用 subtype。这种"宽松匹配"比"严格匹配整桶失败"更利于 pilot 跑通。
- **为什么 question_id 用 DIM_SHORT + mode_short 而不是哈希**：人工排查时第一眼就能从 ID 看出维度与模式（`attr_multi_0001` 一眼就知道是属性绑定的多图题），比纯哈希 ID 友好；同时短前缀避免文件名超长，也方便按前缀做 grep / 分桶统计。
- **为什么 prompt_pattern 渲染失败要回退到 base_prompt_draft 而不是丢弃**：渲染失败的真因往往是某个槽位缺值（例如 attribute_change_slots 没标全）。但 recipe 自带的 `base_prompt_draft` 已经是 Phase 1 用真实 caption 拼出的可读句子，留作兜底比让整道题失败更划算——QC 仍能跑、polish 仍能改写，只是 risk_flags 多一条 `template_render_failed` 提示人工抽查。
- **为什么 evaluator_plan 必须在 Phase 2 落定**：如果 Phase 3 才反向解析自然语言 prompt 推工具集，会引入 NLU 误差且不可复现——同一道题在不同次评测里调用的工具栈可能不一样，分数也就不能横向比。把工具集合编码进 evaluator_plan 是把"评测目标"从自然语言搬到结构化字段，让 Phase 3 变成纯执行器。
- **为什么 prompt_draft 不能直接当 final**：draft 是结构化模板渲染的产物，并未"看图说话"过。如果首帧实际状态与模板槽位有出入（例如模板写"a woman"，但首帧实际是"两个人，左边是女士"），prompt 与图就会错位。最终 prompt 必须经过 Step 4.6 的 VLM 描述对齐才能保证 prompt 与起始状态自洽。

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

**关键设计与底层原理**

- **为什么用 _ROLE_TO_ASSET_TYPE 做间接映射**：题目语义层用 role（`target_subject` / `attribute_reference`），资产存储层用 asset_type（`subject` / `attribute` / `scene_reference_inpainted`）。两层 schema 各自演化（题目可能新增 role、资产库可能新增 type），中间留一张映射表能让任一层调整时不污染另一层；硬编码合并会导致一处改动牵连两套 schema。
- **为什么单图直接复用 TIP 首帧而不是合成**：这是 benchmark 真实性的根基——单图任务的"起始状态"应当是真实视频的首帧，否则评测的就不是"模型对真实首帧的处理能力"，而是"模型对合成首帧的处理能力"，与论文论点错位。Phase 1 已经付出代价做过 manifest 清洗，没必要在这里推倒重来。
- **为什么同角色多候选挑 quality_score 最高**：quality_score 是 Phase 1 综合 bbox 面积、可见度、完整度算出的代理指标，已经融合了多个质量信号；用它做启发式比"随机选一张"或"按时间序选最早"都更稳定，且无额外计算成本。
- **为什么链式回退顺序是 tip → t2i → external**：三级递降的不是只看"清洁度"，更重要的是"reviewer 可审计度"——TIP 资产可追溯到原视频帧（最强）、T2I 可追溯到 prompt+seed（中等）、external 需要单独的 provenance 字段（最弱）。把审计成本作为优先级权重，而不是单纯的视觉质量。
- **为什么 long_edge=1024 + PNG**：1024 是大多数 I2V 模型（Stable Video Diffusion / CogVideoX / Sora-class）的训练分辨率默认值，再大会让 Phase 3 推理开销陡增；PNG 无损保存避免 JPEG 在边缘产生 ringing 伪影——这些伪影会干扰 grounding 评测器的边缘检测，让 P 评分受 codec 噪声影响。
- **为什么 T2I 失败不抛异常**：流水线 8 步，任意一步在单题失败如果中断会让上游全部白跑。让 QC 统一裁决，整体吞吐更稳；T2I API 偶尔挂掉时也不至于让整个 batch 报废。
- **为什么把 quality 元数据存进 manifest 而不是临时计算**：QC（Step 4.5）和 export（Step 4.7）都要消费这些字段，存一次比每步重算更稳定，也避免不同步骤算法略有差异导致同一字段两次算出不同结果。

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

**关键设计与底层原理**

- **为什么用 VLM 而不是规则匹配**：题目合规性是语义级判断（"主体是否还静止"、"目标属性是否还没出现"），传统 CV 规则无法覆盖；VLM 是当前能给出语义级判定的最低成本工具。手写检测器既覆盖不全也维护不动。
- **为什么 confidence ≥ 0.7 才算 fail**：VLM 在边界样本上有抖动，单次低置信度的"否"很可能是模型不自信而非真否决。卡在 0.7 把"明确否决"（≥0.7 的 False）与"模糊判断"（<0.7）分开——前者直接进 retry，后者进人工，两者用不同的处理成本对应不同确定度。
- **为什么 < 0.7 直接进人工而不是再 retry**：retry 的本质是换一张图重新构造，但 VLM 不确定的题目即使换图也大概率再次抖动；让人裁决一次的边际成本，比反复跑 VLM API 既便宜又终局。
- **为什么 HARD_CHECK 集中在 `_HARD_CHECK_NAMES` 常量而不是分散在模板**："哪些 check 一票否决"是评测设计的一等决策，模板作者不应能随意提升某个 check 的权重——否则不同维度的"严格度"会随作者风格漂移。集中常量也避免拼写错误造成静默退化（模板里把 hard check 写错名字，原本应阻断的就被默认软处理）。
- **为什么有 `no_target_attribute_already` / `no_action_already_in_progress` 这类反向检查**：这些是防"题目自我作弊"——如果首帧已经包含目标终态，模型完全静止也能在 E 评分里拿满分。这种"起始即终态"的样本必须从源头剔除，否则 Static Copy baseline 会跟正常模型并列高分，anti-shortcut 实验直接破产。
- **为什么 needs_manual_review 默认不进 Phase 3**：multi-image 维度本来就稀缺，但与其放进去拉低评测可信度，不如等人裁决。论文 reviewer 关心的是"你过滤了多少（怎么保证质量）"，而不是"你保留了多少（怎么凑数）"，所以默认走严格路径。

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
   - **active verb 硬约束**（除 `view_transformation` 外的六个维度）：prompt 必须至少包含一个 active verb（SpaCy POS 标为 `VERB` 且 lemma 不在隐含静态集 `{be, have, exist, remain, stay, look, seem, appear}` 中）。`view_transformation` 题仅要求包含 camera 动作词（`zoom`/`pan`/`tilt`/`dolly`/`orbit` 任一）。未命中记 `failed_check="missing_active_verb"`。
   - **指代检查**：多图必须能解析出 `image 1 / image 2` 或 `the reference X` 这类显式角色指代。
4. **重试与回退**：最多尝试 N 次（`polish_attempts` 字段记录次数），仍不合格则回退到 `prompt_draft`，并在 `risk_flags` 加 `prompt_polish_fallback`。

**输出**

- `benchmark_dataset/prompts/final_prompts.jsonl`：`FinalPromptEntry { question_id, i2v_prompt, polish_attempts, used_fallback, vlm_caption }`。

**关键设计与底层原理**

- **为什么先 VLM caption 再 LLM polish 两步分工**：VLM 负责"看图说话"提供视觉锚点，LLM 负责"按目标改写"做语言润色。合并到一步让 VLM 直接出 prompt 会让结构化约束（forbidden_words、字数、reference 指代）很难管控——VLM 优于看图但弱于遵循复杂格式约束，LLM 反之，分工各取所长。
- **为什么强制 LLM 返回 JSON 而不是裸文本**：裸文本下 LLM 经常自作主张加引号、解释、emoji 或 markdown 标题；JSON 强约束让解析端能严格校验；同时保留 `reasoning` 字段以便事后审查"模型为什么这么改"，调试 polish 失败比直接看 prompt 输出更高效。
- **为什么解析需要 fenced ``` + 括号扫描双层兜底**：LLM 即使被强制要求输出纯 JSON，也常把它包在 markdown 代码块里、或在 JSON 前后加注释。双层解析（先剥 fenced，再扫描第一对配平的大括号）让 polish 不会因为格式细节白费一次 API 调用——单次 LLM 调用成本不低，能解析就尽量解析。
- **为什么字数定 8–25 / interaction 30**：TIP-I2V 真实 prompt 中位数在 12 词左右，8–25 覆盖了约 80% 的真实分布；interaction 题需要描述"主体 A + 主体 B + 因果链"，强行压到 25 词会丢信息（例如把"the man hands the cup to the woman"压到"man hands cup"会让 evaluator 无法定位 woman 角色），所以专门放宽到 30 词。
- **为什么强制 active verb**：T2I-CompBench 与 T2V-CompBench 都明确指出，不含动作动词的 prompt（"a red ball on the table"）让 I2V 模型退化为静态复制却仍能拿高分，评测出现 false positive。把"至少包含一个动作动词"落到 finalize_prompts 的硬检查，是把评测设计意图（"考查指令是否被执行"）从文档约定变成可执行约束。`view_transformation` 作为例外是因为它考的就是 camera 运动，动作动词会跟主维度冲突。
- **为什么禁词命中即重写而不是软警告**：dimension_isolation 是评测可信度的最后防线——motion 题里出现 "pan" 会让模型用相机平移伪装主体位移，spatial 题里出现 "move" 会让 spatial 与 motion 失去边界，论文 §2.3 / §2.4 的硬规则在这里落地。让禁词命中变成强制重写，是把维度边界从设计文档落到可执行约束。
- **为什么 fallback 仍允许 export**：流水线已经把 QC 通过的样本送到这一步，如果因为 polish 失败丢弃整题会让上游 4 步白费。标记 `prompt_polish_fallback` 让审计层（Step 4.8）能统计 fallback 比例并触发后续修复——比起静默丢数据，留痕的质量警示更利于持续改进。

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

**关键设计与底层原理**

- **为什么以 question_id 落盘 join 而不是内存串接**：四个上游模块各自独立写 jsonl，如果靠内存对象串接，一旦中间任何一步崩溃就要从头重跑。落盘 + 文件 join 是流水线最稳的串接方式——任何一步失败都可以单独重跑该步，前后产物保留可复查。
- **为什么三重门控（pass + 非空 prompt + 非空 manifest）**：三种空值各自代表不同的失败模式——`qc_status≠pass` 是质量否决，`i2v_prompt 空` 是 polish 失败且未 fallback（理论上不应发生，是兜底），`manifest 空` 是构图失败。任何一项缺失都会让 Phase 3 报错，三重门控能在 export 阶段就把这些样本滤掉，比让 Phase 3 模型跑完才发现问题节省至少一个量级的算力。
- **为什么多图质量按 worst-case 聚合**：评测可信度的关键是"任何一张参考的弱点都会被模型利用"——三张参考里有一张背景脏，模型可能就靠那张背景做空间推断；最高/平均聚合会掩盖这种风险。worst-case 标记最保守，让 audit 能精准识别出"看似合格但有薄弱点"的多图样本。
- **为什么 phase3_manifest 是 BenchmarkSample 的视图而不是独立结构**：保证两份产物 schema 同源——manifest 字段调整时直接从 Sample 派生，不会出现"加了字段但忘了同步 manifest"的下游事故。如果用独立 schema，schema drift 是迟早的事。
- **为什么按 dimension 分文件而不是单一大文件**：Phase 3 评测器是按维度调度的（Step 5 `evaluators/{dimension}.py`），分文件让"只跑某维度"成为零开销操作（直接读对应 jsonl），不必每次过滤；论文统计也需要按维度切片，分文件让 wc -l 就能统计每维度规模。
- **为什么 source_trace 必带 phase1_sample_ids 与 phase1_asset_ids**：reviewer 一抽样就要能回到 TIP 原视频与原资产去验证"这道题确实源自真实分布"。这是 benchmark "可审计性" 的硬性要求——少了任何一个 ID，"从样本反查 provenance"的链条就断了。

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

**关键设计与底层原理**

- **为什么单独做 audit 而不在 export 里顺便校验**：审计是"独立见证人"角色——审计逻辑与导出逻辑共用代码会让 bug 互相隐藏（同一个错误的字段在两边都按错误规则处理仍能"自洽"通过）。分离两者才能在自检失败时给出可信信号，类似软件工程里的 "test code 不复用 production code 的私有函数"。
- **为什么 audit 不自动修复**：自动修复的本质是"猜测意图"，但每类缺口背后的根因不同——配额缺口可能是因为 Phase 1 数据不够（要补料）、可能是阈值太严（要调参）、也可能是某个上游 step 有 bug（要 debug）。把决策权留给人，比让脚本"按某个默认策略修复"安全得多。
- **为什么六类校验缺一不可**：六个维度对应六类风险——行数对应"完整性"、配额 gap 对应"代表性"、维度统计对应"均衡性"、contrastive 配对对应"实验设计完整性"、质量直方图对应"评测可信度"、QC 直方图对应"成本可控性"。漏掉任何一项都会在 Phase 3 暴露成问题（例如漏 contrastive 校验，到 Phase 3 跑完才发现 anti-shortcut 表算不出来）。
- **为什么 dataset_card.md 用 Markdown 而不是 JSON**：论文素材直接 copy/paste，CI 可读，git diff 友好；JSON 适合机器消费但不适合人审阅。审计输出的主要受众是项目 owner 与论文作者，Markdown 是这个语境下最合适的载体；机器需要的关键数据由 export 阶段已经提供。

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
