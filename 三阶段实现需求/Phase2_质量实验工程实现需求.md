# Phase 2质量实验与正式数据集筛选工程实现需求

> 面向AI Coding Agent的工程规格文档  
> 适用范围：I2V-CompBench数据集构建方（Phase 1 + Phase 2）  
> 最终目标：完成质量对比实验，并从五维候选池中筛选每维300条、共1500条正式样本

---

## 执行总览：问题对应、对比试验与可行性边界

### A. 审视结论

本工程需求对《Phase 2 产物质量问题分析报告》中的主体问题总体覆盖较全：跨阶段 Schema 失配、无效 Prompt 与 fallback、低频修饰词、图像清晰度、宽高比与尺寸适配、主体分布、难度重标以及词汇—主体正交诊断均有对应的工程步骤、输出物和验收入口。原始产物只读、run 目录隔离、断点续跑、API 预算保护、权重显式管理和回归测试等要求也具有工程可行性。

但“问题已经写入需求”不等于“原稿可以按原样全量执行”。本次审视确认了七个必须先修正的可行性问题：

1. 旧要求只验证图像路径“存在”，不能证明 qid 绑定的是正确源图；图像实验前必须增加来源ID、原图—首帧派生谱系及双端哈希门控。
2. 最多14个Prompt条件、8个清晰度条件、8个尺寸条件、20组阈值、240条正交诊断、5个处理层条件和3种集合抽样策略不能同时被“8小时总预算”覆盖；8小时只约束冻结方案的全池生产处理，不包含开发消融、模型部署和人工标注。
3. 词汇—主体正交诊断若要评价文本编码器或视觉长尾能力，必须导入 Phase 3 视频结果或独立人工结果变量；Phase 2 自身只能完成天然四象限分配和供给诊断。
4. Sampling 会改变样本集合，不能与同 qid 的 Text/Image 处理使用配对检验；变换层消融与集合层重采样必须分开。
5. 问题报告的开头总览与综合优先级表对 P2—P6 的编号不一致；本需求统一以“问题名称”为主引用，P 编号只作辅助标识。
6. Coding Agent 可以实现数据处理、任务导出、结果导入和统计报告，但不能替代人工标注者。开发/验证集难度标注、图像方法盲评和1500条终审均是外部人力依赖；没有真实人工结果时，相应实验只能标记为`awaiting_human_annotation`，不得自动生成标签或结论。
7. 当前根目录`requirements.txt`只覆盖原有Phase 1/2基础依赖，尚未声明词频、统计、图像质量、超分辨率、DINO和Grounding等实验依赖；必须增加质量实验专用依赖锁定与模型注册文件，不能把“库可安装、权重可获得”当作默认前提。

本节描述的均为**拟执行实验与待验证命题**，不是已经获得的结果。Coding Agent 必须使用实际结构化产物填写结论，不得把候选阈值、预期方向或分析报告中的建议写成“已证明”。因此，本需求在软件实现层面可行，在完整实验执行层面属于**有条件可行**：是否完成取决于Phase 1资产、模型权重/GPU、API预算和人工标注四类资源是否就绪。

#### 当前仓库可执行性快照（2026-07-13审视）

| 项目 | 当前状态 | 对执行的影响 |
|---|---|---|
| Phase 1/2字段兼容修复 | `aligned_subjects`、顶层slots及v2文件发现逻辑已进入现有代码 | 可作为重新生成plan的代码基础 |
| 无第三方依赖审计与固定划分 | 已有审计脚本、候选报告及development/validation各250条产物 | 可复用，但加入资产绑定后必须重新冻结split hash |
| 正式1500条状态 | 现有审计因结构字段缺失得到0条完整eligible记录，尚未产生official 1500 | 当前只能称为候选池与实验划分，不能称为已完成Benchmark数据集 |
| Phase 1 v2 bundle、canonical源资产清单与派生谱系 | 当前仓库数据目录未发现；旧`configs/phase2.yaml`仍指向不可移植的Windows绝对路径，旧`input_assets_manifest`也未记录输入/输出hash | **P0输入阻断**；取得外部bundle并重建谱系，或完成人工迁移前不能开始图像实验 |
| `src/i2vcompbench/quality/`、质量配置与测试 | 当前尚未创建 | 属于本需求需要Coding Agent新增的主要工程量，不是已有命令的简单重跑 |
| 质量实验依赖、模型权重与GPU | 当前基础依赖文件未完整声明，权重/计算资源待体检 | C2/C3、DINO、Grounding及相关统计在资源就绪前为`blocked` |
| 主体Tier本地资源 | 当前未发现`resources/quality/`及其COCO/LVIS/词频/细粒度资源 | 主体分层和分层抽样在资源落盘、版本及许可证记录完成前为`blocked` |
| 人工标注 | 尚无导入结果 | Coding Agent可先完成任务包和界面；涉及人工金标准的结论为`awaiting_human_annotation` |

因此，建议把“可行”拆成三个里程碑判断：本地审计与工程骨架可立即实现；结构恢复和资产绑定在取得Phase 1证据后可执行；完整图像、难度与终审实验在模型及人工资源就绪后可完成。

### B. 问题报告与实现需求的对应

| 问题名称 | 报告证据 | 本需求对应 | 要证明或验收的内容 | 审视结论 |
|---|---|---|---|---|
| Schema 失配与结构目标失效 | 3517/3517 缺主体 noun 和可消费目标变化 | 第2、8、9、18.4、20.1节 | 主体、Transform 和 Preserve 来自可追溯证据，而非仅仅非空 | 主体与 target relation 已覆盖；本次补充 Preserve 修复和资产绑定 |
| 无效 Prompt、低频表达与 fallback | 121条长度异常、113条重复冠词、18条空槽位、117条 fallback | 第10节 A0—A4核心链、可选A5/A6、Source-Off/On、B0—B3 | 在保持主体、方向、运镜和目标语义时降低无效率及无贡献低频表达 | 覆盖充分；最多14条件只在开发集筛选 |
| 图像清晰度不足 | 低分辨率源图被大倍率插值，增强可能改变身份和纹理 | 第11节 C0—C7 | 清晰度改善时主体、身份和原图语义不发生不可接受下降 | C0—C3为核心；GFPGAN及C5—C7为可选 |
| 宽高比与尺寸适配 | 裁剪、拉伸和填充可能损失主体或引入伪背景 | 第12节 4:3/16:9、D0—D7 | 在固定像素预算下选择主体和场景损失更小的比例与策略 | 主要方案覆盖充分；Outpainting、D7和完整网格为可选 |
| 主体分布失衡 | 长尾主体不能通过改词修复，只能分层和抽样控制 | 第13、16.2—16.3节 | 分层抽样改善覆盖且不引入低质量样本 | 可行；属于集合分布实验，不是同样本处理实验 |
| 旧难度混入质量缺陷 | 模糊、空槽位和评测不确定性不应伪装成 hard | 第14节 G0—G3 | 新标签比旧标签更符合人工对任务本身的难度判断 | 可行；判断难度和语义罕见度应独立报告 |
| 词汇—主体长尾混杂 | 两类长尾自然共现时无法直接解释失败来源 | 第14bis节天然A/B/C/D四象限 | 检查候选供给；导入外部 outcome 后分析关联和交互 | 仅为可选诊断，不阻断1500条交付 |
| 正式1500条筛选 | 质量门槛优先于难度、罕见度和Tier配额 | 第16、20.3节 | 每维300条全部具有正确资产、完整结构、有效Prompt和人工裁决 | 数量和shortfall规则较完整；本次补充资产哈希及全量终审 |

### C. 实现需求实际规定的对比试验

| 实验组 | 对比条件 | 样本与阶段 | 主要指标 | 拟验证命题 | 级别 |
|---|---|---|---|---|---|
| 结构目标恢复 | Phase 1重建优先；残缺项VLM迁移；低置信度人工复核 | 3517条审计，开发集250条人工验收 | Schema完整率、主体/变化/Preserve准确率、维度一致性 | 候选能否恢复为可评测记录 | **前置验收，不是效果对比** |
| Prompt治理消融 | A0—A4嵌套链；可选A5困惑度过滤与A6受控解码；开发集比较Source-Off/On | 开发集250条筛选；验证集只复验基线与前2个冻结候选；全池只运行winner | 无效率、低频修饰率、Zipf、语义保持、维度纯度、成本 | 指令、词频门控及语义后检能否保留任务语义并减少异常表达；PPL或受控解码是否另有增量 | A0—A4 **核心**；A5/A6 **可选** |
| Fallback修复 | B0旧fallback、B1确定性模板、B2定向LLM、B3自动检查与人工队列 | 117条按失败类型/维度/source分成60条开发与57条验证；冻结winner后全量修复 | 结构通过率、语义保持率、人工可用率、API调用/条 | 受约束修复能否安全替代未验证回退 | **核心** |
| 清晰度方法 | 同一canonical原图分别进入C0 Lanczos、C1传统锐化、C2 Real-ESRGAN、C3 SwinIR；C4—C7扩展 | 开发集120张，其中人脸40张；冻结后验证 | 人工清晰度、NIQE/BRISQUE、DINO主体相似、身份变化、检测率、伪影 | 哪种方法在主体和身份非劣时改善首帧可用性 | C0—C3 **核心**；其余**可选** |
| 清晰度组合消融 | C2→C5/C6→C7 | 开发集清晰度子集 | 增量NIQE/BRISQUE、Laplacian、人工伪影 | USM和CLAHE是否有独立收益及边际递减 | **可选** |
| 目标比例 | 4:3 vs 16:9 | 六个宽高比桶×10，共60张pilot；固定像素预算 | 主体/场景保留、形变、填充面积、伪影、人工可用性 | 当前数据分布下哪种推理比例的输入损失更小 | **核心pilot**；无视频结果时不声称生成质量更好 |
| 尺寸适配 | D0—D4与D6保守混合；D5/D7为上界 | 开发集调阈值，验证集120张确认冻结方法 | 主体保留、几何形变、CLIP变化、NIQE、边界伪影、成本 | 主体感知和混合路由是否比单一全局方法更保真 | D0—D4/D6 **核心**；D5/D7和20组网格**可选** |
| 主体采样 | 自然抽样 vs 质量门控后的边际分层抽样 | 合格候选池，多随机种子重复抽样 | Tier/子类型覆盖、配额偏差、唯一来源、重复率、View shortfall | 分层策略能否改善集合覆盖而不降低质量门槛 | **核心集合级实验** |
| 难度标定 | G0旧标签、G1单复杂度、G2固定多因素、G3人工标定有序模型 | 开发集与验证集各250条×至少2人；开发拟合、验证确认 | QWK、macro-F1、ordinal MAE、一致性、混淆矩阵 | 新方法是否比旧标签更符合人工任务难度 | **核心** |
| 处理层组合消融 | P0修复后的共同基线；Baseline、+Text、+Image、+Text+Image、+Text+Image+Aspect | 冻结方案在验证集250条上配对运行 | Prompt/图像/结构硬门控、综合可用率、配对效应量 | 文本、清晰度和比例处理的独立及联合增量 | **核心**；Sampling不作为同qid配对条件 |
| 词汇×主体诊断 | 天然A/B/C/D四象限 | 期望80/60/60/40；不足即shortfall | 候选分布；导入外部outcome后才计算效应量/交互 | 两类长尾因素能否分组报告及是否相关 | **MAY**；缺外部outcome时不运行推断统计 |
| 成本—效果分析 | 各问题域通过硬门槛的方法 | 按方法实际作用样本数 | 质量非劣、耗时、GPU/API、依赖、确定性 | 质量接近时哪个方法更可部署 | **工程决策，不是效果实验** |

### D. 研究问题与证据边界

- **RQ1：**修复 Schema、恢复上游证据并增加资产硬门控后，候选是否成为结构完整、资产正确且可追溯的评测记录？
- **RQ2：**规则预检、异常项定向修复与语义后检，是否能在保持主体、变化和维度语义时减少无效 Prompt 和无贡献低频表达？
- **RQ3：**哪种保守清晰度方法能在身份、主体和语义保真的非劣前提下改善首帧可用性？
- **RQ4：**在统一像素预算下，哪种目标比例和尺寸路由更能保留主体与场景参照并减少形变和伪影？
- **RQ5：**主体分层抽样和人工标定难度是否比自然分布及旧标签提供更可解释的数据分布？
- **RQ6：**在同一P0修复基线上，文本、清晰度和比例处理的独立与联合贡献分别是多少？

本工程最多证明**输入数据质量、结构完整性、任务可评测性和集合分布得到改善**。除非显式导入 Phase 3 生成视频及独立人工标注，不得将任何 Phase 2 实验写成“I2V模型生成能力显著提升”。

### E. 最小可行执行顺序

1. 执行环境、依赖、权重、API预算、Phase 1 bundle和源图哈希体检；缺失核心资源时fail-fast并标记blocked，只有预先定义为MAY的实验才可记录not-run后继续。
2. 执行build-asset-manifest → build-asset-lineage → audit → split，冻结canonical候选、双端资产谱系、开发集和验证集。
3. 优先使用Phase 1 bundle重建，仅对残缺项执行VLM迁移和人工复核；P0结构及资产门控未通过时不得开始后续实验。
4. 在开发集筛选Prompt方法并处理117条fallback；参数冻结后，验证集只用于确认，不再调参。
5. 在开发集完成清晰度、比例和尺寸筛选；在验证集确认冻结方法及阈值。
6. 对质量合格全池只运行已冻结的Prompt和图像方案，不在全池生成全部消融变体。
7. 完成主体Tier、人工难度标定和集合级重采样实验，再执行正式硬门控及每维300条选择。
8. 对拟入选1500条执行全量人工语义终审；发现reject后从合格池补选并重审。
9. 使用冻结方案完成处理层组合消融并生成论文表格；资源允许时再执行GFPGAN、Outpainting、视频探针和正交推断。

**数据划分规则：**开发集用于方法筛选、阈值选择和模型拟合；验证集只用于确认冻结方案及报告泛化结果。任何在验证集上调整的规则都必须回到新的留出集重新验证。

## 0. 文档约定

本文使用以下强制级别：

- **MUST**：违反即不得进入下一阶段；
- **SHOULD**：默认实现，若不实现必须在run manifest中说明原因；
- **MAY**：资源允许时实现的增强项。

本文中“核心实验”表示正式1500条交付前必须完成的证据；“可选/MAY”实验未运行时必须记录`not_run`原因，但不得把它们设为正式集交付的阻断项。

Coding Agent不得将本文中的“候选方案”直接写成实验结论。所有提升数值必须来自实际运行产物。

## 1. 工程目标

### 1.1 核心目标

在不覆盖现有Phase 2原始产物的前提下，建立一条可断点续跑、可审计、可复现的质量实验流水线，完成：

1. 对3517条五维候选执行统一质量审计；
2. 修复或剔除空主体、空目标变化和无效fallback；
3. 比较Prompt低频词治理方案；
4. 比较图像清晰度增强方案；
5. 比较4:3/16:9与尺寸适配方案；
6. 完成主体Tier标注和难度重标；
7. 生成开发集、验证集、人工标注任务与统计报告；
8. 从质量合格候选中按五维等量原则筛选1500条正式数据集；
9. 输出论文第5章可直接引用的表格、JSON统计和失败案例索引。

### 1.2 正式范围

正式维度仅包括：

```python
FORMAL_DIMENSIONS = [
    "attribute_binding",
    "action_binding",
    "motion_binding",
    "background_dynamics",
    "view_transformation",
]
```

`spatial_composition` 和 `interaction_reasoning` **MUST NOT** 进入正式1500条样本。

### 1.3 非目标

本工程不负责：

- 实现Phase 3的E/P/C视频评分器；
- 构建多模型Leaderboard；
- 研究自动视频指标与人类判断的相关性；
- 实现Grounding、DINOv2或光流作为正式视频评分器；
- 用人工拼词或替换主体扩充不存在的recipe；
- 为凑足300条而降低硬质量门槛。

Grounding、DINOv2、CLIP等工具只允许作为**输入数据质量信号**使用。

## 2. 当前基线与已知阻断项

### 2.1 当前候选池

| 维度 | 候选数 | 正式目标 |
|---|---:|---:|
| attribute_binding | 517 | 300 |
| action_binding | 1013 | 300 |
| motion_binding | 676 | 300 |
| background_dynamics | 946 | 300 |
| view_transformation | 365 | 300 |
| **总计** | **3517** | **1500** |

### 2.2 无模型审计结果

当前审计已确认：

- 3517条 `target_subjects[].noun` 缺失；
- 3517条使用泛化主体描述 `the subject`；
- 3517条缺少可供下游直接消费的结构化目标变化；
- 121条prompt字数异常；
- 113条存在重复冠词；
- 99条已有 `failed_check`；
- 18条存在空槽位模式；
- 117条使用fallback；
- 206/3519条Prompt命中已知生僻词，且上游VLM caption已有13.7%命中。

因此，旧 `phase3_manifest.jsonl` **MUST NOT** 直接作为正式数据集发布。

### 2.3 已修复但尚需重跑的代码问题

仓库已修改：

- Phase 2支持读取Phase 1真实字段 `aligned_subjects`；
- Phase 2支持顶层 `attribute_change_slots`、`action_slots`、`motion_slots`、`background_change_slots`、`camera_movement_slots`；
- Phase1Bundle支持 `image_parse_v2.jsonl` 和 `text_parse_v2.jsonl`；
- View维度从“禁止运镜词”修正为“必须包含运镜词”；
- finalize新增空槽位、重复冠词和失败重试；
- export新增结构化目标和failed prompt阻断。

当前仍有三个必须在重跑前关闭的入口：旧`configs/phase2.yaml`中的Phase 1路径是机器相关的Windows绝对路径；`Phase1Bundle`在关键文件缺失时仍可能返回空字典；空slot经过模板渲染后可能退化成`camera performs ''`一类“字符串非空但无实际语义”的target relation。Coding Agent必须把“路径存在、必需文件齐全、记录数非零”设为plan命令的fail-fast前置条件，并对渲染后的结构化关系做语义退化检查，不能依赖普通非空门控兜底。

Coding Agent **MUST NOT** 只对旧question plan重跑export。优先重跑：

```text
plan → finalize → export → audit
```

若Phase 1 bundle不可用，才允许走§9的旧产物迁移修复路径。

## 3. 工程原则

### 3.1 原始产物不可覆盖

以下文件视为只读：

```text
data/benchmark_dataset/phase3_manifest.jsonl
data/benchmark_dataset/question_plans.jsonl
data/benchmark_dataset/prompts/final_prompts.jsonl
data/benchmark_dataset/samples/*.jsonl
data/benchmark_dataset/by_dimension/**
```

实验变体和修复结果 **MUST** 写入独立run目录。

### 3.2 幂等与断点续跑

每个步骤必须：

- 输入相同且配置相同时输出确定；
- 已完成记录默认跳过；
- 支持 `--resume`；
- 对API或模型调用逐条落盘；
- 单条失败不得中断整批任务；
- 失败原因使用强枚举，不写自由文本作为主状态。

### 3.3 API安全

涉及VLM/LLM调用的命令默认dry-run。只有显式传入 `--allow-api` 才能调用外部API。

必须支持：

- `--limit N`：仅处理N条；
- `--max-api-calls N`：API调用上限；
- `--estimated-cost-only`：只输出预计调用数；
- `--resume`：跳过已有有效响应；
- 记录模型名、temperature、prompt hash和原始响应。

### 3.4 依赖与模型权重

根目录现有`requirements.txt`不足以运行本需求中的统计、语义和图像实验。Coding Agent必须新增分组依赖文件并在Task 0中逐项探测：

```text
requirements-quality-core.txt      # numpy/scipy/scikit-learn、wordfreq、OpenCV/scikit-image等
requirements-quality-models.txt    # torch/transformers、语义模型、SR、DINO、Grounding适配器
requirements-quality-optional.txt  # GFPGAN、diffusers/outpainting等MAY方法
requirements-quality.lock.txt      # 在目标运行环境验证后冻结的准确版本
```

依赖文件中的实际包名和版本必须通过最小导入及2条fixture smoke test确认，不能只根据文档示例填写。核心依赖失败时将对应实验标记为`blocked_dependency`；可选依赖失败不阻断核心交付。

代码不得在运行时静默下载大模型。缺少权重时返回 `missing_model_weight` 并给出路径说明。

所有图像增强和视觉工具权重需写入`quality_models.yaml`，至少记录adapter、权重相对路径、SHA-256、库版本、device、dtype和license；run manifest记录实际加载的权重hash。模型来源与许可证不明确时不得进入正式流水线。

## 4. 推荐工程目录

```text
configs/
  quality_experiments.yaml
  quality_models.yaml

requirements-quality-core.txt
requirements-quality-models.txt
requirements-quality-optional.txt
requirements-quality.lock.txt

src/i2vcompbench/quality/
  __init__.py
  cli.py
  schemas.py
  paths.py
  hashing.py
  audit.py
  split.py
  target_repair.py
  prompt_rules.py
  prompt_experiment.py
  image_metrics.py
  image_variants.py
  clarity_experiment.py
  aspect_experiment.py
  subject_tier.py
  difficulty.py
  orthogonal.py
  ablation.py
  cost_decision.py
  human_annotation.py
  statistics.py
  final_selection.py
  report.py

scripts/
  audit_final_candidates.py       # 保留为兼容wrapper，内部调用quality.audit

tests/quality/
  fixtures/
  test_audit.py
  test_split.py
  test_target_repair.py
  test_prompt_rules.py
  test_image_paths.py
  test_subject_tier.py
  test_difficulty.py
  test_final_selection.py
  test_quality_cli_smoke.py

data/benchmark_dataset/quality_experiments/
  <run_id>/
    run_manifest.json
    logs/
    audit/
    splits/
    target_repair/
    prompt/
    clarity/
    aspect/
    subject/
    difficulty/
    orthogonal/
    ablation/
    decision/
    human/
    selection/
    report/
```

## 5. 配置文件规格

新增 `configs/quality_experiments.yaml`：

```yaml
run:
  run_id: null                    # null时生成 YYYYMMDD_HHMMSS
  seed: 20260712
  output_root: data/benchmark_dataset/quality_experiments
  overwrite: false

input:
  benchmark_root: data/benchmark_dataset
  phase1_bundle_dir: null           # MUST由本机配置或CLI显式提供；不得保留机器绝对路径
  source_asset_manifest: null       # source_sample_id与Phase 1原始资产sha256的canonical映射
  asset_lineage_manifest: null      # 原始资产→Phase 2 first_frame派生件的变换与双端hash
  allow_legacy_qid_search: false    # 旧目录仅可用于迁移候选发现，不得静默通过
  manifest: phase3_manifest.jsonl
  question_plans: question_plans.jsonl
  final_prompts: prompts/final_prompts.jsonl
  formal_dimensions:
    - attribute_binding
    - action_binding
    - motion_binding
    - background_dynamics
    - view_transformation

split:
  development_per_dimension: 50
  validation_per_dimension: 50
  stratify_by: [difficulty_old, semantic_rarity]

target_repair:
  prefer_phase1_rebuild: true
  allow_vlm_migration: true
  min_confidence: 0.80
  manual_review_below: 0.90

prompt:
  min_words: 8
  max_words: 25
  zipf_threshold: 3.5
  do_not_replace_pos: [NOUN, PROPN]
  max_llm_retries: 3
  semantic_similarity_min: 0.95
  human_naturalness_min: 4.0
  optional_perplexity_threshold: null  # 仅A5；由开发集冻结
  fallback_development_size: 60
  fallback_validation_size: 57

clarity:
  sample_size: 120
  validation_sample_size: 120
  face_subset_size: 40
  target_long_edge: 854
  output_format: PNG
  output_color_space: sRGB
  methods: [lanczos, unsharp, realesrgan, swinir]
  optional_methods: [realesrgan_gfpgan, realesrgan_unsharp, realesrgan_clahe, realesrgan_unsharp_clahe]
  video_probe_size: 30
  dino_subject_noninferiority_margin: null
  face_identity_noninferiority_margin: null

aspect:
  ratio_stage_sample_size: 60
  ratio_validation_sample_size: 60
  strategy_stage_sample_size: 120
  strategy_validation_sample_size: 120
  ratios: ["4:3", "16:9"]
  common_pixel_budget: null
  target_sizes: {"4:3": null, "16:9": null}
  size_multiple: 16
  max_pixel_budget_relative_error: 0.02
  methods: [stretch, center_crop, letterbox, blur_padding, saliency_crop, hybrid_conservative]
  optional_methods: [outpainting, hybrid_aggressive]
  subject_retention_noninferiority_margin: null
  scene_retention_noninferiority_margin: null
  hybrid:
    minor_ratio_error: 0.04
    saliency_ratio_error: 0.20
    aggressive_saliency_ratio_error: 0.30

subject_tier:
  ratios:
    T1_common: 0.55
    T2_longtail: 0.28
    T3_finegrained: 0.12
    T4_rare_fictional: 0.05

difficulty:
  task_weights:                    # 仅为G2候选初值，最终由G3验证
    target_complexity: 0.35
    subject_localization: 0.30
    background_interference: 0.20
    temporal_complexity: 0.15
  initial_thresholds: [0.33, 0.66]
  report_semantic_rarity_separately: true
  judgeability_review_threshold: 0.70
  feature_min_confidence: 0.80
  prediction_manual_review_below: 0.60
  allow_vlm_for_missing_features: true

human:
  difficulty_annotators_per_sample: 2
  method_blind_raters_per_item: 2
  prompt_human_subset_size: 100
  clarity_human_subset_size: 120
  aspect_human_subset_size: 120
  final_review_annotators_per_sample: 1
  double_review_sets: [development, validation, low_confidence, conflicts]
  require_adjudication: true

selection:
  per_dimension: 300
  difficulty_ratio: {easy: 0.40, medium: 0.35, hard: 0.25}
  rarity_ratio: {common: 0.70, rare: 0.30}
  reject_old_fallback: true
  require_image_exists: true
  require_asset_binding_verified: true
  require_schema_complete: true
  require_preservation_set: true
  require_dimension_verified: true
  require_final_human_accept: true
  draw_seed: 20260712

orthogonal:
  enabled: false                    # MAY；正式1500条交付不依赖该实验
  assignment_only_without_outcome: true
  external_outcome_path: null       # Phase 3得分或独立人工结果；缺失时不得运行推断统计
  quadrant_sizes:
    A_control: 80
    B_lexical_probe: 60
    C_subject_probe: 60
    D_compound_probe: 40
  zipf_threshold_lexical: 4.0
  subject_tier_split: ["T1_common", "T2_longtail"]  # 归入common的tier

ablation:
  enabled: true
  evaluation_set: "validation"  # 在验证集上运行
  processing_conditions: ["baseline", "text_only", "image_only", "text_image", "text_image_aspect"]
  sampling_conditions: ["natural", "quality_only", "quality_stratified"]
  sampling_repeats: 100
  statistical_tests:
    binary_paired: "mcnemar"
    continuous_paired: "wilcoxon"
    ordinal_paired: "wilcoxon"
    set_level: "bootstrap"
  alpha: 0.05
  correction: "bonferroni"

decision:
  weights:
    quality: 0.60
    time: 0.25
    complexity: 0.15
  max_production_run_hours: 8       # 仅指winner全池处理；不含消融、部署、API和人工
  experiment_budget_hours: null     # 由实际资源另行填写
  prefer_deterministic: true
```

`quality_models.yaml`对每个模型资源使用统一记录，至少覆盖`sentence_encoder`、`clip`、`gpt2`、`realesrgan`、`swinir`、`dino`、`grounding`和`face_identity`；MAY方法另列。示例结构如下，`null`资源必须由Task 0报告为未就绪，不得运行时下载：

```yaml
models:
  realesrgan:
    adapter: i2vcompbench.quality.adapters.realesrgan:RealESRGANAdapter
    weight_path: null
    sha256: null
    package: null
    package_version: null
    device: auto
    dtype: fp32
    license: null
    required_for: [clarity_core]
```

配置加载后必须执行比例求和、路径存在性、方法枚举和数值范围校验。

上述非劣界值允许依据开发集标注误差、指标重复性或先验文献在开发阶段冻结；在运行验证集前仍为`null`时，相应验证命令必须fail-fast。不得查看验证结果后反向选择“刚好通过”的margin。

`phase1_bundle_dir=null`时不得静默读取空字典：若`allow_vlm_migration=false`则直接fail-fast；若允许VLM迁移，先输出待迁移条数、API调用上限和估算成本，必须显式`--allow-api`才可继续。开始任何图像实验前，`source_asset_manifest`和`asset_lineage_manifest`必须非空且通过双端hash一致性smoke test。

## 6. 核心Schema

所有Schema使用Pydantic v2或dataclass实现。`SubjectRef`、`TargetRelation`和`PreserveItem`必须直接复用正式Phase 2 Schema或由唯一兼容层导入，禁止质量模块自行定义字段不同的同名类型。

### 6.0 SourceAssetRecord

canonical资产清单必须从Phase 1原始manifest或经人工确认的迁移记录生成，不得从当前qid目录反向猜测：

```python
class SourceAssetRecord(BaseModel):
    upstream_asset_id: str           # canonical全局主键，不使用qid
    source_sample_id: str
    asset_role: Literal["source_first_frame", "reference_asset"]
    upstream_path: str                # POSIX相对路径
    canonical_upstream_sha256: str
    width: int
    height: int
    source_manifest_path: str
    source_manifest_sha256: str
    verification_source: Literal["phase1_manifest", "manual_migration"]
    verified_by: str | None = None
    verified_at: str | None = None
```

使用`manual_migration`时必须保存候选图像、人工裁决和旧/新路径证据。

### 6.0a AssetLineageRecord

Phase 2的`first_frame`通常经过缩放、格式转换或T2I生成，派生文件hash天然不同于Phase 1原始图像hash。因此必须单独保存“上游原始资产→派生首帧”的处理谱系，禁止直接比较原图hash与缩放图hash：

```python
class AssetLineageRecord(BaseModel):
    question_id: str
    role: str
    source_type: Literal["tip_derived_reference"]
    source_sample_id: str
    upstream_asset_id: str
    canonical_upstream_sha256: str
    observed_upstream_sha256: str
    transform_name: str
    transform_params: dict[str, Any]
    transform_spec_sha256: str
    derived_path: str
    expected_derived_sha256: str
    code_version: str
    verification_source: Literal["construction_run", "deterministic_migration", "manual_migration"]
    status: Literal["verified", "needs_manual_review", "source_mismatch", "derived_mismatch"]
```

新一轮`construct_inputs`必须在读取上游字节和写出派生件时同步生成该记录。旧产物迁移只有在“canonical上游hash一致 + 使用冻结变换可重现派生hash”时才能标记为`deterministic_migration`；否则必须人工核验，不能凭视觉相似度自动升级为`verified`。

`asset_lineage_record_sha256`指该qid/role谱系记录按canonical JSON序列化后的hash；`asset_lineage_manifest_sha256`指整份排序后JSONL的文件hash。二者不得混用，run manifest同时保存两者。

正式1500条及核心图像实验只接收可关联到上述canonical记录的`tip_derived_reference`。`t2i_generated`和`external`若未来需要保留，必须另建包含提供方响应字节hash、生成参数/许可证和人工来源裁决的Schema；在本需求内一律标记`unsupported_source_type`，不得用空`source_sample_id`绕过P0门控。

### 6.1 QualityCandidate

```python
class QualityCandidate(BaseModel):
    question_id: str
    dimension: Literal[
        "attribute_binding",
        "action_binding",
        "motion_binding",
        "background_dynamics",
        "view_transformation",
    ]
    difficulty_old: Literal["easy", "medium", "hard"]
    semantic_rarity: Literal["common", "rare"]
    prompt: str
    first_frame_path: str
    source_manifest_hash: str
    source_sample_id: str | None = None
    upstream_asset_id: str | None = None
    canonical_upstream_sha256: str | None = None
    lineage_observed_upstream_sha256: str | None = None
    lineage_expected_derived_sha256: str | None = None
    resolved_derived_sha256: str | None = None
    asset_lineage_record_sha256: str | None = None
    asset_lineage_manifest_sha256: str | None = None
    resolution_source: Literal[
        "manifest_path", "asset_lineage", "legacy_candidate", "unresolved"
    ] = "unresolved"
    asset_binding_status: Literal[
        "verified", "needs_manual_review", "missing_source_hash", "missing_lineage",
        "source_hash_mismatch", "derived_hash_mismatch", "ambiguous", "missing"
    ] = "missing"
    source_plan_hash: str | None = None
    quadrant: Literal[
        "A_control", "B_lexical_probe", "C_subject_probe", "D_compound_probe"
    ] | None = None
```

### 6.2 TargetRepairResult

```python
class TargetRepairResult(BaseModel):
    question_id: str
    target_subjects: list[SubjectRef]
    target_relation: TargetRelation
    preservation_set: list[PreserveItem]
    reviewed_dimension: str
    dimension_consistent: bool
    confidence: float
    repair_source: Literal["phase1_rebuild", "vlm_migration", "manual"]
    status: Literal[
        "pass", "needs_manual_review", "rejected", "api_failed", "invalid_response"
    ]
    raw_response_path: str | None = None
```

五个维度的 `target_relation.type` 固定为：

```python
TARGET_TYPE_BY_DIMENSION = {
    "attribute_binding": "attribute",
    "action_binding": "action",
    "motion_binding": "motion",
    "background_dynamics": "background",
    "view_transformation": "view",
}
```

### 6.3 PromptVariantResult

```python
class PromptVariantResult(BaseModel):
    question_id: str
    method: Literal[
        "A0", "A1", "A2", "A3", "A4", "A5", "A6",
        "B0", "B1", "B2", "B3",
    ]
    source_layer: Literal["P1-Source-Off", "P1-Source-On"] = "P1-Source-Off"
    generator_id: str | None = None
    generator_version: str | None = None
    decoding_config_hash: str | None = None
    parent_method: str | None = None
    prompt_before: str
    prompt_after: str
    word_count: int
    rare_modifier_hits: list[str]
    structural_issues: list[str]
    semantic_similarity: float | None = None
    semantic_preservation_pass: bool | None = None
    dimension_consistent: bool | None = None
    rare_modifier_rate: float | None = None
    mean_zipf_score: float | None = None
    clip_sim_delta: float | None = None
    perplexity_mean: float | None = None
    api_calls: int = 0
    status: Literal[
        "pass", "failed", "needs_manual_review", "unsupported_optional",
        "missing_dependency"
    ]
```

### 6.4 ImageVariantResult

```python
class ImageVariantResult(BaseModel):
    question_id: str
    experiment: Literal["clarity", "aspect"]
    method: str
    source_path: str
    output_path: str
    upstream_asset_id: str
    canonical_upstream_sha256: str
    input_asset_sha256: str
    input_stage: Literal["canonical_source", "clarity_output", "aspect_output"]
    output_sha256: str
    asset_binding_verified: bool
    source_size: tuple[int, int]
    output_size: tuple[int, int]
    metrics: dict[str, float | None]
    status: Literal[
        "pass", "failed", "missing_dependency", "missing_weight",
        "unsupported_optional"
    ]
```

### 6.5 HumanAnnotation

```python
RubricScore = Literal[0.0, 0.5, 1.0]

class AnnotationBase(BaseModel):
    annotation_id: str
    question_id: str
    experiment: str
    method_code: str                 # 盲化码；真实方法保存在独立codebook
    annotator_id: str
    task_version: str
    created_at: str
    comment: str = ""

class TargetRepairAnnotation(AnnotationBase):
    experiment: Literal["target_repair"]
    subject_correct: bool
    relation_correct: bool
    preservation_correct: bool
    dimension_correct: bool

class PromptAnnotation(AnnotationBase):
    experiment: Literal["prompt", "fallback"]
    target_consistent: bool
    dimension_correct: bool
    prompt_naturalness: int                 # 1..5
    overall_usable: bool

class ImageAnnotation(AnnotationBase):
    experiment: Literal["clarity", "aspect"]
    clarity_score: int | None = None         # 1..5，aspect可为空
    subject_complete: bool
    scene_reference_preserved: bool
    geometry_distortion: int                # 1..5，越高越严重
    identity_changed: bool | None = None
    artifact_present: bool
    overall_usable: bool

class DifficultyAnnotation(AnnotationBase):
    experiment: Literal["difficulty"]
    difficulty_label: Literal["easy", "medium", "hard"]
    target_complexity: RubricScore
    subject_localization: RubricScore
    background_interference: RubricScore
    temporal_complexity: RubricScore
    initial_state_visibility: RubricScore
    target_observability: RubricScore
    alternative_explanation_risk: RubricScore
    evaluator_support_gap: RubricScore

class FinalReviewAnnotation(AnnotationBase):
    experiment: Literal["final_review"]
    asset_correct: bool
    dimension_correct: bool
    subject_correct: bool
    transform_complete: bool
    preservation_valid: bool
    target_consistent: bool
    final_accept: bool
    rejection_reasons: list[Literal[
        "asset_mismatch", "wrong_dimension", "subject_incorrect",
        "target_relation_incorrect", "preservation_invalid", "prompt_invalid",
        "image_artifact", "subject_lost", "scene_reference_lost", "other"
    ]]

HumanAnnotation = Annotated[
    TargetRepairAnnotation | PromptAnnotation | ImageAnnotation |
    DifficultyAnnotation | FinalReviewAnnotation,
    Field(discriminator="experiment"),
]

class AdjudicationRecord(BaseModel):
    adjudication_id: str
    question_id: str
    experiment: str
    source_annotation_ids: list[str]
    adjudicator_id: str
    resolved_payload: dict[str, Any]
    rationale: str
    created_at: str
```

### 6.6 SelectionDecision

```python
class SelectionDecision(BaseModel):
    question_id: str
    dimension: str
    eligible: bool
    blocking_reasons: list[str]
    prompt_method: str | None
    clarity_method: str | None
    aspect_method: str | None
    native_first_frame_path: str | None
    native_first_frame_sha256: str | None
    inference_companion_path: str | None
    inference_companion_sha256: str | None
    subject_tier: str | None
    difficulty_new: str | None
    quality_rank_components: dict[str, float | int | None]
    quality_rank_score: float | None
    selection_order_key: str
    selected: bool = False
    bucket_assignments: dict[str, str] = Field(default_factory=dict)
```

### 6.7 DifficultyFeatureRecord

```python
class DifficultyFeatureRecord(BaseModel):
    question_id: str
    target_complexity: float
    subject_localization: float
    background_interference: float
    temporal_complexity: float
    initial_state_visibility: float
    target_observability: float
    alternative_explanation_risk: float
    evaluator_support_gap: float
    d_task: float
    d_judge: float
    evidence: dict[str, list[str]]
    feature_sources: dict[str, Literal["rule", "vlm", "manual"]]
    feature_confidence: dict[str, float]
    status: Literal["pass", "needs_manual_review", "rejected", "missing_evidence"]
    predicted_label: Literal["easy", "medium", "hard"] | None = None
    predicted_probabilities: dict[str, float] | None = None
    prediction_uncertainty: float | None = None
    calibration_model_hash: str | None = None
```

## 7. CLI规格

统一入口：

```bash
python -m i2vcompbench.quality.cli \
  --config configs/quality_experiments.yaml \
  --run-id thesis_v1 \
  <command>
```

必须支持的命令：

```text
audit
build-asset-manifest
build-asset-lineage
split
repair-targets
prompt-variants
prompt-metrics
clarity-variants
clarity-metrics
aspect-variants
aspect-metrics
tag-subjects
calibrate-difficulty
orthogonal-assign
orthogonal-analyze
ablation-run
ablation-analyze
cost-estimate
decision-matrix
prepare-human
import-human
select-final
report
all-local
```

通用参数：

```text
--resume
--limit N
--only-qid QUESTION_ID
--dry-run
--allow-api
--max-api-calls N
--log-level INFO|DEBUG
```

`select-final`另需支持`--stage provisional|commit`；`commit`在缺少完整人工终审导入文件时必须fail-fast。

`all-local`只能执行不需要API和大模型权重的步骤，不得隐式发起网络调用。

## 8. Step 1：审计与划分

### 8.0 build_asset_manifest.py与build_asset_lineage.py

该步骤在候选审计之前分别建立：（1）Phase 1上游原始资产的canonical清单；（2）上游资产到Phase 2首帧派生件的变换谱系。canonical清单的输入只能是Phase 1原始manifest，或带人工裁决记录的旧数据迁移表；不得扫描当前`first_frames/{qid}.png`后将其反向登记为上游真值。

处理要求：

1. 读取每条`source_sample_id`及其原始图像声明路径，生成稳定且全局唯一的`upstream_asset_id`，并解析为POSIX相对路径；
2. 读取图像实际字节并计算SHA-256、宽度和高度；
3. `upstream_asset_id`重复、同一`source_sample_id+asset_role`出现不同图像hash时直接fail-fast并输出冲突记录；
4. 路径缺失、图像损坏或来源ID缺失时进入人工迁移队列，不得自动使用qid同名文件补齐；
5. 新构建流程在读取原始字节后记录`observed_upstream_sha256`，在完成缩放/格式转换后记录`expected_derived_sha256`和完整变换参数；
6. 旧产物优先使用`input_assets_manifest.source_ref_id`关联上游资产，再以冻结代码和参数重放派生过程；重放结果与现有首帧hash一致时才允许自动迁移；
7. 将两份清单自身的hash写入run manifest，后续audit、split、图像实验和正式导出共同引用该版本。

输出：

```text
audit/source_asset_manifest.jsonl
audit/source_asset_manifest_summary.json
audit/source_asset_conflicts.jsonl
audit/source_asset_migration_queue.jsonl
audit/asset_lineage_manifest.jsonl
audit/asset_lineage_summary.json
audit/asset_lineage_conflicts.jsonl
audit/asset_lineage_migration_queue.jsonl
```

### 8.1 audit.py

输入：旧manifest、question plans、final prompts、canonical源资产清单、派生谱系和图像目录。

必须检查：

- 是否属于五个正式维度；
- question_id是否唯一；
- prompt是否为空；
- 是否存在重复冠词、空槽位、占位符、长度异常；
- final prompt是否带 `failed_check`；
- target_subjects是否为空或泛化；
- target_relation是否为空；
- target_relation去除模板套话、引号和标点后是否仍有实际变化语义；
- preservation_set是否为空；
- 图像路径是否存在；
- Windows反斜杠路径是否可转换；
- `source_sample_id`是否存在；
- `upstream_asset_id`是否唯一并能在canonical清单中解析；
- 是否能从canonical资产清单获得上游原图SHA-256；
- 谱系中的上游hash是否与canonical上游hash一致；
- 当前首帧实际hash是否与谱系中的预期派生hash一致；
- 是否出现同一qid对应多个内容不同的候选图像；
- 同一源样本是否产生重复候选。

图像解析必须区分“候选发现”和“资产验证”：

1. 使用`source_sample_id`查询`source_asset_manifest`，验证谱系中的`observed_upstream_sha256`等于canonical原图hash；
2. 优先读取manifest声明的首帧路径，计算派生件实际SHA-256，并与谱系中的`expected_derived_sha256`比较；
3. 禁止将缩放后的首帧hash直接与Phase 1原图hash比较；两者只通过同一条谱系记录建立联系；
4. `first_frames/{qid}.png`、`by_dimension/...`和旧版兼容目录只允许用于**迁移候选发现**；
5. 备用候选只有在与`expected_derived_sha256`一致且匹配唯一时，才可标记`asset_binding_status=verified`；
6. 缺少上游hash、缺少谱系、任一端hash不一致、多个不同内容候选或只能依靠qid猜测时，必须进入`needs_manual_review`或reject，不得以“文件存在”静默通过。

图像清晰度、比例、Prompt图文一致性和正式筛选均以`asset_binding_status=verified`为共同前提。旧数据没有canonical资产映射或派生谱系时，应先输出迁移队列，不得直接开始图像实验。

输出：

```text
audit/candidate_quality_rows.jsonl
audit/candidate_quality_summary.json
audit/blocking_examples.jsonl
audit/asset_binding_rows.jsonl
audit/asset_manual_review_queue.jsonl
```

### 8.2 split.py

开发集和验证集各按维度50条，按 `(difficulty_old, semantic_rarity)` 比例分层。

要求：

- 两个集合不得重叠；
- 每维数量必须精确；
- 同一 `source_sample_id` **MUST NOT** 跨集合；
- 输出包含原始完整行，不只输出qid；
- 输出split hash。

## 9. Step 2：结构化目标修复

### 9.1 首选路径：Phase 1重建

若Phase 1 bundle可用：

1. 验证 `image_parse_v2.jsonl`、`text_parse_v2.jsonl`、`aligned_instances.jsonl`；
2. 运行修正后的build_question_plan；
3. 检查目标主体、noun、target relation和preservation set；
4. 将新plan与旧question_id建立稳定映射；
5. 输出repair overlay，不覆盖旧plan。

若重建后关键字段仍为空，则进入VLM迁移路径。

### 9.2 VLM迁移路径

VLM输入：首帧、final prompt、旧dimension、旧target plan和VLM caption。

只有`asset_binding_status=verified`，或已完成人工资产裁决的记录才允许进入VLM迁移。资产身份不确定时先进入§8迁移队列；不得让VLM在可能绑定错误的首帧上生成“高置信度”结构真值。

VLM必须返回严格JSON：

```json
{
  "target_subjects": [
    {"id": "s1", "noun": "elephant", "description": "cartoon elephant"}
  ],
  "target_relation": {
    "type": "action",
    "value": "lifts its trunk upward",
    "subj": "s1",
    "obj": null
  },
  "preservation_set": [
    {"scope": "s1", "constraint": "identity"},
    {"scope": "background", "constraint": "stable"}
  ],
  "reviewed_dimension": "action_binding",
  "dimension_consistent": false,
  "confidence": 0.94,
  "rationale": "The prompt describes a body action rather than an attribute change."
}
```

硬规则：

- 不允许只根据final prompt猜主体，必须同时观察图像；
- noun不得为 `subject`、`object`、`thing`；
- target relation value不得为空；
- `camera performs ''`、`changes to ''`等“字符串非空但变化槽位为空”的退化关系视为无效；
- preservation set必须沿用正式`PreserveItem{scope,constraint}` Schema，不得另造`type/subject_id/value`字段；不得使用未解析占位，`scope=s1/s2/...`时对应主体必须存在，通用scope只能来自预注册枚举（至少兼容现有`primary_subject/scene/camera/background`）；
- reviewed_dimension不一致时不得自动改写原dimension，进入人工队列；
- confidence < 0.80直接reject；
- 0.80 ≤ confidence < 0.90进入人工复核；
- API原始响应必须保存。

输出：

```text
target_repair/repairs.jsonl
target_repair/manual_review_queue.jsonl
target_repair/rejected.jsonl
target_repair/raw_responses/{qid}.txt
```

### 9.3 验收

- repair结果行数与目标输入数一致；
- `status=pass`的Schema完整率100%；
- 开发集人工主体准确率≥95%；
- 开发集目标变化准确率≥95%；
- 开发集preservation set有效率≥95%；
- 维度冲突样本不得自动通过。

## 10. Step 3：Prompt实验

### 10.1 prompt_rules.py

实现无模型规则：

- 空prompt；
- 未解析 `{slot}`；
- `the the`、`a a`、`an an`；
- 冠词后直接标点；
- 字数范围；
- 维度显式禁词；
- View必须包含运镜线索；
- Prompt必须包含可识别变化谓词；
- 低频词检测时排除NOUN和PROPN。

规则函数必须是纯函数，并有单元测试。

### 10.2 A0—A4核心消融与A5/A6可选对比

设计原则：A0—A4中每层严格包含上一层，可通过逐层对比量化策略增量；A5是可选的困惑度过滤扩展；A6是对A3的本地受控解码分支，不伪装成在线API链路中的相邻步骤。

- **A0**：原prompt（基线，不做任何修改）；
- **A1**：仅修改Polish指令模板（在`prompt_polish.txt`中增加词汇约束规则#9–#11，属于软约束）；
- **A2**：A1 + 确定性同义词映射表替换（`RARE_TO_COMMON`映射，如`ethereal→glowing`）；
- **A3**：A2 + Zipf词频硬门控（wordfreq Zipf < 3.5 触发retry）；
- **A4**：A3 + CLIP-Sim反向验证（polish前后CLIP-Sim下降>0.02则回退）；
- **A5（MAY）**：A4 + GPT-2 Perplexity过滤（开发集冻结阈值后触发retry）；
- **A6（MAY）**：在与A3相同的本地生成器上增加RareWordBlocker解码层硬拒绝（`LogitsProcessor`级别禁止rare token生成）。

#### 源头预洗消融因子

- **P1-Source-Off**：不对Phase 1 VLM输出做预洗（当前状态）；
- **P1-Source-On**：在`_resolve_slots`前插入`_prewash_slot_value`，对VLM返回的修饰词做前置清洗。

核心链形成 **5（A0—A4）×2（源头层）=10个实验条件**；资源允许时增加A5与A6各自的Source-Off/On两臂，最多14个条件。

这些条件只在开发集250条上进行筛选；即使执行全部14臂也不得对3517条候选全量生成约4.9万个变体。开发集冻结方法和阈值后，验证集只运行A0与开发集前2名候选；正式候选池只运行最终winner，并且仅对规则预检标记为异常的Prompt调用LLM。

A0—A5的相邻增量只在底层生成器、解码参数和输入语义完全一致时成立。A6的`LogitsProcessor`仅在本地HuggingFace生成器可控token logits时实现；若Polish使用不暴露logits的在线API，A6必须标记为`unsupported_optional`，不得用事后删词伪装解码层阻断。若A6必须更换生成器，则只能报告为“生成器+约束解码”的方法级对比，不得把A6–A3差异全部归因于RareWordBlocker。

每条变体必须记录`generator_id`、模型/服务版本、temperature、seed（若服务支持）、完整解码配置hash及`parent_method`。A6必须同时运行同一生成器、同一配置但不启用blocker的A3-local控制臂；没有该控制臂时只报告可用率，不计算A6的独立增量。

通过消融可回答：

1. 仅改指令有多大效果？（A1 vs A0）
2. 同义词映射的增量是多少？（A2 vs A1）
3. Zipf硬门控的增量是多少？（A3 vs A2）
4. CLIP-Sim后验检查能减少多少语义漂移？（A4 vs A3）
5. Perplexity过滤的额外贡献？（A5 vs A4）
6. 在生成器和输入完全相同时，解码层硬拒绝在Zipf门控之上是否还有增量？（A6 vs A3）
7. 源头预洗的独立贡献是多少？（Source-On vs Source-Off在各A层的差异）

#### A系列禁止修改项

A2的确定性映射表禁止：

- 修改主体noun；
- 修改动作核心动词；
- 修改方向词；
- 修改相机指令；
- 删除否定词。

A3/A4/A5/A6的生成输入必须包含不可变字段：目标主体、目标关系、目标维度和preservation set。

### 10.3 B0—B3

先按dimension、失败类型和`source_sample_id`分组分层，将117条冻结为60条fallback-development和57条fallback-validation；同一source不得跨组。开发组用于完善模板/重写指令和门槛，验证组只运行冻结条件。winner确定后再用于修复全部117条，但最终全量修复率不得冒充独立验证结果。

两组均比较：

- B0：旧prompt；
- B1：确定性模板；
- B2：LLM重写；
- B3：B2 + 自动检查 + 人工复核队列。

若关键槽位为空，B1必须返回 `missing_required_slot`，不得生成“The subject ...”占位句。

B0—B3必须分别报告开发、验证和winner全量修复三个口径的结构门控通过率、目标语义保持率、人工可用率、平均API调用次数和无法恢复率。选择顺序为：先要求验证集结构有效率和语义保持率均达到95%，再从人工可用率不显著更差的方法中选择成本更低者。任何方法都不允许从空槽位猜测评测真值。

### 10.4 输出

```text
prompt/variants.jsonl
prompt/metrics_by_method.json
prompt/fallback_comparison.json
prompt/fallback_split.json
prompt/manual_annotation_tasks.jsonl
prompt/failure_examples.jsonl
prompt/source_layer_comparison.json
```

每条`variants.jsonl`必须包含以下额外指标字段：

| 指标 | 含义 | 目标 |
|------|------|------|
| `rare_modifier_rate` | 低频修饰词比例 | <1% |
| `mean_zipf_score` | prompt中非名词词汇的平均Zipf分数 | >4.5 |
| `clip_sim_delta` | polish前后CLIP-Sim变化 | ≥-0.02 |
| `perplexity_mean` | GPT-2平均困惑度（A5实际执行时） | 阈值由开发集冻结 |
| `semantic_similarity` | 单条polish前后sentence-transformers cosine | ≥0.95 |
| `semantic_preservation_pass` | 单条是否同时通过相似度与结构化目标后检 | true |

开发集另按五维、异常类型和长度分层抽取100个qid，对实际执行的核心方法做匿名配对自然度与语义保持标注，每个item至少2名标注者。该人工子集用于校验自动指标，不得用同一批人工结果同时调阈值并报告为独立验证。

### 10.5 胜出规则

先应用硬门槛：

- 方法级目标语义保持率（单条`semantic_preservation_pass=true`的比例）≥95%；
- 维度纯度≥95%；
- 无效prompt率为0；
- `clip_sim_delta` ≥ -0.02（不得显著偏离原始语义）；
- 人工自然度和规则流畅性检查达到预注册门槛；若执行A5，再附加开发集冻结的`perplexity_mean`门槛。

满足硬门槛后，按以下优先级排序选择正式方法：

1. `rare_modifier_rate`最低（主目标）；
2. `mean_zipf_score`最高；
3. 方法级目标语义保持率最高；
4. API成本最低。

若多个方法均满足硬门槛且质量差异在95% CI内不显著，优先选择Pareto前沿上处理时间更短、确定性更高的方法（参见§16.4）。

## 11. Step 4：图像清晰度实验

### 11.1 样本选择

从开发集选择120张，分层因素：

- 五个维度；
- 人物/非人物；
- 原始长边区间；
- Laplacian模糊程度；
- 横图/方图/竖图。

人脸子集40张。

所有C条件必须读取谱系已验证的同一canonical上游原图字节，并输出到相同目标长边、色彩空间和文件格式。SR模型若原生输出2×/4×结果，必须再用统一确定性算子归一化到配置目标尺寸，并记录中间尺寸；不得让像素数差异冒充清晰度提升。旧版已经Lanczos放大的`first_frame`只作为C0历史基线的核验对象，不得作为C2/C3的输入；否则会把预插值损伤和双重处理混入方法效应。若只能取得旧首帧而无法取得上游原图，该qid不得进入核心清晰度方法对比。

所有C0—C7只在开发集120条子集上筛选；验证集另取120条，只运行C0与冻结的前2名候选。C4—C7和视频探针均为MAY，缺少权重、GPU或视频生成预算时不阻断C0—C3核心对比及正式集交付。

### 11.2 处理方法

- C0：Lanczos；
- C1：Lanczos + 轻度Unsharp；
- C2：Real-ESRGAN；
- C3：SwinIR；
- C4：Real-ESRGAN + GFPGAN，仅可选消融；
- C5：Real-ESRGAN + Unsharp Masking（sigma=1.5, amount=0.5）；
- C6：Real-ESRGAN + CLAHE（clipLimit=2.0, tileGridSize=8×8）；
- C7：Real-ESRGAN + Unsharp + CLAHE（完整组合流水线，对应分析报告§2.5.6推荐方案）。

每个输出路径：

```text
clarity/images/{method}/{question_id}.png
```

不得覆盖原图。保存方法参数和源/输出hash。

### 11.3 指标

实现：

- Laplacian variance；
- Tenengrad；
- NIQE；
- BRISQUE；
- DINOv2全图与主体区域相似度；
- 人脸身份相似度（仅人脸子集）；
- Grounding目标检测成功率；
- `enhancement_chain_contribution`：每个后处理环节的增量贡献（用NIQE差值量化，仅适用于C5/C6/C7组合流水线）。

缺少bbox时，主体区域指标记为null，不得使用整图指标假装主体指标。

### 11.4 视频探针边界

本模块只导出：

```text
clarity/video_probe_manifest.jsonl
```

包含30条×入选方法的首帧和prompt。视频可由现有共享adapter生成；质量包只导入人工清晰度、身份和伪影标注，不实现Phase 3评分。

### 11.5 胜出规则

身份改变率和DINO主体相似度必须使用预先冻结的非劣界值与95% CI判定；不得将“差异不p显著”等同于“已证明安全”。满足身份、主体和语义保真非劣门槛后，再选择人工清晰度和主体检测成功率更高的方法。人脸子集40张只支持初步安全性筛选，对低发生率身份篡改必须报告区间不确定性。

若多个方法均满足硬门槛且质量差异在95% CI内不显著，优先选择Pareto前沿上处理时间更短、确定性更高的方法（参见§16.4）。

### 11.6 组合消融设计

目标：量化C5/C6/C7中锐化和对比度增强各自的增量贡献，以及组合是否存在边际递减。

消融路径：

```text
路径A：C2 (SR-only) → C5 (SR+Sharpen) → C7 (SR+Sharpen+CLAHE)
路径B：C2 (SR-only) → C6 (SR+CLAHE) → C7 (SR+Sharpen+CLAHE)
```

对每个环节报告：

| 对比 | 量化指标 | 含义 |
|------|---------|------|
| C5 - C2 | ΔNIQE, ΔLaplacian | Unsharp Masking的独立增量 |
| C6 - C2 | ΔNIQE, ΔBRISQUE | CLAHE的独立增量 |
| C7 - C5 | ΔNIQE, ΔBRISQUE | 在已有USM基础上叠加CLAHE的边际收益 |
| C7 - C6 | ΔNIQE, ΔLaplacian | 在已有CLAHE基础上叠加USM的边际收益 |

上述差值必须先转换为“数值越大表示越好”的同方向效应，并使用同图配对bootstrap 95% CI。只有`C7-C5`相对`C6-C2`的增量差异区间支持下降时，才能写为边际递减；仅凭两个点估计大小只能标记为描述性趋势。

输出：

```text
clarity/ablation_chain_metrics.json
clarity/ablation_chain_plot.png
```

## 12. Step 5：尺寸与比例实验

### 12.1 比例阶段

从六类宽高比区间各取10张，共60张。使用同一保守方法输出4:3和16:9版本。

开发集60张用于比例pilot与规则冻结；验证集另取60张、保持相同宽高比分层，只确认冻结比例的输入保持指标，不再更换像素预算、目标尺寸或选择规则。

两种比例必须使用相同总像素预算、相同重采样算子和相同清晰度winner，避免把输出分辨率差异误归因于宽高比。Task 0应依据目标I2V adapter支持的尺寸倍数填写`target_sizes`；两种尺寸的像素数相对差异不得超过2%，不能用“相同长边”冒充相同计算预算。目标尺寸或adapter约束仍为`null`时，比例实验必须标记为blocked。必须报告：主体完整率、场景参照保留率、裁剪/填充面积、几何形变、边界伪影、CLIP图文变化和盲法人工可用性。

比例选择规则必须在查看验证集前冻结：先要求主体和场景参照保留达到预注册非劣界值，再选择综合裁剪/填充面积、形变和人工伪影更小的比例。若没有使用同一I2V adapter的视频探针，结论只能表述为“输入保持性更好”，不得表述为“生成视频质量更好”。

输出：

```text
aspect/ratio/{ratio}/{question_id}.png
aspect/ratio_metrics.json
aspect/ratio_probe_manifest.jsonl
```

### 12.2 策略阶段

在冻结比例下比较：

- D0 stretch；
- D1 center_crop；
- D2 letterbox；
- D3 blur_padding或reflection_padding；
- D4 saliency_crop；
- D5 outpainting，仅可选上界。

D0—D4和D6为核心适配实验；D5、D7和完整20组阈值网格为MAY，不阻断正式1500条交付。开发集用于筛选方法和路由阈值，验证集120张只运行最佳单一方法、D6以及可选D7，不再据验证结果调整阈值。

### 12.3 混合策略整体验证

新增实验臂：

- **D6_hybrid_conservative**：stretch(≤4%) + saliency_crop(4–20%) + blur_padding(>20%)；
- **D7_hybrid_aggressive**：stretch(≤4%) + saliency_crop(4–30%) + outpainting(>30%)。

对比设计：

- 核心比较为D6 vs 最佳核心单一方法（D0–D4）；
- 若实际执行D5/D7，再比较D7 vs D6及D5，不把未运行的可选方法纳入winner搜索；
- 核心条件使用验证集120张全量运行，可选条件按资源状态运行；按宽高比分桶报告（近方图/中等比/极端比三桶）；
- 每桶至少30张有效样本。

输出：

```text
aspect/hybrid_comparison.json
aspect/hybrid_per_bucket.csv
```

### 12.4 阈值敏感性分析

本节为MAY。若资源不足，核心D6实验使用预注册的0.04/0.20候选阈值，并在限制中声明未完成全网格敏感性验证；不得将未执行网格搜索的阈值写成已证明最优。

对混合路由器的两个阈值进行网格搜索：

- `stretch_threshold`：[0.02, 0.04, 0.06, 0.08]
- `saliency_threshold`：[0.15, 0.20, 0.25, 0.30, 0.35]

每组合在60张ratio阶段样本上运行，报告：

| 指标 | 说明 |
|------|------|
| 主体保留率 | GroundingDINO recall（IoU≥0.5） |
| CLIP-Sim(image, prompt) | 图文对齐 |
| NIQE分数 | 无参考图像质量 |
| 各档位实际触发比例 | stretch/saliency_crop/padding各路由实际被触发的比例 |

输出：

```text
aspect/threshold_sensitivity_heatmap.json
aspect/threshold_sensitivity_plot.png
```

最终阈值选择标准：在主体保留率≥95%前提下，选CLIP-Sim最高的组合。

### 12.5 Saliency Crop要求

必须：

- 使用目标主体bbox或显著性区域；
- 裁剪框包含完整主体bbox及安全边距；
- 无bbox时不得假装主体感知，回退到padding并记录原因；
- 输出裁剪前后主体保留率。

### 12.6 Outpainting限制

Outpainting输出不得进入正式主流程，除非人工验证和语义保持实验明确通过。默认仅作对比上界。

### 12.7 正式混合策略

Coding Agent实现可配置路由器：

```python
if ratio_error <= config.aspect.hybrid.minor_ratio_error:
    method = "minor_resize_or_crop"
elif ratio_error <= config.aspect.hybrid.saliency_ratio_error and subject_bbox_available:
    method = "saliency_crop"
else:
    method = "blur_or_reflection_padding"
```

阈值必须来自配置，不得硬编码在多处。

正式尺寸方法的胜出顺序为：（1）主体完整率≥95%；（2）身份、初始状态和场景参照相对原图满足预注册非劣界值；（3）无黑边或明显边界伪影；（4）在前三项通过后，选择人工可用性更高、几何形变和处理成本更低的方法。单一NIQE或CLIP指标不得直接决定winner。

资产采用双轨导出：Phase 1 canonical原图和旧Phase 2产物始终只读；正式manifest的`first_frame_path`指向冻结清晰度方法产生的native-ratio首帧，比例/尺寸winner另生成`inference_companion_path`供I2V adapter读取。由于现有Phase 3 manifest字段固定，伴生件路径与hash写入`selection/inference_assets.jsonl`侧车，不得用同名覆盖native首帧。Phase 3运行时必须显式选择侧车路径，评测与人工终审仍可回看两条资产链。

## 13. Step 6：主体Tier

### 13.1 标注信号

按优先级组合：

1. TIP-I2V内部主体频率；
2. COCO/LVIS本地类别表；
3. 细粒度类别资源；
4. WordNet hypernym；
5. wordfreq兜底。

资源文件放在：

```text
resources/quality/
  coco80.json
  lvis1203.json
  subtlex_top_nouns.json
  finegrained_categories.json
```

不允许运行时从网络静默下载。

### 13.2 多词主体

必须同时保存：

```json
{
  "surface_form": "skeleton king",
  "head_noun": "king",
  "canonical_form": "skeleton king",
  "tier": "T4_rare_fictional",
  "evidence": ["not_in_lvis", "zipf_below_3"]
}
```

不得仅用head noun将“skeleton king”归为常见人物。

### 13.3 输出与验收

```text
subject/subject_tiers.jsonl
subject/distribution_natural.json
subject/distribution_stratified_preview.json
subject/unknown_subjects.jsonl
```

随机抽查每Tier至少30条，报告人工准确率。

### 13.4 自然抽样与Tier边际抽样对比

主体Tier的作用不是将低质量样本提升为高质量，而是在已通过质量门控的候选中改善集合覆盖。整体集合实验比较三种策略，但“Tier是否改善覆盖”的直接对比只使用后两种相同质量池条件：

- Natural：从只通过P0结构与资产正确性、尚未应用Prompt/图像质量门控的池中按维度随机抽样，仅作治理前参照，不得进入正式集；
- Quality-only：从通过全部质量硬门控的池中按预注册种子随机选择每维300条，不优化Tier；
- Quality+Tier：质量规则不变，在合格集内尽量贴近Tier边际目标。

每种方法使用100个预注册随机种子重复抽样，比较质量通过率/质量分布、Tier总变差偏差、唯一主体数、唯一来源数、单一主体最大占比、重复率和View shortfall的分布与95% CI。Natural vs Quality-only估计质量治理带来的集合变化；Quality-only vs Quality+Tier估计分层约束的增量。该实验不使用同qid配对质量检验，不把Tier分布改善解释为I2V模型性能提升。

输出：

```text
subject/sampling_runs.jsonl
subject/sampling_distribution_comparison.json
subject/sampling_bootstrap_intervals.json
```

## 14. Step 7：难度重标

### 14.1 特征

任务难度、判断难度和语义罕见度必须分开保存。每项归一化到 `[0,1]`。

`D_task`只包含任务本身要求模型完成的组合挑战：

- target_complexity；
- subject_localization；
- background_interference；
- temporal_complexity。

四个特征必须使用带锚点的三级量表`{0.0, 0.5, 1.0}`，并在人工界面展示定义：`target_complexity`按单一局部变化—多属性/多阶段变化递增；`subject_localization`按主体显著且唯一—小目标/遮挡/多相似主体递增；`background_interference`按背景静态且无相似干扰—动态或存在强竞争线索递增；`temporal_complexity`按单步终态—连续轨迹、方向或多阶段时序递增。不得用模型输出失败率反向填写这些输入特征。

G2候选任务难度分：

```python
D_task = (
    0.35 * target_complexity
    + 0.30 * subject_localization
    + 0.20 * background_interference
    + 0.15 * temporal_complexity
)
```

权重从配置读取，只是待验证的G2初值，不是已确定的难度规律。

`D_judge`单独描述首帧状态可见性、目标变化可观察性、替代解释风险和评测工具支持缺口。四项同样使用带锚点的三级量表，先归一化到`[0,1]`，其中前两项取反后与后两项求均值；量表锚点、缺失值策略和各项原始分必须保存，不能只保留总分。`D_judge`高于配置阈值时进入人工复核或reject，不得因此把样本标为hard。`semantic_rarity`和`subject_tier`作为独立分析字段报告，默认不进入`D_task`；若论文需要检验罕见度是否改善人工难度拟合，应作为单独消融而不是静默加入正式公式。

#### 全池特征生产与标签推断

Coding Agent不得把全池3517条的8项特征默认为人工已提供。特征生产采用“确定性证据优先、缺失项定向补充”的闭环：

1. 从修复后的target relation、变化原子/阶段数和涉及主体数计算`target_complexity`；
2. 从Phase 1 bbox/遮挡信号、主体相对面积及同类干扰主体数计算`subject_localization`，不得用图像模糊度代替；
3. 从background slot、显著干扰物和前景/背景竞争线索计算`background_interference`；
4. 从动作阶段、轨迹/方向约束、持续性及运镜要求计算`temporal_complexity`；
5. 从首帧解析证据、target relation可观察性、替代解释规则和已注册评测工具覆盖计算四项`D_judge`特征。

每个规则分都必须保存原始证据、规则版本和置信度。证据缺失或置信度低于配置阈值时，先标记`missing_evidence/needs_manual_review`；只有显式允许API时才对缺失项调用VLM，并保存响应，不能让VLM覆盖已有高置信度结构证据。

开发集和验证集的人工作业同时提供：（1）easy/medium/hard金标签；（2）8项锚点分，用于审计自动特征是否可信。G3训练时只以**自动提取的4项`D_task`特征**（及可选维度指示变量）为输入、人工任务难度标签为监督目标；4项`D_judge`只用于门控和误差分析，不得进入正式难度预测。不能用人工特征训练后再假装模型能在全池自动获得同类输入。冻结G3后对全部eligible候选推断，保存三类概率、模型hash和不确定性；最大类别概率低于配置阈值的样本进入人工复核。由此，全池不需要默认人工标注8项特征，但每条必须有可追溯自动结果、人工覆盖结果或明确失败状态。

### 14.2 禁止混入质量缺陷

以下因素不得增加difficulty：

- 图像模糊；
- prompt无效；
- 空主体；
- 图文错配；
- 结构字段缺失。

这些问题必须在质量门控中修复或剔除。

### 14.3 人工标定

开发集250条生成匿名人工任务难度标注任务。至少两名标注者独立给出easy/medium/hard，并明确要求忽略图像模糊、文本缺陷和“难以判断”等质量因素。G3在开发集上拟合并通过交叉验证选择超参数；验证集250条也至少由两名标注者产生独立标签，只用于一次冻结方案确认，不能在看到验证结果后重新调权重或阈值。

比较：

- G0旧标签；
- G1单目标复杂度；
- G2四因素固定权重；
- G3人工标定阈值或有序模型。

G3使用跨五维共享参数的有序模型；可以加入维度指示变量，但不得在每维仅50条开发样本上各自自由拟合全部权重。胜出规则按顺序为：

1. 与人工标签的quadratic weighted kappa（QWK）最高；
2. ordinal MAE最低；
3. macro-F1最高；
4. 在95% CI内无显著差异时选择参数更少、解释更直接的方法。

开发集必须报告交叉验证结果，验证集必须独立报告QWK、ordinal MAE、macro-F1、混淆矩阵和标注者一致性。模型在不同难度组上的生成表现只能作为未来外部效度证据，不得反向用于定义难度。

输出：

```text
difficulty/features.jsonl
difficulty/feature_validation.json
difficulty/human_tasks.jsonl
difficulty/calibration.json
difficulty/full_pool_predictions.jsonl
difficulty/low_confidence_review_queue.jsonl
difficulty/labels_new.jsonl
difficulty/confusion_matrix.csv
```

## 14bis. Step 7b：词汇×主体正交诊断实验

### 14bis.1 目的

本节为MAY诊断。Phase 2首先检查“生僻词”和“生僻主体”的天然四象限供给是否足以支持分组报告。只有显式导入Phase 3视频得分或独立人工结果outcome后，才可进一步检验两类长尾因素与失败结果的关联和交互。拟诊断的两类失败模式为：

- 文本编码器对low-frequency modifier的鲁棒性缺失；
- 视觉概念层对长尾主体的识别能力不足。

### 14bis.2 四象限定义

| 象限 | 提示词 | 主体 | 测量能力 |
|------|--------|------|---------|
| **A (控制组)** | Common（所有修饰词Zipf≥4.0） | T1/T2 (common) | 纯组合能力基线 |
| **B (词汇探针)** | Rare（含≥1个Zipf<4.0修饰词） | T1/T2 (common) | 文本编码器对rare token的鲁棒性 |
| **C (主体探针)** | Common（所有修饰词Zipf≥4.0） | T3/T4 (rare) | 视觉概念层对长尾主体的识别能力 |
| **D (复合探针)** | Rare | T3/T4 (rare) | 交互效应（仅诊断用，不入总分） |

判定规则：

- **Common prompt**：prompt中所有非名词/专名词汇的Zipf分数均≥4.0；
- **Rare prompt**：prompt中存在至少1个非名词/专名词汇Zipf<4.0；
- **Common主体**：subject tier为T1_common或T2_longtail；
- **Rare主体**：subject tier为T3_finegrained或T4_rare_fictional。

### 14bis.3 样本分配

从3517条候选中按以下规则分配：

| 象限 | 期望数量 | 抽样策略 |
|------|---------|---------|
| A控制组 | 80条 | 从quality合格池中随机抽样，确保prompt全为common词 |
| B词汇探针 | 60条 | 仅抽取天然rare修饰词+T1/T2主体样本，不人工注词 |
| C主体探针 | 60条 | 仅抽取天然common prompt+T3/T4主体样本，不为进象限改写prompt |
| D复合探针 | 40条 | 仅抽取天然rare prompt+T3/T4主体样本 |

上述数量是期望供给，不是正式1500条的验收门槛。任一象限不足时输出`orthogonal_shortfall.json`和实际分布，不得注入rare词、替换主体或降低质量门槛。若人工改写样本被另行保留，必须标记为独立synthetic probe，不得进入天然四象限统计或正式1500条。

### 14bis.4 与P1/P4的关系

- 未导入外部outcome时，只比较A/B/C/D的供给数、质量通过率和维度分布，不计算“能力提升”。
- 导入外部outcome后，可检查Prompt治理在B象限的效果是否与A象限不同，以及T3/T4主体的结果差异是否与词汇因素存在交互。这些结果只能解释为当前候选分布中的关联，不自动具有因果含义。

### 14bis.5 统计分析

运行推断统计前，`orthogonal-analyze`必须检查`external_outcome_path`。缺失时输出`not_run_missing_outcome`，不得使用Prompt词频或subject tier本身同时作为自变量和结果变量。

根据outcome类型选择统计模型：连续结果可使用包含词汇、主体和交互项的线性模型，且只有满足残差假设时才报告2×2 ANOVA；二值成败结果使用logistic回归，有序结果使用ordinal regression。统一检查以下对比：

- **主效应1（词汇）**：B-A 和 D-C 的平均差异；
- **主效应2（主体）**：C-A 和 D-B 的平均差异；
- **交互效应**：(D-C) - (B-A) 是否显著≠0。

报告要求：

- 与结果类型一致的effect size（连续结果可用Cohen's d或partial eta squared，二值结果用odds ratio）和95% CI；
- 若交互效应显著（p<0.05），说明两类因素不可独立分离，在论文中需讨论混淆路径；
- 若交互效应不显著，只能说当前样本和统计功效下未检出交互，不得据此证明两类因素相互独立；必须同时报告CI、各象限实际数量和功效限制。

### 14bis.6 输出

```text
orthogonal/quadrant_assignments.jsonl
orthogonal/anova_results.json
orthogonal/effect_sizes.json
orthogonal/interaction_plot.png
orthogonal/orthogonal_shortfall.json
```

### 14bis.7 Schema扩展

在QualityCandidate中增加：

```python
quadrant: Literal["A_control", "B_lexical_probe", "C_subject_probe", "D_compound_probe"] | None = None
```

### 14bis.8 与正式1500条的关系

正交诊断默认为额外或只读分析视图，不是正式1500条的硬配额。天然样本可在不改变Prompt和主体的情况下标注quadrant字段。规则：

- A象限样本同时计入正式样本配额；
- D象限样本**不入总分计算**，仅在论文中作为诊断证据；
- 若作为子集，不得因诊断需要降低正式样本的质量门槛。

## 15. Step 8：人工标注工具

### 15.1 导出

`prepare-human`按实验类型输出JSONL和CSV，字段包含相对图像路径、prompt、任意数量的方法匿名编码、对应量表锚点和标注问题。

方法名必须在每个实验/批次内随机匿名为`M001...MNNN`，不得假设只有3个实验臂。真实方法—盲化码映射单独写入`human/private_method_codebook.json`，不出现在标注页面；同一比较中的展示顺序还需按标注者随机化并记录。

```text
human/tasks/{experiment}/{batch_id}.jsonl
human/tasks/{experiment}/{batch_id}.csv
human/private_method_codebook.json
human/annotation_workload_plan.json
```

### 15.2 导入

`import-human`必须检查：

- annotation_id唯一；
- 标注者ID非空；
- `experiment`能路由到§6.5对应的强类型Schema，量表范围与必填字段合法；
- 每个实验样本是否满足最少标注人数；
- 是否存在缺失方法组。
- `final_accept=false`时`rejection_reasons`非空，选择`other`时comment非空；
- 仲裁记录引用的原始annotation_id全部存在且确有分歧，`resolved_payload`必须再次通过对应experiment的字段Schema校验。

### 15.3 一致性

二值任务实现Cohen's/Fleiss' Kappa；难度及1—5级量表实现加权Kappa，并统一报告bootstrap 95% CI。若SciPy等依赖缺失，应给出清晰错误，不得静默跳过统计。

### 15.4 正式1500条全量语义终审

自动门控通过后，对拟入选的1500条执行至少一次全量人工终审。开发集、验证集、低置信度样本和自动/人工冲突样本至少由两名标注者独立复核，分歧必须仲裁。

终审页面必须同时展示：Phase 1可信原图、Phase 2原始首帧、推理伴生图、Prompt、dimension、target subjects、target relation、preservation set、source sample ID、canonical上游hash、谱系上游hash、预期/实际派生hash和所有自动门控结果。标注者必须明确给出`final_accept`及拒绝原因，不得只留自由文本备注。

输出：

```text
human/final_review_tasks.jsonl
human/final_review_annotations.jsonl
human/final_review_adjudications.jsonl
human/final_review_agreement.json
human/final_review_workload.json
```

被reject的拟入选样本从合格候选池按原配额和选择规则补选，补选样本同样必须经过终审。

## 16. Step 9：正式1500条筛选

筛选采用两阶段提交，避免“必须先入选才能终审、必须终审才能入选”的循环依赖：

1. `select-final --stage provisional`只从自动硬门控合格池生成拟入选名单和候补顺序，并导出§15.4终审任务；
2. 导入真实人工结果后，`select-final --stage commit`仅提交`final_accept=true`样本，reject按冻结候补顺序补选并再次终审；
3. 未完成第二阶段时，任何文件不得命名或声明为official/final 1500。

### 16.1 硬门控

样本必须同时满足：

- 属于五个正式维度；
- 输入资产来源为已建立canonical记录的`tip_derived_reference`；本需求内的T2I/external资产不得进入正式集；
- 图像文件存在且`asset_binding_status=verified`：canonical上游hash与谱系输入hash一致，当前首帧hash与谱系预期派生hash一致；
- target subjects、noun、target relation和preservation set完整；
- dimension review通过；
- prompt结构检查通过；
- 不带failed_check；
- 旧fallback只有在重新修复后才可进入；
- 图文一致性通过；
- 图像质量方案已冻结并成功产出；
- native-ratio首帧与推理伴生件均有独立路径、hash和方法谱系，且未覆盖canonical原图；
- 拟入选后完成§15.4全量人工终审且`final_accept=true`。

### 16.2 目标边际配额

每维300条：

- difficulty：40% / 35% / 25%；
- semantic rarity：70% / 30%；
- subject tier：55% / 28% / 12% / 5%作为参考边际。

不要求三类变量完整笛卡尔积硬配额，避免桶爆炸。

### 16.3 选择优先级

```text
硬质量门槛
  > 每维总量300
  > difficulty边际
  > common/rare边际
  > subject tier边际
  > 预注册种子下的确定性抽取顺序
```

所有候选先通过同一硬质量门槛；同一配额层内使用`SHA256(draw_seed || question_id)`生成稳定抽取顺序，避免用任意加权分挑选“看起来更好”的正式样本。`quality_rank_components/score`只用于描述性与敏感性分析，不直接改变正式选择。不得把任务难度作为质量低的惩罚项，也不得查看验证/正式分布后更换seed。

### 16.4 成本-效果联合决策框架

#### 目的

在质量胜出规则之外引入工程可行性维度，帮助在质量非劣的方法中选择更容易部署的正式方案。8小时预算只约束winner在质量合格全池上的生产处理，不包含开发集/验证集消融、模型安装、权重下载、API排队和人工标注。

#### 成本维度定义

每个实验方法需记录：

```python
class MethodCost(BaseModel):
    method_id: str
    experiment: Literal["prompt", "clarity", "aspect"]
    requires_gpu: bool
    requires_api: bool
    gpu_vram_gb: float | None
    api_calls_per_sample: float
    api_cost_per_call_usd: float | None
    processing_time_per_sample_sec: float  # 实测或估算
    model_weight_size_gb: float | None
    additional_dependencies: list[str]
    deterministic: bool  # 是否每次运行结果相同
    evaluation_stage: Literal["development", "validation", "production"]
    planned_sample_count: int
    total_time_for_scope_min: float
    estimated_production_time_min: float | None
```

#### 预填成本估算表

要求在实验运行前预填估算，运行后更新实测值：

| 方法 | 首次实验阶段 | 作用样本数 | GPU | API | 单条耗时 | 本阶段总时 | 额外依赖 | 确定性 |
|------|------|---:|-----|-----|---------|---------|---------|--------|
| A0 (原prompt) | development | 250 | - | - | [待实测] | [待实测] | - | ✓ |
| A1/A3 (LLM约束/重试) | development | 异常项数[待估算] | - | ✓ | [待实测] | [待实测] | API client | ✗ |
| A2 (确定性映射) | development | 250 | - | - | [待实测] | [待实测] | wordfreq | ✓ |
| A4 (语义后检) | development | 250 | 可选 | - | [待实测] | [待实测] | CLIP | ✓ |
| A5 (PPL后检) | development/MAY | 250 | 可选 | - | [待实测] | [待实测] | transformers | ✓ |
| A6 (RareWordBlocker) | development/MAY | 250 | ✓ | - | [待实测] | [待实测] | local transformers | ✗ |
| C0/C1 | development | 120 | - | - | [待实测] | [待实测] | PIL/OpenCV | ✓ |
| C2/C3 | development | 120 | ✓ | - | [待实测] | [待实测] | Real-ESRGAN/SwinIR | ✓ |
| C4—C7 | development/MAY | 120 | ✓ | - | [待实测] | [待实测] | GFPGAN/OpenCV | 视方法而定 |
| D0—D4/D6 | development | 60或120 | 可选 | - | [待实测] | [待实测] | PIL/GroundingDINO | 视方法而定 |
| D5/D7 | development/MAY | [待估算] | ✓ | - | [待实测] | [待实测] | diffusers | ✗ |

#### 决策矩阵

方案选择使用加权评分：

```python
def method_score(quality_gain: float, cost: MethodCost, weights: dict) -> float:
    """
    quality_gain: 相对基线的质量提升（归一化到[0,1]）
    weights: 从配置读取
    """
    if cost.estimated_production_time_min is None:
        raise ValueError("missing production-time estimate")
    time_penalty = min(1.0, cost.estimated_production_time_min / 480)
    determinism_bonus = 0.1 if cost.deterministic else 0.0
    
    score = (
        weights["quality"] * quality_gain
        - weights["time"] * time_penalty
        - weights["complexity"] * len(cost.additional_dependencies) * 0.05
        + determinism_bonus
    )
    return score
```

权重配置（见§5 `configs/quality_experiments.yaml`）：

```yaml
decision:
  weights:
    quality: 0.60
    time: 0.25
    complexity: 0.15
  max_production_run_hours: 8
  experiment_budget_hours: null
  prefer_deterministic: true
```

#### Pareto前沿分析

对每个问题域（prompt/clarity/aspect），必须为**实际执行且通过硬门槛的核心方法**计算成本表和Pareto frontier JSON；该计算只需比较已有结构化结果，不要求补跑可选方法。覆盖全部MAY方法的完整前沿以及质量—时间散点图为MAY，未执行时记录`not_run_optional`。

`method_score`只作为权重敏感性分析，不直接决定学术winner。质量/时间/复杂度权重和确定性bonus均必须在查看验证结果前冻结，并至少报告两组替代权重下的选择是否变化。

输出：

```text
decision/method_costs.jsonl
decision/pareto_frontier.json
decision/decision_matrix.csv
decision/pareto_plots/{experiment}.png
```

#### 与胜出规则的关系

§10.5/§11.5/§12.7的胜出规则增加一条：

> 若多个方法均满足硬门槛且质量差异在95% CI内不显著，优先选择Pareto前沿上处理时间更短、确定性更高的方法。

### 16.5 View供给风险

View候选只有365条。若硬门控后不足300条：

- 输出shortfall；
- 优先人工修复可修复样本；
- 不降低Schema、维度正确率或图像存在性门槛；
- 不从其他维度补齐；
- 不自动生成新题。

### 16.6 输出

```text
selection/provisional_1500_manifest.jsonl
selection/reserve_order.jsonl
selection/final_1500_manifest.jsonl
selection/inference_assets.jsonl
selection/final_ids.txt
selection/selection_decisions.jsonl
selection/quota_actual_vs_target.json
selection/shortfall_report.json
selection/rejected_examples.jsonl
```

正式manifest路径必须为POSIX风格相对路径，不写Windows反斜杠和机器绝对路径。

## 16bis. Step 9b：组合消融实验

### 16bis.1 目的

本节分成两个统计对象不同的实验：（1）处理层消融，在同一qid上验证Text、Image和Aspect处理的独立及联合收益；（2）集合层重采样，验证质量门控和边际配额对整个样本集覆盖的影响。两者不得共用同一配对检验。

### 16bis.2 处理层消融条件

| 条件 | Text治理 | 清晰度 | 比例/尺寸 | 说明 |
|------|----------|-----------|-----------|------|
| **Baseline** | A0（有效原prompt） | C0 | 统一保守基线 | 已完成P0结构与资产修复，不使用全空旧Schema |
| **+Text** | A_winner | C0 | 同Baseline | 仅替换文本治理 |
| **+Image** | A0 | C_winner | 同Baseline | 仅替换清晰度方法 |
| **+Text+Image** | A_winner | C_winner | 同Baseline | 联合文本和清晰度方法 |
| **+Text+Image+Aspect** | A_winner | C_winner | D_winner | 在前两者基础上加入冻结比例与适配 |

其中`A_winner`、`C_winner`、`D_winner`分别为开发集筛选后冻结、再由验证集确认的§10、§11、§12方法。所有条件共用同一P0修复后canonical记录、正确原图和preservation set。

### 16bis.3 样本与评估

在验证集250条固定qid上运行上述5个条件。每条件输出：

| 指标类别 | 具体指标 |
|---------|---------|
| Prompt质量 | 结构通过、rare_modifier_rate、semantic_similarity；执行A5时另报perplexity |
| 图像质量 | NIQE, Laplacian variance, DINOv2 sim |
| 结构完整率 | Schema pass rate |
| 综合可用率 | 同时通过所有硬门控的比例 |

### 16bis.4 统计检验

- **配对比较**：每对条件使用McNemar检验（pass/fail二分类）或Wilcoxon符号秩检验（连续分数）；
- **多重比较校正**：Bonferroni或FDR（Benjamini-Hochberg）；
- 报告每对比较的p-value和effect size；
- 至少报告以下关键对比的显著性：

| 对比 | 验证目标 |
|------|---------|
| +Text vs Baseline | 文本治理独立效果 |
| +Image vs Baseline | 图像增强独立效果 |
| +Text+Image vs +Text | 图像在文本已修复基础上的增量 |
| +Text+Image+Aspect vs +Text+Image | 比例/尺寸处理的额外增量 |

### 16bis.5 集合层重采样实验

本节复用§13.4的主体Tier抽样产物，并扩展到难度、semantic rarity和多边际联合约束；不得为两个章节分别生成口径不同的Natural或Quality-only基线。

从同一P0结构/资产正确的canonical候选宇宙派生三种集合；其中后两种共享同一完整质量合格池：

| 条件 | 质量门控 | 边际配额 | 目的 |
|---|---|---|---|
| Natural | 仅P0结构/资产门控 | 仅维度总量 | 作为质量治理前的天然供给参照，不进入正式集 |
| Quality-only | 全部质量硬门控 | 仅每维300 | 分离质量门控的集合影响，不引入分层配额 |
| Quality+Stratified | 通过 | 维度+难度+rarity+Tier边际 | 检验分层覆盖改善 |

每种策略使用相同的一组预注册随机种子重复抽样，默认100次；每个seed通过`SHA256(seed || question_id)`改变候选顺序，Quality+Stratified的配额求解器也必须用该顺序打破多解平局。若不同seed意外得到完全相同集合，应如实报告唯一集合数，不得对重复副本计算虚假的95% CI。供给不足时报告实际可运行次数和shortfall。指标包括：质量通过率及质量分数分布、目标边际的总变差或Jensen–Shannon偏差、维度内子类覆盖、唯一来源数、重复率、View shortfall和跨唯一抽样集合的bootstrap 95% CI。Tier/难度/rarity分层的直接增量只比较Quality-only与Quality+Stratified，避免把候选池质量差异误归因于配额。

由于三种Sampling条件的qid集合不同，不得使用单样本配对McNemar/Wilcoxon，也不得把分布偏差改善表述为“单题质量提升”。

### 16bis.6 输出

```text
ablation/conditions.jsonl
ablation/per_sample_scores.jsonl
ablation/pairwise_tests.json
ablation/ablation_summary_table.csv
ablation/contribution_bar_chart.png
ablation/sampling_runs.jsonl
ablation/sampling_distribution_summary.json
ablation/sampling_bootstrap_intervals.json
```

## 17. Step 10：论文报告生成

`report`必须输出：

```text
report/quality_experiment_report.md
report/tables/*.csv
report/figures/*.png
report/method_winners.json
report/final_dataset_card.md
```

至少包含：

1. 3517条候选的基线问题统计；
2. 开发集A0—A4 Prompt核心筛选及可选A5/A6对比（含源头预洗）、验证集冻结候选确认；
3. B0—B3 fallback修复率；
4. C0—C3核心清晰度与身份保持；若实际执行，另报C4—C7可选消融；
5. 4:3/16:9和D0—D4/D6核心尺寸策略对比；若实际执行，另报D5/D7与阈值敏感性；
6. Natural、Quality-only与Quality+Stratified重复抽样的集合覆盖对比；
7. G0—G3难度标定对比；
8. P0共同基线下Baseline、+Text、+Image、+Text+Image、+Text+Image+Aspect处理层消融；
9. 正式1500条的五维、难度、罕见度和主体分布；
10. 失败案例与限制；
11. 若执行正交诊断，报告天然四象限供给和shortfall；仅在导入外部outcome后报告与结果类型匹配的推断统计；未执行时只报告`not_run_optional`；
12. 组合消融贡献柱状图与配对检验结果；
13. 各问题域已执行核心方法的成本表、frontier JSON与最终方案选择依据；完整MAY方法前沿图仅在实际生成时纳入。

报告生成器只读取结构化结果，不重新计算模型指标。

## 18. 测试要求

### 18.1 单元测试

必须覆盖：

- Windows/Unix路径解析；
- canonical上游hash和派生首帧hash分别与谱系对应端一致时才能验证资产绑定；
- 缩放/格式转换后的首帧hash不会被错误地直接与上游原图hash比较；
- 同一`source_sample_id`对应不同hash时资产清单构建失败；
- qid同名但hash不同的图像被拒绝而非静默回退；
- 同一`source_sample_id`不跨开发/验证集；
- 重复冠词和空槽位；
- View运镜线索；
- target relation五维映射；
- 空slot渲染出的非空模板句被判为无效target relation；
- preservation set引用完整性；
- 分层抽样总数与不重叠；
- 最大余数配额；
- subject多词短语；
- 难度分数边界；
- 难度特征证据、预测概率和低置信度路由；
- 人工标注按experiment正确路由到强类型Schema，方法数大于3时仍可盲化；
- `final_accept=false`时拒绝原因必填；
- 二值/连续/有序结果分别选择正确统计检验；
- 硬门控原因；
- View不足300时shortfall而非降级。

### 18.2 集成测试

构造每维2条、共10条fixture，运行：

```text
build-asset-manifest → build-asset-lineage → audit → split → local prompt rules → tag subjects → difficulty → selection preview → report
```

不得调用网络。

### 18.3 API smoke test

在显式 `--allow-api --limit 2 --max-api-calls 2` 下验证target repair或prompt rewrite。默认CI不得调用API。

### 18.4 回归测试

必须为以下已知bug建立测试：

- `aligned_subjects`字段被正确读取；
- `image_parse_v2.jsonl`和`text_parse_v2.jsonl`可发现；
- View prompt含zoom/pan时不被错误拒绝；
- “The the subject.”被拒绝；
- failed fallback不进入正式manifest。
- 只在旧目录中存在的同名qid图像不会在hash不一致时被当作正确资产；
- 缺少preservation set的修复记录不得标记`status=pass`。

## 19. 分阶段开发任务

### Task 0：环境与输入体检

工作：检查依赖、Phase 1 bundle、canonical资产清单、资产派生谱系及双端hash、模型权重、GPU/VRAM、API预算和现有产物。

完成标准：输出 `environment_report.json`、`experiment_availability.json`和`human_resource_plan.json`，明确每个实验是core-ready、optional-ready、awaiting-human还是blocked，不修改数据。

### Task 1：质量包骨架

工作：创建目录、Schema、配置加载、paths、hash和CLI。

完成标准：`all-local --limit 10`可以运行空骨架并生成run manifest。

### Task 2：审计与划分迁移

工作：实现canonical源资产清单与原图—首帧派生谱系构建，将现有 `scripts/audit_final_candidates.py` 逻辑迁入package，保留兼容wrapper。

完成标准：canonical清单可追溯到Phase 1或人工迁移证据，每个verified首帧具有双端hash一致的谱系；审计计数与当前基线一致；资产绑定有独立pass/review/reject结果；开发/验证集各250条且qid和source sample均不重叠。

### Task 3：结构化目标修复

工作：实现Phase 1重建和VLM迁移两条路径、overlay与人工队列。

完成标准：开发集Schema完整率100%，主体、目标变化和preservation set的人工准确率分别达到门槛。

### Task 4：Prompt与fallback实验

工作：在开发集实现A0—A4核心筛选、可选A5困惑度过滤与A6受控解码对比（含源头预洗）、B0—B3、规则、指标和人工任务；验证集只运行冻结候选，全池只运行winner。

完成标准：117条fallback均有明确pass/reject结果；不存在无效prompt通过；开发集A0—A4有完整指标，A5/A6不可用时输出`unsupported_optional`而非伪实现。

### Task 5：图像实验

工作：优先实现C0—C3、4:3/16:9、D0—D4和D6核心变体及指标；权重与预算允许时再实现C4—C7、D5/D7、视频探针和完整阈值敏感性。

完成标准：所有变体不覆盖原图；只对`asset_binding_status=verified`资产运行；核心方法有开发筛选与验证确认；可选权重缺失时输出明确状态且不阻断核心交付。

### Task 6：主体与难度

工作：实现Tier、多词主体、自然/分层集合对比、全池难度证据提取器、`D_task`与`D_judge`分离、G0—G3、人工标定及冻结模型的全池推断。

完成标准：所有eligible候选都有带证据/置信度的Tier、`D_task`、`D_judge`和新难度概率/标签或明确失败原因；G3有开发集交叉验证、独立验证集指标和冻结模型hash，低置信度项进入人工队列。

### Task 7：正式筛选与报告

工作：硬门控、边际配额、1500条拟选、全量人工语义终审与补选、数据卡和论文表格。

完成标准：满足§20验收；不足时输出shortfall而非伪造完成。

### Task 7b：正交诊断

工作（MAY）：使用天然样本实现象限分配与shortfall报告；仅在显式导入外部outcome后运行与结果类型匹配的推断分析。

完成标准：该Task未启动时输出`not_run_optional`即不阻断核心交付；若启动，输出天然四象限实际数量和`orthogonal_shortfall`，缺少外部outcome时输出`not_run_missing_outcome`，不要求伪造ANOVA。

### Task 8：组合消融与决策

工作：实现五个处理层条件的配对消融、三个集合层条件的重复抽样、相应统计、已执行核心方法的成本估算和Pareto frontier JSON；完整可选方法前沿绘图为MAY。

完成标准：五个处理层条件在验证集上有完整配对结果；三个Sampling策略有重复抽样分布报告；学术winner由质量硬门槛和效果统计决定，成本权重只作敏感性分析。

## 20. 最终验收标准

### 20.1 P0验收

- 审计覆盖3517条五维候选；
- 开发集和验证集各250条且无重叠；
- `status=pass`样本结构化字段完整率100%；
- `status=pass`样本的target subjects、target relation和preservation set均有效，target relation不得只是包裹空slot的模板套话；
- 所有进入图像实验的样本均有`source_sample_id`、`upstream_asset_id`、canonical上游hash、派生谱系、预期派生hash和`asset_binding_status=verified`；
- 无failed prompt和旧无效fallback进入eligible池；
- 所有外部调用可断点续跑并有预算保护；
- 每条处理结果可追溯到原始qid和输入hash。

### 20.2 P1验收

- Prompt、fallback、C0—C3清晰度、4:3/16:9及D0—D4/D6尺寸核心方案至少各完成一个验证集对比；
- 方法胜出规则在看验证集结果之前冻结；
- 人工标注一致性有统计报告；
- 图像方法同时报告清晰度和身份保持，不只报告锐度；
- 所有实验结果可由结构化产物重新生成报告；
- G0—G3有开发集交叉验证和独立验证集QWK/ordinal MAE/macro-F1结果；
- 冻结G3已对全eligible池输出概率、标签、模型hash和低置信度人工队列，不存在把500条人工特征直接复制到全池的情况；
- 处理层五个条件有完整配对结果，集合层三个抽样策略有重复抽样报告；
- 正交诊断整体为MAY：未启动时记录`not_run_optional`；若启动但无外部outcome，完成象限供给/shortfall即可；有outcome时才运行类型匹配的效应量和交互分析；
- 已执行核心方法的成本表和frontier JSON为核心；C4—C7、D5/D7、完整阈值网格、视频探针、覆盖全部MAY方法的前沿及绘图为MAY，未执行时必须报告`not_run`原因但不阻断核心交付。

### 20.3 正式集验收

- 五个维度各300条，总计1500条；
- qid唯一；
- 图像路径全部存在且可移植，每条的canonical上游hash、谱系输入hash、预期派生hash和当前首帧hash形成一致链路；
- 结构化目标、保持约束和prompt一致；
- 正式分布和配额偏差有完整报告；
- 最终ID列表、配置、代码版本和输入hash被冻结；
- 1500条全部完成人工语义终审且`final_accept=true`，双人复核子集有一致性与仲裁报告；
- 若任一维不足300条，不得标记任务完成，必须输出shortfall与修复建议。

## 21. Coding Agent禁止事项

1. 不得覆盖原始manifest、prompt或图像；
2. 不得为了配额把其他维度样本重标后直接补入；
3. 不得把模糊、空槽位等质量缺陷标成hard；
4. 不得从final prompt单独猜测结构化目标；
5. 不得对全部低频名词做无条件替换；
6. 不得默认启用GFPGAN或Outpainting进入主数据；
7. 不得只根据NIQE、Laplacian等单指标选图像方案；
8. 不得静默下载模型权重；
9. 不得默认调用付费API；
10. 不得实现或宣称Phase 3评测系统为本工程贡献；
11. 不得在未完成实验时把目标阈值写成已获得结果；
12. 不得在View不足时降低硬质量门槛凑到300条。
13. 不得在缺少上游hash、派生谱系或任一端hash不一致时仅凭qid回退到同名图像；
14. 不得为填满正交象限人工注入rare词、替换主体或改写正式样本；
15. 不得使用验证集重新调参后仍将其报告为独立验证；
16. 不得在缺少外部outcome时宣称正交诊断证明文本编码器或视觉长尾能力；
17. 不得将MAY实验未完成作为核心1500条交付的阻断条件。

## 22. Coding Agent第一轮建议任务

第一轮只完成不依赖外部模型的闭环：

1. 创建 `quality/` package与Schema；
2. 实现配置和run manifest；
3. 迁移audit与split；
4. 实现prompt纯规则检查；
5. 实现canonical上游资产、派生谱系、双端hash比对和禁止qid静默回退；
6. 生成开发/验证集与人工标注模板；
7. 为已知bug补回归测试；
8. 输出第一次environment/audit报告。

第一轮验收通过后，再进入VLM结构修复和图像模型实验，避免在基础数据接口仍不稳定时产生大量API与GPU成本。
