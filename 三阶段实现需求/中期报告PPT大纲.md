# 中期报告PPT大纲

## Benchmark Dataset Construction for Compositional Image-to-Video Generation

> 共13页 | 正文约14.5分钟，预留0.5分钟机动 | PPT文字以英文为主，中文讲解

---

## Slide 1 — Title & Opening

**预计时间**：0.5 min

### 页面文字内容

- **Title**: Benchmark Dataset Construction for Compositional Image-to-Video Generation
- **Subtitle**: Midterm Progress Report
- **Author**: [Name]
- **Affiliation**: [University / Lab]
- **Date**: 2026

### 建议配图/图表

- 背景大图：一张 I2V 生成示例（左侧输入图像 → 右侧生成视频帧序列），营造视觉冲击
- 校徽/院系 Logo 置于右上角

### 对应演讲稿段落

Slide 1: Title & Opening — 开场白介绍姓名、题目、核心问题预告

### 设计建议

- 深色背景（深蓝/藏青）+ 白色标题文字，突出学术感
- 标题字号 40pt+，副标题 24pt
- 底部一行关键词条带：`I2V · Compositional · Benchmark · Quality Control`

---

## Slide 2 — Why Narrow Long-Video Benchmark to I2V?

**预计时间**：1.75 min

### 页面文字内容

1. **Initial Scope**: A benchmark for minute-level long-video generation
2. **What “Long Generation” Often Means**:
   - Strong video generators are commonly short-horizon; StreamingT2V summarizes prior outputs as short clips, up to about **16 s**
   - A major family rolls out clips autoregressively: `tail frame(s) + prompt → next clip` (StreamingT2V, ViD-GPT, Ca2-VDM)
   - Concrete example: STIV conditions each new rollout on the **last two frames** of the previous rollout
3. **Why This Motivates I2V Evaluation**:
   - Each rollout contains an image/few-frame-conditioned local state transition
   - Identity drift, wrong action binding, or camera-motion confusion can propagate across later clips
4. **Boundary**: Not all long-video methods are repeated I2V—FIFO-Diffusion, FreeNoise, and NUWA-XL use queue, window, or hierarchical mechanisms
5. **Feasibility Evidence**: Dedicated long-video data and validated long-horizon evaluators still require separate large-scale work (LVD-2M; MovieBench; Beyond FVD; SLVMEval)
6. **Decision**: Isolate and validate the reusable local capability unit—**mechanism decomposition, not task equivalence**

### 建议配图/图表

- 页面主图采用**分段续写机制图**：
  `Clip 1 (short generator) → [last k frames + prompt] → Clip 2 → [last k frames + prompt] → Clip 3 → …`
- 在箭头上方标注：`StreamingT2V / ViD-GPT / Ca2-VDM / STIV`
- 在链路下方用红色小箭头标出：`local error → propagated drift`
- 右下角放“方法边界”小框：`Other routes: queue / sliding window / hierarchy ≠ repeated I2V`
- 左下角放两项次要可行性约束：`Data Ground Truth`与`Validated Long-Horizon Evaluators`
- 页脚小字：`I2V is an atomic capability in a major extension route—not a synonym for long video`

### 对应演讲稿段落

Slide 2: Why Narrow Long-Video Benchmark to I2V? — 先用技术报告核实“短片段生成器+末帧/末若干帧续写”机制，再说明I2V是可独立验证的局部能力而非长视频同义词

### 设计建议

- 中央约65%用于分段续写链路，右侧约35%放“证据—边界—研究决策”
- 高亮`tail frame(s) + prompt → next clip`与`mechanism decomposition`
- 不使用“Long video = I2V”，也不声称所有模型都固定生成10–15秒
- 论文只显示作者/简称与会议年份，完整引用放页脚或备份页

---

## Slide 3 — I2V Motivation & Benchmark Gap

**预计时间**：1.25 min

### 页面文字内容

1. **Problem**: Visual quality ≠ Instruction following
2. **Conceptual Example**: "Make the white dog sit down"
   - ✗ Brown dog sits | ✗ Camera zoom only | ✓ White dog sits
3. **Existing Benchmarks**:
   - T2I-CompBench: no temporal dynamics
   - T2V-CompBench: text-only input, no first-frame constraint
   - VBench/VBench++: broad quality coverage; compositional failure attribution is not the primary focus
4. **I2V introduces a dual constraint**:
   - Input image → inviolable initial state (subjects, layout, background)
   - Text prompt → selective changes to specific elements
5. **Our Focus**: explicit Preserve + Transform representation for failure attribution

### 建议配图/图表

- 左侧放两只狗的三结果概念示例，并标注`Conceptual illustration — not an experimental result`
- 右侧放紧凑对比：`T2I: no time`、`T2V: no first frame`、`VBench: broad quality`、`Ours: explicit P–T attribution`
- 没有真实实验结果时不展示虚构的FVD/CLIP数值

### 对应演讲稿段落

Slide 3: I2V Motivation & Benchmark Gap — 用两只狗示例解释组合失败，再对比现有Benchmark并引出Preserve–Transform

### 设计建议

- 左图右文，概念示例占页面约55%
- 红色✗/绿色✓突出绑定错误与正确结果
- 页面底部用一句话过渡：`Known first frame makes change and preservation jointly testable`

---

## Slide 4 — Preserve-Transform Evaluation Framework

**预计时间**：1.25 min

### 页面文字内容

1. **Formalization**: I2V = Selective State Transition under Visual Constraints
2. Initial state: C₀ = (Entities, Attributes, Relations, Poses, Background, Camera)
3. Each sample decomposes into:
   - **Transform set T**: What MUST change
   - **Preserve set P**: What MUST remain stable
4. → Supports more explicit *failure attribution*
5. **Five Evaluation Dimensions**:
   - Attribute Binding | Action Binding | Motion Binding | Background Dynamics | View Transformation
6. **Validity condition**: Valid(x) = Identifiable ∧ Observable ∧ Separable ∧ Non-trivial

### 建议配图/图表

- **框架示意图**（页面主视觉）：
  - 中心：输入图像 I₀ → 生成视频 V
  - 左分支标注 "Preserve Set P"（虚线框包裹不变元素图标）
  - 右分支标注 "Transform Set T"（实线框包裹变化元素图标）
  - 下方五个维度图标横排，每个用不同颜色小方块标识
- **五维度图标条**：用5个简洁icon+标签横排展示

### 对应演讲稿段落

Slide 4: Preserve-Transform Framework — 形式化定义、双轴分解、五维度定义、有效性条件

### 设计建议

- 上半页：P-T 双轴示意图（核心创新，视觉突出）
- 下半页：五维度横排彩色标签
- 公式用 LaTeX 风格排版，字号适中不喧宾夺主
- 蓝色=Preserve，橙色=Transform，形成品牌色对比

---

## Slide 5 — Overall Pipeline Architecture

**预计时间**：1 min

### 页面文字内容

1. **Four-Stage Architecture**:
   - Phase 1: Prior Extraction
   - Phase 2: Automated Synthesis
   - Quality Control: P0–P4 Repair
   - Validity Verification
2. **Three Design Principles**:
   - Zero-Assembly Vocabulary — all samples from real user data
   - Dimension Isolation — one intervention per sample
   - Structured Delivery — downstream consumes structured fields directly

### 建议配图/图表

- **横向四阶段流程图**（占页面上 2/3）：
  ```
  [TIP-I2V Real Data] → [Phase 1: Prior Extraction] → [Phase 2: Synthesis] → [Quality Control] → [Validity Verification]
  ```
  - 每阶段用不同颜色方块
  - 阶段间用箭头连接
  - 每个方块下方标注 1-2 个关键产出
- 下方三条原则用图标+文字横排

### 对应演讲稿段落

Slide 5: Overall Pipeline Architecture — 四阶段总览+三条设计原则

### 设计建议

- 流程图从左到右，简洁大气
- 三原则用小 icon（🎯📐📦 风格图标）辅助记忆
- 浅灰背景 + 彩色流程块

---

## Slide 6 — Phase 1: Prior Data Preparation

**预计时间**：1.25 min

### 页面文字内容

1. **Data Source**: TIP-I2V — million-scale real user prompt-image dataset
2. **5-Step Parsing Pipeline**:
   - Step 1: Raw data scanning & cleaning
   - Step 2: VLM visual structure parsing (subjects, background, camera)
   - Step 3: LLM semantic intent parsing (6 boolean intent flags + slots)
   - Step 4: Cross-modal alignment verification
   - Step 5: Prior Package assembly
3. **6 Enhancement Sub-modules** (zero API calls, deterministic, idempotent):
   - patch | align | refbank | priors2 | recipes | audit
4. **Key Algorithm**: Greedy matching for image-text instance alignment (confidence: 1.0 / 0.7 / 0.0)

### 建议配图/图表

- **两层流水线图**：
  - 上层：5 步解析（左→右箭头流，标注模型调用 VLM/LLM）
  - 下层：6 子模块增强（并行/顺序块图，标注"Zero API"徽章）
  - 输入端标注 TIP-I2V，输出端标注 Prior Package
- 或者用表格简化展示 5 步 + 6 子模块

### 对应演讲稿段落

Slide 6: Phase 1 — Prior Data Preparation — TIP-I2V 来源、5步+6子模块、零API约束、贪心匹配

### 设计建议

- 信息量较大，建议分上下两区：上区流程图，下区关键技术点 bullets
- 用"badge"样式标注约束条件（Zero API / Idempotent / Deterministic）
- 蓝灰色调，学术工程感

---

## Slide 7 — Phase 2: Dataset Synthesis

**预计时间**：1.5 min

### 页面文字内容

1. **8-Step Pipeline**: Quota → Sample → Plan → Construct → Verify → Finalize → Export → Audit
2. **Implemented Design Choices**:
   - **Largest Remainder Method**: Integer quota allocation with zero rounding error
   - **Prior Traceability**: Every semantic recipe traces back to a TIP-I2V source
   - **Dual-Track Image Output**: Native aspect ratio + 16:9 inference companion
   - **VQA-based Candidate Check**: Structured verification with 3-state aggregation
   - **Prompt Finalization**: VLM-describe → LLM-polish → rule audit
3. **5 Contrastive Controls**:
   - static_copy | random_motion | global_filter | camera_pan_cheat | subject_swap

### 建议配图/图表

- **8步流水线横向流程图**（紧凑型，每步一个小方块+标签）
- 右侧或下方用**callout框**突出3个创新设计：
  - 最大余数法公式示意
  - 双轨产物示意（原图→两张输出）
  - 对照设计5种类型图标

### 对应演讲稿段落

Slide 7: Phase 2 — Dataset Synthesis — 8步流水线、最大余数法、零拼词表、双轨输出、VQA质检、对照设计

### 设计建议

- 流程图占上半页，创新亮点用高亮 callout 框占下半页
- 每个创新点用不同颜色小标签区分
- 信息密集页，控制文字量，图表为主

---

## Slide 8 — Data Funnel & Current Status

**预计时间**：1 min

### 页面文字内容

1. **Data Funnel**:
   - **4,092** Question Plans
   - **3,519** Prompt Candidates
   - **3,517** Candidate Manifest Rows (**85.9%** of plans; not a final QC pass rate)
   - **1,500** Release Target = 5 dimensions × 300 (**not yet release-ready**)
2. **Image Assets**: **8,184** = 4,092 native + 4,092 16:9 companions
3. **Fallback Usage**: 117 / 3,519 = **3.3%**
4. **Target Interface**: 17-field BenchmarkSample; mandatory fields must pass hard gates before Phase 3 delivery

### 建议配图/图表

- **数据漏斗**（页面核心）：`4,092 plans → 3,519 prompts → 3,517 candidates → 1,500 release target`
- 下方柱状图展示当前候选分布：
  - Attribute: 517 | Action: 1,013 | Motion: 676 | Background: 946 | View: 365
- 右侧小示意：BenchmarkSample JSON Schema 缩略展示
- 对1,500使用虚线框和`Target / Pending final audit`标签，避免被理解为已发布数据集

### 对应演讲稿段落

Slide 8: Data Funnel & Current Status — 区分题目计划、提示词、候选manifest和正式目标集

### 设计建议

- 漏斗和状态标签优先于“大数字成绩单”
- 绿色仅用于真正通过最终质量门控的数据；当前候选使用蓝色或灰色
- 明确写出“文件已生成 ≠ Benchmark已可用”

---

## Slide 9 — Quality Issues: P0 Critical Finding

**预计时间**：1.5 min

### 页面文字内容

1. **Repair Baseline** (frozen old snapshot):
   - 3,517 / 3,517 missing subject nouns and structured changes
2. **Current Workspace Snapshot (2026-07-13)**:
   - Subject noun recovered: **3,263 / 3,517 (92.8%)**
   - Still unresolved: **254** generic/empty subjects
   - `target_relation` still empty: **3,517 / 3,517**
3. **Status**: P0 is partially repaired, but the manifest and preliminary “Final 1500” package are **not release-ready**
4. **Main Causes**: cross-stage field mismatch, unused dimension slots, unstable file-version resolution, and insufficient export hard gates
5. **Next Gate**: relation reconstruction → residual cleanup → full audit → version freeze → package

### 建议配图/图表

- **修复前后对比图**：noun覆盖率 `0% → 92.8%`，同时用红色标出`target_relation = 0% complete`
- **流程断裂图**：Phase 1结构信息 → 字段/版本适配 → Phase 2目标结构 → 导出硬门控
- 底部红色状态条：`Candidate package — not a validated release`

### 对应演讲稿段落

Slide 9: Quality Issues — P0 Critical Finding — 区分旧基线与当前部分修复状态，说明剩余阻断项

### 设计建议

- 红色警告风格头部标题栏（⚠️ P0: BLOCKING）
- 根因分析用编号列表或鱼骨图
- 不使用“Good News”弱化阻断问题；用中性的`Recoverable from upstream evidence`说明可修复性

---

## Slide 10 — Quality Issues: P1–P4

**预计时间**：0.75 min

### 页面文字内容

| Issue | Severity | Finding | Key Metric |
|-------|----------|---------|-----------|
| **P1** Low-frequency Wording | Medium | 5.8% hit the rare-word list; not all are invalid | Separate necessary terms from decorative modifiers |
| **P2** Image Clarity | High | Some sources are 224×126; upscale cannot restore true detail | Blur vs enhancement hallucination |
| **P3** Aspect Ratio | High | Native AR range 0.53–2.52; 68% need crop/pad under target specs | Evaluation inconsistency |
| **P4** Distribution | Medium | Subject frequency & difficulty uncalibrated | Uncontrolled confounds |

### 建议配图/图表

- **四象限/四卡片布局**：每个 P 问题一个卡片
  - P1: 示例生僻词 word cloud 或高亮文本截图
  - P2: 低分辨率 vs 目标分辨率对比图（模糊→清晰）
  - P3: 不同宽高比图像拼贴（方形、竖屏、超宽）
  - P4: 长尾分布曲线示意
- 每个卡片标注严重等级颜色（红/橙/黄）

### 对应演讲稿段落

Slide 10: Quality Issues — P1 to P4 — 四类问题逐一说明

### 设计建议

- 2×2 网格布局，每格一个问题
- 颜色编码严重等级：P2/P3 红色，P1/P4 橙色
- 每格内：标题 + 一句话 + 关键数字 + 小配图

---

## Slide 11 — Future Work: Repair & Validation

**预计时间**：1.5 min

### 页面文字内容

**Thread 1: P0 Closure & Versioning**
- Reconstruct `target_relation`; resolve or reject 254 residual subjects
- Add export hard gates; audit and freeze a new manifest before packaging

**Thread 2: Quality Strategy Comparison**
- P1: keep necessary terms; simplify only replaceable modifiers
- P2: compare interpolation / non-generative sharpening / generative SR with identity-preservation checks
- P3–P4: recorded AR adaptation + frequency-tier-aware sampling

**Thread 3: Ablation & Validity**
| Experiment | Control | Treatment | Metric |
|---|---|---|---|
| Vocabulary | Original | Meaning-preserving simplification | Human equivalence + score shift |
| Clarity | Interpolation | Candidate enhancement | Sharpness + identity similarity + artifact rate |
| Aspect Ratio | Model-default | Recorded unified adaptation | Evaluator variance + target loss rate |

- 50–100 items/dimension × 2–3 **image+text conditioned** I2V models
- Test dimension/difficulty discriminability and failure-mode validity; calibrate automatic metrics against blind human ratings

### 建议配图/图表

- **三线程并行路线图**：
  - 三条水平泳道，每条标注关键里程碑
- 消融实验设计表格（如上所示）
- 有效性验证用简图：抽样 → 模型推理 → 自动评分 → 统计分析

### 对应演讲稿段落

Slide 11: Future Work — Repair & Validation — 三条主线：修复、消融、验证

### 设计建议

- 三条主线用三种颜色区分
- 消融表格紧凑排列
- 右侧可加验证流程小图

---

## Slide 12 — Timeline

**预计时间**：0.5 min

### 页面文字内容

| Weeks | Task | Deliverable |
|-------|------|-------------|
| W1–W2 | Close residual P0 issues + freeze versions | Audited candidate manifest |
| W3–W4 | Compare and calibrate P1–P4 strategies | Frozen quality rules and thresholds |
| W5–W6 | Build and audit the 1,500-sample release candidate | 5 × 300 versioned package |
| W7–W9 | Multi-model validity + human agreement study | Dimension/difficulty/failure statistics |
| W10–W12 | Thesis writing (Ch.3–5) + iteration | Thesis draft + dataset card |

### 建议配图/图表

- **甘特图**（Gantt Chart）：
  - 横轴12周，纵轴5个工作包
  - 每个工作包用不同颜色条带
  - 关键里程碑用菱形标记（如 W2: manifest ready, W8: validation complete）
- 或简化为时间轴（Timeline）横向图

### 对应演讲稿段落

Slide 12: Timeline — 12周时间规划

### 设计建议

- 甘特图占满页面，清晰直观
- 当前位置用竖线标注 "We are here"
- 里程碑用星号/菱形突出

---

## Slide 13 — Conclusion & Interim Contributions

**预计时间**：0.75 min

### 页面文字内容

**Four Interim Outputs:**

1. **Preserve-Transform Dual-Axis Framework**
   - Formally decomposes I2V evaluation into Transform (what must change) + Preserve (what must remain)

2. **Prior-Grounded Candidate Construction Pipeline**
   - Automated candidate generation with traceability to TIP-I2V; final release still requires quality gates

3. **Candidate Scale & Release Target**
   - 3,519 prompt candidates; 8,184 image assets; target release = 1,500 validated samples

4. **Auditable Data Quality Workflow**
   - P0–P4 taxonomy across schema, prompt, image, aspect ratio, and distribution

**Next Phase**: Close P0 → compare quality strategies → validate metrics → freeze a release candidate

### 建议配图/图表

- **四贡献图标列表**：每条贡献配一个简洁icon，纵向排列
- 或用**四象限贡献矩阵**：
  - 横轴：理论贡献 vs 工程贡献
  - 纵轴：已完成 vs 进行中
- 底部 "Thank you & Questions" + 联系方式

### 对应演讲稿段落

Slide 13: Conclusion & Contributions — 四项阶段性产出、明确未完成边界、下阶段方向与致谢

### 设计建议

- 回归深色背景（与 Slide 1 呼应），形成首尾闭环
- 四条贡献用数字编号+金色/白色高亮
- 最后一行大字 "Thank You · Questions Welcome"
- 底部可加二维码（GitHub 仓库链接）

---

## 整体设计建议汇总

### 配色方案
- 主色：深蓝 (#1A237E) — 学术权威感
- 辅色：橙色 (#FF6F00) — 标注创新点/Transform轴
- 辅色：青蓝 (#00ACC1) — 标注Preserve轴
- 警告色：红色 (#D32F2F) — P0等关键问题
- 成功色：绿色 (#388E3C) — 通过指标/解决方案

### 字体建议
- 标题：Arial Black / Montserrat Bold
- 正文：Calibri / Source Sans Pro
- 代码/数据：Consolas / JetBrains Mono

### 布局原则
- 每页信息层次不超过3层
- 关键数字放大2-3倍突出
- 图表面积 ≥ 50%，文字面积 ≤ 50%
- 留白充分，避免信息过载

### 动画建议
- 仅在必要时使用淡入（Fade In）
- 流程图可分步显示
- 避免花哨转场效果
