# 中期报告PPT大纲

## Benchmark Dataset Construction for Compositional Image-to-Video Generation

> 共13+1页 | 总时长约15分钟 | PPT文字以英文为主，学术答辩风格

---

## Slide 1 — Title & Opening

**预计时间**：1 min

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

## Slide 2 — Research Motivation

**预计时间**：2 min

### 页面文字内容

1. I2V generation has achieved remarkable visual quality (SVD, CogVideoX, DynamiCrafter…)
2. **Problem**: Visual quality ≠ Instruction following
3. Example: "Make the white dog sit down"
   - ✗ Wrong subject performs action (brown dog sits)
   - ✗ No action occurs (camera zoom only)
   - ✓ Traditional metrics (FVD, CLIPScore) give high scores
4. **Gap**: Cannot diagnose *compositional failures* — wrong binding of actions/attributes/motions to subjects

### 建议配图/图表

- **对比示例图**（页面核心视觉）：
  - 上方：输入图像（两只狗：白+棕）+ 文本指令 "Make the white dog sit down"
  - 下方三列对比：
    - Column A: ✗ Brown dog sits（绑定错误）
    - Column B: ✗ Camera zoom only（无动作）
    - Column C: ✓ White dog sits（正确）
  - 每列下方标注 FVD/CLIP 分数（A/B 高分但错误，形成反差）
- 可用伪截图或示意简图代替

### 对应演讲稿段落

Slide 2: Research Motivation — 从 I2V 模型进步引出"视觉好看≠指令遵循"，用两只狗的例子具体说明

### 设计建议

- 左右分栏布局：左侧文字要点，右侧对比图
- 红色✗/绿色✓突出对错对比
- 关键短语 "Compositional Failures" 加粗高亮（橙色/红色）

---

## Slide 2.5 — Long Video Generation → I2V Transition

**预计时间**：1 min

### 页面文字内容

1. **Initial research focus**: Long video generation benchmark (1–5 min videos)
2. **Key Insight from literature review**:
   - Three main technical routes: Chunked Synthesis (StreamingT2V, MicroCinema, MovieDreamer), Autoregressive (LongLive, FreeNoise), Hierarchical (NUWA-XL, CogVideo)
   - **All methods share the same fundamental paradigm**:
     - Segment N: Generate short video conditioned on **last frame of Segment N−1**
     - This IS an I2V task
3. **Implication**: Long video generation = Chain of I2V tasks
4. **Pivot rationale**:
   - Performance bottleneck lies in single-segment I2V quality
   - Existing metrics (FVD, CLIPScore, IS) designed for 16–64 frames — cannot assess long video global coherence or attribute failures to specific causes
   - Long video human evaluation: high cost, attention decay, subjective inconsistency
   - I2V evaluation: higher diagnostic precision, fully reproducible, low compute cost
   - I2V enables automated evaluation pipelines (Grounding DINO, optical flow, DINOv2) — infeasible at long-video granularity
   - Avoids conflation of stitching strategy quality with base model capability

### 建议配图/图表

- **长视频分段生成示意图**（页面核心视觉）：
  - 横向时间轴，分为3-4个片段
  - 每个片段显示为短视频帧序列
  - 片段之间用粗箭头连接，箭头上标注“末帧 → 首帧”
  - 每个箭头下方标注“I2V Generation”
  - 用红色虚线框圈出“这就是I2V”的核心洞察
- 右下角callout框：“Long Video Quality ≥ Single I2V Quality”

### 对应演讲稿段落

Slide 2: Research Motivation — “I want to briefly share how this project evolved”段落（课题从长视频Benchmark转向I2V Benchmark的动机说明）

### 设计建议

- 左侧放置分段生成示意图（占页面60%）
- 右侧放置3个bullet point总结转向理由
- 用橙色高亮“= Chain of I2V”关键结论
- 底部用绿色callout强调“I2V benchmark = evaluate the atomic building block”
- 与前后页风格一致，用箭头表明从Silde 2动机自然过渡到Slide 3 gap分析

---

## Slide 3 — Gap in Existing Benchmarks

**预计时间**：1 min

### 页面文字内容

1. **T2I-CompBench**: Compositional evaluation for T2I — no temporal dynamics
2. **T2V-CompBench**: Extends to video — text-only input, no first-frame constraint
3. **VBench / VBench++**: Comprehensive video quality — lacks fine-grained compositional diagnosis
4. **Key Insight**: I2V introduces *dual constraint*
   - Input image → inviolable initial state (subjects, layout, background)
   - Text prompt → selective changes to specific elements
5. **No existing benchmark models this Preserve-Transform duality**

### 建议配图/图表

- **对比表格**（核心视觉元素）：

| | T2I-CompBench | T2V-CompBench | VBench++ | **Ours** |
|---|:---:|:---:|:---:|:---:|
| Input | Text | Text | Text/Image | **Image+Text** |
| Core Target | Static composition | Dynamic + text align | General quality | **Selective compositional change** |
| First-frame modeling | ✗ | ✗ | Partial | **Explicit P-T** |
| Data quality control | ✗ | ✗ | ✗ | **Systematic** |

### 对应演讲稿段落

Slide 3: Gap in Existing Benchmarks — 逐项对比现有 Benchmark，引出 Preserve-Transform 对偶性

### 设计建议

- 表格占页面中心 60%，用颜色区分最后一列（蓝色/绿色高亮"Ours"）
- 上方一句话定位：*"Where does our work fit?"*
- 表格下方一句总结引出下页

---

## Slide 4 — Preserve-Transform Evaluation Framework

**预计时间**：2 min

### 页面文字内容

1. **Formalization**: I2V = Selective State Transition under Visual Constraints
2. Initial state: C₀ = (Entities, Attributes, Relations, Poses, Background, Camera)
3. Each sample decomposes into:
   - **Transform set T**: What MUST change
   - **Preserve set P**: What MUST remain stable
4. → Enables precise *failure attribution*
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

**预计时间**：2 min

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

**预计时间**：2 min

### 页面文字内容

1. **8-Step Pipeline**: Quota → Sample → Plan → Construct → Verify → Finalize → Export → Audit
2. **Key Innovations**:
   - **Largest Remainder Method**: Integer quota allocation with zero rounding error
   - **Zero-Assembly Principle**: Every item traces back to TIP-I2V source
   - **Dual-Track Image Output**: Native aspect ratio + 16:9 inference companion
   - **VQA-based QC**: Structured verification with 3-state aggregation
   - **Prompt Finalization**: VLM-describe → LLM-polish + forbidden-word constraints
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

## Slide 8 — Current Scale & Output

**预计时间**：0.5 min

### 页面文字内容

1. **Dataset Scale**:
   - **3,519** prompts
   - **8,184** image assets (4,092 native + 4,092 × 16:9 companions)
2. **Quality Metrics**:
   - VQA pass rate: **85.9%** (3,517 / 4,092)
   - Fallback prompt usage: only **3.3%** (117 items)
3. **Delivery Format**: 17-field flat BenchmarkSample schema → directly consumable by Phase 3
4. **Dimension Coverage**: Attribute | Action | Motion | Background | View

### 建议配图/图表

- **大数字展示**（Dashboard 风格）：
  - 三个大数字卡片横排：`3,519 Prompts` | `8,184 Images` | `85.9% Pass`
- 下方饼图或柱状图：五维度样本分布
  - Attribute: 502 | Action: 948 | Motion: 658 | Background: 927 | View: 361
- 右侧小示意：BenchmarkSample JSON Schema 缩略展示

### 对应演讲稿段落

Slide 8: Current Scale & Output — 数据规模、通过率、Schema 结构

### 设计建议

- Dashboard/数据面板风格，大数字突出
- 绿色高亮通过率，蓝色标注规模
- 简洁，让数字自己说话

---

## Slide 9 — Quality Issues: P0 Critical Finding

**预计时间**：1.5 min

### 页面文字内容

1. **P0: Structured Evaluation Targets Completely Empty** (Blocking)
2. Full-schema audit of 3,517 candidates reveals:
   - 3,517/3,517 `target_subjects[].noun` = EMPTY
   - Subject descriptions degraded to "the subject" (generic placeholder)
   - No structured `target_relation` for downstream consumption
3. → Phase 3 evaluators **cannot consume** current manifest
4. **Root Causes** (5 factors):
   - ① Field name mismatch: `target_instances` vs `aligned_subjects`
   - ② Dimension-specific slot fields unused by Phase 2
   - ③ v2 filename not recognized by loader
   - ④ NL polish masks structural emptiness
   - ⑤ View Transformation constraint logic inverted
5. **Good news**: Candidate pool sufficient after fixing — code bugs, not data scarcity

### 建议配图/图表

- **根因鱼骨图/因果链图**：
  - 中心：P0 现象（目标为空）
  - 5条分支指向5个根因
- 或用**流程断裂图**：Phase 1 产出 → [断裂点标红] → Phase 2 消费失败
- 底部：候选池余量柱状图（5维度，标注各维度通过数 vs 目标300）

### 对应演讲稿段落

Slide 9: Quality Issues — P0 Critical Finding — P0 现象、根因五因素、候选池充足

### 设计建议

- 红色警告风格头部标题栏（⚠️ P0: BLOCKING）
- 根因分析用编号列表或鱼骨图
- 底部"Good News"用绿色 callout 缓和严肃氛围

---

## Slide 10 — Quality Issues: P1–P4

**预计时间**：1 min

### 页面文字内容

| Issue | Severity | Finding | Key Metric |
|-------|----------|---------|-----------|
| **P1** Rare Vocabulary | Medium | 5.8% prompts contain rare words (malevolence, sclera…) | Tests encoder robustness, not composition |
| **P2** Image Clarity | High | Source 224×126 → 3.8× upscale to 854×480; info density 6.9% | Severe blurring |
| **P3** Aspect Ratio | High | Native AR range 0.53–2.52; only 2% near standard | Evaluation inconsistency |
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

**预计时间**：2 min

### 页面文字内容

**Thread 1: Systematic Repair**
- P0: 6 targeted code fixes → full pipeline re-run
- P1: 3-layer rare word defense (Zipf < 3.5 threshold)
- P2: Real-ESRGAN + GFPGAN enhancement pipeline
- P3: Intelligent AR adaptation (resize / center-crop / letterbox)
- P4: Frequency-tier-aware sampling + difficulty calibration

**Thread 2: Ablation Experiments**
| Experiment | Control | Treatment | Metric |
|---|---|---|---|
| Vocabulary | Rare-word prompts | Simplified prompts | CLIP-Sim, inter-model variance |
| Clarity | LANCZOS upscale | Real-ESRGAN | Laplacian Var, FID |
| Aspect Ratio | Mixed AR | Unified 854×480 | Evaluator stability |

**Thread 3: Validity Verification**
- Sample 50–100 items/dimension × 2–3 I2V models (SVD, CogVideoX, DynamiCrafter)
- Prove: dimension discriminability, difficulty discriminability, failure mode predictability

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
| W1–W2 | P0 code fixes + pipeline re-run | Structurally complete manifest |
| W3–W4 | P1–P4 systematic repairs | Repaired dataset artifacts |
| W5–W6 | Quality experiments + ablation | Experiment data & analysis |
| W7–W8 | Validity verification | Dimension/difficulty/failure statistics |
| W9–W12 | Thesis writing (Ch.3–5) + iteration | Thesis draft |

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

## Slide 13 — Conclusion & Contributions

**预计时间**：1 min

### 页面文字内容

**Four Main Contributions:**

1. **Preserve-Transform Dual-Axis Framework**
   - Formally decomposes I2V evaluation into Transform (what must change) + Preserve (what must remain)

2. **Fully Automated, Prior-Grounded Synthesis Pipeline**
   - Produces benchmark samples traceable to real user data (TIP-I2V)

3. **Dataset Scale: 3,519 Prompts × 8,184 Images**
   - Five compositional dimensions with structured delivery

4. **Systematic Data Quality Methodology**
   - P0–P4 issue taxonomy — first discussion in benchmark construction literature

**Next Phase**: Complete repairs → Ablation studies → Validated, publication-ready benchmark

### 建议配图/图表

- **四贡献图标列表**：每条贡献配一个简洁icon，纵向排列
- 或用**四象限贡献矩阵**：
  - 横轴：理论贡献 vs 工程贡献
  - 纵轴：已完成 vs 进行中
- 底部 "Thank you & Questions" + 联系方式

### 对应演讲稿段落

Slide 13: Conclusion & Contributions — 四项贡献总结、下阶段方向、致谢

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
