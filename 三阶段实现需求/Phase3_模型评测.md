# I2V-CompBench Phase 3 实现需求（模型评测）

> 本文档面向 AI coding agent，是《I2V-CompBench Phase 1/2/3 实现需求（Coding Agent 版）》的 Phase 3 拆分文档。Phase 1、Phase 2 内容请参见同目录下的另外两份文档。
>
> Phase 3 的目标：消费 Phase 2 输出的 `benchmark_dataset/phase3_manifest.jsonl`，对待评 I2V 模型生成视频，并按 7 维度执行**执行门控评分** `S = E * (0.6P + 0.4C)`，输出可审计、可复现的评测结果。

---

## 0. Phase 3 总目标

```text
Phase 3 数据流：
  benchmark_dataset/{phase3_manifest.jsonl, first_frames/, ref_images/}
        +
  待评模型生成视频
        ↓
  归一化（fps/分辨率 + fps=8 motion 副本 + 输入图归一化）
        ↓
  tool features（grounding / tracking / depth / identity 一次性产出）
        ↓
  per-dimension evaluators（输入：sample + 视频 + 输入图 + tool features）
        ↓
  scoring（exec-gated S = E·(λP + (1-λ)C)）
        ↓
  human_validation（Spearman ρ ≥ 0.5 门控）
        ↓
  aggregate（仅聚合通过门控的维度）
        ↓
  report
```

必须支持：

- 7 个评测维度：attribute, action, motion, spatial, background, view, interaction。
- 2 种输入模式：single_image, multi_image，且**多图属于主维度评测**，不是 stress-only。
- 统一执行门控评分公式：`S = E * (lambda * P + (1 - lambda) * C)`，默认 `lambda = 0.6`。
- 不允许使用线性总分 `0.45E + 0.35P + 0.20C`。
- 工具失败必须记 `tool_uncertain`，不要简单记 `E=0`。
- 必须保存完整 E/P/C、辅助指标、failure_modes，不能只保存最终总分。
- **首帧 / 参考图必须作为 evaluator 的显式入参**（P 维度本质是「生成视频 vs 输入首帧」的差异，不传首帧无法计算）。
- **multi_image 样本必须先做 identity binding**：把参考图主体与视频中的主体匹配上，否则等于「猫存在 ≠ 是参考图那只猫」，多图考点失效。

### 0.1 vs T2V-CompBench / VBench-I2V 的评测方法论差异化

本阶段评测器与常见参考框架的关键区别：

| 评测项 | T2I-CompBench | T2V-CompBench | 本框架 (Phase 3) |
|------|------|------|------|
| **语义判定** | Disentangled BLIP-VQA 拆问 | Grid-LLaVA / D-LLaVA + CoT | **采用 Disentangled VQA 拆问**（详见 §4.5 总说） |
| **Spatial 判断** | UniDet bbox + 水平距离>垂直距离 ∧ IoU<0.1 | G-Dino + bbox geometry | **GroundingDINO + IoU<0.1 几何公式**（详见 §4.5.4） |
| **Motion 轨迹** | 不评（T2I） | DOT tracking | CoTracker + RAFT + 背景差公式 `delta_p_rel = delta_p_fg - delta_p_bg` |
| **对照检验** | 无 | static_copy | static_copy + random_motion + global_filter + camera_pan_cheat + **subject_swap_inverse 配对检验** |
| **VLM 采样** | 单次 | 未明说 | **vlm_repeat = 3 取均**（详见 §4.6） |
| **总分公式** | 各维度独立 | 加权平均 | **执行门控乘法** `S = E·(0.6P + 0.4C)` |
| **人评验收** | 无明确门槛 | Spearman 报呈 | 每维度 ≥ 50 样本且 Spearman ρ ≥ 0.5（详见 §4.9） |

---

## 1. 推荐目录结构（Phase 3 部分）

```text
i2v_compbench/
  configs/
    phase3.yaml
    dimensions.yaml
    evaluator_weights.yaml   # ★ 冻结本次评测器版本的权重清单（hash + path），见 §A
  data/
    benchmark_dataset/    # Phase 2 输出，作为 Phase 3 输入
  src/
    i2vcompbench/
      phase3/
        generate_videos.py
        preprocess_videos.py
        extract_tool_features.py   # ★ 新增：一次性产出 grounding / tracking / depth / identity
        identity_binding.py        # ★ 新增：multi_image 参考图 ↔ 视频主体匹配
        evaluators/
          attribute.py
          action.py
          motion.py
          spatial.py
          background.py
          view.py
          interaction.py
        scoring.py
        baselines.py              # 独立 QA 流程，不在主评测主线
        aggregate.py
        human_validation.py
        report.py
      schemas/
        phase3.py                 # BenchmarkSample / EvalResult / FailureMode / ToolFeatures
      utils/
        io.py
        video.py
        geometry.py
        vlm.py
        preservation.py           # ★ 新增：P 维度共享子项（identity / non-target / background drift）
        coherence.py              # ★ 新增：C 维度共享子项（frame-yes-ratio / signal-std）
  runs/
    eval/
```

### 1.1 直接复用 T2V-CompBench / T2I-CompBench 的代码模块

以下模块**直接 fork 论文开源代码**，不重造。改造点用 `# CHANGED:` 标注；论文原行号引自 [PROJECT_ANALYSIS.md](../PROJECT_ANALYSIS.md) §9。

| 我们的模块 | T2V-CompBench 源文件 | 函数 + 行号 | 改造点 |
|---|---|---|---|
| `evaluators/spatial.py` 2D 几何 | `compbench_eval_spatial_relationships.py` | `spatial_judge()` L222、`pick_max_2d()` L259、`calculate_iou()` L173、`filter_box()` L200 | bbox 格式从 `[xc,yc,w,h]` → `[x1,y1,x2,y2]`；按 §4.5.4 加 IoU<0.1 阈值 |
| `evaluators/spatial.py` 3D 深度 | 同上 | `pick_max_3d()` L295 | Depth backbone 由项目方在 §A 指定；**保留 `depth ↑ ⇔ 距相机 ↑ 近` 的反直觉语义**（KILLER-1） |
| `evaluators/spatial.py` 帧 → 视频 | 同上 | `combine_frame()` L304、`combine_csv_and_cal_model_score()` L357 | 在原非线性映射外加 `tool_uncertain` 出口（KILLER-3 双轨） |
| `evaluators/motion.py` 前景/背景分割 | `compbench_motion_binding_seg.py` | `foreground_background_mask()` L261、`save_mask_foreground()` L250、`save_mask_data()` L220 | SAM 版本由项目方在 §A 指定 |
| `evaluators/motion.py` 跟踪与评分 | `compbench_eval_motion_binding.py` | `foreground()` L632、`background()` L498、`combine_fore_back()` L304、`object_score()` L347 | **去掉硬编码 `W=856, H=480`**，改读 video metadata（PROJECT_ANALYSIS §10.4 已指出此技术债） |
| `evaluators/motion.py` fps=8 重采样 | `compbench_motion_binding_seg.py` | `Video_preprocess.convert_video_to_standard_video()` | DOT 训练分布在 fps=8，必须重采样（KILLER-2） |
| `evaluators/motion.py` 最终调整 | `compbench_eval_motion_binding.py` | `model_score()` L450 | 兼容路径：`score >= 0 → score*0.8 + 0.2` 写进 §4.10 兼容 CSV |
| `utils/video.py` Video_preprocess | `compbench_run_depth.py` L17 / `compbench_motion_binding_seg.py` L17 | `extract_frames(num_frames=16)`、`convert_video_to_grid(num_image=6)`、`merge_grid()`、`convert_video_to_standard_video()` | 两文件中代码完全重复，我们抽到 utils 仅 fork 一次 |
| `extract_tool_features.py` GroundingDINO 调用 | `compbench_eval_spatial_relationships.py` | `load_model()` L54、`get_grounding_output()` L65 | backbone 由项目方在 §A 指定；阈值见 §4.3 |
| `evaluators/*` MLLM 拆问 | `LLaVA/llava/eval/compbench_eval_consistent_attr.py` 等 4 个脚本 | 全文件 | **整体替换为项目方指定的 MLLM**，但保留：3-seed 多次采样、3 轮对话顺序（描述 → 物体1 → 物体2）、A/B/C/D 评级映射 |
| `aggregate.py` Spearman 相关 | T2V-CompBench README 报告流程 | — | 我们额外把它前置为「评测器门控」（§4.9） |

---

## 2. 全局枚举与边界规则（Phase 3 必须遵守）

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

### 2.3 Motion / Spatial 评测边界

- Motion evaluator：处理位移类（Type A 绝对方向、Type B 相对位移、Type C 多图轨迹）。
- Spatial evaluator：**不评估"移动过程"**。如果题目出现移动过程，应归 Motion。

### 2.4 Action / Interaction 评测边界

- Action evaluator：单主体肢体动作。
- Interaction evaluator：多主体因果/社交/功能事件。Interaction 自动评测置信度通常较低，**应默认进入更高比例 human validation**。

### 2.5 FAILURE_MODES 枚举（evaluator 输出只能从本表取值）

```python
FAILURE_MODES = [
    "static_copy",        # 视频几乎等于首帧，没执行变化
    "global_filter",      # 整图调色/滤镜，未做局部变化
    "camera_pan_cheat",   # 整体平移冒充主体运动
    "object_missing",     # 目标主体未出现
    "wrong_attribute",    # 属性错（颜色/形状/材质/状态）
    "wrong_direction",    # 运动方向错
    "wrong_relation",     # 空间关系错
    "wrong_camera",       # 镜头运动错（zoom/pan/tilt 类型或方向错）
    "identity_lost",      # 主体身份漂（变成另一只猫/人）
    "non_target_drift",   # 非目标主体被动了
    "background_drift",   # 背景被改写
    "artifact_severe",    # 严重失真 / 闪烁 / 结构坍塌
    "timing_wrong",       # 因果时序错（交互题专用）
    "identity_unbound",   # multi_image 参考图与视频主体未能绑定
    "tool_uncertain",     # 工具失败——与上面正交，不代表模型失败
]
```

`failure_diagnostics.json` 按本枚举聚合；coding agent 不允许写表外自由字符串。

### 2.6 BenchmarkSample schema（Phase 2 输出 → Phase 3 消费）

```json
{
  "question_id": "spatial_0042",
  "dimension": "spatial_composition",
  "input_mode": "multi_image",
  "first_frame_path": "first_frames/spatial_0042.jpg",
  "input_image_paths": ["ref_images/spatial_0042_ref0.jpg", "ref_images/spatial_0042_ref1.jpg"],
  "prompt": "A cat sits on a sofa.",
  "target_subjects": [
    {"id": "s1", "noun": "cat",  "ref_image_idx": 0},
    {"id": "s2", "noun": "sofa", "ref_image_idx": 1}
  ],
  "target_relation": {"type": "spatial", "value": "on", "subj": "s1", "obj": "s2"},
  "preservation_set": ["s2.appearance", "background.global"],
  "contrastive_pair_id": "spatial_pair_0021",
  "contrastive_role": "original",   // or "inverse"
  "evaluator_tools": ["grounding", "depth", "vlm_existence"],
  "expected_failure_modes": ["wrong_relation", "object_missing"],
  "subtype": "static_relation",
  "difficulty": "medium",
  "semantic_rarity": "common",
  "source_type": "tip_i2v_real"
}
```

**硬约束**：

- `target_subjects` / `target_relation` / `preservation_set` 是三个必须字段。evaluator **不允许从 prompt 猜实体与关系**，只能读这三个结构化字段。
- `evaluator_tools` 不是提示，是**调度列表**：extract_tool_features 只跳它里面的工具，节省 GPU。
- `contrastive_pair_id` 非空时，original/inverse 两条 sample 必须同时评测才能计 `paired_E_diff`（§4.5.4）。

---

## 3. Phase 3 输入与输出

### 3.1 输入

```text
data/benchmark_dataset/
  phase3_manifest.jsonl       # 每行一条 BenchmarkSample（schema 见 §2.6）
  first_frames/               # 首帧图，与 question_id 对齐
  ref_images/                 # multi_image 参考图
model_adapter config
```

**字段完整性验收**：Phase 3 启动时必须先验证 manifest 每行含 §2.6 列出的全部必须字段，缺字段直接 abort。

### 3.2 输出

```text
runs/eval/{model_name}/
  generations/
  normalized_videos/
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

---

## 4. 模块详细规格

### 4.1 Module: `generate_videos.py`

功能：

- 调用待评 I2V 模型。
- 输入单图或多图 + prompt。
- 保存 mp4 与 generation metadata。

接口建议：

```python
class ModelAdapter:
    def generate(self, sample: BenchmarkSample) -> GeneratedVideo:
        ...
```

`GeneratedVideo`：

```json
{
  "question_id": "attr_0001",
  "model_name": "ModelX",
  "video_path": "runs/eval/ModelX/generations/attr_0001_seed42.mp4",
  "status": "success",
  "generation_params": {
    "seed": 42,
    "num_frames": 81,
    "fps": 16,
    "resolution": "720p"
  },
  "errors": []
}
```

**生成参数依据**：

- `num_frames=81 / fps=16` 对齐当前主流 I2V 模型默认 5s@16fps 输出。模型原生输出不同时，以原生为准，由 §4.2 统一重采样。
- `resolution=720p` 仅为默认；evaluator 底层几何（§4.5.4）**不允许硬编码分辨率**，必须读 video metadata。

**多种子策略**（应对 I2V 随机性）：

- 默认 `gen_seed_count = 1`（seed=42），控制评测成本。
- 论文主结果表必须跑 `gen_seed_count = 3`（seed=1/2/3），**最终分取 mean(S)**，同时报 std 体现模型稳定性；不允许 best-of-3（避免「挑最好」嫌疑）。
- `EvalResult` 纪录 `gen_seed`；aggregate 阶段同时输出按 seed 分组的 std。

验收：

- generation failure 要记录，不得中断全局评测。
- 多图模型不支持时，记录 `unsupported_input_mode`，**不允许静默退化为单图输入**。

### 4.2 Module: `preprocess_videos.py`

功能：

- 统一 fps、帧数、分辨率→ `normalized_videos/main/`。
- **额外产出 fps=8 标准化副本**（motion evaluator 专用，KILLER-2）→ `normalized_videos/standard_fps8/`。
  - 重采样公式：`model_frames = total_frame / fps * 8`（fork T2V-CompBench `convert_video_to_standard_video()`）。
  - 原因：DOT 训练分布在 fps=8，fps 偏离会让跟踪位移数值口径变化。
- 抽帧：默认 `extract_frames(num_frames=16)`（对齐 T2V-CompBench）。
- **输入图归一化**：首帧 / 参考图 同时 resize 到与 normalized_videos 同分辨率，以便 P 维度逐帧比对。
- 记录视频基础质量（闪烁 / 黑帧 / 均值亮度等）。

输出：

```json
{
  "question_id": "attr_0001",
  "normalized_video_path": "normalized_videos/main/attr_0001.mp4",
  "standard_fps8_video_path": "normalized_videos/standard_fps8/attr_0001.mp4",
  "frames_dir": "frames/attr_0001/",
  "first_frame_normalized_path": "normalized_videos/first_frames/attr_0001.jpg",
  "ref_images_normalized_paths": ["normalized_videos/ref/attr_0001_ref0.jpg"],
  "num_frames": 81,
  "fps": 16,
  "resolution": [1280, 720]
}
```

### 4.3 Module: §Tool features 上游依赖详规

不单独设 `grounding.py` / `tracking.py` 主调，所有调用交给 §4.4.5 的 `extract_tool_features.py` 一次性产出，evaluator 只读不算（避免同一视频被 GroundingDINO 跑 4-7 次）。本小节仅定义默认参数。

```yaml
# configs/phase3.yaml 必须显式定义，evaluator 不允许硬编码
grounding:
  box_threshold: 0.35       # 对齐 T2V-CompBench
  text_threshold: 0.25
  iou_threshold_2d: 0.9     # 2D bbox 去重阈值
  iou_threshold_3d: 0.95    # 3D bbox 去重阈值
  backbone: "<在 §A 填>"   # GroundingDINO 版本由项目方决定
sam:
  backbone: "<在 §A 填>"
depth:
  backbone: "<在 §A 填>"
  semantics: "value_higher_is_closer"   # ★ 必须显式说明（KILLER-1）
tracking:
  primary: "DOT"            # 主，fork T2V-CompBench
  fallback: "<在 §A 填>"     # 备：CoTracker / 其他
flow:
  backbone: "RAFT"          # 仅用于背景差 delta_p_bg
vlm:
  backbone: "<在 §A 填>"     # 项目方后续确定
  vlm_repeat: 3             # §4.6 详述
```

### 4.4 其他底层工具说明

- **跟踪主算法为 DOT**（T2V-CompBench Table 3 报告 DOT 在密集轨迹上 Spearman 高于 CoTracker）；CoTracker 仅作为 fallback 或单点稀疏跟踪。
- **Depth 语义反直觉**（KILLER-1）：Depth Anything 系列输出「值越大 = 离相机越近」。下游所有 `in front of / behind` 判定必须遵守该语义（详见 §4.5.4）。

### 4.4.5 Module: `extract_tool_features.py`（新增、evaluator 上游唯一入口）

功能：为每条 sample 一次性产出下游所需的全部工具特征，evaluator 只读。

输出 schema：

```json
{
  "question_id": "spatial_0042",
  "grounding": {
    "per_frame": [
      {"frame_idx": 0, "subjects": [{"id": "s1", "bbox": [x1,y1,x2,y2], "conf": 0.91}, ...]}
    ],
    "first_frame": {...},
    "ref_images": [{...}]
  },
  "tracking": {
    "tubes": [{"id": "s1", "trajectory": [[x,y], ...], "conf": 0.85}],
    "backend": "DOT"
  },
  "depth": {
    "per_frame_path": "depth/spatial_0042/",
    "semantics": "value_higher_is_closer"
  },
  "identity": {
    "ref_to_video": [
      {"ref_idx": 0, "matched_subject_id": "s1", "identity_match_score": 0.78}
    ]
  },
  "tool_status": {
    "grounding": "valid",
    "tracking": "valid",
    "depth": "valid",
    "identity": "valid"
  }
}
```

调度规则：

- 必须根据 sample.evaluator_tools 按需调用，不在列表里的工具跳过（节省 GPU）。
- multi_image 样本必须额外调 `identity_binding.py`（§4.4.6）。
- 任一子工具失败 → `tool_status.<tool> = "tool_uncertain"`，**不开 E=0**。

### 4.4.6 Module: `identity_binding.py`（新增、multi_image 专用）

功能：将参考图中的主体与生成视频中的主体进行身份匹配，输出 `identity_match_score ∈ [0, 1]`。

算法：

```text
for ref in input_image_paths:
    ref_feat = DINOv2(ref) ⊕ CLIP(ref)        # 拼接特征
    for subject in video_grounded_subjects:
        sub_feat = mean over tube frames of DINOv2(crop) ⊕ CLIP(crop)
        sim = cosine(ref_feat, sub_feat)
    最佳匹配 → identity_match_score
```

判定与下游影响：

- `identity_match_score ≥ 0.7`：身份绑定成功，E 不受惩罚。
- `0.5 ≤ score < 0.7`：软惩罚，E *= 0.7。
- `score < 0.5`：视为 `identity_unbound`（§2.5），**E *= 0.5** 且 failure_modes 加该项。
- 所有分数进入 `tool_confidence.identity`，供 aggregate 诊断。

验收：

- 对于同一题，每个 `target_subjects[*].ref_image_idx` 绑定不能冲突（一个 video subject 不允许被两个 ref 同时绑上）。
- single_image 样本跳过本模块。

### 4.5 Module: per-dimension evaluators

**总说：Disentangled VQA 拆问机制（所有使用 VLM 的 evaluator 必须遵循）**

T2I-CompBench 指出：直接问 VLM "Does this video match the prompt 'a red apple to the left of a green book'?" 会让模型把多个判定压成一个趋向于说 "yes" 的平均判断，丢失边界信息。本框架所有 VLM 问题必须拆成三类独立问题，每题独立调用、独立记分：

1. **Object existence 问题**（Yes/No 二值）：“Is there a red apple in the video? Yes/No”、“Is there a green book in the video? Yes/No”——验证所有提及主体都出现。
2. **Attribute / Action / State 问题**（A/B/C/D 四档评级）：“How well does the apple match the color 'red'? A: perfect / B: mostly / C: partially / D: not at all”——验证属性绑定与动作发生程度。
3. **Relation / Composition 问题**（Yes/No 二值）：“Is the apple to the left of the book? Yes/No”，仅在前两类问题都过后才问。

**评分规则**：

```text
E_existence ∈ {0, 1}      # 任一主体 No 则 0
E_attr_score ∈ {1.0, 0.67, 0.33, 0}  # A/B/C/D 映射
E_relation ∈ {0, 1}

E = E_existence × E_attr_score × E_relation
```

该机制遵循 T2I-CompBench Disentangled BLIP-VQA 设计、及 T2V-CompBench D-LLaVA 拆问设计。A/B/C/D 评级映射对齐 T2V-CompBench `compbench_eval_consistent_attr.py` 的评级设计。

**3 轮对话 priming 顺序**（fork T2V-CompBench，对小模型尤其重要）：

```text
轮 1：让 VLM 描述视频内容一句话（进入视觉上下文，不记分）
轮 2：问物体 1 的存在 / 属性 / 动作
轮 3：问物体 2 的存在 / 属性 / 动作
三轮在同一个会话内完成（保持上下文）。
```

Relation 问题可追加在轮 3 后，仅当轮 2/3 都过时才问。

**VLM 输入模式分流**（平衡成本与时序敏感度）：

| 维度 | 输入模式 | 说明 |
|---|---|---|
| Attribute / Background | **Grid 拼图**（3×2，6 帧） | fork T2V-CompBench `convert_video_to_grid(num_image=6)`，单次调用省成本。适用粗粒度静态判定 |
| Action / Motion / Interaction | **原生帧序列**（16 帧） | 需要时序信息，由 MLLM 原生视频 token 输入（如 backbone 不支持 16 帧原生输入，退化为 4×4 拼图 + frame index annotation） |
| Spatial / View | **末帧单图** | 空间终态判定不需时序，View 镜头判断亦以首+末帧对比为主 |

3×2 拼图规格与 T2V-CompBench `compbench_eval_consistent_attr.py` 严格对齐（论文已 ablate）；16 帧输入与抽帧仅仅是拼图的剩余帧，gridding 代码可复用。

**VLM 多次采样**：每个子问题调用 `vlm_repeat=3` 次（详见 §4.6 评分规则），平均后软阈值化（Yes/No 问题 ≥ 0.5 计为 Yes；A/B/C/D 问题取众数，3 次互不一致走 low_confidence）。

**multi_image 身份绑定硬门**：evaluator 执行前必须检查 `tool_features.identity.identity_match_score`：

- score < 0.5：failure_modes 加 `identity_unbound`，**E *= 0.5** 后再进入本 evaluator 后续判定。
- 0.5 ≤ score < 0.7：E *= 0.7。
- score ≥ 0.7：E 不受惩罚。

每个 evaluator 输入（★ 入参变更）：

```python
evaluate(
    sample,              # BenchmarkSample（§2.6）
    video,               # 预处理后的 main 版本视频 + standard_fps8 副本
    input_images,        # ★ 首帧 + 参考图（multi_image 为 list）——不传不能计 P
    tool_features,       # ★ extract_tool_features 产出（grounding/tracking/depth/identity）
) -> EvalResult
```

`EvalResult` schema：

```json
{
  "question_id": "motion_0001",
  "model_name": "ModelX",
  "evaluator_version": "v1.0.0",
  "dimension": "motion_binding",
  "input_mode": "single_image",
  "gen_seed": 42,
  "E": 0.76,
  "P": 0.63,
  "C": 0.82,
  "S": 0.59,
  "aggregation": "exec_gated",
  "lambda": 0.6,
  "tool_status": "valid",
  "tool_confidence": {
    "grounding": 0.92,
    "tracking": 0.85,
    "flow": 0.79,
    "vlm": null,
    "vlm_std": 0.12,
    "identity": 0.81
  },
  "failure_modes": [],
  "paired_E_diff": null,
  "auxiliary": {
    "direction_correct": true,
    "relative_displacement": 0.12
  }
}
```

`evaluator_version` 字段记录本次评测使用的评测器快照版本（对应 `configs/evaluator_weights.yaml`），仅同一版本下的分数可跨论文比较。

#### 4.5.1 Attribute evaluator

E（颜色 / 形状 / 材质 / 状态 拆问）：

- 目标区域属性是否达到目标值：**Disentangled VQA 为主**，HSV/Lab 仅验证次。
- multi-image 属性是否来自正确 reference：依赖 identity_binding。

P：

- **非目标主体属性变化幅度以 HSV/Lab 距离为主**，VLM 仅在 HSV 边界态走仲裁。
- 目标主体身份和未指定属性保持（DINOv2/CLIP）。
- 背景变化幅度（调用 `utils/preservation.py`）。

C（具体公式）：

```text
C = (#frames where attribute_VLM_yes) / total_frames     # 主：fork T2V-CompBench consistent_attribute
或者
C = 1 - std(HSV_color_distance_to_target_per_frame) / max_signal
两者乘积：C = C_vlm_yes_ratio × C_signal_stable
```

**颜色判定双轨明确分工**（T2I-CompBench 经验：HSV 在阴影/反光下完全失败，VQA 远为最准）：

```text
E（是否变为目标颜色）：Disentangled VQA "is the apple red?" 主，HSV 验证次
P（其他主体颜色变化）：HSV/Lab 距离主，VLM 仅在 HSV 边界态仲裁
C（颜色稳定不闪烁）：HSV 帧间 std
```

工具：

- SAM2 / grounding。
- HSV / Lab / brightness / texture stats。
- DINOv2 / CLIP feature similarity。
- 结构化 VLM 多选题（3 轮 priming + A/B/C/D 评级）。

#### 4.5.2 Action evaluator

E：

- tube-level 动作证据。
- 动作持续时间达到阈值。

P：

- 非目标主体动作泄漏。
- 动作互换。
- 身份保持。

C：

- 动作过程完整、连续。

工具：

- DWPose / ViTPose。
- 原子动作规则或 action classifier。
- tube-level pose sequence。

#### 4.5.3 Motion evaluator

包含 Type A / B / C。

**Step 1：fps=8 重采样（§4.2 已产出）**（KILLER-2）

```text
model_frames = total_frame / fps * 8       # fork T2V-CompBench convert_video_to_standard_video()
例：原视频 81 帧 / 16fps → 提取 81/16*8 = 40 帧，重组为 fps=8 视频。
```

原因：DOT 训练分布在 fps=8，fps 偏离会让跟踪位移数值口径变化，T2V-CompBench 报告同。下游跟踪一律读 `standard_fps8_video_path`。

**Step 2：GroundingDINO + SAM 在第 1 帧分割前景 / 背景 → DOT 跟踪 → 净运动向量**

E：

- Type A：相对背景位移方向正确，位移超过阈值。
- Type B：目标主体移动后达到目标相对关系。
- Type C：多图参考主体在场景中沿目标轨迹运动。

P：

- 背景拖动低（调 `utils/preservation.background_drift`）。
- 非目标主体位移低。
- 无 camera pan cheat（背景全局光流 vs 主体净运动比例阈值）。

C（具体公式）：

```text
C_traj = 1 - mean(轨迹二阶差 | per frame) / max_signal
C_dir  = (#frames where instantaneous direction ∈ ±15° 于目标) / total_frames
C = C_traj × C_dir
```

核心公式：

```text
delta_p_rel = delta_p_foreground - delta_p_background
坐标系：图像坐标系（x→右、y→下）
net_left = back_x - fore_x      # 正值 = 主体向左
net_up   = back_y - fore_y      # 正值 = 主体向上
```

**帧 → 视频评分（双轨输出、KILLER-3）**：

```text
# 轨道 A：fork T2V-CompBench combine_frame() L304 + model_score() L450（仅作论文兼容 CSV 输出，详见 §4.10）
frame_score ∈ {-2, -1, 0, >0}：
  -2 → 0      # 两主体都未检测到
  -1 → 0.2    # 一个未检测到
   0 → 0.4    # 检测到但方向错
  >0 → frame_score * 0.6 + 0.4
video_score = mean(frame_scores)
final_score = video_score * 0.8 + 0.2 if video_score >= 0 else 0
# → 仅写入 T2V-CompBench 兼容 CSV（§4.10），不进 E/P/C 主分。

# 轨道 B：我们的主分
E_per_frame ∈ [0, 1]    # 方向判定 + 位移阈值软评分
E = mean(E_per_frame for frame in valid_frames)
  + tool_uncertain 出口：若 valid_frame_ratio < 0.7，tool_status=tool_uncertain, S=null
```

**路线选择**：主分走轨道 B（与 `tool_uncertain` 机制一致，更干净）；轨道 A 仅作为 T2V-CompBench leaderboard 兼容 CSV 输出供横比。

工具：

- **DOT (Dense Optical Tracking)** # 主：fork T2V-CompBench `compbench_eval_motion_binding.py`
- CoTracker # 备：单点稀疏 / DOT 不可用时
- RAFT / 光流 # 仅用于背景差 delta_p_bg
- GroundingDINO / SAM2 第 1 帧分割。
- DepthAnything（状态依 §4.5.4 语义）用于 toward/away 或 front/behind。

#### 4.5.4 Spatial evaluator

E：

- 终态静态关系成立。
- 多图参考主体数量、身份和关系正确。

> ⚠️ **KILLER-1：Depth Anything 语义方向反直觉**
>
> Depth Anything（v1/v2 系列）输出**值越大 = 离相机越近**，与「深度=距离」的常识相反。下游所有 `in front of / behind / closer / farther` 判定必须遵守该语义。`configs/phase3.yaml` 已写死 `depth.semantics: "value_higher_is_closer"`，evaluator 必须读这个开关而不是凭直觉。

**判定公式**（遵循 T2I-CompBench UniDet 几何设计 + T2V-CompBench `spatial_judge()` L222 / `pick_max_2d()` L259 / `pick_max_3d()` L295，适配到视频末帧）：

```text
# 设主体 A bbox = (xA1, yA1, xA2, yA2)、主体 B bbox = (xB1, yB1, xB2, yB2)
# 中心点 cA = ((xA1+xA2)/2, (yA1+yA2)/2)，cB 同理

"A is to the left of B":
  cA.x < cB.x
  AND |cA.x - cB.x| > |cA.y - cB.y|       # 水平主走向
  AND IoU(bboxA, bboxB) < 0.1               # 避免重叠造成伪判定

"A is above B":
  cA.y < cB.y
  AND |cA.y - cB.y| > |cA.x - cB.x|
  AND IoU(bboxA, bboxB) < 0.1

# 3D 深度判定（KILLER-1：depth↑ ⇔ 距相机↑近）：
let depth_A = median(DepthAnything(maskA))
let depth_B = median(DepthAnything(maskB))

"A is in front of B":   depth_A > depth_B   AND  IoU(maskA, maskB) < 0.5
"A is behind B":        depth_A < depth_B   AND  IoU(maskA, maskB) < 0.5
"A is closer than B":   depth_A > depth_B
"A is farther than B":  depth_A < depth_B
```

三个条件全部成立计 E_geom=1，任一不成立计 E_geom=0。`E = E_geom * E_existence`（后者来自 §4.5 拆问中的 object existence 子问题）。

**帧 → 视频评分（双轨输出，与 §4.5.3 一致、KILLER-3）**：

```text
# 轨道 A：fork T2V-CompBench combine_frame() L304 + combine_csv_and_cal_model_score() L357
frame_score ∈ {-2, -1, 0, >0}：
  -2 → 0     # 两主体都未检测到
  -1 → 0.2   # 一个未检测到
   0 → 0.4   # 检测到但方向错
  >0 → frame_score * 0.6 + 0.4
video_score = mean(frame_scores)
# → 仅写入 §4.10 兼容 CSV，不进 E/P/C 主分。

# 轨道 B：我们的主分（与 tool_uncertain 机制一致）
在后 50% 帧上采样 5 帧：
  E_per_frame = E_geom × E_existence ∈ {0, 1}
  E = mean(E_per_frame)
  若有 ≥ 2 帧 grounding 失败 → tool_status=tool_uncertain, S=null
```

**主分走轨道 B**；轨道 A 仅作 leaderboard 兼容 CSV 输出。

P：

- 参考主体身份/外观保持（调 `utils/preservation.identity`）。
- 背景与非目标关系保持（调 `utils/preservation.background_drift`）。

C（具体公式）：

```text
# 空间关系稳定、不闪烁、不消失（fork T2V-CompBench consistent_attribute 思路）
C_yes_ratio = (#frames where 几何判定成立) / total_sampled_frames
C_signal_stable = 1 - std(IoU(maskA, maskB) per frame) / max(IoU)   # 关系几何抖动
C = C_yes_ratio × C_signal_stable
# 在后 50% 帧上采样 5 帧；若高于 4/5 帧维持判定，C_yes_ratio = 1。
```

**Subject swap inverse 配对处理**：若样本 `contrastive_pair_id` 非空，evaluator 必须输出 `paired_E_diff = E_original - E_inverse`。Phase 3 aggregate 阶段计算模型在该维度上的 `swap_sensitivity = mean(max(0, paired_E_diff))`，低于 0.2 认为模型在该维度上不遵循方向/角色指令。Motion / Interaction evaluator 同样要求输出 `paired_E_diff`。

工具：

- GroundingDINO / SAM2（fork §1.1 表）。
- DepthAnything（语义见上方 ⚠️ 警告）。
- bbox / mask geometry（fork `spatial_judge()` L222）。
- CLIP / DINO identity similarity。

注意：

- 不评估"移动过程"。如果题目出现移动过程，应归 Motion。

#### 4.5.5 Background evaluator

E：

```text
E_bg = S_visible * S_localized * S_semantic
```

- 目标背景变化可见。
- 变化主要在目标背景区域。
- 语义符合 prompt（Disentangled VQA："is the sky cloudy in the video?"）。

P：

- 前景主体保持（调 `utils/preservation.identity`）。
- 非目标背景区域不过度变化（调 `utils/preservation.background_drift`，要求非目标区域 LPIPS 低于阈值）。

C（具体公式）：

```text
C_yes_ratio = (#frames where VLM_yes about target background) / total_frames
C_signal_stable = 1 - std(LPIPS(target_region_t, target_region_t-1)) / max_signal   # 渐进而非突变
C = C_yes_ratio × C_signal_stable
```

工具：

- SAM2 前景/背景 mask。
- CLIP / LPIPS / SSIM 区域变化（调 `utils/coherence.signal_stable`）。
- VLM 结构化语义判断（3×2 grid 模式，§4.5 总说）。
- anti-global-filter ratio：若整图变化与目标背景区域变化比 > 0.6 → 触发 `global_filter` failure_mode。

#### 4.5.6 View evaluator

E：

- camera type 正确：zoom / pan / tilt / static（VLM 末帧+首帧对比 + homography 验证）。
- 方向正确。
- 幅度达到阈值。

P：

- 主体身份和属性保持（调 `utils/preservation.identity`）。
- 刚性背景内容不被重写（homography 残差低于阈值）。

C（具体公式）：

```text
# 运镜平滑、无 jitter
C_jitter = 1 - mean(||H_t - H_{t-1}||_F) / max_signal     # 帧间 homography 抖动
C_smooth = 1 - std(camera_motion_magnitude_per_frame) / max_signal
C = C_jitter × C_smooth
```

工具：

- RAFT 光流。
- homography estimation（用于刚性背景拟合 + jitter 度量）。
- camera motion decomposition（zoom / pan / tilt 分解）。
- identity similarity（DINOv2 / CLIP）。

#### 4.5.7 Interaction evaluator

E：

- actor / object 正确。
- interaction event 发生。
- expected effect 成立。

P：

- 非参与主体不被卷入（调 `utils/preservation.non_target_drift`）。
- 主体身份保持（调 `utils/preservation.identity`）。
- 背景保持（调 `utils/preservation.background_drift`）。

C（具体公式）：

```text
# 因果顺序合理，无结果提前出现
C_temporal = 1 if (t_cause < t_effect) else 0     # 关键事件帧序列
C_yes_ratio = (#frames where interaction_VLM_yes after t_cause) / (#frames after t_cause)
C = C_temporal × C_yes_ratio
# t_cause / t_effect 由 grounding 接触帧 + 状态规则（knocked_over 等）联合判定。
```

**配对题输出**：交互题大量来自 subject_swap_inverse 配对（actor / patient 互换），evaluator 必须输出 `paired_E_diff`，aggregate 计 `swap_sensitivity`（§4.8）。

工具：

- 结构化 MLLM QA（16 帧原生输入模式，§4.5 总说）。
- grounding / tracking 接触或距离变化。
- 状态规则，例如 knocked_over、handed_over、opened、broken、wobble。

注意：

- Interaction 自动评测置信度通常低，应默认进入更高比例 human validation。

### 4.6 Module: `scoring.py`

实现统一评分：

```python
def exec_gated_score(E: float, P: float, C: float, lam: float = 0.6) -> float:
    return E * (lam * P + (1 - lam) * C)
```

规则：

- E/P/C 必须 clamp 到 [0, 1]。
- `tool_status=tool_uncertain` 时 S 可为 null，不进入主分（aggregate 阶段单独统计 `tool_uncertain_rate`）。
- `low_confidence` 进入主分但在报告中单独统计。
- **`P` 与 `C` 子项必须由共享 utility 计算**：所有 evaluator 的 P 子项（identity / non_target_drift / background_drift）调 `utils/preservation.py`；C 子项（frame-yes-ratio / signal-stable）调 `utils/coherence.py`，详见 §4.11。evaluator 不允许各自重写。

**VLM 多次采样与均值**（适用于所有 evaluator 中调用 VLM 的子问题）：

- 默认 `vlm_repeat = 3`，每个子问题独立调用 3 次，使用不同 `seed`（1/2/3）但保持输入同一帧；
- 二值问题的输出取平均 ∈ [0,1]，软阈值 ≥ 0.5 记为 Yes；
- 多选问题取 mode（众数），如果三次互不一致记 `vlm_inconsistent=true` 并让该项走 `low_confidence`；
- 每个子问题三次调用的标准差记在 `tool_confidence.vlm_std`，>0.3 视为 `low_confidence`。

**为什么调 3 次取均**：T2I-CompBench Table 5 与 T2V-CompBench Table 3 都报告了 VLM 在边界样本上的表现具有随机性，单次采样会拉低人评相关性。三次取均是平衡评测成本（3× API 调用）与评测稳定性的默认值，资源受限时可降到 1，但 §4.9 人评验收要求不变。

### 4.7 Module: `baselines.py`

**定位**：基线是**评测器质量审计工具**，不是被评模型。它独立于主评测主线，单独 CLI 调度（§5），跑出来的分数不写入 `per_sample_scores.jsonl`，只写入 `runs/eval/_baselines/{baseline_name}/` 子目录，供 §4.9 / §4.10 引用。

必须实现：

- `static_copy_baseline`：直接把首帧重复成静态视频。
- `random_motion_baseline`：把首帧加随机平移/缩放。
- `global_filter_baseline`：整图调色滤镜。
- `camera_pan_cheat_baseline`：整体 pan 冒充主体运动。

验收（这些是**对评测器的反向验收**，不通过则评测器有缺陷）：

- Static Copy 在需要变化的样本上 overall S < 0.10。
- Global Filter 在 Background / Attribute 中不能获得高 E/P。
- Camera Pan Cheat 不能在 Motion 中获得高分。

以上任何一条不达标，必须先修评测器再跑被评模型——否则后续模型分数失去意义。

### 4.8 Module: `aggregate.py`

输出：

```json
{
  "model_name": "ModelX",
  "evaluator_version": "v1.0.0",
  "dimension_scores": {
    "attribute_binding": {"S": 0.58, "E": 0.72, "P": 0.69, "C": 0.64, "S_std_over_seeds": 0.04}
  },
  "input_mode_scores": {
    "single_image": {"S": 0.61},
    "multi_image": {"S": 0.44}
  },
  "overall": {
    "S": 0.53,
    "E": 0.68,
    "P": 0.62,
    "C": 0.59
  },
  "tool_uncertain_rate": 0.07,
  "swap_sensitivity": {
    "spatial_composition": 0.07,
    "motion_binding": 0.15,
    "interaction_reasoning": 0.31,
    "direction_blind_dimensions": ["spatial_composition", "motion_binding"]
  }
}
```

必须按以下分组聚合：

- dimension
- input_mode
- subtype
- difficulty
- semantic_rarity
- source_type
- contrastive_pair（计 `swap_sensitivity`，详见下方规则）
- gen_seed（计 `S_std_over_seeds`）

**`tool_uncertain` 处理规则**（与 §4.6 一致，硬性写在 aggregate）：

```text
样本 S = null 时：
  1. 不参与 dimension_scores / input_mode_scores / overall 的均值
  2. 计入 tool_uncertain_rate = #(S==null) / total_samples
  3. 该样本完整保留在 per_sample_scores.jsonl，failure_modes 含 "tool_uncertain"
  4. 若某维度 tool_uncertain_rate > 0.3，该维度报告中标注 "low_coverage=true"
     （提示评测器在该维度上工具失败率过高，需补 fallback）
```

**`swap_sensitivity` 计算规则**（仅 contrastive_pair_id 非空的样本参与）：

```text
for pair in contrastive_pairs:
    paired_E_diff = E(pair.original) - E(pair.inverse)
    swap_sensitivity_per_dim = mean(max(0, paired_E_diff))

direction_blind_dimensions = [dim for dim in DIMENSIONS
                              if swap_sensitivity[dim] < 0.2]
```

论文表格里被标 `direction_blind_dimensions` 的维度建议用红字标记。

### 4.9 Module: `human_validation.py`

**定位**：human validation **前置于 aggregate**——必须先通过 Spearman ρ ≥ 0.5 的可信门控，aggregate 才被允许把该维度写入论文主表。未通过的维度只能进 supplementary。

功能：

- 抽取人工标注子集。
- 导出标注表。
- 计算自动 E/P/C 与人工 E/P/C 的 Spearman / Kendall。
- **执行 lambda ablation**：在 50 样本人评子集上扫 `lambda ∈ {0.4, 0.5, 0.6, 0.7}`，取使 Spearman ρ(S, human_S) 最大者作为最终 lambda 写回 `configs/phase3.yaml`。审稿人会问 0.6 怎么定的，这一步必须留存证据。

**抽样量与验收门槛**（遵循 T2V-CompBench Table 2 人评检验设计）：

- **每维度 ≥ 50 样本**：7 维 × 50 = 350 条起。common/rare 按 80/20 分层抽样，正确面 / 错误面 / 中间面的比例保持 1:1:2（避免人评集中间区间过稀）。人工打分 0/0.5/1 三级。
- **每样本 ≥ 3 名标注员**：取中位数为 ground truth，汇报 inter-annotator agreement（Krippendorff’s α ≥ 0.6 为可接受）。
- **Spearman ρ 验收门槛**：自动 E、P、C 与人工中位数在其维度内的 Spearman ρ 均 ≥ **0.5**，未达标的维度报告中必须标注为 `human_validation_failed=true`，且该维度评分不能进入论文主表，只能作为 supplementary。该阈值与 T2I-CompBench Table 5（0.4–0.7 的人评相关范围）中区间取下限。
- **复现性验收**：Human validation 必须使用与模型评分 lockstep 的同一抽样集（通过 `human_subset_manifest.jsonl` 固定 `question_id` 列表），避免抽样漂移。

输出：

```text
human_subset_manifest.jsonl
human_annotations_template.csv
human_correlation_report.json
```

标注字段：

- human_E
- human_P
- human_C
- failure_modes
- artifact_severity
- annotator_id

验收：

- 至少支持按维度均匀抽样。
- 支持按 artifact severity 分桶相关性。
- **Spearman ρ ≥ 0.5 作为评测器的可发表验收条件**（上述）。
- **每维度 ≥ 50 样本 × 3 名标注员 ≥ 1050 人次标注**，成本预估进报告。

### 4.10 Module: `report.py`

功能：

- 汇总 per_sample_scores、dimension_scores、input_mode_scores、failure_diagnostics、tool_uncertainty_report、human_correlation_report。
- 自动生成论文图表所需可视化与表格。
- 输出 `overall_report.json` 与 `visualizations/`。
- **导出 T2V-CompBench leaderboard 兼容 CSV**（详见下方）。

**T2V-CompBench leaderboard 兼容导出**：

```python
def export_t2v_compbench_compat_csv(model_name, dimension_scores, output_dir):
    """
    覆盖能与 T2V-CompBench 对齐的 5 个维度：
      attribute_binding / motion_binding / spatial_composition
      / action_binding / interaction_reasoning
    每个维度一个 CSV，最后一行格式严格为 "Score: {value}"，可直接打包 .zip 提交
    HuggingFace T2V-CompBench leaderboard 做横向对比。

    注意：
      - Motion / Spatial 维度的 "Score" 取 §4.5.3 / §4.5.4 的「轨道 A」非线性映射结果，
        而非我们的主分 S（兼容 T2V-CompBench combine_frame() L304 + model_score() L450）
      - Attribute / Action / Interaction 用我们主分 S 即可（T2V-CompBench 这三项亦无非线性映射）
      - 输出目录：runs/eval/{model_name}/leaderboard_compat/
          - attribute_binding.csv
          - motion_binding.csv
          - spatial_composition.csv
          - action_binding.csv
          - interaction_reasoning.csv
          - leaderboard_compat.zip
    """
```

这一项让我们的模型可以直接打 T2V-CompBench leaderboard 横比，免去审稿人「是否真的比 T2V-CompBench 更难/更好」的质疑。主分仍以 §4.6 执行门控 S 为准，本 CSV 只是兼容输出。

### 4.11 Module: 共享 utility

所有 evaluator 的 P / C 子项必须调用以下共享模块，禁止各自重写。fork 自 T2V-CompBench `consistent_attribute/` 与 T2I-CompBench `BLIPvqa_eval/` 公共逻辑。

#### `utils/preservation.py`（P 维度共享）

```python
def identity(video_subject_features, ref_features) -> float:
    """DINOv2 ⊕ CLIP cosine 相似度，对 tube 帧聚合后输出 [0,1]。"""

def non_target_drift(video, sample, tool_features) -> float:
    """非目标主体的位移 / 属性变化幅度。
    位移阈值 0.05 * frame_diag；属性以 HSV 距离 > 30 计为漂。
    返回 1 - drift_ratio ∈ [0,1]，越高越好。"""

def background_drift(video, sample, tool_features) -> float:
    """非目标背景区域的 LPIPS / SSIM 漂移。
    阈值 LPIPS > 0.3 计为 background_drift failure_mode。
    返回 1 - drift_ratio ∈ [0,1]。"""
```

#### `utils/coherence.py`（C 维度共享）

```python
def frame_yes_ratio(per_frame_yes_list) -> float:
    """#frames where target judgment is Yes / total_frames。
    fork T2V-CompBench consistent_attribute/ 公式。"""

def signal_stable(signal_per_frame, max_signal=None) -> float:
    """1 - std(signal) / max_signal，归一化到 [0,1]。
    通用稳定性度量，适用 HSV 距离 / LPIPS / IoU 等任意逐帧信号。"""

def temporal_order(t_cause, t_effect) -> int:
    """因果时序判定：t_cause < t_effect 返回 1，否则 0。Interaction 专用。"""
```

以上函数被 §4.5.1–§4.5.7 的 evaluator 反复调用，evaluator 内部不允许重写同名逻辑。

---

## 5. Phase 3 CLI 总流程

**主评测主线**（严格顺序，后一步依赖前一步产出）：

```bash
python -m i2vcompbench.phase3.generate_videos       --config configs/phase3.yaml --model ModelX
python -m i2vcompbench.phase3.preprocess_videos     --config configs/phase3.yaml --model ModelX
python -m i2vcompbench.phase3.extract_tool_features --config configs/phase3.yaml --model ModelX  # ★ §4.4.5一次性产出下游全部工具特征
python -m i2vcompbench.phase3.evaluate              --config configs/phase3.yaml --model ModelX
python -m i2vcompbench.phase3.human_validation      --config configs/phase3.yaml --model ModelX  # ★ §4.9 门控在 aggregate 之前
python -m i2vcompbench.phase3.aggregate             --config configs/phase3.yaml --model ModelX
python -m i2vcompbench.phase3.report                --config configs/phase3.yaml --model ModelX
```

**评测器审计主线**（独立于主评测，评测器发布前跑一次即可）：

```bash
python -m i2vcompbench.phase3.baselines             --config configs/phase3.yaml
# 该命令会依次调起 static_copy / random_motion / global_filter / camera_pan_cheat 四个基线，
# 输出到 runs/eval/_baselines/，验收结果写入 baselines_audit.json，不干扰主评测。
```

可以先实现一个 unified CLI：

```bash
i2vcompbench phase3 --config configs/phase3.yaml --model ModelX
```

---

## 6. Phase 3 Pilot 实现优先级

### P0：先跑通闭环

1. 定义 Pydantic/dataclass schemas（`BenchmarkSample`、`GeneratedVideo`、`ToolFeatures`、`EvalResult`）。
2. 实现 scoring 框架（exec_gated_score） + 共享 utility（§4.11 的 preservation/coherence 骨架）。
3. 实现 `extract_tool_features.py`（§4.4.5） + `identity_binding.py`（§4.4.6）骨架，评测器全部走 tool_features 不走原始调用。
4. 至少实现 Attribute / Motion / Spatial / View 四个 evaluator 的简化版本，含 KILLER-1/2/3 修正。
5. 实现 Static Copy baseline（独立 CLI），验证门控评分接近 0。

### P1：完整评测能力

1. 补齐 Action / Background / Interaction evaluator。
2. Tool confidence 全量输出。
3. Multi-image 输入 model adapter 支持。

### P2：完整评测可信度

1. Failure mode 分类。
2. Human validation 导出与相关性计算。
3. Baseline 全量实现（random_motion / global_filter / camera_pan_cheat）。
4. 论文图表自动生成。

---

## 7. Phase 3 最小验收标准

- 对至少 1 个模型或 mock model 生成 `per_sample_scores.jsonl`。
- 每条有效样本输出 E/P/C/S/tool_status/failure_modes。
- Static Copy baseline overall S < 0.10。
- `dimension_scores.json` 和 `input_mode_scores.json` 可生成。

---

## 8. Phase 3 常见错误清单

1. 不要在 Phase 3 使用 `0.45E + 0.35P + 0.20C` 线性总分。
2. 不要把 tool failure 简单记作 E=0；应标记 `tool_uncertain`。
3. 不要只保存最终总分；必须保存 E/P/C、辅助指标和失败模式。
4. 不要让 generation failure 中断全局评测——单条失败要记录后继续跑。
5. 不要让多图模型在不支持多图输入时静默退化为单图——必须记 `unsupported_input_mode`。
6. 不要让 Spatial evaluator 评估"移动过程"——位移过程应归 Motion。
7. 不要让 Interaction 仅靠自动评分得出结论——应默认进入更高比例 human validation。

---

## 9. Coding Agent 的第一步任务建议（Phase 3）

1. 创建 schemas：`BenchmarkSample`、`GeneratedVideo`、`ToolFeatures`、`EvalResult`。
2. 实现 scoring.exec_gated_score 与最小 aggregate；同步起骨 `utils/preservation.py` 与 `utils/coherence.py`。
3. 实现 `extract_tool_features.py` 与 `identity_binding.py` 骨架，mock 返回常量动起环走通。
4. 用 Static Copy mock video 跑 Phase 3，验证门控评分接近 0。
5. 接通 Attribute / Motion / Spatial / View 四个 evaluator 的简化版本（含 KILLER-1/2/3 修正），跑通 21 条 pilot 样本。

完成 Phase 3 闭环后，再接真实 VLM、I2V 模型与完整评测器，并启动 human validation 与 baselines 全量。

---

## A. 附录：环境与模型权重清单

### A.1 运行环境

```text
Python 3.11
CUDA 12.1+
PyTorch 2.3+
单 conda 环境（区别于 T2V-CompBench 需 LLaVA Py3.10 + Detection Py3.12 双环境）
GPU 需求：
  - extract_tool_features：单卡 24 GB（GroundingDINO + SAM + Depth + DOT 串行）
  - VLM：按选型决定（在 §A.2 填），72B 级调 4×A100 80GB 或 vLLM 分布式
```

### A.2 模型与权重清单（`configs/evaluator_weights.yaml`冻结）

```yaml
# 本表为占位模板，具体 backbone 与版本由项目方后续填入；
# 一旦填入，该评测器版本被冻结为 evaluator_version v1.0.0。
# 跨 evaluator_version 的分数不可直接比较。

mllm:
  name: "<项目方后续确定>"        # 需与 Disentangled VQA 拆问 + 3 轮对话 + A/B/C/D 评级兑现
  weight_path: "<填>"
  hash: "<填>"
  context_length: "<填，必须 ≥ 16 帧原生输入>"

grounding_dino:
  name: "<项目方后续确定>"        # 参考 T2V-CompBench 同款：groundingdino_swint_ogc.pth
  weight_path: "<填>"
  hash: "<填>"

sam:
  name: "<项目方后续确定>"        # SAM / SAM2
  weight_path: "<填>"
  hash: "<填>"

depth:
  name: "<项目方后续确定>"        # Depth Anything v1/v2
  weight_path: "<填>"
  hash: "<填>"
  semantics: "value_higher_is_closer"   # ⚠️ KILLER-1、不允许修改

tracking_primary:
  name: "DOT"                            # T2V-CompBench Table 3 报告 Spearman 高于 CoTracker
  weight_path: "<填，参考同款 cvo_raft_patch_8.pth>"
  hash: "<填>"

tracking_fallback:
  name: "<项目方后续确定>"        # CoTracker / 其他
  weight_path: "<填>"
  hash: "<填>"

flow:
  name: "RAFT"                           # 仅用于背景差 delta_p_bg
  weight_path: "<填>"
  hash: "<填>"

pose:
  name: "<项目方后续确定>"        # DWPose / ViTPose——Action evaluator 依赖
  weight_path: "<填>"
  hash: "<填>"

identity:
  dinov2: "<填>"
  clip: "<填>"
```

### A.3 evaluator_version 语义

- `evaluator_version` 写入每条 `EvalResult` 与 `aggregate` 输出，与 `configs/evaluator_weights.yaml` 哈希一一对应。
- 任何 backbone、阈值、VLM 提示词变化都必须递增 `evaluator_version`，避免论文表桌上出现“同名不同心”的数。
- 跨 `evaluator_version` 的分数：仅可作参考，不得直接在论文主表里并列。
