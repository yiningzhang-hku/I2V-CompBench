# I2V-CompBench Phase 3 实现需求（模型评测）

> 本文档面向 AI coding agent，是《I2V-CompBench Phase 1/2/3 实现需求（Coding Agent 版）》的 Phase 3 拆分文档。Phase 1、Phase 2 内容请参见同目录下的另外两份文档。
>
> Phase 3 的目标：消费 Phase 2 输出的 `benchmark_dataset/phase3_manifest.jsonl`，对待评 I2V 模型生成视频，并按 7 维度执行**执行门控评分** `S = E * (0.6P + 0.4C)`，输出可审计、可复现的评测结果。

---

## 0. Phase 3 总目标

```text
Phase 3: benchmark_dataset/ + model outputs -> runs/eval/
```

必须支持：

- 7 个评测维度：attribute, action, motion, spatial, background, view, interaction。
- 2 种输入模式：single_image, multi_image，且**多图属于主维度评测**，不是 stress-only。
- 统一执行门控评分公式：`S = E * (lambda * P + (1 - lambda) * C)`，默认 `lambda = 0.6`。
- 不允许使用线性总分 `0.45E + 0.35P + 0.20C`。
- 工具失败必须记 `tool_uncertain`，不要简单记 `E=0`。
- 必须保存完整 E/P/C、辅助指标、failure_modes，不能只保存最终总分。

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
  data/
    benchmark_dataset/    # Phase 2 输出，作为 Phase 3 输入
  src/
    i2vcompbench/
      phase3/
        generate_videos.py
        preprocess_videos.py
        grounding.py
        tracking.py
        evaluators/
          attribute.py
          action.py
          motion.py
          spatial.py
          background.py
          view.py
          interaction.py
        scoring.py
        baselines.py
        aggregate.py
        human_validation.py
        report.py
      schemas/
        phase3.py
      utils/
        io.py
        video.py
        geometry.py
        vlm.py
  runs/
    eval/
```

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

---

## 3. Phase 3 输入与输出

### 3.1 输入

```text
data/benchmark_dataset/phase3_manifest.jsonl
model_adapter config
```

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
  "video_path": "runs/eval/ModelX/generations/attr_0001.mp4",
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

验收：

- generation failure 要记录，不得中断全局评测。
- 多图模型不支持时，记录 `unsupported_input_mode`。

### 4.2 Module: `preprocess_videos.py`

功能：

- 统一 fps、帧数、分辨率。
- 抽帧。
- 记录视频基础质量。

输出：

```json
{
  "question_id": "attr_0001",
  "normalized_video_path": "...",
  "frames_dir": "...",
  "num_frames": 81,
  "fps": 16,
  "resolution": [1280, 720]
}
```

### 4.3 Module: `grounding.py`

功能：

- 根据 sample metadata 定位输入图像和视频帧中的目标主体。
- 支持 GroundingDINO/SAM2/VLM fallback。
- 输出每个主体的 frame-level bbox/mask。

验收：

- 每个 target_subject 输出 grounding confidence。
- grounding 失败时标记 `tool_uncertain`，**不要简单给 E=0**。

### 4.4 Module: `tracking.py`

功能：

- 使用 CoTracker / 光流 / SAM2 propagation 跟踪主体。
- 输出轨迹、mask tube、tracking confidence。

验收：

- 对 Motion/Spatial/Action/Interaction 至少输出 target tube。
- tracking 丢失比例进入 tool confidence。

### 4.5 Module: per-dimension evaluators

**总说：Disentangled VQA 拆问机制（所有使用 VLM 的 evaluator 必须遵循）**

T2I-CompBench 指出：直接问 VLM "Does this video match the prompt 'a red apple to the left of a green book'?" 会让模型把多个判定压成一个趋向于说 "yes" 的平均判断，丢失边界信息。本框架所有 VLM 问题必须拆成三类独立问题，每题独立调用、独立记分：

1. **Object existence 问题**：“Is there a red apple in the video? Yes/No”，“Is there a green book in the video? Yes/No”——验证所有提及主体都出现。
2. **Attribute / Action / State 问题**：“Is the apple red? Yes/No”、“Is the person waving? Yes/No”——验证属性绑定与动作发生。
3. **Relation / Composition 问题**：“Is the apple to the left of the book? Yes/No”，仅在前两类问题都过后才问。

**评分规则**：E = 三类问题的二进制答案乘积（任一为 No 则 E=0），P 和 C 另外多轮问。该机制遵循 T2I-CompBench Disentangled BLIP-VQA 设计、及 T2V-CompBench D-LLaVA 拆问设计。

**VLM 多次采样**：每个子问题调用 `vlm_repeat=3` 次（详见 §4.6 评分规则），平均后软阈值化（≥ 0.5 计为 Yes）。

每个 evaluator 输入：

```python
evaluate(sample, video, grounding_result, tracking_result) -> EvalResult
```

`EvalResult` schema：

```json
{
  "question_id": "motion_0001",
  "model_name": "ModelX",
  "dimension": "motion_binding",
  "input_mode": "single_image",
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
    "vlm": null
  },
  "failure_modes": [],
  "auxiliary": {
    "direction_correct": true,
    "relative_displacement": 0.12
  }
}
```

#### 4.5.1 Attribute evaluator

E：

- 目标区域属性是否达到目标值。
- multi-image 属性是否来自正确 reference。

P：

- 非目标主体属性变化幅度。
- 目标主体身份和未指定属性保持。
- 背景变化幅度。

C：

- 属性曲线稳定，无闪烁和回跳。

工具：

- SAM2 / grounding。
- HSV / Lab / brightness / texture stats。
- CLIP / DINO feature similarity。
- 结构化 VLM 多选题。

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

E：

- Type A：相对背景位移方向正确，位移超过阈值。
- Type B：目标主体移动后达到目标相对关系。
- Type C：多图参考主体在场景中沿目标轨迹运动。

P：

- 背景拖动低。
- 非目标主体位移低。
- 无 camera pan cheat。

C：

- 轨迹平滑，无瞬移。

核心公式：

```text
delta_p_rel = delta_p_foreground - delta_p_background
```

工具：

- CoTracker。
- RAFT / 光流。
- GroundingDINO / SAM2。
- DepthAnything 用于 toward/away 或 front/behind。

#### 4.5.4 Spatial evaluator

E：

- 终态静态关系成立。
- 多图参考主体数量、身份和关系正确。

**判定公式**（遵循 T2I-CompBench UniDet 几何设计，适配到视频末帧）：

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

"A is in front of B" / "A is behind B":
  DepthAnything(maskA).median  vs  DepthAnything(maskB).median
  AND IoU(maskA, maskB) < 0.5
```

三个条件全部成立计 E_geom=1，任一不成立计 E_geom=0。`E = E_geom * E_existence`（后者来自 §4.5 拆问中的 object existence 子问题）。

P：

- 参考主体身份/外观保持。
- 背景与非目标关系保持。

C：

- 空间关系稳定，不闪烁、不消失。
- 在后 50% 帧上采样 5 帧，高于 4/5 帧维持判定公式计 C=1，否则按命中比例线性按分。

**Subject swap inverse 配对处理**：若样本 `contrastive_pair_id` 非空，evaluator 必须输出 `paired_E_diff = E_original - E_inverse`。Phase 3 aggregate 阶段计算模型在该维度上的 `swap_sensitivity = mean(max(0, paired_E_diff))`，低于 0.2 认为模型在该维度上不遵循方向/角色指令。Motion / Interaction evaluator 同样要求输出 `paired_E_diff`。

工具：

- GroundingDINO / SAM2。
- DepthAnything。
- bbox / mask geometry。
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
- 语义符合 prompt。

P：

- 前景主体保持。
- 非目标背景区域不过度变化。

C：

- 渐进变化，无突变。

工具：

- SAM2 前景/背景 mask。
- CLIP / LPIPS / SSIM 区域变化。
- VLM 结构化语义判断。
- anti-global-filter ratio。

#### 4.5.6 View evaluator

E：

- camera type 正确：zoom / pan / tilt / static。
- 方向正确。
- 幅度达到阈值。

P：

- 主体身份和属性保持。
- 刚性背景内容不被重写。

C：

- 运镜平滑，无 jitter。

工具：

- RAFT 光流。
- homography estimation。
- camera motion decomposition。
- identity similarity。

#### 4.5.7 Interaction evaluator

E：

- actor / object 正确。
- interaction event 发生。
- expected effect 成立。

P：

- 非参与主体不被卷入。
- 主体身份保持。
- 背景保持。

C：

- 因果顺序合理，无结果提前出现。

工具：

- 结构化 MLLM QA。
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
- `tool_status=tool_uncertain` 时 S 可为 null，不进入主分。
- `low_confidence` 进入主分但在报告中单独统计。

**VLM 多次采样与均值**（适用于所有 evaluator 中调用 VLM 的子问题）：

- 默认 `vlm_repeat = 3`，每个子问题独立调用 3 次，使用不同 `seed`（1/2/3）但保持输入同一帧；
- 二值问题的输出取平均 ∈ [0,1]，软阈值 ≥ 0.5 记为 Yes；
- 多选问题取 mode（众数），如果三次互不一致记 `vlm_inconsistent=true` 并让该项走 `low_confidence`；
- 每个子问题三次调用的标准差记在 `tool_confidence.vlm_std`，>0.3 视为 `low_confidence`。

**为什么调 3 次取均**：T2I-CompBench Table 5 与 T2V-CompBench Table 3 都报告了 VLM 在边界样本上的表现具有随机性，单次采样会拉低人评相关性。三次取均是平衡评测成本（3× API 调用）与评测稳定性的默认值，资源受限时可降到 1，但 §4.9 人评验收要求不变。

### 4.7 Module: `baselines.py`

必须实现：

- `static_copy_baseline`
- `random_motion_baseline`
- `global_filter_baseline`
- `camera_pan_cheat_baseline`

验收：

- Static Copy 在需要变化的样本上 overall S < 0.10。
- Global Filter 在 Background/Attribute 中不能获得高 E/P。
- Camera Pan Cheat 不能在 Motion 中获得高分。

### 4.8 Module: `aggregate.py`

输出：

```json
{
  "model_name": "ModelX",
  "dimension_scores": {
    "attribute_binding": {"S": 0.58, "E": 0.72, "P": 0.69, "C": 0.64}
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
  "tool_uncertain_rate": 0.07
}
```

必须按以下分组聚合：

- dimension
- input_mode
- subtype
- difficulty
- semantic_rarity
- source_type
- contrastive_pair

### 4.9 Module: `human_validation.py`

功能：

- 抽取人工标注子集。
- 导出标注表。
- 计算自动 E/P/C 与人工 E/P/C 的 Spearman / Kendall。

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

---

## 5. Phase 3 CLI 总流程

```bash
python -m i2vcompbench.phase3.generate_videos --config configs/phase3.yaml --model ModelX
python -m i2vcompbench.phase3.preprocess_videos --config configs/phase3.yaml --model ModelX
python -m i2vcompbench.phase3.evaluate --config configs/phase3.yaml --model ModelX
python -m i2vcompbench.phase3.baselines --config configs/phase3.yaml
python -m i2vcompbench.phase3.aggregate --config configs/phase3.yaml --model ModelX
python -m i2vcompbench.phase3.human_validation --config configs/phase3.yaml --model ModelX
python -m i2vcompbench.phase3.report --config configs/phase3.yaml --model ModelX
```

可以先实现一个 unified CLI：

```bash
i2vcompbench phase3 --config configs/phase3.yaml --model ModelX
```

---

## 6. Phase 3 Pilot 实现优先级

### P0：先跑通闭环

1. 定义 Pydantic/dataclass schemas（`GeneratedVideo`、`EvalResult`）。
2. 实现 scoring 框架（exec_gated_score）。
3. 至少实现 Attribute / Motion / Spatial / View 四个 evaluator 的简化版本。
4. 实现 Static Copy baseline，验证门控评分接近 0。

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

1. 创建 schemas：`GeneratedVideo`、`EvalResult`。
2. 实现 scoring.exec_gated_score 与最小 aggregate。
3. 用 Static Copy mock video 跑 Phase 3，验证门控评分接近 0。
4. 接通 Attribute / Motion / Spatial / View 四个 evaluator 的简化版本，跑通 21 条 pilot 样本。

完成 Phase 3 闭环后，再接真实 VLM、I2V 模型与完整评测器，并启动 human validation 与 baselines 全量。
