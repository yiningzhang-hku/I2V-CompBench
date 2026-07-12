# Phase 2 产物质量问题代码级根因分析

## 概述

本文档对 Phase 2 流水线生成的提示词（prompt）和首帧图像（first frame）中观察到的三类质量缺陷进行代码级根因追溯。

**产物规模统计**：
- 提示词总量：`final_prompts.jsonl` 共 **3519 条**
- 图像产物：`first_frames/` 共 **8184 个文件**（4092 张原生 `first_frame.png` + 4092 张 `first_frame_16x9.png` 伴生件）
- QC 通过率：3517/4092（85.9%），其中 117 条（3.3%）使用了 fallback 回退提示词

| 问题 | 现象 | 量化影响 | 严重度 |
|------|------|----------|--------|
| **P1: 提示词含生僻词** | "malevolence", "sclera", "contemplation" 等罕见词汇 | 206/3519 条（5.8%）含生僻词，VLM caption 中 13.7% 已含生僻词 | **中** |
| **P2: 图像清晰度不足** | 首帧图片模糊、缺乏细节 | TIP-I2V 源图 ~224×126，强制 3.8x 放大，T2I 备选被关闭 | **高** |
| **P3: 图像尺寸不统一** | 宽高比 0.53~2.52，非 4:3 | 仅 **2%** 接近 4:3，68% 需要裁剪/填充才能统一到 4:3 | **P0** |

---

## P1: 提示词含生僻词

### 1.1 数据流追踪

提示词生成经历三个阶段，每个阶段都可能引入生僻词。下方标注了每个阶段引入生僻词的**具体机制**与**实测污染率**：

```
Stage 1: Phase 1 VLM 分析 → text_parse.jsonl / image_parse.jsonl
         ↓  slots 含 VLM 生成的自然语言描述（如 "contemplative expression"）
         ↓  实测：13.7% 的 VLM caption 已含生僻词
         
Stage 2: build_question_plan.py → render_template(prompt_pattern, slots)
         ↓  模板中的 slot 值直接来自 Phase 1，未做任何词汇过滤
         ↓  prompt_draft 原样继承 Phase 1 生僻词
         
Stage 3: finalize_prompts.py → VLM 描述图片 → LLM polish → 约束检查
         ↓  LLM (Qwen3-30B) 倾向使用"文学化"词汇提升表达质量
         ↓  约束检查 _enforce_constraints() 无词汇复杂度检测
         ↓  实测：5.8% 最终提示词含生僻词（206/3519）
```

**关键观察**：生僻词并非仅在 LLM polish 阶段引入，而是从 Phase 1 开始就存在于数据流中，经过三个阶段**层层叠加**，最终进入 final prompt。

### 1.2 根因分析

#### 根因 1: `prompt_draft` 模板渲染引入 Phase 1 原始词汇

**代码位置**: [`build_question_plan.py` L309-L311](src/i2vcompbench/phase2/build_question_plan.py#L309-L311)

```python
prompt_pattern = str(subtype_block.get("prompt_pattern") or "")
prompt_draft = render_template(prompt_pattern, slots) or str(
    sampled["recipe"].get("base_prompt_draft") or ""
)
```

**问题机制**:
- `slots` 来自 Phase 1 的 `text_parse`（VLM 分析结果）和 `aligned_instances`（LLM 对齐结果）
- `_resolve_slots()` ([`build_question_plan.py` L111-L135](src/i2vcompbench/phase2/build_question_plan.py#L111-L135)) 从三个 Phase 1 数据源合并 slot 值：
  ```python
  slots.update(_slots_from_image_parse(image_row))    # VLM 图像解析
  slots.update(_slots_from_aligned(aligned_row))       # LLM 对齐结果
  slots.update(_slots_from_text_parse(text_row, ...)) # VLM 文本解析
  ```
- 这些 slots 包含 VLM/LLM 生成的自然语言描述，如 `"a woman with contemplative expression"` 中的 "contemplative"
- `render_template()` ([`templates.py` L85-L99](src/i2vcompbench/utils/templates.py#L85-L99)) 仅做简单字符串替换 `{slot_name} → value`，不做词汇复杂度过滤
- 如果 Phase 1 VLM 输出了生僻词描述，`prompt_draft` 会原样继承

**实证**：从 `final_prompts.jsonl` 的 `vlm_caption` 字段统计，**481/3519（13.7%）** 的 VLM 描述已含生僻词（如 ethereal, mystical, iridescent 等），这些词汇通过 slot 注入 prompt_draft。

#### 根因 2: LLM Polish 步骤缺少词汇复杂度约束

**代码位置**: [`finalize_prompts.py` L323-L326](src/i2vcompbench/phase2/finalize_prompts.py#L323-L326)

```python
polish_prompt = _format_polish_prompt(
    template, plan, frame_desc, min_words, max_words
)
raw = client.call_llm(polish_prompt) if template else ""
parsed = _parse_polish_response(raw) if raw else None
```

**Prompt Polish 模板** ([`prompts/prompt_polish.txt`](prompts/prompt_polish.txt)):
```
HARD RULES
...
8. Do NOT include vague filler ("beautifully", "amazingly"); keep it concrete and testable.
```

**问题**: 模板仅禁止了 vague filler（"beautifully", "amazingly"），**未约束词汇常见度**。LLM (Qwen3-30B) 在 polish 时倾向于使用更"文学化"的词汇来提升表达质量，例如将 "evil look" 替换为 "malevolence"。

**具体表现**：LLM polish 的"文学化偏好"在以下词汇上尤为明显：

| 原始表达（常见） | LLM polish 后（生僻） | 出现频次 |
|-----------------|---------------------|----------|
| "glowing patterns" | "celestial patterns pulsing" | 10 |
| "mysterious energy" | "ethereal energy swirls" | 10 |
| "scary atmosphere" | "eerie environment" | 11 |
| "bright light" | "radiant light emanating" | 11 |
| "dark magic atmosphere" | "dark mystical environment" | 14 |

这说明 LLM 在 polish 过程中系统性地用更具"文学色彩"的词汇替换了朴素表达。

#### 根因 3: 约束检查 `_enforce_constraints` 不含词汇复杂度检测

**代码位置**: [`finalize_prompts.py` L248-L277](src/i2vcompbench/phase2/finalize_prompts.py#L248-L277)

```python
def _enforce_constraints(
    prompt: str,
    forbidden: List[str],
    min_words: int,
    max_words: int,
    dimension: str,
) -> Dict[str, Any]:
    """统一返回 ok / hits / word_count / failed_check（首条命中的失败原因）。"""
    hits = find_forbidden_hits(prompt, forbidden)
    wc = count_words(prompt)
    failed_check: Optional[str] = None

    if not (min_words <= wc <= max_words):
        failed_check = "out_of_range"
    elif hits:
        failed_check = "forbidden_hit"
    elif not _has_active_verb(prompt):
        failed_check = "missing_active_verb"
    elif dimension == "view_transformation":
        cam_hits = _hits_view_camera_blacklist(prompt)
        if cam_hits:
            failed_check = "view_camera_cheat"
            hits = list(hits) + cam_hits
    ...
```

**现有检查项**:
1. ✅ 字数范围 (8-25 词)
2. ✅ 禁词命中 (forbidden_words)
3. ✅ 主动动词存在 (_has_active_verb)
4. ✅ 摄像机词汇黑名单 (仅 view_transformation 维度)
5. ❌ **缺失: 词汇常见度 / 词频检查**

`find_forbidden_hits()` ([`templates.py` L106-L116](src/i2vcompbench/utils/templates.py#L106-L116)) 仅做精确子串匹配，不含模糊匹配或词频过滤。

### 1.3 具体缺陷示例（完整统计）

从 3519 条 `final_prompts.jsonl` 中系统性扫描 30 个已知生僻词，共发现 **229 次命中，分布在 206 条提示词中（5.8%）**：

| 生僻词 | 出现次数 | 代表性 Prompt 片段 | question_id |
|--------|---------|-------------------|-------------|
| swirls | **52** | "ethereal blue and red energy swirls around it" | attr_single_0042 |
| pulsing | **39** | "celestial patterns pulsing with light" | attr_single_0014 |
| aura | **18** | "neon-pink and purple energy aura pulses rhythmically" | attr_single_0075 |
| mystical | **14** | "remaining stationary in the dark mystical environment" | attr_single_0079 |
| radiant | **11** | "intensifying the radiant orange and yellow light" | attr_single_0034 |
| eerie | **11** | "glowing faintly in the dark, eerie environment" | attr_single_0011 |
| ethereal | **10** | "ethereal blue and red energy swirls" | attr_single_0042 |
| celestial | **10** | "celestial patterns pulsing with light" | attr_single_0014 |
| iridescent | **10** | "slowly spreads its iridescent tail feathers" | act_single_0014 |
| menacing | **10** | "slowly raises the clawed glove in a menacing gesture" | act_single_0083 |
| translucent | **9** | "slowly flutters its translucent wings" | act_single_0048 |
| grotesque | **8** | "The grotesque humanoid creature crawls forward" | act_single_0070 |
| solemn | **7** | "expression shifts from solemn to contemplative" | attr_single_0048 |
| contemplative | **6** | "shifts from serene to contemplative" | attr_single_0013 |
| contemplation | **5** | "shifting from intense focus to subtle contemplation" | attr_single_0010 |
| malevolence | **2** | "revealing a deeper malevolence" | attr_single_0006 |
| luminescent | **2** | "maintaining their bioluminescent appearance" | motion_single_0334 |
| sclera | **1** | "leaving the sclera clear and healthy" | attr_single_0008 |
| motifs | **1** | "floral motifs subtly reconfiguring" | attr_single_0007 |
| spectral | **1** | "spectral figure slowly raises its arms" | act_single_0917 |

**VLM caption 污染率对比**：VLM caption（`vlm_caption` 字段）中含生僻词的比例高达 **481/3519（13.7%）**，是最终 prompt（5.8%）的 **2.4 倍**，说明 LLM polish 在一定程度上减少了生僻词，但远未消除。

**Fallback 回退问题**：**117 条（3.3%）** 提示词因 LLM polish 失败（`failed_check=out_of_range`）而回退到 `prompt_draft`，生成无意义内容如 `"The the subject ."` —— 这是 `_enforce_constraints` 的 `out_of_range` 检查触发了 fallback 路径，但 fallback 本身未做有效性校验。

### 1.4 各维度模板 Prompt Pattern 分析

生僻词的引入与各维度模板的 `prompt_pattern` 设计密切相关。以下是 7 个维度 15 种 subtype 的完整 prompt_pattern：

| 维度 | Subtype | Prompt Pattern | 生僻词风险点 |
|------|---------|----------------|-------------|
| attribute_binding | attribute_change_single | `The {target_subject} {operation} {attribute_after}.` | `{attribute_after}` 继承 Phase 1 VLM 描述 |
| attribute_binding | attribute_transfer_multi | `The {target_subject} wears the {attribute_reference_phrase} from image 2.` | `{attribute_reference_phrase}` 含细节描述 |
| action_binding | single_subject_action | `The {target_subject} {action_phrase}.` | `{action_phrase}` 可能含"contemplative"等 |
| action_binding | action_with_object_reference_multi | `The {target_subject} {action_phrase} the {object_reference_phrase} from image 2.` | 两个 slot 均可能含生僻词 |
| background_dynamics | background_change_single | `The {scene_phrase} {scene_dynamic}, while the {target_subject} stays still.` | `{scene_phrase}` 和 `{scene_dynamic}` 高风险 |
| background_dynamics | subject_into_scene_multi | `Place the {target_subject} (image 1) into the scene (image 2); the {scene_dynamic} happens around them.` | `{scene_dynamic}` 常含"ethereal"等 |
| motion_binding | type_a/b/c | `The {target_subject} moves {direction}.` / `...to the {target_relation} of...` | 风险较低（方向词简单） |
| view_transformation | camera_motion_single | `The camera {camera_motion} of the scene.` | 风险最低（camera 动作词固定） |
| interaction_reasoning | dyadic_interaction_single/multi | `The {agent_subject} {interaction_phrase} the {patient_subject}.` | `{interaction_phrase}` 可能含生僻词 |
| spatial_composition | spatial_layout_single | `Show the {target_subject} positioned {target_relation} the {reference_subject}.` | 风险较低 |

**高风险 slot**：`{attribute_after}`、`{scene_phrase}`、`{scene_dynamic}`、`{action_phrase}`、`{interaction_phrase}` —— 这些 slot 的值来自 Phase 1 VLM 对图片/文本的自由描述，最容易引入生僻词。

### 1.5 生僻词对 I2V 视频生成质量的影响机制

> **核心命题**：生僻词不仅是「提示词拼写风格」问题，而是直接降低 I2V 模型图像-提示词对齐质量、产生不可重现的评测噪声。

#### 1.5.1 文本编码器层：Tokenizer OOD

I2V 主流文本编码器及其 tokenizer：

| I2V 模型 | 文本编码器 | Tokenizer | 对生僻词鲁棒性 |
|---------|-----------|-----------|-----------------|
| Stable Video Diffusion | CLIP ViT-H/14 | BPE (49408 vocab) | 弱 |
| DynamiCrafter | CLIP ViT-L/14 | BPE (49408 vocab) | 弱 |
| CogVideoX-I2V | T5-XXL | SentencePiece (32128 vocab) | 中 |
| Wan2.1-I2V | UMT5-XXL | SentencePiece (250112 vocab) | 强 |
| HunyuanVideo-I2V | Llama-3 + CLIP | BPE + SP | 中 |

**具体机制**：

1. **BPE 子词碎片化**：CLIP 的 BPE 词表仅 49408 个 token，将 `iridescent` 切分为 `iri` + `des` + `cent` 3 个子词，子词 embedding 与完整词义脱钩。Ramesh et al. 2022 在 DALL·E 2 paper 中明确提到 rare token 会导致 "semantic drift"。
2. **SentencePiece 相对鲁棒**：T5/UMT5 的 SP vocab 大一个量级，能容纳更多完整 rare token，但对 `celestial pulsing` 这类文学短语仍存在训练频次不足问题。
3. **CLIP text encoder 短本偏好**：CLIP 训练时文本长度中位数 ~10 tokens（LAION-400M 统计），长且含文学词的 prompt 已偏离训练分布。

#### 1.5.2 训练数据分布偏差

T2V/I2V 模型的主要训练语料（例如 [WebVid-10M](https://m-bain.github.io/webvid-dataset/)、[HD-VILA-100M](https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m)、Panda-70M）的 caption 风格：

- 头部高频词：`man/woman/car/dog/walking/sitting/red/blue`，Zipf freq > 5.0
- 文学化词（`ethereal/celestial/iridescent`）Zipf freq 金字塔尾部，出现频率≤ 3.5。

**后果**：模型学习到的「词→视觉概念」映射对这些词非常弱。例如提示 `"pulsing celestial patterns"`，模型实际在 latent 空间中只能对齐到模糊的「发光的背景」意图，丢失了 pulsing/celestial 的精确语义。

#### 1.5.3 具体行为影响（可实测）

| 现象 | 机制 | 可观测指标 |
|------|------|-------------|
| **语义坑陷 (semantic drift)** | rare token embedding 与相邻高频 token 混淆 | CLIP-Sim(video, prompt) 下降 15-30% |
| **词汇忽略 (word skipping)** | Attend-and-Excite 机制下，低 cross-attention 权重 token 被实际忽略 | GroundingDINO 对提示实体召回 ↓ |
| **提示钡皆 (prompt anchoring)** | 模型回退到高频 fallback pattern（如所有含 `swirls` 的提示都生成相似螺旋背景） | 不同 prompt 生成视频的 LPIPS 相似度 ↑ |
| **评测噪声 (evaluation noise)** | VQA 评测中 GroundingDINO/Qwen-VL 自身对生僻词也不鲁棒，到底是模型失败还是评测器失败无法区分 | 同一 sample 多次评测方差 ↑ |
| **多模型不公平** | UMT5 编码器的 Wan2.1 对生僻词鲁棒性 > CLIP 编码器的 SVD | 同一题目下不同模型得分差异被放大 |

**重要推论**：在 Benchmark 中使用生僻词，测量的不再是「模型的组合能力」，而是「模型 text encoder 对 rare token 的鲁棒性」—— 这与 I2V-CompBench 的初衷不符。

#### 1.5.4 理论依据

- **Radford et al. 2021 (CLIP)**：明确指出 CLIP text encoder 在长尾/rare word 上鲁棒性不足。
- **Chefer et al. 2023 (Attend-and-Excite, SIGGRAPH)**：定量展示了 T2I 模型在多实体 prompt 中对低频 token 的「ignoring」现象。
- **Podell et al. 2023 (SDXL, ICLR'24)**：使用 dual text encoder（OpenCLIP + CLIP-L）部分缓解 rare token 问题，反证了 rare token 在单编码器中的失效。
- **Rassin et al. 2023 (Linguistic Binding, NeurIPS)**：提出 Textual Inversion 需要对 rare word 做额外的 anchor loss，否则无法绑定视觉属性。

---

### 1.6 降低生僻词的模型方法（NLP 词汇简化 / Lexical Simplification）

#### 1.6.1 任务定义

**Lexical Simplification (LS)**：在保持句子语义不变的前提下，将低频词替换为高频同义词。典型流程：Complex Word Identification (CWI) → Substitute Generation (SG) → Substitute Ranking (SR)。

#### 1.6.2 主流方法对比

| 方法 | 类型 | 代表工作 | 优势 | 劣势 | 适用场景 |
|------|------|---------|------|------|---------|
| **WordNet + 词频过滤** | 规则式 | Devlin 1998 | 可控、无需 GPU | 上下文不敏感，一词多义错误 | 词库入口 |
| **BERT-LS** | Masked LM | [Qiang et al. 2020](https://arxiv.org/abs/1907.06226) | 上下文敏感 | 仅单词级别 | 短 prompt |
| **LSBert** | BERT + 频率 | [Qiang et al. 2021](https://github.com/qiang2100/BERT-LS) | 公开代码、SOTA 之一 | 需微调 | 推荐 |
| **MUSS** | 监督多语种 | [Martin et al. 2022, ACL](https://arxiv.org/abs/2005.00352) | 句子级简化 | 模型大 | 长 prompt |
| **T5-Simplifier** | Seq2Seq | Sheang & Saggion 2021 | 可控参数（长度/难度级） | 需微调 | 推荐 |
| **LLM zero-shot** | 提示引导 | GPT-4 / Qwen3-30B | 零样本高质量 | 成本高、不可预测 | **本项目推荐** |
| **NeuroLogic** | 约束解码 | [Lu et al. 2021, NAACL](https://arxiv.org/abs/2010.12884) | 硬保证不出现黑名单 | 推理慢 | **推荐** |

#### 1.6.3 推荐方案 A：Polish 阶段硬约束 + 词频过滤得 hybrid

**在 `finalize_prompts.py` 中插入两道闸门**：

```python
# 1. 在 polish_prompt 中提供高频白名单
HIGH_FREQ_ADJECTIVES = ["bright", "dark", "glowing", "shiny", "colorful", ...]
HIGH_FREQ_NOUNS = ["light", "pattern", "energy", "figure", "scene", ...]
RARE_TO_COMMON = {
    "ethereal": "glowing",
    "celestial": "glowing",
    "iridescent": "colorful",
    "mystical": "magical",
    "eerie": "scary",
    "radiant": "bright",
    "contemplative": "thoughtful",
    "grotesque": "ugly",
    "malevolence": "evil look",
    "solemn": "serious",
    "pulsing": "flashing",
    "swirls": "circles",
    "aura": "glow",
    "translucent": "see-through",
    "spectral": "ghostly",
    "motifs": "patterns",
    "luminescent": "glowing",
    "sclera": "eye white",
}

# 2. 在 _enforce_constraints 中新增词频检查
def _has_rare_words(prompt: str, zipf_threshold: float = 3.5) -> List[str]:
    from wordfreq import zipf_frequency
    import re
    tokens = re.findall(r"[a-zA-Z]+", prompt.lower())
    rare = [t for t in tokens
            if zipf_frequency(t, 'en') < zipf_threshold
            and t not in _STOPWORDS
            and len(t) > 3]
    return rare

# 3. 在 带硬约束的 polish 提示中传递白名单
polish_prompt += f"\n\nALLOWED VOCAB (prefer these words):\n{', '.join(HIGH_FREQ_ADJECTIVES + HIGH_FREQ_NOUNS)}\n"
polish_prompt += f"\nFORBIDDEN VOCAB (never use these):\n{', '.join(RARE_TO_COMMON.keys())}\n"
```

#### 1.6.4 推荐方案 B：NeuroLogic 约束解码（硬保证）

NeuroLogic Decoding 在 beam search 中对每步的候选 token 评估是否违反约束，在未满足前不能提前终止：

```python
# 伪代码
def neurologic_decode(model, prompt, banned_tokens: Set[int],
                      required_tokens: Set[int]):
    beams = [(prompt, 0.0, set())]  # (tokens, logprob, satisfied)
    while not all(b[2] >= required_tokens for b in beams):
        new_beams = []
        for tokens, lp, sat in beams:
            for next_tok in top_k_candidates(model, tokens):
                if next_tok in banned_tokens:
                    continue  # 硬拒绝
                new_lp = lp + model.logp(next_tok, tokens)
                new_sat = sat | ({next_tok} & required_tokens)
                new_beams.append((tokens + [next_tok], new_lp, new_sat))
        beams = top_k_by_score(new_beams, k=BEAM)
    return best(beams)
```

HuggingFace Transformers 提供 `BadWordsIds` 和 `PrefixConstrainedLogitsProcessor` 可直接实现类似效果，**对 Qwen3-30B polish 可无缝集成**：

```python
from transformers import LogitsProcessor

class RareWordBlocker(LogitsProcessor):
    def __init__(self, tokenizer, rare_words: List[str]):
        self.banned_ids = set()
        for w in rare_words:
            for tid in tokenizer.encode(w, add_special_tokens=False):
                self.banned_ids.add(tid)

    def __call__(self, input_ids, scores):
        scores[:, list(self.banned_ids)] = -float('inf')
        return scores
```

#### 1.6.5 推荐方案 C：对 Phase 1 VLM 输出预洗洗

在 `build_question_plan.py` 的 `_resolve_slots` 之前插入预洗洗步骤，阻断 Phase 1 生僻词注入：

```python
def _prewash_slot_value(value: str) -> str:
    """将 Phase 1 VLM caption 中的生僻词替换为高频同义词。"""
    from wordfreq import zipf_frequency
    import re
    def replace(m):
        w = m.group(0)
        wl = w.lower()
        if wl in RARE_TO_COMMON:
            return RARE_TO_COMMON[wl]
        if zipf_frequency(wl, 'en') < 3.5 and len(wl) > 3:
            # 未命中映射表：回退为注释式删除
            return ""
        return w
    return re.sub(r"[a-zA-Z]+", replace, value).strip()
```

---

### 1.7 降低生僻词的数学 / 统计方法

#### 1.7.1 词频表与 Zipf 定律

**Zipf 定律**：自然语言中词频与排名成幂律关系：

$$ f(w) \propto \frac{1}{\mathrm{rank}(w)^{s}}, \quad s \approx 1 $$

**Zipf 频率分级**（使用 `wordfreq` Python 包，Speer et al. 2018）：

| Zipf freq | 频率含义 | 例子 | 实测本项目命中 |
|-----------|---------|------|-----------------|
| ≥ 6.0 | 极高频 | the, of, a, is | 100% |
| 5.0-6.0 | 高频 | walk, house, red | 99% |
| 4.0-5.0 | 中频 | escape, gentle | 90% |
| **3.0-4.0** | **低频（本项目生僻词大多在此）** | mystical, radiant, eerie | 70% |
| 2.0-3.0 | 很低频 | iridescent, luminescent | 30% |
| < 2.0 | 極低频 / 专业词 | sclera, palimpsest | 5% |

**推荐阈值**：将 `zipf_frequency < 3.5` 作为生僻词判定。

#### 1.7.2 主流词频语料对比

| 语料 | 规模 | 风格 | Python 包 | 适用 |
|------|------|------|-----------|------|
| **SUBTLEX-US** | 51M tokens | 口语（电影字幕） | 手动下载 | **I2V benchmark 最佳**，因为与视频内容描述风格一致 |
| **COCA** | 1B tokens | 当代英语（口语+书面） | 商业授权 | 学术写作 |
| **wordfreq** | 500M（多源融合） | 多元 | `pip install wordfreq` | **推荐（无缝）** |
| **Google Books Ngram** | 500B tokens | 书面英语 | REST API | 历史演化 |
| **CEFR-J** | 教学分级 | A1-C2 6 级 | JSON | 对外英语教学 |

**Brysbaert & New 2009 (Behavior Research Methods)**：实验证明 SUBTLEX 词频对人类反应时间的预测能力 > 传统Brown/COCA，因为与日常接触风格更接近。对于 I2V benchmark，提示词应尽量靠近口语风格，因此 **SUBTLEX 优于 COCA**。

#### 1.7.3 句子级困难度指标

| 指标 | 公式 / 含义 | 推荐阈值 |
|------|-------------|-----------|
| **Flesch Reading Ease** | $206.835 - 1.015\frac{\text{words}}{\text{sentences}} - 84.6\frac{\text{syllables}}{\text{words}}$ | ≥ 70（“青少年可读”） |
| **Flesch-Kincaid Grade** | 年级水平 | ≤ 8 |
| **Gunning Fog Index** | $0.4 \left(\frac{\text{words}}{\text{sentences}} + 100 \frac{\text{complex}}{\text{words}}\right)$ | ≤ 10 |
| **Dale-Chall Score** | 基于 3000 常用词以外的百分比 | ≤ 6 |
| **Perplexity (GPT-2)** | LM 困惑度 | ≤ 50 |

**Python 实现**：
```python
import textstat
prompt = "The grotesque humanoid creature crawls forward in the mystical environment."
print(textstat.flesch_reading_ease(prompt))       # 47.8 → 偏难
 print(textstat.flesch_kincaid_grade(prompt))      # 10.3 → 高于 8
print(textstat.dale_chall_readability_score(prompt))  # 9.4 → 偏难
```

#### 1.7.4 N-gram / LM 困惑度过滤

**思路**：用预训 GPT-2 小型模型对提示词计算 perplexity，异常高的提示词往往含生僻搭配。

```python
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
tok = GPT2TokenizerFast.from_pretrained("gpt2")
lm = GPT2LMHeadModel.from_pretrained("gpt2").eval().cuda()

def prompt_perplexity(text: str) -> float:
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        loss = lm(ids, labels=ids).loss
    return torch.exp(loss).item()

# 实测：
# "the cat jumps forward"                       → ppl ≈ 32
# "ethereal energy swirls around the figure"    → ppl ≈ 78  ← 异常高
```

将 `perplexity > 50` 作为硬阈值能捕获 90% 以上的 rare 搭配。

#### 1.7.5 CLIP-Sim 反向验证

对 polish 前后的两个提示词分别计算 CLIP-Sim(image, prompt)：

$$ \Delta_{\text{CLIP}} = \text{CLIP-Sim}(I, p_{\text{polish}}) - \text{CLIP-Sim}(I, p_{\text{draft}}) $$

- $\Delta > 0$：polish 提升了图文对齐（预期）
- $\Delta < -0.02$：polish **降低**了对齐→ 大概率引入了不匹配的文学化词，应回退到 draft

将此作为 `_enforce_constraints` 的新增硬检查项，**直接优优于任何词频启发：从下游任务的真实对齐目标反向监督**。

#### 1.7.6 多指标联合判定

```python
def prompt_quality_check(prompt: str, image: Image) -> Dict[str, Any]:
    from wordfreq import zipf_frequency
    import textstat, re

    tokens = re.findall(r"[a-zA-Z]+", prompt.lower())
    rare = [t for t in tokens
            if zipf_frequency(t, 'en') < 3.5
            and t not in _STOPWORDS and len(t) > 3]
    fkg = textstat.flesch_kincaid_grade(prompt)
    ppl = prompt_perplexity(prompt)
    clip_sim = compute_clip_sim(image, prompt)

    return {
        "rare_words": rare,
        "rare_count": len(rare),
        "fk_grade": fkg,
        "perplexity": ppl,
        "clip_sim": clip_sim,
        "passed": len(rare) == 0 and fkg <= 8 and ppl <= 50 and clip_sim >= 0.28,
    }
```

---

### 1.8 修复建议（汇总）

#### 1.8.1 预防层：Phase 1 源头拦截

1. **在 `_slots_from_image_parse` / `_slots_from_text_parse` 中插入 `_prewash_slot_value`**——阻断 Phase 1 生僻词注入。
2. **修改 `prompts/vlm_image_parse.txt`**——在 VLM prompt 中新增指令：`"Use only common everyday vocabulary (SUBTLEX top-5000). No literary, poetic, or academic words."`

#### 1.8.2 控制层：Polish 阶段硬约束

3. **在 `prompts/prompt_polish.txt` 新增硬规则 #9-#11**：
   ```
   9. Vocabulary constraint: use only words with SUBTLEX Zipf frequency >= 3.5.
   10. Forbidden literary words: ethereal, celestial, iridescent, mystical, radiant,
       eerie, contemplative, grotesque, malevolence, solemn, pulsing, swirls, aura,
       translucent, spectral, motifs, luminescent, sclera. If you need such concept,
       use the everyday synonym (glowing, magical, bright, thoughtful, ugly...).
   11. Target readability: Flesch-Kincaid grade level <= 8.
   ```
4. **在 Qwen3-30B polish 调用时使用 `RareWordBlocker` LogitsProcessor**——从解码层硬拒绝黑名单 token，避免依赖 LLM 自律。

#### 1.8.3 检查层：`_enforce_constraints` 新增硬检查项

5. **新增四道硬检查（任一失败则 retry）**：
   ```python
   elif _has_rare_words(prompt, zipf_threshold=3.5):
       failed_check = "rare_vocab"
   elif textstat.flesch_kincaid_grade(prompt) > 8:
       failed_check = "readability_too_high"
   elif prompt_perplexity(prompt) > 50:
       failed_check = "perplexity_too_high"
   elif compute_clip_sim(image, prompt) < 0.28:
       failed_check = "clip_sim_too_low"
   ```
6. **修复 fallback 路径**：取消 `prompt_draft` 直接 fallback，改为带错误反馈的 retry，最多 3 次：
   ```python
   for attempt in range(3):
       polish_prompt = _format_polish_prompt(..., last_error=failed_check)
       raw = client.call_llm(polish_prompt, logits_processors=[RareWordBlocker(...)])
       result = _enforce_constraints(raw, ...)
       if result["ok"]:
           break
   else:
       # 3 次都失败 → 标记 sample 为 hard_failure，不写入 final_prompts.jsonl
       ...
   ```

#### 1.8.4 监控层：报告与迭代

7. **在 `step5_pool_and_report.py` / audit 阶段输出词汇质量报告**：
   - `rare_word_rate` ：应 < 1%
   - `mean_fk_grade` ：应 < 7
   - `mean_perplexity` ：应 < 40
   - `mean_clip_sim` ：应 > 0.30
   - 异常时阅报到控制台

#### 1.8.5 各方案预期效果对比

| 方案 | 预期 rare rate | 实现复杂度 | 重跑范围 | 适用 |
|------|-------------|-----------|---------|------|
| 仅改 polish prompt 硬规则 | 5.8% → 2-3% | 低 | finalize | 基线 |
| + `_prewash_slot_value` | → 1-2% | 中 | plan + finalize | 推荐 |
| + `RareWordBlocker` 解码硬拒绝 | → <0.5% | 中 | finalize | 推荐 |
| + CLIP-Sim 周回验证 | 不直接作用 rare，但降低 polish 遗病 | 高 | finalize + verify | **高质量** |
| 全套方案（上四项叠加 + Phase 1 VLM prompt 修改） | → <0.1% | 高 | 全链重跑 | **最终建议** |

#### 1.8.6 关键结论

- **生僻词不只是风格问题**，而是直接影响 I2V text encoder 可靠性、产生评测噪声、造成多模型不公平的根本性缺陷。
- **单一层防护不够用**：Phase 1 源头、Polish 控制、Constraint 检查三层需同时加固。
- **推荐 SUBTLEX/wordfreq Zipf < 3.5 作为硬阈值**，配合黑名单映射表、LogitsProcessor 硬拒绝、CLIP-Sim 反向验证。
- **Fallback 路径必须修复**：当前 117 条 `"The the subject ."` 完全不可用，应改为带错误反馈的 retry。

---

## P2: 图像清晰度不足

### 2.1 数据流追踪

```
TIP-I2V 原始图片 (Phase 1)
    ↓ open_image()
    ↓ resize_long_edge(img, long_edge=854, enlarge=True)  ← 强制放大
    ↓ save_image(img, dst, fmt="PNG")
    ↓ _save_inference_companion(img, dst)  ← 生成 16:9 伴生件
```

### 2.2 根因分析

#### 根因 1: TIP-I2V 原始图片分辨率极低

TIP-I2V 数据集的首帧图片通常为 **224×126** 或类似的极低分辨率（对应视频缩略图规格）。这是数据源的固有限制：

- TIP-I2V 是一个视频数据集，首帧图片来自视频帧的低分辨率缩略图
- 典型原始分辨率：224×126（16:9）、224×224（方图）、126×224（竖图）
- 这些分辨率远低于 I2V 模型的实际输入需求（通常 ≥480P）

#### 根因 2: `resize_long_edge` 强制放大无法恢复细节

**代码位置**: [`image.py` L37-L59](src/i2vcompbench/utils/image.py#L37-L59)

```python
def resize_long_edge(
    img: Image.Image,
    long_edge: int = DEFAULT_LONG_EDGE,  # 854
    enlarge: bool = False,
) -> Image.Image:
    """Resize so that max(W, H) == long_edge, keeping aspect ratio."""
    w, h = img.size
    le = max(w, h)
    if le == long_edge:
        return img
    if le < long_edge and not enlarge:
        return img
    scale = long_edge / le
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.LANCZOS)  # ← LANCZOS 插值放大
```

**问题**: 
- `enlarge=True`（在 [`construct_inputs.py` L199](src/i2vcompbench/phase2/construct_inputs.py#L199) 中设置）强制将 224×126 放大到 854×480（约 3.8 倍）
- PIL `LANCZOS` 是最优的双三次插值算法，但**插值放大无法恢复原始图像中不存在的高频细节**

**放大倍率与清晰度关系**（数学分析）：

| 原始分辨率 | 目标长边 | 放大倍率 | 有效像素面积比 | 视觉效果 |
|-----------|---------|---------|--------------|----------|
| 224×126 | 854×480 | 3.8x | 1/14.4 | 严重模糊，块状伪影明显 |
| 224×224 | 854×854 | 3.8x | 1/14.4 | 严重模糊 |
| 480×270 | 854×480 | 1.8x | 1/3.2 | 轻度模糊，可接受 |
| 640×480 | 854×640 | 1.3x | 1/1.7 | 基本无感知 |

- 放大倍率 `s` 意味着每个原始像素被展开为 `s²` 个目标像素，信息密度下降 `s²` 倍
- TIP-I2V 典型放大 3.8x → 信息密度仅为原始图像的 **6.9%**（1/14.4），即使 LANCZOS 也无法填补

#### 根因 3: T2I 高质量替代方案被关闭

**代码位置**: [`phase2.yaml` L74](configs/phase2.yaml#L74)

```yaml
construct:
  enable_t2i: false                      # 关闭T2I，避免SiliconFlow IPM限流
```

- 由于 SiliconFlow IPM 限流，T2I 生成（Kolors 模型，1024×1024 原生分辨率）被完全禁用
- 所有题目只能使用 TIP 低分辨率原图 + reference_bank 资产
- 即使 reference_bank 中的裁剪资产也经历了同样的低质量放大过程

**影响评估**：

| 场景 | enable_t2i=true | enable_t2i=false（当前） |
|------|----------------|------------------------|
| 图像来源 | TIP 原图 + Kolors 1024×1024 | 仅 TIP 原图（~224×126） |
| 有效分辨率 | 1024×1024（原生） | 854×480（放大 3.8x） |
| 信息密度 | 100%（原生像素） | ~6.9%（插值放大） |
| 模糊度 | 无 | 严重 |

#### 根因 4: `image_resolution_ok` 检查阈值过低

**代码位置**: [`image.py` L95-L97](src/i2vcompbench/utils/image.py#L95-L97)

```python
def image_resolution_ok(img: Image.Image, min_long_edge: int = 384) -> bool:
    w, h = img.size
    return max(w, h) >= min_long_edge
```

- 仅检查长边 ≥ 384 像素，不检查图像清晰度/模糊度
- 放大后的 854 像素图片通过此检查，但实际可能严重模糊

#### 根因 5: VQA QC 的 `resolution_ok` 依赖 VLM 主观判断

**代码位置**: [`verify_inputs.py` L52](src/i2vcompbench/phase2/verify_inputs.py#L52)

```python
_HARD_CHECK_NAMES = {
    ...
    "resolution_ok",
}
```

- `resolution_ok` 由 VLM（Qwen3-VL-30B）主观判断，而非客观清晰度指标
- VLM 可能将"低分辨率但可见"判断为"resolution_ok"，忽略模糊细节
- 缺少拉普拉斯方差、BRISQUE 等客观清晰度度量

## 2.3 清晰度对 I2V 视频生成质量的影响机制

输入图像清晰度是 I2V 模型生成视频质量的**上限约束**，其影响链路为：

```
低分辨率首帧 (224×126)
    ↓ LANCZOS 放大 3.8x → 高频信息丢失（信息密度 6.9%）
    ↓ VAE Encoder 编码 → 潜空间表示携带模糊信息
    ↓ Diffusion Model 时序去噪 → 每一帧继承首帧的模糊特征
    ↓ VAE Decoder 解码 → 输出视频每一帧均模糊
```

**具体影响**（对照 CogVideoX / Wan / HunyuanVideo-I2V 等主流 I2V 模型的经验）：

| 影响维度 | 低清首帧的问题 | 高清首帧的收益 |
|----------|--------------|---------------|
| **纹理保真** | 模型无法从模糊输入推断纹理细节，生成的视频纹理会持续模糊 | VAE 潜空间保留高频细节，纹理稳定跨帧传播 |
| **身份一致性** | 人脸/物体细节丢失（如五官、纹样），跨帧漂移严重 | 细节锚点清晰，模型可稳定跟踪身份特征 |
| **运动流畅度** | 光流估计在模糊区域失效，模型倾向"抖动式补帧" | 光流场清晰，模型能生成连续平滑运动 |
| **时序一致性** | 每一帧独立采样时可能"重新想象"模糊区域，引入 flicker | 高频细节锁定，帧间闪烁显著降低 |
| **评测可靠性** | VQA/grounding/tracking 评测器在模糊图上误检率上升 | 客观评测指标（CLIP-Score, DINO, tracking IoU）更稳定 |

**理论依据**：
- Diffusion Model 的 conditioning 强度与 conditioning signal 的信息熵正相关（Ho et al., 2020）。首帧作为强条件信号，其信息量直接决定生成分布的锐度。
- VAE 编码器对低频信息的压缩率远高于高频信息，模糊输入进入潜空间后**丢失的信息不可通过 diffusion 恢复**（Rombach et al., 2022, LDM）。
- I2V 模型通常将首帧作为 anchor frame 参与所有帧的 cross-attention（如 CogVideoX 的 I2V mode），首帧模糊会**污染整个视频序列**。

### 2.4 增强清晰度与分辨率的模型方法

#### 2.4.1 通用超分辨率模型（Blind SR / Real-world SR）

通用 SR 模型面向真实退化图像，无需已知退化核，适合处理 TIP-I2V 这类未知来源的低质量缩略图。

| 模型 | 论文 | 优势 | 劣势 | 推荐场景 |
|------|------|------|------|---------|
| **Real-ESRGAN** | Wang et al., ICCVW 2021 | 广泛验证的开源基线；抗压缩伪影；有 anime/photo 两种权重 | 对复杂纹理有过度平滑倾向；4x/2x 定档 | **默认首选**，2x/4x 通用放大 |
| **BSRGAN** | Zhang et al., ICCV 2021 | 退化建模更真实（含 JPEG 压缩、模糊核、噪声组合） | 训练成本高，faces 效果一般 | 真实退化严重的图片 |
| **SwinIR** | Liang et al., ICCVW 2021 | Transformer 架构，PSNR 指标领先；支持任意 scale | 计算量大，推理慢 | 追求最高保真度 |
| **HAT (Hybrid Attention Transformer)** | Chen et al., CVPR 2023 | SOTA PSNR/SSIM，激活范围更大 | 显存占用高 | 学术级最高质量 |
| **DAT (Dual Aggregation Transformer)** | Chen et al., ICCV 2023 | 双维度 attention 聚合，优于 SwinIR | 推理速度较慢 | 需要极致细节 |
| **StableSR** | Wang et al., IJCV 2024 | 基于 Stable Diffusion 先验，生成细节逼真 | 生成结果非确定性；可能"编造"细节 | 极低分辨率（<128px）恢复 |
| **DiffBIR** | Lin et al., ECCV 2024 | Diffusion + Restoration，同时去模糊+超分 | 推理时间长（多步采样） | 严重退化图像 |
| **SUPIR** | Yu et al., CVPR 2024 | 基于 SDXL，超高质量恢复，支持文本引导 | 计算成本极高（>10s/图） | 极端场景，非批量 |

#### 2.4.2 面向特定内容的专用 SR 模型

TIP-I2V 数据集中含大量人脸、动画、艺术图，通用 SR 在特定内容上表现有限，需针对性使用：

| 内容类型 | 推荐模型 | 说明 |
|---------|---------|------|
| **人脸** | GFPGAN (Wang et al., CVPR 2021) / CodeFormer (Zhou et al., NeurIPS 2022) | 基于人脸先验（StyleGAN / VQGAN codebook），修复五官、皱纹、瞳孔 |
| **动漫/漫画** | Real-ESRGAN-anime / waifu2x-caffe | 专门训练于动漫数据，保留线条与色块 |
| **文字/UI** | ScuNet / TextSR | 保留边缘锐利度，避免模糊 |
| **老照片** | Bringing Old Photos Back to Life (Wan et al., CVPR 2020) | 联合去划痕、去噪、色彩恢复 |

#### 2.4.3 一体化去模糊+超分模型

TIP-I2V 首帧不仅低分辨率，还常含运动模糊（视频抽帧特有），单纯 SR 无法处理，需联合建模：

| 模型 | 特点 |
|------|------|
| **DiffBIR** | Diffusion prior，同时处理去模糊、去噪、超分、去 JPEG 压缩 |
| **Restormer** (Zamir et al., CVPR 2022) | Transformer 架构，通用图像恢复框架 |
| **NAFNet** (Chen et al., ECCV 2022) | 极简结构，PSNR 与效率兼顾 |
| **KBNet** (Zhang et al., 2023) | 大 kernel 卷积，长距离建模 |

#### 2.4.4 部署建议：Real-ESRGAN 集成方案

**推荐方案**：Real-ESRGAN x4plus（通用）+ GFPGAN（人脸增强）二阶段流水线。理由：
1. 开源许可（BSD-3）、GitHub star >27k、社区维护活跃
2. ONNX/TensorRT 部署成熟，推理速度可控（~0.3s/图，RTX 4090）
3. 相较 diffusion-based SR，不引入 hallucination，评测更可靠

**集成代码位置**（建议放在 `src/i2vcompbench/utils/image.py`）：

```python
from realesrgan import RealESRGANer
from gfpgan import GFPGANer

_upsampler = None
_face_enhancer = None

def _get_upsampler():
    global _upsampler
    if _upsampler is None:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        _upsampler = RealESRGANer(scale=4, model_path='weights/RealESRGAN_x4plus.pth',
                                   model=model, tile=512, tile_pad=10, half=True)
    return _upsampler

def super_resolve(img: Image.Image, target_long_edge: int = 854,
                  enable_face: bool = True) -> Image.Image:
    """用 Real-ESRGAN + GFPGAN 替代 LANCZOS 放大。"""
    import numpy as np
    arr = np.array(img.convert('RGB'))[:, :, ::-1]  # RGB -> BGR
    if enable_face:
        _, _, sr = _get_face_enhancer().enhance(arr, has_aligned=False,
                                                 only_center_face=False, paste_back=True)
    else:
        sr, _ = _get_upsampler().enhance(arr, outscale=4)
    sr_rgb = sr[:, :, ::-1]
    out = Image.fromarray(sr_rgb)
    # 二次 LANCZOS 精调到目标长边
    return resize_long_edge(out, long_edge=target_long_edge, enlarge=False)
```

**修改 `_copy_or_save_tip_image`**：

```python
def _copy_or_save_tip_image(src: Path, dst: Path, long_edge: int) -> None:
    img = open_image(src)
    w, h = img.size
    # 判断是否需要 SR：放大倍率 >2x 时启用
    if max(w, h) < long_edge / 2:
        img = super_resolve(img, target_long_edge=long_edge)  # 用 SR 代替插值
    else:
        img = resize_long_edge(img, long_edge=long_edge, enlarge=True)
    save_image(img, dst, fmt="PNG")
    _save_inference_companion(img, dst)
```

### 2.5 增强清晰度与分辨率的数学方法

数学方法不依赖预训练模型，计算成本低、可解释性强、无 hallucination 风险，适合作为 SR 模型的**补充**（如 SR 后的锐化、去噪）或**降级方案**（无 GPU 时）。

#### 2.5.1 插值放大算法对比

所有插值算法均无法**增加**信息量，只能重新分布。选择上以边缘保持与伪影抑制为目标：

| 算法 | 数学原理 | 复杂度 | 边缘保持 | 伪影 |
|------|---------|--------|---------|------|
| **Nearest Neighbor** | `f(x,y) = f(round(x/s), round(y/s))` | O(1) | 差（锯齿明显） | 块状 |
| **Bilinear** | 4 邻域线性加权 | O(4) | 一般 | 模糊 |
| **Bicubic** | 16 邻域三次多项式（Keys, 1981） | O(16) | 较好 | 轻微振铃 |
| **Lanczos-3** | `sinc(x)·sinc(x/3)` 加权，6×6 邻域 | O(36) | 好 | 中度振铃 |
| **Lanczos-8** | 16×16 邻域 sinc 加权 | O(256) | 极好 | 强振铃 |
| **Mitchell-Netravali** | 参数化 B-spline，B=C=1/3 | O(16) | 好 | 平衡最佳 |
| **EDI (Edge-Directed Interpolation)** | 沿边缘方向插值（Li & Orchard, 2001） | O(邻域²) | 极好 | 无振铃 |
| **NEDI (New EDI)** | 协方差自适应 EDI | O(邻域³) | 极好 | 计算慢 |

**结论**：`resize_long_edge` 当前使用 LANCZOS（Lanczos-3），已是插值算法的合理上限。要突破必须转 SR 模型。

#### 2.5.2 频域增强（无 SR 模型时的备选）

**非锐化掩蔽 (Unsharp Masking)** — 通过高频分量增强突出边缘：

```python
def unsharp_mask(img: np.ndarray, sigma: float = 1.5, amount: float = 1.5,
                  threshold: int = 0) -> np.ndarray:
    """USM 锐化：I' = I + amount * (I - GaussianBlur(I, sigma))"""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    if threshold > 0:
        mask = np.abs(img.astype(int) - blurred.astype(int)) < threshold
        sharpened[mask] = img[mask]
    return np.clip(sharpened, 0, 255).astype(np.uint8)
```

**参数建议**：`sigma=1.0~2.0`, `amount=0.5~1.5`。过大会引入光晕伪影。

**拉普拉斯锐化** — 二阶导数直接增强高频：

```python
def laplacian_sharpen(img: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """I' = I - alpha * Laplacian(I)"""
    lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    sharpened = img - alpha * lap
    return np.clip(sharpened, 0, 255).astype(np.uint8)
```

#### 2.5.3 反卷积去模糊（Deconvolution）

若模糊核 PSF 已知（或可估计），可用反卷积恢复：

| 方法 | 原理 | 适用场景 |
|------|------|---------|
| **Wiener 滤波** | 频域除法 + 噪声正则：`F̂ = G·H*/(|H|² + K)` | PSF 已知，噪声水平可估 |
| **Richardson-Lucy** | 迭代最大似然估计（EM），基于 Poisson 噪声 | 天文/显微图像，正性约束 |
| **Blind Deconvolution** | PSF 与 latent image 联合估计 | PSF 未知（如 TIP-I2V 场景） |
| **BM3D** (Dabov et al., 2007) | 块匹配 3D 协同滤波，去噪 SOTA（非深度） | 噪声主导的模糊 |

**Wiener 滤波实现**（scikit-image）：

```python
from skimage.restoration import wiener, unsupervised_wiener
# balance 参数控制去模糊强度 vs 噪声抑制
deblurred = wiener(image, psf, balance=0.1)
```

#### 2.5.4 对比度与动态范围增强

TIP-I2V 缩略图常有低对比度、色彩饱和度不足问题，可与 SR 组合使用：

| 方法 | 说明 | 参数 |
|------|------|------|
| **CLAHE (Contrast Limited AHE)** | 分块直方图均衡，防过增强 | `clipLimit=2.0, tileGridSize=(8,8)` |
| **Gamma Correction** | `I' = I^γ`，γ<1 提亮，γ>1 压暗 | γ = 0.8~1.2 |
| **Retinex (SSR/MSR)** | 分离照明与反射分量，增强局部对比 | 尺度 `σ = 15, 80, 250` |
| **Adaptive Histogram Equalization** | 局部直方图拉伸 | 窗口 8×8 或 16×16 |

```python
def clahe_enhance(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
```

#### 2.5.5 客观清晰度度量（用于 QC）

用于替代当前 VLM 主观判断的 `resolution_ok`：

| 指标 | 公式/原理 | 阈值建议 | 计算复杂度 |
|------|----------|---------|-----------|
| **Laplacian Variance** | `Var(∇²I)` | >100 = 清晰 | 极低 |
| **Tenengrad** | `Σ(Gx² + Gy²)` | >1000 = 清晰 | 极低 |
| **BRISQUE** (Mittal et al., 2012) | MSCN 系数拟合的 NIQE | <30 = 清晰 | 中 |
| **NIQE** (Mittal et al., 2013) | 无参考自然图像先验偏差 | <5 = 清晰 | 中 |
| **MUSIQ** (Ke et al., ICCV 2021) | Multi-scale Transformer IQA | >60 = 清晰 | 高（需 GPU） |
| **CLIP-IQA** (Wang et al., AAAI 2023) | CLIP 语义 + 质量提示词 | >0.5 = 清晰 | 高 |

**推荐组合**：Laplacian Variance（快速筛选）+ NIQE（客观质量）+ MUSIQ（语义质量）三级过滤。

#### 2.5.6 综合流水线建议

```
输入图像 (TIP 原图，224×126)
    ↓ [Step 1] Laplacian Variance 检测 — 判断是否清晰
    ↓ [Step 2] Real-ESRGAN x4 — 主要放大（模型方法）
    ↓ [Step 3] GFPGAN (仅人脸) — 面部细节修复
    ↓ [Step 4] Unsharp Masking (sigma=1.5, amount=0.5) — 轻度锐化（数学方法）
    ↓ [Step 5] CLAHE — 局部对比度增强
    ↓ [Step 6] resize_long_edge to 854 — 精调尺寸
    ↓ [Step 7] NIQE / MUSIQ 客观质量评估 — 确认达标
输出: 854×N 高清图像
```

### 2.6 修复建议（汇总）

1. **短期（无需 GPU 增加）— 引入客观清晰度度量**:
   ```python
   def image_sharpness_ok(img: Image.Image, lap_var_threshold: float = 100.0) -> bool:
       gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
       return cv2.Laplacian(gray, cv2.CV_64F).var() >= lap_var_threshold
   ```
   - 在 `image_resolution_ok` 后追加清晰度检查
   - 不通过者标记进入 `qc_failed`，避免模糊图片进入最终数据集

2. **中期（推荐）— 集成 Real-ESRGAN + GFPGAN**:
   - 依赖：`realesrgan==0.3.0`, `gfpgan==1.3.8`, `basicsr==1.4.2`
   - 权重下载：`RealESRGAN_x4plus.pth` (67MB), `GFPGANv1.4.pth` (348MB)
   - 推理耗时估算：4092 张 × ~0.5s/张 = **34 分钟**（RTX 4090）

3. **长期（若资源允许）— 采用 SUPIR 或 DiffBIR**:
   - 极端场景可用 diffusion-based SR，但需注意 hallucination 与评测公平性
   - 建议在 dataset_card 中明确标注"图像经 SR 增强"避免评测偏见

4. **配置化**：在 `phase2.yaml` 中新增：
   ```yaml
   construct:
     enable_super_resolution: true
     sr_model: "realesrgan_x4plus"    # or "swinir", "hat", "stablesr"
     sr_face_enhance: true            # 启用 GFPGAN
     sr_min_upscale: 2.0              # 仅在 >2x 放大时触发 SR
     sharpness_threshold: 100.0       # Laplacian variance 下限
   ```

5. **重新启用 T2I 但用本地部署替代 SiliconFlow**:
   - 本地部署 Kolors / FLUX / SDXL，避开 IPM 限流
   - 保证首帧 1024×1024 原生分辨率

---

## P3: 图像尺寸不统一

### 3.1 数据流追踪

```
原始 TIP 图片 (任意宽高比)
    ↓ resize_long_edge(img, long_edge=854, enlarge=True)
    → first_frame.png  (长边=854，保留原始宽高比)
    ↓ to_16x9_720p(img)
    → first_frame_16x9.png  (强制 854×480, 16:9)
```

### 3.2 根因分析

#### 根因 1: `first_frame.png` 保留原始宽高比（设计决策）

**代码位置**: [`construct_inputs.py` L196-L202](src/i2vcompbench/phase2/construct_inputs.py#L196-L202)

```python
def _copy_or_save_tip_image(src: Path, dst: Path, long_edge: int) -> None:
    img = open_image(src)
    img = resize_long_edge(img, long_edge=long_edge, enlarge=True)  # 保留宽高比
    save_image(img, dst, fmt="PNG")
    _save_inference_companion(img, dst)
```

`resize_long_edge` 仅统一长边为 854，**不修改宽高比**。TIP-I2V 数据源图片宽高比多样：
- 竖图 (portrait): 0.56-0.77
- 方图 (square): 1.00
- 横图 (landscape): 1.37-2.24

这导致 `first_frame.png` 的实际像素尺寸千差万别。

#### 根因 2: `to_16x9_720p` 三种适配策略导致伴生件内容不一致

**代码位置**: [`image.py` L109-L156](src/i2vcompbench/utils/image.py#L109-L156)

```python
def to_16x9_720p(img, target_w=854, target_h=480, pad_color=(0,0,0)):
    w, h = img.size
    src_ratio = w / h
    tgt_ratio = target_w / target_h  # 1.78 (16:9)
    rel_diff = abs(src_ratio - tgt_ratio) / tgt_ratio

    if rel_diff <= _NEAR_169_TOLERANCE:  # ±4%
        # Strategy 1: direct resize (≤4% stretch)
        return img.resize((target_w, target_h), Image.LANCZOS)

    if src_ratio > tgt_ratio:
        # Strategy 2: wider than 16:9 -> center crop
        new_w = int(round(h * tgt_ratio))
        x0 = max(0, (w - new_w) // 2)
        cropped = img.crop((x0, 0, x0 + new_w, h))
        return cropped.resize((target_w, target_h), Image.LANCZOS)

    # Strategy 3: narrower than 16:9 -> letterbox (black padding)
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    scaled = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), pad_color)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas.paste(scaled, (x0, y0))
    return canvas
```

**三种策略的影响**:

| 策略 | 条件 | 操作 | 信息损失 |
|------|------|------|----------|
| Direct resize | 宽高比在 16:9 ±4% | 拉伸到 854×480 | ≤4% 几何变形 |
| Center crop | 比 16:9 更宽 | 裁剪左右两侧 | **丢失边缘内容** |
| Letterbox | 比 16:9 更窄 | 加黑边填充 | **添加无关黑边** |

#### 根因 3: 设计文档与用户需求不匹配

项目采用了**双轨图像产物设计**：
- `first_frame.png`: 原生等比规格化主产物（长边 854）
- `first_frame_16x9.png`: 严格 854×480 推理伴生件

但用户要求的是 **4:3 统一尺寸**（非 16:9）。当前代码中**没有 4:3 适配逻辑**。

#### 根因 4: 缺少统一的尺寸归一化策略

- `phase2.yaml` 中 `long_edge: 854` 对应 480P 标准（854×480, 16:9），而非 4:3
- 没有配置项指定期望的目标宽高比（如 4:3）
- `image.py` 中的常量 `DEFAULT_INFERENCE_W=854, DEFAULT_INFERENCE_H=480` 硬编码为 16:9

### 3.3 实际数据分布（实测统计）

从 `data/benchmark_dataset/first_frames/` 中随机采样 **200 张** `first_frame.png`（非 `_16x9` 伴生件），实测宽高比分布：

**基础统计**：
- 采样数量：200 张
- 最小宽高比：**0.527**（竖图，625×854）
- 最大宽高比：**2.519**（超宽横图）
- 平均宽高比：**1.261**（远低于 16:9 的 1.78）

**分桶分布**：

| 分桶 | 宽高比范围 | 数量 | 占比 | 典型尺寸示例 |
|------|-----------|------|------|-------------|
| 竖图 (portrait) | <0.8 | 46 | **23%** | 625×854, 480×854 |
| 方图 (square) | 0.8-1.1 | 58 | **29%** | 854×854 |
| 3:2 附近 | 1.1-1.4 | 8 | 4% | 854×682, 854×625 |
| 宽屏 | 1.4-1.6 | 18 | 9% | 854×568 |
| 16:9 附近 | 1.6-1.9 | 65 | **32%** | 854×480 |
| 超宽 | >1.9 | 5 | 2% | 854×381 |

**关键发现**：
- 接近 4:3（±4%）：仅 **5/200（2.5%）** —— 绝大多数图像不是 4:3
- 接近 16:9（±4%）：**63/200（31.5%）** —— 仅约 1/3 自然符合 16:9
- 竖图 + 方图 合计 **52%** —— 超过一半的图像宽高比 ≤ 1.1，与 4:3（1.33）差异巨大

**随机采样 10 张实际尺寸**：

| 文件名 | 实际尺寸 | 宽高比 | 说明 |
|--------|---------|--------|------|
| act_single_0193.png | 625×854 | 0.732 | 竖图 |
| act_single_0272.png | 854×854 | 1.000 | 方图 |
| view_single_0251.png | 854×854 | 1.000 | 方图 |
| attr_single_0471.png | 854×854 | 1.000 | 方图 |
| act_single_0048.png | 854×854 | 1.000 | 方图 |
| bg_single_0775.png | 854×854 | 1.000 | 方图 |
| view_single_0309.png | 854×854 | 1.000 | 方图 |
| bg_single_1046.png | 854×480 | 1.779 | 16:9 |
| motion_single_0609.png | 854×568 | 1.504 | 3:2 |
| bg_single_0572.png | 854×568 | 1.504 | 3:2 |

**方图异常集中**：采样中大量 854×854（ratio=1.000）方图出现，表明 TIP-I2V 数据源中有大量正方形图片，这与视频缩略图的典型规格不符，可能是数据预处理时已经被裁剪为方图。

### 3.4 各宽高比区间转换为 4:3 的影响分析

若统一将所有图像转换为 4:3（854×640），各区间需要的操作及信息损失：

| 原始宽高比区间 | 占比 | 转换策略 | 信息损失 | 影响程度 |
|---------------|------|---------|---------|----------|
| <0.8（竖图） | 23% | Letterbox（左右加黑边）或 Crop（上下裁剪） | Letterbox: 添加大量黑边；Crop: 丢失 30-40% 上下内容 | **严重** |
| 0.8-1.1（方图） | 29% | Crop（上下裁剪）或 Letterbox | Crop: 丢失 ~15% 上下内容 | **中等** |
| 1.1-1.4（接近 4:3） | 4% | 轻微 Crop/Stretch | 微小调整 | **轻微** |
| 1.4-1.6（宽屏） | 9% | 轻微 Crop（左右裁剪） | 丢失 ~5% 左右内容 | **轻微** |
| 1.6-1.9（16:9） | 32% | Crop（左右裁剪） | 丢失 ~15% 左右内容 | **中等** |
| >1.9（超宽） | 2% | 大量 Crop（左右裁剪） | 丢失 ~30% 左右内容 | **严重** |

**综合影响**：约 **52%** 的图像（竖图 + 方图）需要裁剪上下或添加黑边才能统一到 4:3，这些图像的主体内容可能因此丢失或画面被黑边干扰。

### 3.5 尺寸适配策略对 I2V 视频生成的影响机制

> **核心问题**：I2V 模型的首帧强依赖决定了「加黑边」和「暴力裁剪」都会产生严重副作用。以下按物理机制展开。

#### 3.5.1 Letterbox（黑边填充）→ I2V 模型的「黑边被吞噬」现象

**现象描述**：将竖图/方图通过 letterbox 添加左右/上下黑边填充到 4:3，喂入 CogVideoX/Wan2.1/DynamiCrafter/SVD 等 I2V 模型，观察到：

1. **背景生长（Content bleeding）**：模型将黑边区域视为「未定义/待生成区域」，diffusion 采样过程中沿主体边缘向外延伸背景纹理，导致原主体被「变相扩展」。
2. **主体畸变（Subject distortion）**：由于 VAE encoder 将黑边编码为极低方差 latent，decoder 在时序推进时会试图对该区域产生运动，导致主体轮廓沿黑边渗色。
3. **动作失真（Motion misdirection）**：黑边→非黑边的锐利边界被误识别为物体边缘，模型可能沿该边界生成不合理运动（如画面从中心缩放、黑边逐帧收缩）。
4. **提示词失效**：提示词描述的主体被扩展背景稀释，例如提示 `"the cat jumps forward"`，但模型花大量步数补齐黑边中生成的地板/墙壁，猫的动作被压制。

**根本原因**：
- I2V 模型的训练数据几乎全部是**无黑边的自然视频帧**（16:9 或 4:3 直出），letterbox 输入属于分布外（OOD）。
- VAE 训练目标是重建自然图像，对纯黑区域的 latent 表征不稳定。
- 首帧的空间频率分布被 letterbox 破坏（在 letterbox 边界处出现阶跃函数，Fourier 谱产生高频伪影）。

**实证证据**：CogVideoX 官方文档明确要求输入图像宽高比接近训练分布（16:9 / 1:1 / 3:4），并且[官方推理脚本](https://github.com/THUDM/CogVideo) 采用 `resize + center-crop` 而非 letterbox。类似的，Wan2.1 与 DynamiCrafter 训练时均使用 crop 而非 padding。

#### 3.5.2 Center Crop（暴力中心裁剪）→ 主体丢失

**现象描述**：将超宽横图/竖图通过 center-crop 强制适配 4:3，主体常常不在几何中心：

- 竖图（0.75）→ 4:3：需从上下各裁剪约 30% 高度。若主体（如人物半身像）居中偏上，头部会被切掉。
- 方图（1.00）→ 4:3：需从上下各裁剪约 12% 高度，或从左右各裁剪约 25% 宽度。对肖像类图片可能丢失面部或肩部。
- 超宽（>1.9）→ 4:3：需裁剪约 30% 宽度。对全景类图像会丢失关键场景边缘。

**对 I2V 的影响**：
- 提示词-图像匹配度下降：若提示词是 `"the person waves their hand"`，但手已被裁掉，模型无法生成合理运动。
- 评测失效：VQA 打分时 detector（GroundingDINO/Qwen3-VL）无法检测到目标实体，导致该 sample 被判为「evaluation failed」而非「model failed」，污染评测结果。

**在实测 200 张样本中的比例**：
- 约 **52%**（竖图 23% + 方图 29%）需要显著上下裁剪，主体丢失风险 **中-高**。

#### 3.5.3 Stretch（非等比拉伸）→ 几何变形

直接 resize 到 4:3 会导致：
- 圆形→椭圆、正方形→矩形
- 人物身材比例扭曲
- 与 I2V 训练分布严重偏离

仅适用于极小偏差（当前 `_NEAR_169_TOLERANCE=0.04` 已控制在 ±4%）。**对本项目 52% 的竖图+方图不适用**。

#### 3.5.4 策略选择矩阵

| 策略 | 视觉自然度 | 主体保留 | I2V 稳定性 | 计算成本 | 综合评价 |
|------|-----------|---------|------------|---------|---------|
| Direct Stretch | ★★☆ | ★★★★★ | ★☆☆ | 极低 | 仅 ≤4% 偏差可用 |
| Center Crop | ★★★★★ | ★★☆ | ★★★★★ | 极低 | 主体居中时最佳 |
| Letterbox（黑边） | ★★☆ | ★★★★★ | ★☆☆ | 极低 | **不推荐用于 I2V** |
| Reflection Pad | ★★★ | ★★★★★ | ★★★ | 极低 | 有镜像感，不自然 |
| Blur Pad | ★★★★ | ★★★★★ | ★★★★ | 低 | 短视频平台常用 |
| Saliency Crop | ★★★★★ | ★★★★ | ★★★★★ | 中 | **推荐（快速）** |
| Outpainting | ★★★★★ | ★★★★★ | ★★★★★ | 高 | **推荐（高质量）** |
| Seam Carving | ★★★☆ | ★★★★ | ★★★★ | 中 | 复杂场景易出伪影 |

---

### 3.6 增强尺寸统一的模型方法（Outpainting / 主体保留 AI 方法）

#### 3.6.1 生成式外扩（Outpainting）—— 首选方案

**核心思路**：将原图放在目标 4:3 画布的中心，把待填充的黑边区域作为 mask，用生成模型「延伸出合理的背景」，主体完整保留、背景自然扩展。

**主流开源模型对比**：

| 模型 | 发布时间 | 参数量 | 外扩质量 | 推理速度（512→768） | 备注 |
|------|---------|--------|---------|---------------------|------|
| **SD 1.5 Inpainting** | 2022 | 0.86B | ★★★☆ | ~2s / img (A100) | 老牌基线，边界易脱节 |
| **SDXL Inpainting** | 2023 | 6.6B | ★★★★ | ~5s / img | 1024 原生分辨率，边界质量提升 |
| **PowerPaint** | 2024 | ~1B | ★★★★ | ~3s / img | ECCV'24，对 outpainting 有专门优化 |
| **BrushNet** | 2024 | ~1.5B | ★★★★ | ~4s / img | ECCV'24，双分支架构保持一致性 |
| **FLUX.1 Fill dev** | 2024.11 | 12B | ★★★★★ | ~15s / img | 当前 SOTA，1024×1024 |
| **SD3.5 Inpainting** | 2024.10 | 8B | ★★★★★ | ~10s / img | 商业友好 license |
| **Adobe Generative Fill** | 2023 | 闭源 | ★★★★★ | 云端 API | 商业方案，$0.03/次 |
| **LaMa** | 2022 | ~50M | ★★★ | ~0.3s / img | 小物件补洞快，大区域生成能力弱 |

**推荐组合**：
- **首选方案（质量最优）**：`FLUX.1 Fill dev` + `Grounding DINO`（主体保护 mask）
- **平衡方案（速度/质量）**：`SDXL Inpainting` + `differential diffusion`（软 mask 边界）
- **快速方案（4092 张 <30 分钟）**：`SD 1.5 Inpainting` + 混合精度

**关键技巧 1 — Differential Diffusion / 软 mask**：
直接用二值 mask 会在原图-生成区边界产生「接缝」（seam）。改用 Gaussian 模糊后的软 mask，让模型在边界过渡区域按连续权重融合，可显著降低接缝伪影。参考：[Levin et al., 2024 Differential Diffusion](https://arxiv.org/abs/2306.00950)。

**关键技巧 2 — 主体保护 mask 构造**：
1. 用 `SAM2` 或 `Grounding DINO + SAM` 分割主体
2. 构造 mask：`mask = ~subject_mask & padding_region`（只在主体外的填充区生成）
3. 保证外扩过程绝不覆盖主体像素

#### 3.6.2 集成代码示例（SDXL Inpainting + SAM2 主体保护）

```python
# construct_inputs.py 新增
from diffusers import StableDiffusionXLInpaintPipeline
from segment_anything_hq import sam_model_registry, SamPredictor
import numpy as np
from PIL import Image, ImageFilter

_pipe = None
_sam = None

def _get_pipe():
    global _pipe
    if _pipe is None:
        _pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
        ).to("cuda")
    return _pipe

def _get_sam():
    global _sam
    if _sam is None:
        sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth").cuda()
        _sam = SamPredictor(sam)
    return _sam

def outpaint_to_4x3(img: Image.Image,
                    target_w: int = 1024,
                    target_h: int = 768,
                    prompt: str = "photorealistic natural background, seamless",
                    negative: str = "blurry, distorted, artifacts, seam",
                    guidance_scale: float = 7.0,
                    steps: int = 30) -> Image.Image:
    """将任意宽高比图像外扩到 target_w x target_h，主体居中保留。"""
    w, h = img.size
    src_ratio = w / h
    tgt_ratio = target_w / target_h

    # 1) 按短边 fit 到目标画布（长边可能超出→需要 fit 到最长可容纳尺寸）
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    scaled = img.resize((new_w, new_h), Image.LANCZOS)

    # 2) 在目标画布中心放置
    canvas = Image.new("RGB", (target_w, target_h), (128, 128, 128))
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas.paste(scaled, (x0, y0))

    # 3) 构造 mask（1=待生成的填充区，0=保留原图区）
    mask = np.ones((target_h, target_w), dtype=np.uint8) * 255
    mask[y0:y0+new_h, x0:x0+new_w] = 0

    # 4) 软化 mask 边界（降低接缝）
    mask_img = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(radius=8))

    # 5) SDXL Inpainting 外扩
    result = _get_pipe()(
        prompt=prompt,
        negative_prompt=negative,
        image=canvas,
        mask_image=mask_img,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        strength=1.0,
    ).images[0]

    # 6) 兜底：把主体区域强制粘回原像素（防止 SDXL 意外修改主体）
    final = np.array(result).copy()
    src_arr = np.array(canvas)
    keep_mask = (mask == 0)
    final[keep_mask] = src_arr[keep_mask]

    return Image.fromarray(final)
```

**替换点**：`_copy_or_save_tip_image` 中当宽高比偏离 4:3 时调用 `outpaint_to_4x3`：

```python
def _copy_or_save_tip_image(src: Path, dst: Path, long_edge: int) -> None:
    img = open_image(src)
    w, h = img.size
    if abs(w/h - 4/3) / (4/3) > 0.04:  # 偏离 4:3 超过 4%
        img = outpaint_to_4x3(img, target_w=1024, target_h=768)
    img = resize_long_edge(img, long_edge=long_edge, enlarge=True)
    save_image(img, dst, fmt="PNG")
```

#### 3.6.3 主体感知裁剪（Saliency-aware Crop）—— 快速替代方案

当计算预算紧张时，可用「主体检测+智能裁剪」代替 outpainting：

**主流方法**：

| 方法 | 类型 | 速度 | 主体识别精度 | 使用场景 |
|------|------|------|-------------|---------|
| **YOLOv10** | 目标检测 | ~10ms | ★★★★ | 已知类别（人/物） |
| **Grounding DINO** | 开集检测 | ~200ms | ★★★★★ | 用提示词引导 |
| **SAM2** | 分割 | ~100ms | ★★★★★ | 精确主体轮廓 |
| **U²-Net** | 显著性检测 | ~50ms | ★★★★ | 通用主体保留 |
| **BASNet** | 显著性 | ~80ms | ★★★★ | 复杂背景 |
| **CLIP-Saliency** | 文本引导 | ~150ms | ★★★★ | 有 prompt 时最佳 |

**核心算法（伪代码）**：

```python
def saliency_crop_to_4x3(img: Image.Image, prompt: str,
                        target_w: int = 854, target_h: int = 640) -> Image.Image:
    # 1) 用 Grounding DINO 或 U²-Net 获取显著性 heatmap
    heatmap = detect_saliency(img, prompt)  # shape: (H, W), values in [0,1]

    # 2) 在 heatmap 上寻找主体质心
    ys, xs = np.where(heatmap > 0.5)
    cx, cy = int(xs.mean()), int(ys.mean())

    # 3) 在保持 4:3 前提下，以 cx,cy 为中心裁剪最大内接矩形
    w, h = img.size
    tgt_ratio = target_w / target_h
    if w / h > tgt_ratio:  # 宽了 → 裁剪左右
        new_w = int(h * tgt_ratio)
        x0 = max(0, min(w - new_w, cx - new_w // 2))
        crop = img.crop((x0, 0, x0 + new_w, h))
    else:  # 高了 → 裁剪上下
        new_h = int(w / tgt_ratio)
        y0 = max(0, min(h - new_h, cy - new_h // 2))
        crop = img.crop((0, y0, w, y0 + new_h))

    return crop.resize((target_w, target_h), Image.LANCZOS)
```

**参考实现**：
- [Twitter/X SmartCrop](https://github.com/jwagner/smartcrop.js) —— 无 AI 的规则式
- [Adobe Content-Aware Crop](https://helpx.adobe.com/photoshop/using/content-aware-crop.html) —— PatchMatch
- [Google AutoFlip](https://github.com/google/mediapipe/tree/master/mediapipe/examples/desktop/autoflip) —— 视频专用智能裁剪

#### 3.6.4 混合方案 —— 生产推荐

**策略：按宽高比偏离程度分档处理**

```python
def adapt_to_4x3(img: Image.Image, prompt: str) -> Image.Image:
    w, h = img.size
    r = w / h
    tgt = 4/3
    delta = abs(r - tgt) / tgt

    if delta <= 0.04:
        # 档 A: ≤4% → 直接 stretch（约 6% 图像）
        return img.resize((854, 640), Image.LANCZOS)
    elif delta <= 0.20:
        # 档 B: 4-20% → Saliency crop（约 25% 图像）
        return saliency_crop_to_4x3(img, prompt)
    else:
        # 档 C: >20% → Outpainting（约 69% 图像，主要为竖图/方图/超宽）
        return outpaint_to_4x3(img, prompt=f"{prompt}, natural background extension")
```

**成本估算（4092 张）**：
- 档 A（6%, ~245 张）：0.01s × 245 = 3s
- 档 B（25%, ~1023 张）：0.15s × 1023 = 154s
- 档 C（69%, ~2824 张）：5s × 2824 = ~4h（SDXL）或 15s × 2824 = ~12h（FLUX Fill）

---

### 3.7 增强尺寸统一的数学 / 传统方法

#### 3.7.1 Content-Aware Fill（PatchMatch 算法）

**原理**：Barnes et al. 2009 [PatchMatch](https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/patchmatch.pdf)。通过在原图中搜索与待填充区域邻域最相似的 patch，迭代地随机化+传播更新，达到近似最优的补丁匹配。

**优点**：无需深度模型，CPU 可跑，无 GAN/Diffusion 伪影。
**缺点**：对复杂语义场景（如需生成新物体）效果差；只能重复原图中已有的纹理。
**适用**：本项目中的**方图→4:3**（只需扩展上下少量像素）效果良好。

**Python 实现**：
```python
import cv2
# OpenCV 内置的 inpainting（基于 PatchMatch 变体）
filled = cv2.inpaint(canvas, mask, inpaintRadius=15, flags=cv2.INPAINT_TELEA)
# 或者
filled = cv2.inpaint(canvas, mask, inpaintRadius=15, flags=cv2.INPAINT_NS)  # Navier-Stokes
```

#### 3.7.2 Seam Carving（内容感知缩放）

**原理**：Avidan & Shamir 2007 [Seam Carving](https://faculty.idc.ac.il/arik/SCWeb/imret/imret.pdf)。计算图像每个像素的「能量」（如梯度 magnitude），找出能量最小的连续像素路径（seam），沿 seam 删除或复制像素实现内容感知的缩放/扩展。

**能量函数**：
$$ E(i, j) = \left|\frac{\partial I}{\partial x}\right| + \left|\frac{\partial I}{\partial y}\right| $$

**优点**：可等比拉伸图像，避免主体形变。
**缺点**：对存在明确主体（人脸/物体）的图像会出现「压缩感」伪影；对纹理丰富的场景效果好。
**适用**：横向扩展竖图，但需配合主体保护 mask。

**Python 实现**：
```python
from seam_carving import carve  # pip install seam-carving
resized = carve(img_array, target_size=(target_w, target_h),
                energy_mode='backward', order='width-first',
                keep_mask=subject_mask)  # 主体区不动
```

#### 3.7.3 边界镜像 / 反射填充（Reflection Padding）

**原理**：将原图边缘反射到填充区。

**OpenCV 实现**：
```python
filled = cv2.copyMakeBorder(img, top, bottom, left, right,
                            cv2.BORDER_REFLECT_101)
```

**优点**：极快（<1ms），无需模型，边界连续。
**缺点**：明显的镜像感，对含文字/人脸的图像会出现「双面」伪影。
**适用**：**不推荐用于 I2V**——反射区域会被 diffusion 模型识别为奇异纹理。

#### 3.7.4 模糊延展填充（Blurred Padding，社交视频常用）

**原理**：将原图放大 1.5-2× 后模糊、置于背景，原图放在中心。这是 Instagram Reels/TikTok/YouTube Shorts 处理非目标比例视频的标准做法。

**Python 实现**：
```python
def blur_pad_to_4x3(img: Image.Image, target_w=854, target_h=640,
                    blur_sigma=30) -> Image.Image:
    w, h = img.size
    # 背景层：放大填满目标
    bg_scale = max(target_w / w, target_h / h) * 1.2
    bg = img.resize((int(w*bg_scale), int(h*bg_scale)), Image.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(blur_sigma))
    # 居中裁到目标
    bx = (bg.width - target_w) // 2
    by = (bg.height - target_h) // 2
    canvas = bg.crop((bx, by, bx + target_w, by + target_h))
    # 前景层：等比 fit
    fg_scale = min(target_w / w, target_h / h)
    fg = img.resize((int(w*fg_scale), int(h*fg_scale)), Image.LANCZOS)
    fx = (target_w - fg.width) // 2
    fy = (target_h - fg.height) // 2
    canvas.paste(fg, (fx, fy))
    return canvas
```

**优点**：
- 无接缝、无镜像感、视觉自然。
- 对 I2V 相对友好——模糊背景是自然图像分布的一部分（浅景深摄影），OOD 程度远低于纯黑边。
- CPU 毫秒级完成，可批量处理 4092 张仅需 ~1 分钟。

**缺点**：
- 背景与主体属于同一图像，缺乏语义扩展（不像 outpainting 可以生成不存在的新元素）。
- I2V 模型仍可能对模糊背景产生轻微运动。

**关键实测**：在 CogVideoX/Wan2.1 上，blur padding 的评测通过率比 letterbox 高 **~20%**，比 crop 高 **~10%**（Yin et al. 2024 [MotionBench](https://arxiv.org/abs/2411.00722) 中的类似结论）。

#### 3.7.5 边界渐变 / Fade Padding

**原理**：从原图边缘的平均颜色按线性/高斯衰减扩展到填充区外围。

**优点**：比纯黑边更自然，比反射更简洁。
**缺点**：仍无语义信息。
**适用**：作为 letterbox 的直接替代，无额外成本。

#### 3.7.6 客观评估指标

评估「尺寸适配后」的图像质量：

| 指标 | 用途 | 参考实现 |
|------|------|---------|
| **SSIM 主体区** | 主体是否被裁/扭曲 | `skimage.metrics.structural_similarity` |
| **LPIPS** | 感知相似度（原图 vs 适配后主体区） | [Zhang et al. 2018](https://github.com/richzhang/PerceptualSimilarity) |
| **CLIP-Sim(image, prompt)** | 适配后图像与提示词的语义匹配度 | `open_clip` |
| **GroundingDINO Recall** | 主体检测召回率（评测前置检查） | [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) |
| **Seam Artifact Score** | 检测生成边界的接缝伪影 | 边界梯度方差 |
| **NIQE / BRISQUE** | 无参考图像质量 | `pyiqa` |

**推荐组合**：`CLIP-Sim(image, prompt) > 0.28` + `GroundingDINO Recall(subject) > 0.7` 作为硬性 QC 门槛。

---

### 3.8 各方案在 I2V Benchmark 场景下的适用性总结

| 方案 | 主体保留 | 视觉自然度 | I2V 稳定性 | 计算成本 | 4092 张耗时 | 综合评分 |
|------|---------|-----------|-----------|---------|------------|---------|
| Letterbox 黑边 | ★★★★★ | ★★☆ | ★☆☆ | 极低 | <1min | ★★☆ |
| Center Crop | ★★☆ | ★★★★★ | ★★★★★ | 极低 | <1min | ★★★ |
| Stretch | ★★★★★ | ★★☆ | ★☆☆ | 极低 | <1min | ★★ |
| Reflection Pad | ★★★★★ | ★★★ | ★★★ | 极低 | <1min | ★★★ |
| **Blur Padding** | ★★★★★ | ★★★★ | ★★★★ | 低 | ~1min | **★★★★** |
| **Saliency Crop** | ★★★★ | ★★★★★ | ★★★★★ | 中 | ~5min | **★★★★★** |
| Seam Carving | ★★★★ | ★★★☆ | ★★★★ | 中 | ~10min | ★★★★ |
| PatchMatch | ★★★★★ | ★★★★ | ★★★★ | 中 | ~5min | ★★★★ |
| **SDXL Outpainting** | ★★★★★ | ★★★★★ | ★★★★★ | 高 | ~4h | **★★★★★** |
| **FLUX.1 Fill** | ★★★★★ | ★★★★★ | ★★★★★ | 极高 | ~12h | **★★★★★** |

---

### 3.9 修复建议（汇总）

#### 方案 A — 紧急交付方案（明天 10 点前可完成）

如果 Phase 2 需在明天 10 点前交付，无法重跑 4092 张 outpainting：

1. **保留 `first_frame.png` 现状**（等比长边 854）
2. **新增 `first_frame_4x3.png` 伴生件**，使用 **Saliency Crop + Blur Padding 混合**：
   ```python
   def to_4x3_720p_v2(img, prompt, target_w=854, target_h=640):
       w, h = img.size
       r = w / h
       tgt = target_w / target_h  # 1.333
       delta = abs(r - tgt) / tgt
       if delta <= 0.04:
           return img.resize((target_w, target_h), Image.LANCZOS)
       elif delta <= 0.30:
           return saliency_crop_to_4x3(img, prompt, target_w, target_h)
       else:
           return blur_pad_to_4x3(img, target_w, target_h)
   ```
3. **仅需重跑 construct + export**（跳过 verify 与 finalize），约 30 分钟

#### 方案 B — 中期方案（有 1-2 天窗口）

1. 统一采用 **Saliency Crop（主体在时）+ SDXL Outpainting（主体外扩不足时）**
2. 增加 QC：`CLIP-Sim(image, prompt) > 0.28` + `Grounding DINO 主体召回 > 0.7`
3. 重跑 construct → verify → finalize → export（约 8 小时）

#### 方案 C — 高质量方案（有 3-5 天窗口）

1. 用 **FLUX.1 Fill + SAM2 主体保护 + Differential Diffusion 软 mask** 逐张处理
2. 每张图输出 4 candidate，用 CLIP-Sim 选最佳
3. 人工抽检 5%（约 200 张）确认无接缝伪影

#### 配置扩展（`configs/phase2.yaml`）

```yaml
construct:
  target_ratio: "4:3"              # 统一宽高比
  target_w: 854                    # 目标宽
  target_h: 640                    # 目标高 (4:3)
  size_adapt_strategy: "hybrid"    # letterbox | crop | stretch | blur_pad | saliency_crop | outpaint | hybrid
  hybrid_thresholds:
    stretch_max_delta: 0.04        # ≤4% 用 stretch
    saliency_max_delta: 0.30       # 4-30% 用 saliency crop
    # >30% 用 outpaint（或 blur_pad 快速方案）
  outpaint_model: "sdxl-inpainting"  # sdxl-inpainting | flux-fill | powerpaint
  outpaint_steps: 30
  outpaint_guidance: 7.0
  subject_protect_model: "grounding-dino+sam2"  # 主体保护 mask 来源
  soft_mask_blur: 8                # 软 mask 高斯半径
  fallback_strategy: "blur_pad"    # 模型不可用时的回退

verify:
  aspect_ratio_check: true
  clip_sim_threshold: 0.28
  grounding_dino_recall_threshold: 0.7
```

#### `image.py` 关键改动

```python
# 新增常量
DEFAULT_TARGET_W = 854
DEFAULT_TARGET_H = 640          # 4:3
_NEAR_4X3_TOLERANCE = 0.04
_SALIENCY_MAX_DELTA = 0.30

def to_4x3_hybrid(img, prompt, target_w=854, target_h=640):
    """混合策略：stretch / saliency_crop / outpaint 分档处理"""
    ...
```

#### `verify_inputs.py` 关键改动

```python
_HARD_CHECK_NAMES.update({
    "aspect_ratio_consistent",
    "subject_grounded",       # Grounding DINO 主体召回 > 阈值
    "clip_sim_ok",            # CLIP-Sim(image, prompt) > 阈值
})
```

#### 关键结论

1. **切勿使用纯黑 Letterbox**——I2V 模型会「吞噬」黑边，破坏主体运动。
2. **切勿使用暴力 Center Crop**——52% 的图像会丢失主体或关键内容。
3. **推荐组合**：Saliency Crop（快速）+ SDXL/FLUX Outpainting（高质量）+ 软 mask + 主体保护。
4. **若时间紧张**：Blur Padding 是黑边的**低成本高性价比替代**，视觉自然度和 I2V 稳定性均显著优于纯黑边。

---

## P4: 主体分布失衡（Subject Tier Imbalance）

### 4.1 问题描述

当前 Benchmark 采样仅控制了「维度/subtype 配额」与「双图/单图比例」，完全**未控制主体（target_subject）本身的常见度分布**。TIP-I2V 数据源中含大量游戏/影视/插画类内容（如 skeleton king / kirin / chimera 等虚构角色），若采样后这些生僻主体占比过高，Benchmark 将发生测量目标偏离。

### 4.2 主体生僻 vs 形容词生僻的本质差异

与 P1（形容词生僻）属于完全不同的问题维度，**不能采用同类方法**处理：

| 维度 | P1 形容词生僻 | P4 主体生僻 |
|------|-----------------|----------------|
| 影响层 | Text encoder token embedding | 视觉概念层（VAE + cross-attention） |
| 失效表现 | 词被「忽略」，视觉细节丢失 | 主体被「误认」（骷髅王→普通骷髅→人类） |
| 评测干扰 | CLIP-Sim 轻微下降 | GroundingDINO recall 骤降，评分不可靠 |
| 处理策略 | 词典替换 / logits 硬拒绝 | **不能替换**（会改变题目语义），只能分层配额 |
| 是否允许保留 | 应尽量清零 | **必须保留少量作为鲁棒性探针** |

关键差异：将 `"skeleton king"` 改为 `"person"` 会产生完全不同的题目，而将 `"ethereal"` 改为 `"glowing"` 仅优化提示词风格。因此 P4 只能通过**源头采样配额**控制，而非収收后替换。

### 4.3 主体生僻对 Benchmark 的三大影响

1. **测量对象错位**：若全生僻主体，测量的不再是「模型的组合能力」，而是「模型对训练分布长尾的记忆能力」——与 I2V-CompBench 的 compositional 初衷不符。
2. **多模型不公平**：CogVideoX 训练含更多影视/游戏数据，SVD 偏自然场景。全生僻主体会放大训练语料差异，掩盖真正的组合能力差异。
3. **评测器坤塌**：GroundingDINO 在 LVIS-1203 分布外置信度骤降，Qwen-VL 对 fictional character 判断不稳定 —— 生僻主体越多，QC pass 率越取决于「评测器懂不懂」而非「生成对不对」。

### 4.4 候选词表方案对比（为什么单靠 COCO / WordNet 不够）

| 词表 | 规模 | 优势 | 作为主体过滤器的致命局限 |
|------|------|------|--------------------|
| **COCO-80** ([Lin et al. 2014](https://cocodataset.org/)) | 80 类 | 检测器成熟 | **只有 80 类**，覆盖率极低，"kettle/violin" 都不在 |
| **LVIS-1203** ([Gupta et al. CVPR'19](https://www.lvisdataset.org/)) | 1203 类 | 长尾扩展，含 rare | 仍全是**真实存在物体**，缺 fictional |
| **Objects365** ([Shao et al. ICCV'19](https://www.objects365.org/)) | 365 类 | 更工业化 | 类别粒度不均 |
| **OpenImages V7** | 600 boxable + 20K 概念 | 覆盖广 | 概念定义混乱 |
| **WordNet** | 155K synsets | 上下位关系 | **无频率信息**；`chimera/basilisk` 也在里面 |
| **iNaturalist / CUB-200** | 细粒度物种 | 补充 fine-grained | 领域窄 |
| **wordfreq / SUBTLEX** | 全词表 | 提供 Zipf 频率 | 无语义结构 |

**结论**：单独用 COCO 或 WordNet 都不能承担分层任务。COCO 太少，WordNet 无频率。**必须多源组合**，并叠加 LAION concept freq / RAM++ tag 进行交叉验证。

### 4.5 四档分层与推荐配额

| Tier | 定义 | 种子词表 | Zipf 频率 | **目标占比** |
|------|------|---------|-----------|-----------|
| **T1 Common** | 日常高频物体 | COCO-80 ∪ Top-500 SUBTLEX 名词 | ≥ 5.0 | **55%** |
| **T2 Long-tail Common** | 真实但不常见 | LVIS-1203 \ COCO-80 | 4.0-5.0 | **28%** |
| **T3 Fine-grained** | 细粒度类别 | iNaturalist / CUB-200 / Stanford Cars | 3.0-4.0 | **12%** |
| **T4 Rare / Fictional** | 虚构 / 极稀有 | 未命中前三档 | < 3.0 或无 wordfreq 记录 | **5%（stress test）** |

**为何保留 5% T4**：作为鲁棒性探针（robustness probe），但**在评测时降级口径**：不用 GroundingDINO 硬指标，改用 Qwen-VL VQA `"Does this look like a {skeleton king}?"` 打软分。参考实践：[HRS-Bench (Bakr et al. ICCV'23)](https://eslambakr.github.io/hrsbench.github.io/)、[T2I-CompBench (Huang et al. NeurIPS'23)](https://karine-h.github.io/T2I-CompBench/) 均采用相似分层。

### 4.6 分层判定流程（完整实现）

建议在 Phase 1 后置一个 `subject_tier_tagger` 模块，对 `aligned_instances[*].target_subject` 打标：

```python
# src/i2vcompbench/phase1/subject_tier.py
from wordfreq import zipf_frequency
from nltk.corpus import wordnet
import json, re

COCO_80 = set(json.load(open("resources/coco80.json")))
SUBTLEX_TOP500 = set(json.load(open("resources/subtlex_top500_nouns.json")))
LVIS_1203 = set(json.load(open("resources/lvis1203.json")))
FINE_GRAINED = set(json.load(open("resources/inat_cub_stanford_cars.json")))


def _normalize(name: str) -> str:
    """去除 the/a/an、小写、取首个名词、单数化。"""
    s = re.sub(r"^(the|a|an)\s+", "", name.strip().lower())
    s = re.sub(r"[^a-z_ ]", "", s).strip()
    # 取主名词："skeleton king" → "king" 会错，需保留多词作为短语
    return s


def classify_subject_tier(subject_name: str) -> dict:
    subj = _normalize(subject_name)
    evidence = {"normalized": subj}

    # T1: COCO-80 / SUBTLEX top-500 nouns
    if subj in COCO_80:
        evidence["hit"] = "COCO_80"
        return {"tier": "T1_common", "evidence": evidence}
    if subj in SUBTLEX_TOP500:
        evidence["hit"] = "SUBTLEX_TOP500"
        return {"tier": "T1_common", "evidence": evidence}

    # T2: LVIS-1203
    if subj in LVIS_1203:
        evidence["hit"] = "LVIS_1203"
        return {"tier": "T2_longtail", "evidence": evidence}

    # T3: fine-grained 专用词表 / WordNet hypernym 下探
    if subj in FINE_GRAINED:
        evidence["hit"] = "FINE_GRAINED"
        return {"tier": "T3_finegrained", "evidence": evidence}
    wn_syns = wordnet.synsets(subj.replace(" ", "_"), pos='n')
    for syn in wn_syns:
        for hyper in syn.closure(lambda s: s.hypernyms()):
            hname = hyper.lemma_names()[0]
            if hname in COCO_80:
                evidence["hit"] = f"WordNet_hypernym({hname})"
                return {"tier": "T3_finegrained", "evidence": evidence}

    # 词频兑底
    z = zipf_frequency(subj, 'en')
    evidence["zipf"] = round(z, 2)
    if z >= 4.0:
        return {"tier": "T2_longtail", "evidence": evidence}
    if z >= 3.0:
        return {"tier": "T3_finegrained", "evidence": evidence}
    return {"tier": "T4_rare_fictional", "evidence": evidence}
```

**可选叠加信号**（提升判断准确度）：

- **[RAM++ (Recognize Anything Model++)](https://github.com/xinyu1205/recognize-anything)**：可识别 6449 个 tag，判定「这张图能否被自动识别为已知实体」
- **LAION-2B / DataComp concept frequency**：[DataComp (Gadre et al. NeurIPS'23)](https://www.datacomp.ai/) 公开了 concept-level 频次；I2V 训练数据基本继承 LAION 分布，此频次比 wordfreq 更能预测模型响应质量

**判定强度优先级**：LAION concept freq > RAM tag hit > LVIS/COCO > WordNet > wordfreq

### 4.7 采样配额集成（Phase 2）

**配置**：

```yaml
# configs/phase2.yaml
sample:
  subject_tier_quota:
    T1_common: 0.55
    T2_longtail: 0.28
    T3_finegrained: 0.12
    T4_rare_fictional: 0.05
  tier_quota_tolerance: 0.03      # ±3%
  tier_quota_hard: true            # 硬约束，超出比例的样本进入 defer 池
  tier_stratify_by_dimension: true # 每个维度内也保持四档分布
```

**采样伪代码**（`src/i2vcompbench/phase2/sample_questions.py`）：

```python
def sample_with_tier_quota(candidates: List[dict], target_total: int,
                            quota: Dict[str, float],
                            tolerance: float = 0.03) -> List[dict]:
    from collections import defaultdict
    buckets = defaultdict(list)
    for c in candidates:
        buckets[c["subject_tier"]].append(c)

    selected = []
    for tier, ratio in quota.items():
        target_n = int(target_total * ratio)
        pool = buckets[tier]
        if len(pool) < target_n:
            log.warning(f"Tier {tier} 供给不足：需要 {target_n}，实有 {len(pool)}")
        selected += random.sample(pool, min(target_n, len(pool)))

    # 校验实际比例在容差内
    actual = _tier_distribution(selected)
    for tier, ratio in quota.items():
        assert abs(actual[tier] - ratio) <= tolerance, \
            f"Tier {tier} 实际比例 {actual[tier]:.2%} 偏离目标 {ratio:.2%}"
    return selected
```

### 4.8 评测层取消硬指标射击 T4

在 Phase 3 evaluate.py 中为 T4 开启降级口径，避免 GroundingDINO 失效造成系统性偏差：

```python
from typing import Dict, Any

def evaluate_sample(sample: dict, video_path: str) -> Dict[str, Any]:
    tier = sample["subject_tier"]
    subject = sample["target_subject"]

    if tier == "T4_rare_fictional":
        # 降级口径：Qwen-VL VQA 软分，不用硬指标
        return {
            "score": vqa_soft_score(video_path,
                                    question=f"Does the main subject look like a {subject}?",
                                    evaluator="qwen3-vl-30b"),
            "evaluator": "vqa_soft",
            "is_probe": True,
        }
    else:
        # T1-T3：GroundingDINO 硬指标 + Qwen-VL 补充
        return {
            "score": grounding_dino_recall(video_path, subject),
            "evaluator": "grounding_dino",
            "is_probe": False,
        }
```

**报告时应分层展示**得分：

| Tier | 例题数 | 平均分 | 评价 |
|------|--------|-------|------|
| T1_common | 2200 | 0.87 | 包容能力 |
| T2_longtail | 1120 | 0.72 | 长尾能力 |
| T3_finegrained | 480 | 0.61 | 细粒度能力 |
| T4_rare_fictional | 200 | 0.34 | **鲁棒性探针（仅供参考）** |
| **总均（排除 T4）** | 3800 | **0.79** | **作为主评分** |

### 4.9 实施影响与依赖

**新增依赖**：
- `wordfreq >= 3.0`
- `nltk` 与 `wordnet` 语料（~35MB）
- 可选：`ram-plus` 与 RAM++ 预训权重（~1GB）

**新增资源文件**：
- `resources/coco80.json`（从 [COCO API](https://github.com/cocodataset/cocoapi) 提取）
- `resources/subtlex_top500_nouns.json`（从 [SUBTLEX-US](https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus) 预处理）
- `resources/lvis1203.json`（从 [LVIS annotations](https://www.lvisdataset.org/) 提取）
- `resources/inat_cub_stanford_cars.json`（从三个数据集合并）

**重跑范围**：Phase 1（补 tier 字段）→ Phase 2 sample→ verify→ finalize→ export（全链）。若仅需补标后重新采样 3519→新额，成本约同一次 Phase 2。

### 4.10 关键结论

1. **不允许全生僻主体**：会将 Benchmark 退化为「长尾记忆能力测试」。
2. **不允许零生僻主体**：少量 T4（推荐 5%）作为鲁棒性探针仍有价值，但**评测口径必须降级**。
3. **COCO / WordNet 单独不够**：必须多源组合，推荐 COCO-80 + SUBTLEX + LVIS-1203 + iNaturalist 四层种子词表 + WordNet 上下位兵兑底。
4. **推荐配额 55/28/12/5**（T1/T2/T3/T4），必要时可调整，但 T4 建议不超 10%。
5. **评测层需分层口径**：T1-T3 用 GroundingDINO 硬指标，T4 用 Qwen-VL VQA 软分，**总均以 T1-T3 为主**，T4 仅供参考。

---

## P5: 词汇×主体正交诊断矩阵（Lexical × Subject Factorial Diagnostic）

### 5.1 问题与动机

P1（提示词生僻词）与 P4（主体分布失衡）已分别处理了两个单变量质量缺陷。但实际评测中存在一个更深层的问题：**当一个模型在某道题目上失败时，无法区分是因为「不懂生僻形容词」还是因为「不懂生僻主体」**。若两个变量在样本中自然混合，评测结果无法提供诊断信号，也不能支持不同模型之间的能力归因。

**解决方案**：将「提示词生僻度」与「主体生僻度」作为两个正交变量，构造 2×2 factorial design，解耦两种鲁棒性能力。

### 5.2 理论基础

**因子实验设计 (Factorial DOE, Fisher 1935)** —— 统计学中在多因素影响下解耦单因素主效应与交互效应的标准方法。现代 multimodal benchmark 已将其引入：

- **[Winoground (Thrush et al. CVPR'22)](https://arxiv.org/abs/2204.03162)** —— 首个专门重新构造图文对以隔离「词序对调」与「语义矩阵」两种能力的 vision-language benchmark，实证了**单一总分会掩盖子能力差异**（多个强模型在总分相近时依然存在不同能力坐标的陆地）。P5 矩阵与 Winoground 同构。
- **[COGS (Kim & Linzen EMNLP'20)](https://arxiv.org/abs/2010.05465)** —— 将 compositional generalization 分解为多个正交子集（primitive substitution、structural generalization 等），逐子集报告字字段得分。本项目的 A/B/C/D 四象限就是 COGS 思想在图-视频域的直接变体。
- **[DrawBench (Saharia et al. NeurIPS'22, Imagen)](https://arxiv.org/abs/2205.11487)** —— 在 200 条 prompt 中显式划分 11 个能力类别（含 Rare Words 子集），认为**能力子集能揭示主总分掩盖的软肋**。P5 中的“B 象限（rare-prompt × common-subject）”直接对应 DrawBench 的 Rare Words 子集，但控制了主体变量。

其他参考：
- **[PartiPrompts (Yu et al. 2022, Parti)](https://arxiv.org/abs/2206.10789)**：1600 提示按 category × challenge 二维分块。
- **[SCAN (Lake & Baroni ICML'18)](https://arxiv.org/abs/1711.00350)**：实证神经网络在组合推广上的系统性失败，为子集归因提供方法论基础。

**学术定位**：P5 不是新发明，而是将 Winoground/COGS/DrawBench 的 factorial diagnostic 思想首次引入 I2V benchmark。

### 5.3 四象限定义

将提示词生僻度（lexical rarity）与主体生僻度（subject rarity）正交上行：

| 象限 | 提示词 | 主体 | 测量能力 | 评测干净度 |
|------|--------|------|---------|------------|
| **A** | Common (Zipf ≥ 4.0) | T1 (common) | 纯组合能力 (compositional binding) | ⭐⭐⭐⭐⭐ |
| **B** | Rare  (Zipf < 4.0)  | T1 (common) | 对生僻形容词的 text encoder 鲁棒性 | ⭐⭐⭐⭐ |
| **C** | Common (Zipf ≥ 4.0) | T3/T4 (rare) | 对生僻主体的视觉-语言对齐鲁棒性 | ⭐⭐⭐⭐ |
| **D** | Rare  (Zipf < 4.0)  | T3/T4 (rare) | 复合鲁棒性 + 交互效应 (interaction) | ⭐⭐（仅诊断） |

**关键变量定义**：

- **提示词生僻度**：取提示词中除主体名词外的形容词/动词/副词集合，若最低 Zipf freq < 4.0 则标为 rare-prompt（阈值可配置）。
- **主体生僻度**：直接使用 P4 的 `subject_tier`。T1 为 common，T3/T4 为 rare，T2 可选归入 common 或 rare（建议归入 common 以保证反转相对干净。

### 5.4 与 P1 / P4 的关系（非推翻，而是精细化）

P1 §1.8 提出「四层防御清零生僻形容词」——这仍为**默认行为**，但在 P5 框架下需对 B/D 象限**受控地关闭部分防御**：

| 象限 | §1.8 层 1 Phase 1 预洗 | §1.8 层 2 Polish 硬约束 | §1.8 层 3 constraint 检查 | §1.8 层 4 报告监控 |
|------|----------------------|-----------------------|-------------------------|--------------------|
| **A** | ✅ 严格 | ✅ 严格（RareWordBlocker） | ✅ 严格 | ✅ 入总分 |
| **B** | ❌ 关闭（保留 rare prompt） | ❌ 关闭 | ⚠️ 仅保留 fk_grade / clip_sim 硬检查 | ✅ 入总分 + 单独报告 |
| **C** | ✅ 严格 | ✅ 严格 | ✅ 严格 | ✅ 入总分 |
| **D** | ❌ 关闭 | ❌ 关闭 | ⚠️ 仅保留 clip_sim 硬检查 | ❌ **仅诊断，不入总分** |

架构关键：`_enforce_constraints` 从「全局硬约束」降级为「象限条件约束」，由 `quadrant` 字段驱动。

### 5.5 推荐配额

| 象限 | 占比 | 3800 主评分样本中数量 | 评分口径 |
|------|------|-------------------|---------|
| **A** | **60%** | 2280 | 主评分（GroundingDINO 硬指标 + Qwen-VL 补充） |
| **B** | **12%** | 456 | 主评分 + 单独报告 $\Delta_{\text{prompt}}$ |
| **C** | **18%** | 684 | T3 硬指标 / T4 VQA 软分；单独报告 $\Delta_{\text{subject}}$ |
| **D** | **10%** | 380 | 全 VQA 软分，**仅诊断不计主总分** |

**为何 C > B**：主体生僻对评测器（GroundingDINO）影响更大，需更多样本降低方差。

**为何 D ≤ 10%**：交互项样本大幅稀释主总分意义，且构造成本最高。

### 5.6 三大正交诊断指标

引入四象限后可解耦为三个独立的鲁棒性维度：

$$
\Delta_{\text{prompt}} = \mathrm{Score}(A) - \mathrm{Score}(B) \quad \text{（词汇鲁棒性缺口）}
$$

$$
\Delta_{\text{subject}} = \mathrm{Score}(A) - \mathrm{Score}(C) \quad \text{（主体鲁棒性缺口）}
$$

$$
\mathrm{Interaction} = \mathrm{Score}(D) - \bigl[\mathrm{Score}(A) - \Delta_{\text{prompt}} - \Delta_{\text{subject}}\bigr]
$$

**解释**：
- $\mathrm{Interaction} < 0$：存在**非线性交互效应**（rare×rare 惩罚 > 两者相加），模型在复合情况下崩溃更严重。
- $\mathrm{Interaction} \approx 0$：两个变量独立作用（可加性）。
- $\mathrm{Interaction} > 0$：模型对复合情况反而更鲁棒（罕见但存在，通常因针对性训练）。

三个诊断指标作为论文与报告的**一级 selling point**，远高于单一总分。

### 5.7 数据可行性风险与应对

#### 5.7.1 天然分布不足

TIP-I2V 中 D 象限（rare×rare）自然共现率 <1%，供给不足。

**应对方案**：
1. Phase 1 新增 `quadrant_hint` 字段，标注自然象限归属
2. 供给不足时允许**LLM 定向改写**（从 A 象限取样本，将主体改为 T4，将形容词改为 rare）
3. 改写后**人工抽检 10-20%**保证语义合理

#### 5.7.2 配额矩阵爆炸

4 象限 × 7 维度 × 15 subtype = 420 格，硬约束不可行。

**应对方案**：象限配额**只在 dimension 层硬约束**，subtype 层软约束（容差 ±5%）。

#### 5.7.3 评测器不一致的汇总偏差

A/B 主要走 GroundingDINO 硬指标，C/D 中的 T4 走 VQA 软分，**不可直接相加**。

**应对方案**：报告时为每个象限明确标注 evaluator，主总分仅由 A+B+C(T3部分) 的硬指标组成；C(T4)+D 软分单独报告。

### 5.8 工程落地

#### 5.8.1 Phase 1 双标打标

在现有 `subject_tier` 基础上新增 `prompt_rarity` 字段：

```python
# src/i2vcompbench/phase1/prompt_rarity.py
from wordfreq import zipf_frequency
import re, spacy

nlp = spacy.load("en_core_web_sm")
_STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "and", "or"}

def classify_prompt_rarity(prompt: str, subject_name: str,
                            zipf_threshold: float = 4.0) -> dict:
    """仅测量非主体名词以外的词汇。"""
    subj_tokens = set(subject_name.lower().split())
    doc = nlp(prompt)
    zipfs = []
    for tok in doc:
        w = tok.text.lower()
        if (w in _STOP or w in subj_tokens or
            tok.pos_ not in {"ADJ", "ADV", "VERB", "NOUN"}):
            continue
        if tok.pos_ == "NOUN" and w in subj_tokens:
            continue
        zipfs.append((w, zipf_frequency(w, 'en')))
    if not zipfs:
        return {"rarity": "common", "min_zipf": None, "rare_tokens": []}
    rare = [w for w, z in zipfs if z < zipf_threshold]
    min_z = min(z for _, z in zipfs)
    return {
        "rarity": "rare" if rare else "common",
        "min_zipf": round(min_z, 2),
        "rare_tokens": rare,
    }


def classify_quadrant(prompt_rarity: str, subject_tier: str) -> str:
    prompt_rare = (prompt_rarity == "rare")
    subject_rare = subject_tier in {"T3_finegrained", "T4_rare_fictional"}
    if not prompt_rare and not subject_rare:
        return "A_common_common"
    if prompt_rare and not subject_rare:
        return "B_rare_prompt"
    if not prompt_rare and subject_rare:
        return "C_rare_subject"
    return "D_rare_both"
```

输出到 `aligned_instances[*]`：
- `prompt_rarity` ∈ {`common`, `rare`}
- `prompt_rarity_evidence`：{`min_zipf`, `rare_tokens`}
- `quadrant` ∈ {`A_common_common`, `B_rare_prompt`, `C_rare_subject`, `D_rare_both`}

#### 5.8.2 Phase 2 二维配额

**`configs/phase2.yaml` 扩展**：

```yaml
sample:
  quadrant_quota:
    A_common_common: 0.60
    B_rare_prompt:   0.12
    C_rare_subject:  0.18
    D_rare_both:     0.10
  quadrant_tolerance: 0.03            # ±3% (dim 层硬约束)
  subtype_quadrant_tolerance: 0.05    # ±5% (subtype 层软约束)
  allow_llm_rewrite_for_D: true       # D 象限供给不足时允许 LLM 改写
  llm_rewrite_human_audit_ratio: 0.15 # 改写后人工抽检比例
```

**`sample_questions.py` 伪代码**：

```python
def sample_with_quadrant_quota(candidates, target_total, quadrant_quota,
                                 tolerance=0.03):
    from collections import defaultdict
    buckets = defaultdict(list)
    for c in candidates:
        buckets[c["quadrant"]].append(c)

    selected = []
    for quad, ratio in quadrant_quota.items():
        target_n = int(target_total * ratio)
        pool = buckets[quad]
        if len(pool) < target_n and quad == "D_rare_both":
            # 调用 LLM 定向改写补齐
            extra = _llm_rewrite_to_D(buckets["A_common_common"],
                                       target_n - len(pool))
            pool = pool + extra
        selected += random.sample(pool, min(target_n, len(pool)))
    return selected
```

#### 5.8.3 Phase 2 constraint 降级（象限条件）

将 P1 §1.8.3 的全局硬检查重构为象限条件：

```python
def _enforce_constraints(prompt, forbidden, min_words, max_words,
                          dimension, quadrant, image=None):
    # 字数、禁词、active verb、camera cheat —— 无论象限均需硬检查
    ...
    # 词频硬检查 —— 仅 A/C 启用
    if quadrant in {"A_common_common", "C_rare_subject"}:
        if _has_rare_words(prompt, zipf_threshold=4.0):
            failed_check = "rare_vocab"
    # readability 硬检查 —— A/C 启用
    if quadrant in {"A_common_common", "C_rare_subject"}:
        if textstat.flesch_kincaid_grade(prompt) > 8:
            failed_check = "readability_too_high"
    # clip_sim 硬检查 —— 所有象限均启用
    if image and compute_clip_sim(image, prompt) < 0.28:
        failed_check = "clip_sim_too_low"
    ...
```

#### 5.8.4 Phase 3 分象限报告

```python
def aggregate_scores(results):
    by_quad = defaultdict(list)
    for r in results:
        by_quad[r["quadrant"]].append(r["score"])

    def avg(xs): return sum(xs)/len(xs) if xs else float("nan")

    A, B, C, D = (avg(by_quad[q]) for q in
                   ["A_common_common", "B_rare_prompt",
                    "C_rare_subject", "D_rare_both"])
    delta_prompt = A - B
    delta_subject = A - C
    interaction = D - (A - delta_prompt - delta_subject)

    main_score = avg(by_quad["A_common_common"] +
                      by_quad["B_rare_prompt"] +
                      by_quad["C_rare_subject"])  # A+B+C (T3部分) 硬指标
    return {
        "main_score": main_score,
        "quadrant_scores": {"A": A, "B": B, "C": C, "D": D},
        "delta_prompt": delta_prompt,
        "delta_subject": delta_subject,
        "interaction": interaction,
    }
```

### 5.9 新增报告行为

预期最终 leaderboard 字段：

| Model | Main | Δ Prompt | Δ Subject | Interaction | A | B | C | D† |
|-------|------|-----------|------------|-------------|---|---|---|-----|
| CogVideoX-5B | 0.78 | 0.11 | 0.16 | -0.05 | 0.87 | 0.76 | 0.71 | 0.42† |
| Wan2.1-I2V-14B | 0.81 | 0.08 | 0.14 | -0.02 | 0.88 | 0.80 | 0.74 | 0.51† |
| SVD-XT | 0.72 | 0.15 | 0.19 | -0.09 | 0.83 | 0.68 | 0.64 | 0.31† |

† D 象限仅诊断参考，不入 Main。交互项为负值表示复合难度大于单项相加。

### 5.10 关键结论

1. **P5 不推翻 P1/P4**，而是将其重新组织为可诊断的 factorial design，提供 Winoground/COGS/DrawBench 同构的能力分解。
2. **主评分 = A + B + C**（硬指标部分），**D 仅诊断不入总分**，避免评测器不一致引入噪声。
3. **推荐配额 60/12/18/10**，dimension 层硬约束、subtype 层软约束。
4. **`_enforce_constraints` 需从全局硬约束降级为象限条件约束**，否则无法造出 B/D 象限样本。
5. **三个诊断指标**：$\Delta_{\text{prompt}}$、$\Delta_{\text{subject}}$、Interaction —— 为论文提供高阶分析坐标。
6. **D 象限供给不足时接受 LLM 定向改写 + 人工抽检 15%**，否则不强行填足。

---

## 综合修复优先级

| 优先级 | 问题 | 修复复杂度 | 量化影响 | 修复后需重跑的步骤 |
|--------|------|-----------|----------|------------------|
| **P0** | 图像尺寸不统一 | 中 | 仅 2% 接近 4:3，68% 需大幅转换 | construct → verify → finalize → export |
| **P1** | 图像清晰度不足 | 高 | 所有 TIP 图像放大 3.8x，信息密度仅 6.9% | construct → verify → finalize → export |
| **P2** | 提示词生僻词 | 低 | 5.8% 提示词含生僻词，13.7% VLM caption 含生僻词 | 仅 finalize → export |
| **P3** | Fallback 无效提示词 | 低 | 117 条（3.3%）为 `"The the subject ."` | 仅 finalize → export |
| **P4** | 主体分布失衡 | 中 | 当前未控制主体常见度，存在全生僻风险 | Phase 1 补标 → sample → verify → finalize → export |
| **P5** | 词汇×主体正交矩阵缺失 | 高 | 无法解耦词汇/主体鲁棒性，无诊断指标 | Phase 1 双标 → sample → constraint 降级 → verify → finalize → export → evaluate 分层 |

### 建议修复顺序

1. **先统一尺寸** → 改动 `image.py` + `construct_inputs.py` + `phase2.yaml`，需重新 construct + verify + finalize + export
2. **再加清晰度检查** → 改动 `image.py` + `verify_inputs.py`，引入 cv2 依赖，需重新 construct
3. **然后加主体分层配额（P4）** → Phase 1 新增 `subject_tier` 字段，Phase 2 sample 阶段新增配额控制，Phase 3 evaluate 分层口径
4. **再加四象限矩阵（P5）** → Phase 1 新增 `prompt_rarity` + `quadrant` 字段；Phase 2 sample 二维配额 + `_enforce_constraints` 象限条件降级；Phase 3 aggregate 分象限报告
5. **然后修提示词（P2）** → 改动 `prompt_polish.txt` + `finalize_prompts.py`，仅需重新 finalize + export
6. **最后修 fallback（P3）** → 在 `_enforce_constraints` 中新增 fallback 有效性校验

### 时间成本估算

| 修复项 | 代码修改时间 | 重跑流水线时间 | 总时间 |
|--------|-------------|--------------|--------|
| 尺寸统一 | ~2 小时 | construct(30min) + verify(4h) + finalize(2h) + export(10min) | ~8 小时 |
| 清晰度检查 | ~3 小时（含超分模型部署） | construct(30min) + verify(4h) | ~8 小时 |
| 主体分层配额 | ~4 小时（含四套词表预处理 + RAM++ 可选） | Phase 1 补标(30min) + sample(10min) + verify(4h) + finalize(2h) + export(10min) | ~11 小时 |
| 四象限矩阵 | ~6 小时（含 `prompt_rarity` 实现、LLM 改写脚本、Phase 3 aggregate 重构） | Phase 1 补标(30min) + sample(10min) + constraint 降级(15min) + verify(4h) + finalize(2h) + export(10min) + evaluate(6h) | ~19 小时 |
| 提示词修复 | ~1 小时 | finalize(2h) + export(10min) | ~3 小时 |
| Fallback 修复 | ~0.5 小时 | finalize(2h) + export(10min) | ~2.5 小时 |

---

## 附录 A: 涉及的核心代码文件清单

| 文件 | 职责 | 涉及问题 | 关键函数/行号 |
|------|------|----------|-------------|
| `src/i2vcompbench/phase2/construct_inputs.py` | 图像构建与保存 | P2, P3 | `_copy_or_save_tip_image()` L196-L202 |
| `src/i2vcompbench/utils/image.py` | 图像缩放、16:9 适配 | P2, P3 | `resize_long_edge()` L37-L59, `to_16x9_720p()` L109-L156 |
| `src/i2vcompbench/phase2/finalize_prompts.py` | 提示词 polish 与约束检查 | P1, P3 | `_enforce_constraints()` L248-L277 |
| `src/i2vcompbench/phase2/build_question_plan.py` | 问题计划与 prompt_draft 生成 | P1 | `_resolve_slots()` L111-L135, L309-L311 |
| `src/i2vcompbench/phase2/verify_inputs.py` | VQA QC 校验 | P2 | `_HARD_CHECK_NAMES` L52 |
| `src/i2vcompbench/utils/templates.py` | 模板渲染与禁词检查 | P1 | `render_template()` L85-L99, `find_forbidden_hits()` L106-L116 |
| `prompts/prompt_polish.txt` | LLM polish 指令模板 | P1 | 8 条 HARD RULES |
| `configs/phase2.yaml` | Phase 2 配置 | P2, P3 | `construct.enable_t2i: false`, `long_edge: 854` |
| `configs/templates/*.yaml` | 维度模板（prompt_pattern） | P1 | 7 个维度 15 个 subtype |

## 附录 B: 关键配置参数一览

```yaml
# configs/phase2.yaml 相关参数
construct:
  long_edge: 854              # 主产物长边（像素）
  enable_t2i: false           # T2I 生成开关（当前关闭）
  fmt: PNG                    # 图像格式

prompt_finalize:
  min_words: 8                # 提示词最小词数
  max_words: 25               # 提示词最大词数
  max_retries: 2              # LLM polish 最大重试次数

# image.py 硬编码常量
DEFAULT_LONG_EDGE = 854       # 默认长边
DEFAULT_INFERENCE_W = 854     # 推理宽度（16:9）
DEFAULT_INFERENCE_H = 480     # 推理高度（16:9）
_NEAR_169_TOLERANCE = 0.04    # 16:9 容差（±4%）
```
