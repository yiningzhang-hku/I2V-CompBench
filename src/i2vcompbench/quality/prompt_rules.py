"""
Prompt 纯规则检查模块 — 确定性、无副作用、不依赖任何模型。

每个 check_* 函数接受 prompt（及少量配置参数），返回 issue 标识符列表。
空列表 == 通过。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ============================================================
# 常量
# ============================================================

VIEW_CAMERA_CUES: List[str] = [
    "camera pans", "camera pan", "camera tilts", "camera tilt",
    "camera moves", "camera sweeps", "camera slides",
    "zoom in", "zoom out", "zooms in", "zooms out",
    "pan left", "pan right", "pans left", "pans right",
    "moves left", "moves right", "sweeps left", "sweeps right",
    "tilt up", "tilt down", "tilts up", "tilts down",
    "dolly", "tracking shot", "aerial view", "bird's eye",
    "low angle", "high angle", "rotating", "orbit",
    "crane shot", "steady cam", "steadicam",
]

STATIC_VERBS = {
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "having",
    "does", "do", "did",
}

# 动词后缀启发式（用于 check_change_verb）
_VERB_SUFFIX_RE = re.compile(
    r"\b[a-zA-Z]{3,}(?:ing|s|ed|es|ies|ied)\b", re.IGNORECASE
)

# 占位符正则
_PLACEHOLDER_RE = re.compile(r"\{[^}]+\}")

# 冠词
_ARTICLES = {"the", "a", "an"}

# ============================================================
# Zipf 分数（带 fallback）
# ============================================================

try:
    from wordfreq import zipf_frequency as _zipf_frequency
    _WORDFREQ_AVAILABLE = True
except ImportError:
    _WORDFREQ_AVAILABLE = False
    _zipf_frequency = None  # type: ignore


def zipf_score(word: str) -> Optional[float]:
    """使用 wordfreq 计算 Zipf 分数，库不可用时返回 None。"""
    if not _WORDFREQ_AVAILABLE:
        return None
    return _zipf_frequency(word.lower(), "en")  # type: ignore


# ============================================================
# 维度禁用词加载
# ============================================================

_DEFAULT_TEMPLATES_DIR = Path(__file__).resolve().parents[3] / "configs" / "templates"


def load_dimension_forbidden_words(
    templates_dir: str | None = None,
) -> Dict[str, List[str]]:
    """从 configs/templates/*.yaml 加载各维度的 forbidden_words。"""
    tpl_dir = Path(templates_dir) if templates_dir else _DEFAULT_TEMPLATES_DIR
    result: Dict[str, List[str]] = {}
    if not tpl_dir.exists():
        return result
    for yaml_path in sorted(tpl_dir.glob("*.yaml")):
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        dimension = data.get("dimension", yaml_path.stem)
        # 收集所有 subtypes 的 forbidden_words 并去重
        words: List[str] = []
        for st in data.get("subtypes", []):
            for w in st.get("forbidden_words", []):
                if w and w not in words:
                    words.append(w)
        result[dimension] = words
    return result


# ============================================================
# 规则检查函数
# ============================================================


def check_empty_prompt(prompt: str) -> List[str]:
    """空 prompt 或纯空白 → ['empty_prompt']"""
    if not prompt or not prompt.strip():
        return ["empty_prompt"]
    return []


def check_unresolved_placeholders(prompt: str) -> List[str]:
    """含 {xxx} 未解析占位符 → ['unresolved_placeholder']"""
    if _PLACEHOLDER_RE.search(prompt):
        return ["unresolved_placeholder"]
    return []


def check_repeated_articles(prompt: str) -> List[str]:
    """连续重复冠词 the the / a a / an an → ['repeated_article']"""
    tokens = prompt.lower().split()
    for i in range(len(tokens) - 1):
        if tokens[i] in _ARTICLES and tokens[i] == tokens[i + 1]:
            return ["repeated_article"]
    return []


def check_article_before_punctuation(prompt: str) -> List[str]:
    """冠词后直接跟标点(.,;:!?) → ['article_before_punctuation']"""
    # 匹配: 冠词后紧跟（可能有空格）标点
    pattern = re.compile(r"\b(the|a|an)\s*[.,;:!?]", re.IGNORECASE)
    if pattern.search(prompt):
        return ["article_before_punctuation"]
    return []


def check_word_count(
    prompt: str, min_words: int = 8, max_words: int = 25
) -> List[str]:
    """单词数不在范围内 → ['word_count_too_short'] 或 ['word_count_too_long']"""
    if not prompt or not prompt.strip():
        return []  # 由 check_empty_prompt 处理
    count = len([w for w in re.split(r"\s+", prompt.strip()) if w])
    if count < min_words:
        return ["word_count_too_short"]
    if count > max_words:
        return ["word_count_too_long"]
    return []


def check_dimension_forbidden_words(
    prompt: str, dimension: str, forbidden_words: List[str]
) -> List[str]:
    """包含维度禁用词 → ['forbidden_word:<word>']
    注意：view_transformation 的逻辑相反，此函数不检查该维度。"""
    if dimension == "view_transformation":
        return []
    if not prompt or not forbidden_words:
        return []
    low = prompt.lower()
    issues: List[str] = []
    for w in forbidden_words:
        if not w:
            continue
        if w.lower() in low:
            issues.append(f"forbidden_word:{w}")
    return issues


def check_view_camera_cues(prompt: str, dimension: str) -> List[str]:
    """view_transformation 维度必须包含运镜线索 → ['missing_camera_cue']"""
    if dimension != "view_transformation":
        return []
    if not prompt:
        return ["missing_camera_cue"]
    low = prompt.lower()
    for cue in VIEW_CAMERA_CUES:
        if cue.lower() in low:
            return []
    return ["missing_camera_cue"]


def check_change_verb(prompt: str) -> List[str]:
    """Prompt 必须包含可识别的变化/动作谓词 → ['missing_change_verb']

    通过后缀检测动词形态（-ing, -s, -ed 等），排除静态词。
    """
    if not prompt or not prompt.strip():
        return ["missing_change_verb"]
    tokens = re.findall(r"\b[a-zA-Z]+\b", prompt)
    for token in tokens:
        low = token.lower()
        if low in STATIC_VERBS:
            continue
        # 检查动词形态后缀
        if _VERB_SUFFIX_RE.match(token):
            return []
    return ["missing_change_verb"]


def check_rare_modifiers(prompt: str, zipf_threshold: float = 3.5) -> List[str]:
    """非 NOUN/PROPN 词汇的 Zipf 分数低于阈值 → ['rare_modifier:<word>']

    排除规则：
    - 全大写词（可能是缩写）
    - 含数字的 token
    - 长度 <= 2 的词
    - 首字母大写且非句首位置的词（视为专名跳过）

    如果 wordfreq 不可用，返回 ['rare_modifier_check_skipped']。
    """
    if not _WORDFREQ_AVAILABLE:
        return ["rare_modifier_check_skipped"]
    if not prompt or not prompt.strip():
        return []

    tokens = re.findall(r"\b[a-zA-Z]+\b", prompt)
    issues: List[str] = []
    seen: set = set()

    for i, token in enumerate(tokens):
        # 排除：长度 <= 2
        if len(token) <= 2:
            continue
        # 排除：全大写（缩写）
        if token.isupper():
            continue
        # 排除：含数字（不会被 findall 匹配到，但以防万一）
        if any(c.isdigit() for c in token):
            continue
        # 排除：首字母大写且非句首 → 视为专名
        if i > 0 and token[0].isupper():
            continue

        low = token.lower()
        if low in seen:
            continue

        score = zipf_score(low)
        if score is not None and score < zipf_threshold:
            issues.append(f"rare_modifier:{low}")
            seen.add(low)

    return issues


# ============================================================
# 聚合函数
# ============================================================


def check_all_rules(
    prompt: str,
    dimension: str,
    forbidden_words: Optional[List[str]] = None,
    min_words: int = 8,
    max_words: int = 25,
    zipf_threshold: float = 3.5,
) -> List[str]:
    """运行所有规则检查，返回合并的 issues 列表。"""
    issues: List[str] = []

    issues.extend(check_empty_prompt(prompt))
    # 空 prompt 无需继续检查
    if issues:
        return issues

    issues.extend(check_unresolved_placeholders(prompt))
    issues.extend(check_repeated_articles(prompt))
    issues.extend(check_article_before_punctuation(prompt))
    issues.extend(check_word_count(prompt, min_words, max_words))
    issues.extend(
        check_dimension_forbidden_words(prompt, dimension, forbidden_words or [])
    )
    issues.extend(check_view_camera_cues(prompt, dimension))
    issues.extend(check_change_verb(prompt))
    issues.extend(check_rare_modifiers(prompt, zipf_threshold))

    return issues


# ============================================================
# 批量运行函数
# ============================================================


def run_prompt_rules(
    candidates: List[Dict[str, Any]],
    dimension_forbidden_words: Dict[str, List[str]],
    config: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """对候选列表运行全部规则，返回结构化报告。

    每个 candidate dict 需包含:
      - question_id: str
      - prompt: str
      - dimension: str

    Returns:
        {
            "total": int,
            "checked": int,
            "issues_found": int,
            "clean_count": int,
            "issue_distribution": {"empty_prompt": 5, ...},
            "rare_words_found": ["ethereal", ...],
            "results": [{"question_id": ..., "issues": [...], "rare_hits": [...]}]
        }
    """
    cfg = config or {}
    min_words = cfg.get("min_words", 8)
    max_words = cfg.get("max_words", 25)
    zipf_threshold = cfg.get("zipf_threshold", 3.5)

    items = candidates[:limit] if limit else candidates
    total = len(candidates)
    checked = len(items)

    results: List[Dict[str, Any]] = []
    issue_distribution: Dict[str, int] = {}
    rare_words_set: set = set()
    issues_found = 0
    clean_count = 0

    for cand in items:
        qid = cand.get("question_id", "")
        prompt = cand.get("prompt", "")
        dimension = cand.get("dimension", "")
        forbidden = dimension_forbidden_words.get(dimension, [])

        issues = check_all_rules(
            prompt,
            dimension,
            forbidden_words=forbidden,
            min_words=min_words,
            max_words=max_words,
            zipf_threshold=zipf_threshold,
        )

        # 统计分布
        rare_hits: List[str] = []
        for iss in issues:
            # 归类 key（去掉 :value 后缀用于统计）
            key = iss.split(":")[0] if ":" in iss else iss
            issue_distribution[key] = issue_distribution.get(key, 0) + 1
            if iss.startswith("rare_modifier:"):
                word = iss.split(":", 1)[1]
                rare_hits.append(word)
                rare_words_set.add(word)

        if issues:
            issues_found += 1
        else:
            clean_count += 1

        results.append({
            "question_id": qid,
            "issues": issues,
            "rare_hits": rare_hits,
        })

    return {
        "total": total,
        "checked": checked,
        "issues_found": issues_found,
        "clean_count": clean_count,
        "issue_distribution": issue_distribution,
        "rare_words_found": sorted(rare_words_set),
        "results": results,
    }
