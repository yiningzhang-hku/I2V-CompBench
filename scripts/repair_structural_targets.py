"""
P0 结构化修复脚本：修复 question_plans.jsonl 中 target_subjects[].noun / target_relation 为空的问题。

根因：Phase 1 外接硬盘不可用，导致 Phase 2 生成时无法读取 image_parse 数据。
修复方案：对每条记录的 first_frame 图像调用 VLM 进行分析，提取主体名词、描述和关系。

用法：
    python scripts/repair_structural_targets.py                  # 全量修复
    python scripts/repair_structural_targets.py --dry-run        # 仅处理前 10 条
    python scripts/repair_structural_targets.py --resume         # 从 checkpoint 恢复
    python scripts/repair_structural_targets.py --batch-size 4   # 自定义并发数
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加项目 src 到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from i2vcompbench.utils.api_client_phase1 import SiliconFlowClient
from i2vcompbench.utils.io_utils import parse_json_from_text

# ============================================================
# 常量
# ============================================================

DATA_DIR = PROJECT_ROOT / "data" / "benchmark_dataset"
INPUT_JSONL = DATA_DIR / "question_plans.jsonl"
OUTPUT_JSONL = DATA_DIR / "question_plans_repaired.jsonl"
CHECKPOINT_JSONL = DATA_DIR / "question_plans_repair_checkpoint.jsonl"
FIRST_FRAMES_DIR = DATA_DIR / "first_frames"

# VLM 配置
VLM_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
BATCH_SIZE = 32
TIMEOUT = 120
RETRY_COUNT = 5
RATE_LIMIT_DELAY = 0.1

# 维度感知 prompt 片段
DIMENSION_HINTS: Dict[str, str] = {
    "attribute_binding": (
        "Focus especially on visual ATTRIBUTES of each subject: color, size, material, "
        "texture, shape, state (open/closed/wet/dry). These attributes are critical."
    ),
    "action_binding": (
        "Focus especially on the CURRENT ACTION or POSE of each subject. Describe what "
        "each subject is doing right now, as the video will show them performing actions."
    ),
    "motion_binding": (
        "Focus especially on the MOTION POTENTIAL of each subject. Note their current "
        "position, orientation, and any implied direction of motion."
    ),
    "spatial_composition": (
        "Focus especially on SPATIAL RELATIONSHIPS between subjects: relative positions, "
        "distances, and arrangement within the frame."
    ),
    "background_dynamics": (
        "Focus especially on BACKGROUND ELEMENTS that could exhibit dynamics: water, "
        "clouds, fire, smoke, leaves, flags. Note which are potentially dynamic vs rigid."
    ),
    "view_transformation": (
        "Focus especially on the CAMERA/VIEW properties: shot type, angle, depth cues, "
        "and rigid reference structures that would reveal camera motion."
    ),
    "interaction_reasoning": (
        "Focus especially on INTERACTIONS between subjects: physical contact, gaze "
        "direction, causal relationships, and social dynamics."
    ),
}


# ============================================================
# VLM Prompt 构建
# ============================================================

def build_vlm_prompt(dimension: str) -> str:
    """构建维度感知的 VLM prompt，要求输出结构化 JSON。"""
    dimension_hint = DIMENSION_HINTS.get(dimension, "")

    prompt = f"""You are a professional image structural analysis assistant. Analyze this image carefully and output a structured JSON description.

Requirements:
1. Identify ALL "thing-type" subject instances (people, animals, vehicles, objects, etc. — countable entities), excluding background elements (sky, ground, grass, etc.)
2. For each subject, describe:
   - name: category noun in English (e.g., "dog", "woman", "car", "cup")
   - description: brief instance-specific description in English
   - attributes: color, size, material/texture, state, and any distinctive visual features
3. If there are multiple subjects, describe their spatial/interaction relationships
4. Identify scene type and any camera motion hints from the static frame

{dimension_hint}

Output STRICTLY in the following JSON format, with NO other text:
```json
{{
  "subjects": [
    {{
      "name": "<category noun, English, lowercase>",
      "description": "<brief instance description, English>",
      "attributes": {{
        "color": ["<color>"],
        "size": "<tiny/small/medium/large/huge>",
        "material_texture": ["<material or texture>"],
        "state": ["<state: open/closed/wet/dry/etc>"],
        "wearing": ["<wearing items if applicable>"],
        "pose_action": "<current pose or action>"
      }}
    }}
  ],
  "relations": [
    {{
      "subject1": "<name of subject 1>",
      "relation": "<spatial or interaction relation: left_of/right_of/above/below/on_top_of/next_to/holding/riding/near/in_front_of/behind>",
      "subject2": "<name of subject 2>"
    }}
  ],
  "scene_type": "<indoor/outdoor/studio/abstract/etc>",
  "camera_motion_hint": "<static/slow_pan_possible/zoom_possible/tilt_possible/orbit_possible>"
}}
```

Rules:
- All descriptive text must be in English
- "name" must be a generic category NOUN (e.g., "dog" not "golden retriever running")
- "description" should be specific enough to distinguish this instance from others
- If only one subject, "relations" should be an empty array []
- Do NOT miss any visible subject instances
- Do NOT include background elements (sky, grass, road, wall) as subjects"""

    return prompt


# ============================================================
# 响应解析与字段填充
# ============================================================

def parse_vlm_response(raw_text: str) -> Optional[Dict[str, Any]]:
    """解析 VLM 响应文本为结构化字典。"""
    if not raw_text:
        return None
    parsed = parse_json_from_text(raw_text)
    return parsed


def fill_record(record: Dict[str, Any], vlm_result: Dict[str, Any]) -> Dict[str, Any]:
    """将 VLM 解析结果填充到 question_plan 记录中。

    填充逻辑：
    - target_subjects[].noun ← vlm_result.subjects[i].name
    - target_subjects[].description ← vlm_result.subjects[i].description
    - target_relation ← 从 vlm_result.relations 提取
    """
    subjects_from_vlm = vlm_result.get("subjects", [])
    target_plan = record.get("target_plan", {})
    target_subjects = target_plan.get("target_subjects", [])

    # 填充 target_subjects
    for i, ts in enumerate(target_subjects):
        if i < len(subjects_from_vlm):
            vlm_subj = subjects_from_vlm[i]
            # 仅填充空字段
            if not ts.get("noun"):
                ts["noun"] = vlm_subj.get("name", "")
            if not ts.get("description") or ts.get("description") in ("the subject", ""):
                ts["description"] = vlm_subj.get("description", "")
        elif subjects_from_vlm:
            # 如果 target_subjects 多于 VLM 检测到的主体，用第一个填充
            vlm_subj = subjects_from_vlm[0]
            if not ts.get("noun"):
                ts["noun"] = vlm_subj.get("name", "")
            if not ts.get("description") or ts.get("description") in ("the subject", ""):
                ts["description"] = vlm_subj.get("description", "")

    target_plan["target_subjects"] = target_subjects

    # 填充 target_relation（仅当当前为空且 VLM 检测到关系时）
    relations_from_vlm = vlm_result.get("relations", [])
    current_relation = target_plan.get("target_relation")

    if relations_from_vlm and len(target_subjects) > 1:
        # 有多主体且有关系时填充
        if current_relation is None or (
            isinstance(current_relation, dict)
            and not current_relation.get("type")
            and not current_relation.get("value")
        ):
            rel = relations_from_vlm[0]
            target_plan["target_relation"] = {
                "type": _infer_relation_type(rel.get("relation", "")),
                "value": rel.get("relation", ""),
                "subj": "s1",
                "obj": "s2",
            }

    record["target_plan"] = target_plan
    return record


def _infer_relation_type(relation_value: str) -> str:
    """从关系值推断关系类型。"""
    spatial_relations = {
        "left_of", "right_of", "above", "below", "on_top_of",
        "next_to", "in_front_of", "behind", "near",
    }
    interaction_relations = {
        "holding", "riding", "carrying", "pulling", "pushing",
        "feeding", "chasing", "leading", "touching",
    }
    if relation_value.lower() in spatial_relations:
        return "spatial"
    elif relation_value.lower() in interaction_relations:
        return "interaction"
    return "spatial"


# ============================================================
# question_id → first_frame 图像路径
# ============================================================

def resolve_image_path(question_id: str) -> Optional[Path]:
    """从 question_id 推导出 first_frame 图像路径。

    命名规则：question_id 直接对应文件名，如：
      attr_single_0001 → attr_single_0001.png
    """
    # 优先原始 png
    img_path = FIRST_FRAMES_DIR / f"{question_id}.png"
    if img_path.exists():
        return img_path

    # 尝试 16x9 变体
    img_path_16x9 = FIRST_FRAMES_DIR / f"{question_id}_16x9.png"
    if img_path_16x9.exists():
        return img_path_16x9

    # 尝试 jpg
    img_path_jpg = FIRST_FRAMES_DIR / f"{question_id}.jpg"
    if img_path_jpg.exists():
        return img_path_jpg

    return None


# ============================================================
# Checkpoint 管理
# ============================================================

def load_checkpoint() -> set:
    """加载已处理的 question_id 集合。"""
    if not CHECKPOINT_JSONL.exists():
        return set()

    processed = set()
    with open(CHECKPOINT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                processed.add(obj["question_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    logger.info(f"从 checkpoint 加载 {len(processed)} 条已处理记录")
    return processed


def save_checkpoint(question_id: str, status: str, vlm_raw: str = ""):
    """追加一条 checkpoint 记录。"""
    with open(CHECKPOINT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "question_id": question_id,
            "status": status,
            "timestamp": time.time(),
            "vlm_raw_len": len(vlm_raw),
        }, ensure_ascii=False) + "\n")


# ============================================================
# 异步批量处理
# ============================================================

async def process_one(
    client: SiliconFlowClient,
    record: Dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """处理单条记录：调用 VLM 并填充字段。"""
    question_id = record["question_id"]
    dimension = record.get("dimension", "")

    # 找到图像
    img_path = resolve_image_path(question_id)
    if img_path is None:
        logger.warning(f"[{question_id}] 图像文件不存在，跳过")
        save_checkpoint(question_id, "no_image")
        return record

    # 构建 prompt
    prompt = build_vlm_prompt(dimension)

    # 调用 VLM（禁用thinking以加速）
    async with semaphore:
        raw_text = await client.async_call_vlm(str(img_path), prompt, disable_thinking=True)

    # 解析响应
    vlm_result = parse_vlm_response(raw_text)
    if vlm_result is None:
        logger.warning(f"[{question_id}] VLM 响应解析失败，保留原始数据")
        save_checkpoint(question_id, "parse_failed", raw_text)
        return record

    # 填充字段
    record = fill_record(record, vlm_result)
    save_checkpoint(question_id, "success", raw_text)
    return record


async def process_batch(
    client: SiliconFlowClient,
    batch: List[Dict[str, Any]],
    semaphore: asyncio.Semaphore,
) -> List[Dict[str, Any]]:
    """并发处理一批记录。"""
    tasks = [process_one(client, record, semaphore) for record in batch]
    return await asyncio.gather(*tasks)


async def run_repair(records: List[Dict[str, Any]], args: argparse.Namespace):
    """主修复流程。"""
    # 初始化客户端
    config = {
        "api": {
            "api_key_env": "SILICONFLOW_API_KEY",
            "base_url": "https://api.siliconflow.cn/v1",
            "vlm": {
                "model": VLM_MODEL,
                "max_tokens": 1024,
                "temperature": 0.0,
            },
            "llm": {
                "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "max_tokens": 1500,
                "temperature": 0.0,
            },
            "batch_size": args.batch_size,
            "retry_count": RETRY_COUNT,
            "retry_delay": 2,
            "timeout": TIMEOUT,
            "rate_limit_delay": RATE_LIMIT_DELAY,
        }
    }
    client = SiliconFlowClient(config)
    batch_size = args.batch_size
    semaphore = asyncio.Semaphore(batch_size)

    # Checkpoint 恢复
    processed_ids = set()
    if args.resume:
        processed_ids = load_checkpoint()

    # 过滤已处理和无需修复的记录
    todo_records = []
    already_done = []

    for rec in records:
        qid = rec["question_id"]
        if qid in processed_ids:
            already_done.append(rec)
            continue

        # 检查是否真的需要修复（target_subjects[].noun 全为空）
        target_plan = rec.get("target_plan", {})
        target_subjects = target_plan.get("target_subjects", [])
        needs_repair = any(
            not ts.get("noun") for ts in target_subjects
        )
        if needs_repair:
            todo_records.append(rec)
        else:
            already_done.append(rec)

    logger.info(
        f"总记录数: {len(records)} | "
        f"已处理/无需修复: {len(already_done)} | "
        f"待修复: {len(todo_records)}"
    )

    if not todo_records:
        logger.info("所有记录已修复或无需修复")
        return records

    # Dry-run 模式只取前 10 条
    if args.dry_run:
        todo_records = todo_records[:10]
        logger.info(f"[DRY-RUN] 仅处理前 {len(todo_records)} 条")

    # 分批处理
    repaired_map: Dict[str, Dict[str, Any]] = {}
    total_batches = (len(todo_records) + batch_size - 1) // batch_size
    pbar = tqdm(total=len(todo_records), desc="VLM 结构修复")

    start_time = time.time()
    last_report_time = start_time

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(todo_records))
        batch = todo_records[batch_start:batch_end]

        results = await process_batch(client, batch, semaphore)

        for rec in results:
            repaired_map[rec["question_id"]] = rec

        pbar.update(len(batch))

        # 每 10 分钟汇报进度
        current_time = time.time()
        if current_time - last_report_time >= 600:
            elapsed = current_time - start_time
            done_count = batch_end
            remaining = len(todo_records) - done_count
            speed = done_count / elapsed if elapsed > 0 else 0
            eta_min = (remaining / speed / 60) if speed > 0 else 0
            logger.info(
                f"[进度汇报] 已处理: {done_count}/{len(todo_records)} | "
                f"速度: {speed:.1f} 条/秒 | "
                f"预估剩余: {eta_min:.1f} 分钟"
            )
            last_report_time = current_time

    pbar.close()

    # 合并结果：用修复后的记录替换原记录
    final_records = []
    for rec in records:
        qid = rec["question_id"]
        if qid in repaired_map:
            final_records.append(repaired_map[qid])
        else:
            final_records.append(rec)

    return final_records


# ============================================================
# 输出
# ============================================================

def write_output(records: List[Dict[str, Any]], output_path: Path):
    """写入修复后的 JSONL。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"输出修复后文件: {output_path} ({len(records)} 条)")


# ============================================================
# 统计报告
# ============================================================

def print_repair_stats(records: List[Dict[str, Any]]):
    """输出修复统计。"""
    total = len(records)
    noun_filled = 0
    noun_empty = 0
    relation_filled = 0

    for rec in records:
        target_plan = rec.get("target_plan", {})
        target_subjects = target_plan.get("target_subjects", [])
        for ts in target_subjects:
            if ts.get("noun"):
                noun_filled += 1
            else:
                noun_empty += 1

        tr = target_plan.get("target_relation")
        if tr and isinstance(tr, dict) and tr.get("value"):
            relation_filled += 1

    logger.info("=" * 60)
    logger.info("修复统计报告")
    logger.info("=" * 60)
    logger.info(f"总记录数:           {total}")
    logger.info(f"noun 已填充:        {noun_filled}")
    logger.info(f"noun 仍为空:        {noun_empty}")
    logger.info(f"relation 已填充:    {relation_filled}")
    logger.info("=" * 60)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="P0 结构化修复: 填充 question_plans.jsonl 中的空 target_subjects 字段"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅处理前 10 条记录用于测试"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="从 checkpoint 恢复处理"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"并发批大小 (默认: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_JSONL),
        help=f"输出文件路径 (默认: {OUTPUT_JSONL})"
    )
    args = parser.parse_args()

    # 加载 .env
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"已加载环境变量: {env_path}")

    # 检查 API KEY
    api_key = os.environ.get("SILICONFLOW_API_KEY", "")
    if not api_key:
        logger.error("SILICONFLOW_API_KEY 环境变量未设置，请检查 .env 文件")
        sys.exit(1)

    # 读取输入数据
    logger.info(f"读取输入文件: {INPUT_JSONL}")
    records = []
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"共读取 {len(records)} 条记录")

    # 统计当前空字段情况
    null_noun_count = sum(
        1 for rec in records
        for ts in rec.get("target_plan", {}).get("target_subjects", [])
        if not ts.get("noun")
    )
    logger.info(f"当前 noun 为空的 subject 数: {null_noun_count}")

    # 检查图像目录
    if not FIRST_FRAMES_DIR.exists():
        logger.error(f"首帧图像目录不存在: {FIRST_FRAMES_DIR}")
        sys.exit(1)

    # 运行修复
    repaired_records = asyncio.run(run_repair(records, args))

    # 写入输出
    output_path = Path(args.output)
    write_output(repaired_records, output_path)

    # 打印统计
    print_repair_stats(repaired_records)


if __name__ == "__main__":
    main()
