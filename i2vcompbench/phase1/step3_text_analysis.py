"""
Step 3: LLM-based text prompt analysis.
Calls SiliconFlow LLM API to classify intent and extract semantic slots.
Supports async concurrency and resume from checkpoint.
"""

import asyncio
from collections import Counter
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from i2vcompbench.utils.api_client_phase1 import SiliconFlowClient
from i2vcompbench.utils.io_utils import (
    append_jsonl,
    ensure_dir,
    load_prompt_template,
    parse_json_from_text,
    read_jsonl,
    read_processed_ids,
    write_freq_csv,
)
from i2vcompbench.schemas.phase1_legacy import (
    ActionSlot,
    AttributeChangeSlot,
    BackgroundChangeSlot,
    CameraMovementSlot,
    ManifestItem,
    MotionSlot,
    SpatialRelationSlot,
    TextAnalysisResult,
)


def _parse_llm_response(sample_id: str, prompt_text: str, raw_text: str) -> TextAnalysisResult:
    """Parse LLM raw text into TextAnalysisResult."""
    if not raw_text:
        return TextAnalysisResult(
            sample_id=sample_id,
            prompt_text=prompt_text,
            primary_intent="ambiguous",
            raw_llm_output="",
            parse_success=False,
            parse_error="Empty LLM response",
        )

    parsed = parse_json_from_text(raw_text)
    if parsed is None:
        return TextAnalysisResult(
            sample_id=sample_id,
            prompt_text=prompt_text,
            primary_intent="ambiguous",
            raw_llm_output=raw_text,
            parse_success=False,
            parse_error="JSON parse failed",
        )

    try:
        # Parse semantic slots
        attr_slots = [
            AttributeChangeSlot(**s) for s in parsed.get("attribute_change_slots", []) if isinstance(s, dict)
        ]
        action_slots = [
            ActionSlot(**s) for s in parsed.get("action_slots", []) if isinstance(s, dict)
        ]
        motion_slots = [
            MotionSlot(**s) for s in parsed.get("motion_slots", []) if isinstance(s, dict)
        ]
        spatial_slots = [
            SpatialRelationSlot(**s) for s in parsed.get("spatial_relation_slots", []) if isinstance(s, dict)
        ]
        bg_slots = [
            BackgroundChangeSlot(**s) for s in parsed.get("background_change_slots", []) if isinstance(s, dict)
        ]
        cam_slots = [
            CameraMovementSlot(**s) for s in parsed.get("camera_movement_slots", []) if isinstance(s, dict)
        ]

        return TextAnalysisResult(
            sample_id=sample_id,
            prompt_text=prompt_text,
            primary_intent=parsed.get("primary_intent", "ambiguous"),
            subject_sub_intent=parsed.get("subject_sub_intent"),
            background_sub_intent=parsed.get("background_sub_intent"),
            camera_sub_intent=parsed.get("camera_sub_intent"),
            involves_attribute_change=parsed.get("involves_attribute_change", False),
            involves_action=parsed.get("involves_action", False),
            involves_directed_motion=parsed.get("involves_directed_motion", False),
            involves_spatial_relation_change=parsed.get("involves_spatial_relation_change", False),
            involves_background_change=parsed.get("involves_background_change", False),
            involves_camera_movement=parsed.get("involves_camera_movement", False),
            attribute_change_slots=attr_slots,
            action_slots=action_slots,
            motion_slots=motion_slots,
            spatial_relation_slots=spatial_slots,
            background_change_slots=bg_slots,
            camera_movement_slots=cam_slots,
            nouns=parsed.get("nouns", []) or [],
            verbs=parsed.get("verbs", []) or [],
            adjectives=parsed.get("adjectives", []) or [],
            raw_llm_output=raw_text,
            parse_success=True,
            parse_error=None,
        )
    except Exception as e:
        return TextAnalysisResult(
            sample_id=sample_id,
            prompt_text=prompt_text,
            primary_intent="ambiguous",
            raw_llm_output=raw_text,
            parse_success=False,
            parse_error=f"Schema construction error: {str(e)}",
        )


def _compute_statistics(results: list, output_dir: str) -> None:
    """Compute and write frequency statistics from text analysis results."""
    od = Path(output_dir)

    intent_freq = Counter()
    subject_intent_freq = Counter()
    target_subject_freq = Counter()
    action_verb_freq = Counter()
    attr_change_type_freq = Counter()
    motion_dir_freq = Counter()
    spatial_pred_freq = Counter()
    camera_cmd_freq = Counter()
    env_term_freq = Counter()
    noun_freq = Counter()
    verb_freq = Counter()
    prompt_lengths = []

    for r in results:
        if not r.parse_success:
            continue

        intent_freq[r.primary_intent] += 1
        if r.subject_sub_intent:
            subject_intent_freq[r.subject_sub_intent] += 1

        prompt_lengths.append(len(r.prompt_text.split()))

        # Attribute change slots
        for slot in r.attribute_change_slots:
            target_subject_freq[slot.target_subject.lower()] += 1
            attr_change_type_freq[slot.attribute_type] += 1

        # Action slots
        for slot in r.action_slots:
            target_subject_freq[slot.target_subject.lower()] += 1
            action_verb_freq[slot.action_verb.lower()] += 1

        # Motion slots
        for slot in r.motion_slots:
            target_subject_freq[slot.target_subject.lower()] += 1
            motion_dir_freq[slot.direction] += 1

        # Spatial relation slots
        for slot in r.spatial_relation_slots:
            spatial_pred_freq[slot.target_predicate] += 1

        # Camera slots
        for slot in r.camera_movement_slots:
            camera_cmd_freq[slot.command] += 1

        # Background slots
        for slot in r.background_change_slots:
            env_term_freq[f"{slot.change_type}:{slot.to_state}"] += 1

        # Word frequencies
        for n in r.nouns:
            noun_freq[n.lower()] += 1
        for v in r.verbs:
            verb_freq[v.lower()] += 1

    write_freq_csv(str(od / "intent_distribution.csv"), dict(intent_freq), "intent")
    write_freq_csv(str(od / "subject_intent_distribution.csv"), dict(subject_intent_freq), "sub_intent")
    write_freq_csv(str(od / "target_subject_freq.csv"), dict(target_subject_freq), "subject")
    write_freq_csv(str(od / "action_verb_freq.csv"), dict(action_verb_freq), "verb")
    write_freq_csv(str(od / "attribute_change_type_freq.csv"), dict(attr_change_type_freq), "attribute_type")
    write_freq_csv(str(od / "motion_direction_freq.csv"), dict(motion_dir_freq), "direction")
    write_freq_csv(str(od / "spatial_predicate_freq.csv"), dict(spatial_pred_freq), "predicate")
    write_freq_csv(str(od / "camera_command_freq.csv"), dict(camera_cmd_freq), "command")
    write_freq_csv(str(od / "environment_term_freq.csv"), dict(env_term_freq), "term")
    write_freq_csv(str(od / "noun_freq.csv"), dict(noun_freq), "noun")
    write_freq_csv(str(od / "verb_freq.csv"), dict(verb_freq), "verb")

    # Prompt length stats
    if prompt_lengths:
        import statistics
        length_stats = {
            "mean": round(statistics.mean(prompt_lengths), 2),
            "median": round(statistics.median(prompt_lengths), 2),
            "min": min(prompt_lengths),
            "max": max(prompt_lengths),
            "stdev": round(statistics.stdev(prompt_lengths), 2) if len(prompt_lengths) > 1 else 0,
        }
        import pandas as pd
        pd.DataFrame([length_stats]).to_csv(
            str(od / "prompt_length_stats.csv"), index=False, encoding="utf-8-sig"
        )
        logger.info(f"Prompt length stats: {length_stats}")


async def _process_batch(
    client: SiliconFlowClient,
    batch: list,
    prompt_template: str,
    output_path: str,
    failed_path: str,
    semaphore: asyncio.Semaphore,
) -> list:
    """Process a batch of samples concurrently."""

    async def _process_one(item: ManifestItem):
        async with semaphore:
            # Use clean_prompt_text (Pika params removed) for LLM analysis
            prompt_for_analysis = item.clean_prompt_text or item.prompt_text
            filled_prompt = prompt_template.replace("{prompt_text}", prompt_for_analysis)
            raw_text = await client.async_call_llm(filled_prompt)
            result = _parse_llm_response(item.sample_id, item.prompt_text, raw_text)

            if result.parse_success:
                append_jsonl(output_path, result)
            else:
                append_jsonl(output_path, result)
                append_jsonl(failed_path, {
                    "sample_id": item.sample_id,
                    "prompt_text": item.prompt_text,
                    "error": result.parse_error,
                })
            return result

    tasks = [_process_one(item) for item in batch]
    return await asyncio.gather(*tasks)


def analyze_texts(config: dict) -> None:
    """
    Main entry: LLM text analysis with async concurrency and resume support.
    """
    manifest_path = str(Path(config["paths"]["manifest_dir"]) / "manifest_clean.jsonl")
    output_dir = str(Path(config["paths"]["output_dir"]) / "text_analysis")
    prompt_path = str(Path(config["paths"]["prompt_dir"]) / "llm_text_parse.txt")
    ensure_dir(output_dir)

    output_path = str(Path(output_dir) / "text_parse.jsonl")
    failed_path = str(Path(output_dir) / "failed_samples.jsonl")

    # Load data
    manifest_items = read_jsonl(manifest_path, ManifestItem)
    if not manifest_items:
        logger.error("No clean manifest items found. Run step1 first.")
        return

    # Load prompt template
    prompt_template = load_prompt_template(prompt_path)

    # Resume support
    processed_ids = read_processed_ids(output_path)
    todo_items = [item for item in manifest_items if item.sample_id not in processed_ids]
    logger.info(
        f"Total clean samples: {len(manifest_items)}, "
        f"Already processed: {len(processed_ids)}, "
        f"Remaining: {len(todo_items)}"
    )

    if not todo_items:
        logger.info("All samples already processed. Skipping LLM calls.")
    else:
        # Initialize API client
        client = SiliconFlowClient(config)
        batch_size = client.batch_size
        semaphore = asyncio.Semaphore(batch_size)

        async def _run_all():
            pbar = tqdm(total=len(todo_items), desc="LLM Text Analysis")
            for i in range(0, len(todo_items), batch_size):
                batch = todo_items[i : i + batch_size]
                await _process_batch(
                    client, batch, prompt_template, output_path, failed_path, semaphore
                )
                pbar.update(len(batch))
            pbar.close()

        asyncio.run(_run_all())

    # Compute statistics
    logger.info("Computing text analysis statistics...")
    all_results = read_jsonl(output_path, TextAnalysisResult)
    _compute_statistics(all_results, output_dir)

    # Summary
    success_count = sum(1 for r in all_results if r.parse_success)
    fail_count = len(all_results) - success_count
    logger.info("=" * 60)
    logger.info("Text Analysis Summary")
    logger.info("=" * 60)
    logger.info(f"Total analyzed:     {len(all_results)}")
    logger.info(f"Parse success:      {success_count}")
    logger.info(f"Parse failed:       {fail_count}")
    logger.info(f"Output: {output_path}")
