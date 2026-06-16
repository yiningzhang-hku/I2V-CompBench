"""
Step 2: VLM-based image structural analysis.
Calls SiliconFlow VLM API to parse each image into structured JSON.
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
    BackgroundElement,
    BackgroundInfo,
    CameraBaseline,
    ImageAnalysisResult,
    ManifestItem,
    SubjectAttributes,
    SubjectInfo,
    SubjectRelation,
)


def _parse_vlm_response(sample_id: str, raw_text: str) -> ImageAnalysisResult:
    """Parse VLM raw text into ImageAnalysisResult."""
    if not raw_text:
        return ImageAnalysisResult(
            sample_id=sample_id,
            raw_vlm_output="",
            parse_success=False,
            parse_error="Empty VLM response",
        )

    parsed = parse_json_from_text(raw_text)
    if parsed is None:
        return ImageAnalysisResult(
            sample_id=sample_id,
            raw_vlm_output=raw_text,
            parse_success=False,
            parse_error="JSON parse failed",
        )

    try:
        # Parse subjects
        subjects = []
        for s in parsed.get("subjects", []):
            attrs_data = s.get("attributes", {})
            attrs = SubjectAttributes(
                color=attrs_data.get("color", []) or [],
                size=attrs_data.get("size"),
                material_texture=attrs_data.get("material_texture", []) or [],
                state=attrs_data.get("state", []) or [],
                wearing=attrs_data.get("wearing", []) or [],
            )
            subjects.append(SubjectInfo(
                id=s.get("id", f"subj_{len(subjects)}"),
                name=s.get("name", "unknown"),
                instance_description=s.get("instance_description", ""),
                count=s.get("count", 1),
                attributes=attrs,
                current_pose_action=s.get("current_pose_action", ""),
                position_in_frame=s.get("position_in_frame", ""),
                is_animate=s.get("is_animate", False),
            ))

        # Parse relations
        relations = []
        for r in parsed.get("subject_relations", []):
            relations.append(SubjectRelation(
                subject_a=r.get("subject_a", ""),
                predicate=r.get("predicate", ""),
                subject_b=r.get("subject_b", ""),
            ))

        # Parse background
        bg_data = parsed.get("background", {})
        bg_elements = []
        for e in bg_data.get("elements", []):
            bg_elements.append(BackgroundElement(
                name=e.get("name", ""),
                type=e.get("type", "rigid"),
                current_state=e.get("current_state", ""),
                region=e.get("region", ""),
            ))
        background = BackgroundInfo(
            elements=bg_elements,
            lighting=bg_data.get("lighting", ""),
            weather=bg_data.get("weather", ""),
            time_of_day=bg_data.get("time_of_day", ""),
            foreground_background_separability=bg_data.get("foreground_background_separability", ""),
            rigid_background_ratio=bg_data.get("rigid_background_ratio", ""),
        )

        # Parse camera baseline
        cam_data = parsed.get("camera_baseline", {})
        camera_baseline = CameraBaseline(
            shot_type=cam_data.get("shot_type", ""),
            framing=cam_data.get("framing", ""),
            camera_angle=cam_data.get("camera_angle", ""),
            estimated_depth=cam_data.get("estimated_depth", ""),
            has_rigid_reference_structure=cam_data.get("has_rigid_reference_structure", False),
            scene_depth_complexity=cam_data.get("scene_depth_complexity", ""),
        )

        return ImageAnalysisResult(
            sample_id=sample_id,
            subjects=subjects,
            subject_count=parsed.get("subject_count", len(subjects)),
            has_multiple_subjects=parsed.get("has_multiple_subjects", len(subjects) > 1),
            has_same_category_instances=parsed.get("has_same_category_instances", False),
            subjects_clearly_distinguishable=parsed.get("subjects_clearly_distinguishable", True),
            subject_relations=relations,
            background=background,
            camera_baseline=camera_baseline,
            raw_vlm_output=raw_text,
            parse_success=True,
            parse_error=None,
        )
    except Exception as e:
        return ImageAnalysisResult(
            sample_id=sample_id,
            raw_vlm_output=raw_text,
            parse_success=False,
            parse_error=f"Schema construction error: {str(e)}",
        )


def _compute_statistics(results: list, output_dir: str) -> None:
    """Compute and write frequency statistics from image analysis results."""
    od = Path(output_dir)

    subject_cat_freq = Counter()
    subject_attr_freq = Counter()
    pose_action_freq = Counter()
    relation_freq = Counter()
    subject_count_freq = Counter()
    scene_setting_freq = Counter()
    bg_element_freq = Counter()
    bg_type_freq = Counter()
    shot_type_freq = Counter()
    camera_angle_freq = Counter()

    for r in results:
        if not r.parse_success:
            continue

        subject_count_freq[str(r.subject_count)] += 1

        for s in r.subjects:
            subject_cat_freq[s.name.lower()] += 1
            for c in s.attributes.color:
                subject_attr_freq[f"color:{c.lower()}"] += 1
            if s.attributes.size:
                subject_attr_freq[f"size:{s.attributes.size}"] += 1
            for st in s.attributes.state:
                subject_attr_freq[f"state:{st.lower()}"] += 1
            for w in s.attributes.wearing:
                subject_attr_freq[f"wearing:{w.lower()}"] += 1
            if s.current_pose_action:
                pose_action_freq[s.current_pose_action.lower()] += 1

        for rel in r.subject_relations:
            relation_freq[rel.predicate] += 1

        # Background
        if r.background.lighting:
            scene_setting_freq[f"lighting:{r.background.lighting}"] += 1
        if r.background.weather:
            scene_setting_freq[f"weather:{r.background.weather}"] += 1
        if r.background.time_of_day:
            scene_setting_freq[f"time:{r.background.time_of_day}"] += 1

        for e in r.background.elements:
            bg_element_freq[e.name.lower()] += 1
            bg_type_freq[e.type] += 1

        # Camera
        if r.camera_baseline.shot_type:
            shot_type_freq[r.camera_baseline.shot_type] += 1
        if r.camera_baseline.camera_angle:
            camera_angle_freq[r.camera_baseline.camera_angle] += 1

    write_freq_csv(str(od / "subject_category_freq.csv"), dict(subject_cat_freq), "category")
    write_freq_csv(str(od / "subject_attribute_freq.csv"), dict(subject_attr_freq), "attribute")
    write_freq_csv(str(od / "subject_pose_action_freq.csv"), dict(pose_action_freq), "pose_action")
    write_freq_csv(str(od / "subject_relation_freq.csv"), dict(relation_freq), "relation")
    write_freq_csv(str(od / "subject_count_distribution.csv"), dict(subject_count_freq), "subject_count")
    write_freq_csv(str(od / "scene_setting_freq.csv"), dict(scene_setting_freq), "setting")
    write_freq_csv(str(od / "bg_element_freq.csv"), dict(bg_element_freq), "element")
    write_freq_csv(str(od / "bg_element_type_distribution.csv"), dict(bg_type_freq), "type")
    write_freq_csv(str(od / "shot_type_distribution.csv"), dict(shot_type_freq), "shot_type")
    write_freq_csv(str(od / "camera_angle_distribution.csv"), dict(camera_angle_freq), "camera_angle")


async def _process_batch(
    client: SiliconFlowClient,
    batch: list,
    vlm_prompt: str,
    output_path: str,
    failed_path: str,
    semaphore: asyncio.Semaphore,
) -> list:
    """Process a batch of samples concurrently."""

    async def _process_one(item: ManifestItem):
        async with semaphore:
            raw_text = await client.async_call_vlm(item.image_path, vlm_prompt)
            result = _parse_vlm_response(item.sample_id, raw_text)

            if result.parse_success:
                append_jsonl(output_path, result)
            else:
                append_jsonl(output_path, result)
                append_jsonl(failed_path, {
                    "sample_id": item.sample_id,
                    "error": result.parse_error,
                })
            return result

    tasks = [_process_one(item) for item in batch]
    return await asyncio.gather(*tasks)


def analyze_images(config: dict) -> None:
    """
    Main entry: VLM image analysis with async concurrency and resume support.
    """
    manifest_path = str(Path(config["paths"]["manifest_dir"]) / "manifest_clean.jsonl")
    output_dir = str(Path(config["paths"]["output_dir"]) / "image_analysis")
    prompt_path = str(Path(config["paths"]["prompt_dir"]) / "vlm_image_parse.txt")
    ensure_dir(output_dir)

    output_path = str(Path(output_dir) / "image_parse.jsonl")
    failed_path = str(Path(output_dir) / "failed_samples.jsonl")

    # Load data
    manifest_items = read_jsonl(manifest_path, ManifestItem)
    if not manifest_items:
        logger.error("No clean manifest items found. Run step1 first.")
        return

    # Load prompt template
    vlm_prompt = load_prompt_template(prompt_path)

    # Resume support
    processed_ids = read_processed_ids(output_path)
    todo_items = [item for item in manifest_items if item.sample_id not in processed_ids]
    logger.info(
        f"Total clean samples: {len(manifest_items)}, "
        f"Already processed: {len(processed_ids)}, "
        f"Remaining: {len(todo_items)}"
    )

    if not todo_items:
        logger.info("All samples already processed. Skipping VLM calls.")
    else:
        # Initialize API client
        client = SiliconFlowClient(config)
        batch_size = client.batch_size
        semaphore = asyncio.Semaphore(batch_size)

        # Process in batches with progress bar
        async def _run_all():
            pbar = tqdm(total=len(todo_items), desc="VLM Image Analysis")
            for i in range(0, len(todo_items), batch_size):
                batch = todo_items[i : i + batch_size]
                await _process_batch(
                    client, batch, vlm_prompt, output_path, failed_path, semaphore
                )
                pbar.update(len(batch))
            pbar.close()

        asyncio.run(_run_all())

    # Compute statistics from all results
    logger.info("Computing image analysis statistics...")
    all_results = read_jsonl(output_path, ImageAnalysisResult)
    _compute_statistics(all_results, output_dir)

    # Summary
    success_count = sum(1 for r in all_results if r.parse_success)
    fail_count = len(all_results) - success_count
    logger.info("=" * 60)
    logger.info("Image Analysis Summary")
    logger.info("=" * 60)
    logger.info(f"Total analyzed:     {len(all_results)}")
    logger.info(f"Parse success:      {success_count}")
    logger.info(f"Parse failed:       {fail_count}")
    logger.info(f"Output: {output_path}")
