"""
Step 1: Build manifest from TIP-I2V raw data.
Supports two modes:
  1. HuggingFace datasets (download directly from tipi2v/TIP-I2V)
  2. Local files (parquet/jsonl/csv in raw_dir)
"""

import io
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from PIL import Image
from tqdm import tqdm

from i2vcompbench.utils.io_utils import clean_pika_prompt, ensure_dir, write_jsonl
from i2vcompbench.schemas.phase1_legacy import ManifestItem


def _validate_image(image_path: str) -> bool:
    """Check if image file exists and is readable."""
    path = Path(image_path)
    if not path.exists():
        return False
    try:
        img = Image.open(path)
        img.verify()
        return True
    except Exception:
        return False


def _load_from_huggingface(split: str = "Eval", max_samples: int = None) -> Optional[list]:
    """Load TIP-I2V dataset from HuggingFace using streaming to avoid downloading full dataset."""
    try:
        from datasets import load_dataset
        logger.info(f"Loading TIP-I2V from HuggingFace (split={split}, streaming=True)...")
        ds = load_dataset("tipi2v/TIP-I2V", split=split, streaming=True)
        
        # Collect samples using streaming (avoids downloading all parquet files)
        samples = []
        count = 0
        limit = max_samples or float('inf')
        for sample in ds:
            samples.append(sample)
            count += 1
            if count >= limit:
                break
            if count % 500 == 0:
                logger.info(f"  Streamed {count} samples...")
        
        logger.info(f"Loaded {len(samples)} samples from HuggingFace (streaming)")
        return samples
    except Exception as e:
        logger.warning(f"Failed to load from HuggingFace: {e}")
        return None


def _load_from_local(raw_dir: Path) -> Optional[pd.DataFrame]:
    """Try to load local files: parquet > jsonl > csv."""
    # Parquet
    parquet_files = list(raw_dir.glob("*.parquet"))
    if parquet_files:
        dfs = []
        for pf in parquet_files:
            logger.info(f"Loading parquet: {pf}")
            dfs.append(pd.read_parquet(pf))
        return pd.concat(dfs, ignore_index=True)

    # JSONL
    jsonl_files = list(raw_dir.glob("*.jsonl"))
    if jsonl_files:
        dfs = []
        for jf in jsonl_files:
            logger.info(f"Loading JSONL: {jf}")
            dfs.append(pd.read_json(jf, lines=True))
        return pd.concat(dfs, ignore_index=True)

    # CSV
    csv_files = list(raw_dir.glob("*.csv"))
    if csv_files:
        dfs = []
        for cf in csv_files:
            logger.info(f"Loading CSV: {cf}")
            dfs.append(pd.read_csv(cf))
        return pd.concat(dfs, ignore_index=True)

    return None


def _save_hf_image(img, output_path: Path) -> bool:
    """Save a HuggingFace PIL Image to file."""
    try:
        if img is None:
            return False
        if isinstance(img, Image.Image):
            if img.mode not in ("RGB",):
                img = img.convert("RGB")
            img.save(output_path, format="JPEG", quality=95)
            return True
        elif isinstance(img, dict) and "bytes" in img:
            pil_img = Image.open(io.BytesIO(img["bytes"]))
            if pil_img.mode not in ("RGB",):
                pil_img = pil_img.convert("RGB")
            pil_img.save(output_path, format="JPEG", quality=95)
            return True
        elif isinstance(img, bytes):
            pil_img = Image.open(io.BytesIO(img))
            if pil_img.mode not in ("RGB",):
                pil_img = pil_img.convert("RGB")
            pil_img.save(output_path, format="JPEG", quality=95)
            return True
        return False
    except Exception as e:
        logger.debug(f"Failed to save image: {e}")
        return False


def build_manifest(config: dict) -> None:
    """
    Main entry: scan raw data, validate, and produce manifest files.
    Tries HuggingFace first, then falls back to local files.
    """
    raw_dir = Path(config["paths"]["raw_dir"])
    manifest_dir = Path(config["paths"]["manifest_dir"])
    ensure_dir(str(manifest_dir))
    image_dir = manifest_dir / "images"
    ensure_dir(str(image_dir))

    max_samples = config.get("sampling", {}).get("max_samples", None)
    hf_split = config.get("sampling", {}).get("hf_split", "Eval")

    # ---- Try loading data ----
    hf_dataset = None
    local_df = None

    # Check if local files exist first
    has_local = any(raw_dir.glob("*.parquet")) or any(raw_dir.glob("*.jsonl")) or any(raw_dir.glob("*.csv"))

    if has_local:
        logger.info("Found local data files, loading from local...")
        local_df = _load_from_local(raw_dir)
    else:
        logger.info("No local data found, attempting HuggingFace download...")
        hf_dataset = _load_from_huggingface(hf_split, max_samples)

    if hf_dataset is None and local_df is None:
        logger.error(
            f"No data found. Either:\n"
            f"  1. Place TIP-I2V data files in {raw_dir}\n"
            f"  2. Ensure internet access for HuggingFace download"
        )
        return

    # ---- Process HuggingFace dataset (list of dicts) ----
    if hf_dataset is not None:
        total = len(hf_dataset)
        logger.info(f"Processing {total} samples from HuggingFace dataset...")

        all_items = []
        clean_items = []
        bad_items = []

        for idx in tqdm(range(total), desc="Building manifest"):
            row = hf_dataset[idx]

            # Extract fields (TIP-I2V column names)
            sample_id = row.get("UUID", f"sample_{idx:06d}")
            raw_prompt = str(row.get("Text_Prompt", "")).strip()
            subject_field = row.get("Subject", None)
            image_data = row.get("Image_Prompt", None)

            # If image_data is a PIL Image (HuggingFace returns PIL directly for image columns)
            # keep as-is, _save_hf_image handles PIL

            # Clean Pika prompt
            pika_result = clean_pika_prompt(raw_prompt)
            clean_text = pika_result["clean_text"]
            pika_camera = pika_result["pika_camera"]
            pika_motion = pika_result["pika_motion"]

            # Save image
            image_path = str(image_dir / f"{sample_id}.jpg")
            image_saved = _save_hf_image(image_data, Path(image_path))

            # Validate
            status = "ok"
            error_detail = None

            if not clean_text or len(clean_text) <= 3:
                status = "empty_prompt"
                error_detail = f"Clean prompt too short: '{clean_text}' (raw: '{raw_prompt[:100]}')"
            elif not image_saved:
                status = "missing_image"
                error_detail = "Failed to extract/save image"

            item = ManifestItem(
                sample_id=sample_id,
                prompt_text=raw_prompt,
                clean_prompt_text=clean_text,
                image_path=image_path if image_saved else "",
                subject_field=subject_field,
                pika_camera=pika_camera,
                pika_motion=pika_motion,
                status=status,
                error_detail=error_detail,
            )

            all_items.append(item)
            if status == "ok":
                clean_items.append(item)
            else:
                bad_items.append(item)

    # ---- Process local DataFrame ----
    elif local_df is not None:
        total = len(local_df)
        if max_samples and max_samples < total:
            local_df = local_df.head(max_samples)
            total = max_samples
            logger.info(f"Sampled to {max_samples} rows")

        logger.info(f"Processing {total} samples from local files...")

        # Auto-detect columns
        cols = local_df.columns.tolist()
        col_lower = {c.lower(): c for c in cols}

        prompt_col = None
        for c in ["text_prompt", "prompt", "text", "caption"]:
            if c in col_lower:
                prompt_col = col_lower[c]
                break

        image_col = None
        for c in ["image_prompt", "image", "img", "image_path"]:
            if c in col_lower:
                image_col = col_lower[c]
                break

        id_col = None
        for c in ["uuid", "id", "sample_id"]:
            if c in col_lower:
                id_col = col_lower[c]
                break

        subject_col = col_lower.get("subject", None)

        if prompt_col is None:
            logger.error(f"Cannot find prompt column. Available: {cols}")
            return

        logger.info(f"Column mapping: prompt={prompt_col}, image={image_col}, id={id_col}")

        all_items = []
        clean_items = []
        bad_items = []

        for idx, row in tqdm(local_df.iterrows(), total=total, desc="Building manifest"):
            raw_prompt = str(row.get(prompt_col, "")).strip()
            sample_id = str(row[id_col]) if id_col else f"sample_{idx:06d}"
            subject_field = str(row[subject_col]) if subject_col and subject_col in row.index else None

            pika_result = clean_pika_prompt(raw_prompt)
            clean_text = pika_result["clean_text"]
            pika_camera = pika_result["pika_camera"]
            pika_motion = pika_result["pika_motion"]

            # Handle image
            image_path = ""
            image_ok = False
            if image_col and image_col in row.index:
                img_data = row[image_col]
                if isinstance(img_data, str) and Path(img_data).exists():
                    image_path = img_data
                    image_ok = _validate_image(img_data)
                elif isinstance(img_data, (dict, bytes)):
                    image_path = str(image_dir / f"{sample_id}.jpg")
                    image_ok = _save_hf_image(img_data, Path(image_path))
                elif isinstance(img_data, Image.Image):
                    image_path = str(image_dir / f"{sample_id}.jpg")
                    image_ok = _save_hf_image(img_data, Path(image_path))

            status = "ok"
            error_detail = None
            if not clean_text or len(clean_text) <= 3:
                status = "empty_prompt"
                error_detail = f"Clean prompt too short: '{clean_text}'"
            elif not image_ok:
                status = "missing_image"
                error_detail = "Image not available or unreadable"

            item = ManifestItem(
                sample_id=sample_id,
                prompt_text=raw_prompt,
                clean_prompt_text=clean_text,
                image_path=image_path,
                subject_field=subject_field,
                pika_camera=pika_camera,
                pika_motion=pika_motion,
                status=status,
                error_detail=error_detail,
            )

            all_items.append(item)
            if status == "ok":
                clean_items.append(item)
            else:
                bad_items.append(item)

    # ---- Write outputs ----
    manifest_path = str(manifest_dir / "manifest.jsonl")
    clean_path = str(manifest_dir / "manifest_clean.jsonl")
    bad_path = str(manifest_dir / "bad_samples.jsonl")

    write_jsonl(manifest_path, all_items)
    write_jsonl(clean_path, clean_items)
    write_jsonl(bad_path, bad_items)

    # ---- Pika parameter statistics ----
    from collections import Counter
    camera_freq = Counter()
    motion_freq = Counter()
    for item in all_items:
        if item.pika_camera:
            camera_freq[item.pika_camera] += 1
        if item.pika_motion is not None:
            motion_freq[str(item.pika_motion)] += 1

    # ---- Print summary ----
    logger.info("=" * 60)
    logger.info("Manifest Build Summary")
    logger.info("=" * 60)
    logger.info(f"Total samples scanned:  {len(all_items)}")
    logger.info(f"Clean samples (ok):     {len(clean_items)}")
    logger.info(f"Bad samples:            {len(bad_items)}")
    if bad_items:
        status_counts = Counter(item.status for item in bad_items)
        for s, c in status_counts.most_common():
            logger.info(f"  - {s}: {c}")
    logger.info(f"Pika camera commands found: {sum(camera_freq.values())}")
    for cmd, cnt in camera_freq.most_common(10):
        logger.info(f"  - {cmd}: {cnt}")
    logger.info(f"Pika motion params found: {sum(motion_freq.values())}")
    logger.info(f"Output files:")
    logger.info(f"  {manifest_path}")
    logger.info(f"  {clean_path}")
    logger.info(f"  {bad_path}")
