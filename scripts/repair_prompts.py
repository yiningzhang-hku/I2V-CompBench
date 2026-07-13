"""
P1 Prompt Quality Repair Script - fixes prompt issues in phase3_manifest.jsonl.

Usage:
    python scripts/repair_prompts.py --phase rule
    python scripts/repair_prompts.py --phase llm
    python scripts/repair_prompts.py --phase all
"""
from __future__ import annotations
import argparse, asyncio, json, os, re, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
from wordfreq import zipf_frequency

MANIFEST_PATH = Path("data/benchmark_dataset/phase3_manifest.jsonl")
CHECKPOINT_PATH = Path("data/benchmark_dataset/prompt_repair_checkpoint.json")
ZIPF_THRESHOLD = 3.5
LLM_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
LLM_BASE_URL = "https://api.siliconflow.cn/v1"
LLM_BATCH_SIZE = 10
LLM_TIMEOUT = 180
LLM_MAX_RETRIES = 3
MEGA_BATCH_SIZE = 10  # prompts per single API call

# Camera cue words that should NOT be replaced (pans/tilts removed - we handle them)
CAMERA_CUE_WORDS = {"pan", "tilt", "zoom", "zooms", "dolly",
                    "rotating", "orbit", "crane", "tracking", "aerial", "steadicam"}

# All replacement words verified to have Zipf >= 3.5
SYNONYM_MAP: Dict[str, str] = {
    # Adverbs
    "subtly": "slightly", "dynamically": "quickly", "rhythmically": "slowly",
    "confidently": "quickly", "dimly": "barely", "urgently": "quickly",
    "vigorously": "quickly", "warmly": "softly", "vividly": "clearly",
    "vertically": "straight up", "gracefully": "gently",
    # Visual motion verbs
    "swirling": "spinning", "swirls": "spins", "swirl": "spin",
    "flicker": "flash", "flickers": "flashes", "flickering": "flashing",
    "pulses": "beats", "pulsing": "beating", "ripples": "waves",
    "ripple": "wave", "rippling": "flowing", "glows": "shines",
    "glides": "slides", "shimmering": "shining", "shimmer": "shine",
    "twinkling": "shining", "twinkle": "shine", "luminous": "bright",
    "translucent": "thin",
    # Weather/Nature
    "misty": "cloudy", "stormy": "dark", "sunlit": "bright", "turbulent": "wild",
    # Intensity
    "intensifies": "grows stronger", "intensifying": "growing stronger",
    "intensify": "grow stronger", "darkens": "grows darker",
    "deepens": "grows deeper", "thickens": "grows thicker",
    # Urban/Scene
    "futuristic": "modern", "cityscape": "city view", "skyline": "city line",
    # Body/Character
    "humanoid": "human-like", "hooded": "covered", "silhouette": "dark shape",
    "sways": "rocks", "flaps": "beats", "lowers": "drops",
    # Nature
    "foliage": "leaves", "turquoise": "blue", "teal": "blue", "hue": "color",
    # High-frequency verbs
    "adjusts": "shifts", "transforms": "changes", "accelerates": "speeds up",
    "accumulates": "builds up", "accumulating": "building up",
    "activates": "turns on", "activating": "turning on",
    "adorned": "covered", "alters": "changes", "ambiance": "feeling",
    "animates": "brings to life", "anthropomorphic": "human-like",
    "appendages": "arms", "arched": "curved", "arches": "curves",
    "cascading": "falling down", "cobblestone": "stone",
    "dislodges": "moves", "disperses": "spreads",
    "dissipates": "fades away", "dissipating": "fading away",
    "drifts": "moves slowly", "elongates": "stretches", "elongating": "stretching",
    "emerges": "appears", "envelops": "covers",
    "erupts": "bursts", "evaporates": "fades away", "expands": "grows",
    "fiery": "burning", "flutters": "moves softly", "fluttering": "moving softly",
    "glistens": "shines", "glistening": "shining",
    "hovering": "floating", "hovers": "hangs in the air",
    "illuminated": "lit up", "illuminates": "lights up", "illuminating": "lighting up",
    "jagged": "sharp", "lush": "green", "meanders": "flows",
    "ornate": "detailed", "oscillates": "moves back and forth",
    "oscillating": "moving back and forth", "overcast": "cloudy",
    "plumes": "clouds", "propels": "pushes", "quivers": "shakes",
    "radiates": "sends out", "radiating": "sending out",
    "recedes": "moves back", "receding": "moving back",
    "reverberates": "shakes", "rugged": "rough", "scorched": "burned",
    "serene": "calm", "shimmers": "shines", "shrouded": "covered",
    "sprawling": "spreading", "submerges": "goes under",
    "submerged": "under water", "surges": "rushes", "surging": "rushing",
    "swaying": "moving slowly", "tendrils": "thin lines",
    "towering": "tall", "trembles": "shakes", "trembling": "shaking",
    "undulates": "waves", "undulating": "waving",
    "unfolds": "opens up", "unfurls": "opens out", "unfurling": "opening out",
    "weathered": "worn", "withered": "dried", "writhing": "twisting",
    "writhe": "twist", "tousled": "messy", "tattered": "torn",
    "ethereal": "pale", "bustling": "busy",
    "billowing": "flowing", "billows": "flows",
    "bioluminescent": "bright", "cobbled": "stone",
    "crumbles": "breaks apart", "crumbling": "breaking apart",
    "desolate": "empty", "dilapidated": "broken down",
    "engulfs": "covers", "enveloping": "covering",
    "fissure": "crack", "fissures": "cracks",
    "fluorescent": "bright", "frosted": "cold", "glacial": "icy",
    "holographic": "light-based", "incandescent": "bright hot",
    "labyrinth": "maze", "molten": "melted", "monolithic": "massive",
    "otherworldly": "strange", "parched": "dry", "pristine": "clean",
    "plummets": "falls fast", "plummeting": "falling fast",
    "pulsates": "beats", "pulsating": "beating", "radiant": "bright",
    "retracts": "pulls back", "retracting": "pulling back",
    "spectral": "pale", "stalactites": "hanging rocks",
    "subterranean": "underground", "verdant": "green", "volcanic": "burning",
    "vortex": "spinning mass", "warps": "twists",
    "windswept": "wind-blown", "zenith": "top",
    # Camera terms (safe since we added "camera moves"/"moves left" to cues)
    "pans": "moves", "tilts": "angles",
    # Additional common rare words
    "starry": "star-lit", "snowy": "white",
    "cyberpunk": "high-tech", "hues": "colors",
    "silhouetted": "outlined", "hazy": "cloudy",
    "mossy": "green", "grassy": "green",
    "cybernetic": "mechanical", "bearded": "hairy",
    "cloaked": "covered", "foggy": "cloudy",
    "faintly": "softly", "ghostly": "pale",
    "dusk": "evening", "rhythmic": "steady",
    "outstretched": "spread out", "joyful": "happy",
    "eerie": "strange", "dreamy": "soft",
    "moonlit": "night-lit", "spiky": "sharp",
    "smoky": "dark", "shadowy": "dark",
    "menacing": "threatening", "dappled": "spotted",
    "stylized": "designed", "figurine": "small figure",
    # Round 2 - post-LLM remaining rare words
    "blinks": "closes eyes", "blossoms": "flowers",
    "fades": "goes away", "wiggles": "moves",
    "wags": "shakes", "lanterns": "lights",
    "goggles": "glasses", "robes": "long clothes",
    "candlelight": "candle light", "raindrops": "drops",
    "fireflies": "small lights", "splashing": "hitting water",
    "swims": "moves in water", "bony": "thin",
    "dunes": "sand hills", "mane": "hair",
}


def fix_repeated_articles(prompt: str) -> str:
    prompt = re.sub(r'\b(the)\s+\1\b', r'\1', prompt, flags=re.IGNORECASE)
    prompt = re.sub(r'\b(a)\s+\1\b', r'\1', prompt, flags=re.IGNORECASE)
    prompt = re.sub(r'\b(an)\s+\1\b', r'\1', prompt, flags=re.IGNORECASE)
    return prompt


def fix_article_before_punctuation(prompt: str) -> str:
    return re.sub(r'\b(the|a|an)\s*([.,;:!?])', r'\2', prompt, flags=re.IGNORECASE)


def fix_rare_words_by_synonym(prompt: str, dimension: str = "") -> Tuple[str, List[str]]:
    """Replace rare words using synonym map. Skips camera cue words contextually."""
    replaced = []
    result = prompt
    for rare_word, replacement in SYNONYM_MAP.items():
        if rare_word in CAMERA_CUE_WORDS:
            low_prompt = result.lower()
            camera_patterns = ["camera " + rare_word, rare_word + " left", rare_word + " right",
                             rare_word + " up", rare_word + " down"]
            if any(cp in low_prompt for cp in camera_patterns):
                continue
        pattern = re.compile(r'\b' + re.escape(rare_word) + r'\b', re.IGNORECASE)
        if pattern.search(result):
            repl = replacement
            def replace_match(m, repl=repl):
                original = m.group(0)
                if original[0].isupper() and repl[0].islower():
                    return repl[0].upper() + repl[1:]
                return repl
            new_result = pattern.sub(replace_match, result)
            if new_result != result:
                replaced.append(rare_word)
                result = new_result
    return result, replaced


def check_still_has_rare_words(prompt: str) -> List[str]:
    tokens = re.findall(r'\b[a-zA-Z]+\b', prompt)
    rare = []
    seen = set()
    for i, token in enumerate(tokens):
        if len(token) <= 2 or token.isupper():
            continue
        if i > 0 and token[0].isupper():
            continue
        low = token.lower()
        if low in seen:
            continue
        score = zipf_frequency(low, "en")
        if score < ZIPF_THRESHOLD:
            rare.append(low)
            seen.add(low)
    return rare


def create_llm_client():
    import httpx
    from openai import OpenAI
    api_key = os.environ.get("SILICONFLOW_API_KEY", "")
    if not api_key:
        raise ValueError("SILICONFLOW_API_KEY not set")
    http_client = httpx.Client(timeout=httpx.Timeout(connect=30, read=120, write=30, pool=30))
    return OpenAI(api_key=api_key, base_url=LLM_BASE_URL, http_client=http_client)


def _parse_numbered_response(content: str, expected_count: int) -> List[str]:
    """Parse numbered LLM response like '1. prompt text here'."""
    lines = content.strip().split("\n")
    parsed = [""] * expected_count
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(\d+)[.):\s]+(.+)', line)
        if m:
            idx = int(m.group(1)) - 1
            text = m.group(2).strip().strip('"').strip("'").strip()
            if 0 <= idx < expected_count and text:
                parsed[idx] = text
    return parsed


def llm_mega_batch(client, records: List[Dict[str, Any]], issue_type: str,
                   rec_map: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Process records via LLM using mega-batching with checkpoint support."""
    results = []
    total_batches = (len(records) + MEGA_BATCH_SIZE - 1) // MEGA_BATCH_SIZE
    consecutive_failures = 0
    for mb_idx, mb_start in enumerate(range(0, len(records), MEGA_BATCH_SIZE)):
        mb = records[mb_start:mb_start + MEGA_BATCH_SIZE]
        
        if issue_type == "rare_words":
            header = ("You are a video description editor. Rewrite each prompt below "
                     "replacing uncommon/rare words with simpler, common English alternatives.\n"
                     "RULES: Keep same meaning. 8-25 words each. Keep proper nouns. "
                     "Output: numbered lines '1. rewritten prompt'\n")
            items = "\n".join(f"{i+1}. [{','.join(r.get('_rare_words',[])[:5])}] {r['prompt']}" 
                           for i, r in enumerate(mb))
        elif issue_type == "too_short":
            header = ("Expand each too-short prompt to 8-20 words keeping same meaning.\n"
                     "Add motion details. Use common words. Output: numbered '1. expanded'\n")
            items = "\n".join(f"{i+1}. {r['prompt']}" for i, r in enumerate(mb))
        elif issue_type == "missing_verb":
            header = ("Add an action/motion verb to each prompt.\n"
                     "Keep 8-25 words. Use common words. Output: numbered '1. fixed'\n")
            items = "\n".join(f"{i+1}. {r['prompt']}" for i, r in enumerate(mb))
        elif issue_type == "missing_camera":
            header = ("Add a camera movement cue to each prompt.\n"
                     "Use: camera moves, zoom in, zoom out, tracking shot. "
                     "Keep 8-25 words. Output: numbered '1. fixed'\n")
            items = "\n".join(f"{i+1}. {r['prompt']}" for i, r in enumerate(mb))
        else:
            continue
        
        mega_prompt = header + "\n" + items
        batch_ok = False
        
        for attempt in range(LLM_MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": mega_prompt}],
                    max_tokens=1500, temperature=0.3, timeout=LLM_TIMEOUT,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                content = response.choices[0].message.content or ""
                parsed = _parse_numbered_response(content, len(mb))
                for idx, rec in enumerate(mb):
                    new_p = parsed[idx] if idx < len(parsed) else ""
                    if new_p and 3 <= len(new_p.split()) <= 30:
                        results.append({"question_id": rec["question_id"], "new_prompt": new_p})
                    else:
                        results.append({"question_id": rec["question_id"], "new_prompt": rec["prompt"]})
                batch_ok = True
                consecutive_failures = 0
                break
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg:
                    wait_time = 30 * (2 ** attempt)
                    print(f"    Rate limit, waiting {wait_time}s...", flush=True)
                    time.sleep(wait_time)
                elif "timeout" in err_msg.lower() or "timed out" in err_msg.lower():
                    print(f"    Timeout (attempt {attempt+1}), retrying...", flush=True)
                    time.sleep(5)
                else:
                    print(f"    LLM error (attempt {attempt+1}): {err_msg[:100]}", flush=True)
                    time.sleep(3 * (2 ** attempt))
                if attempt == LLM_MAX_RETRIES - 1:
                    for rec in mb:
                        results.append({"question_id": rec["question_id"], "new_prompt": rec["prompt"]})
        
        if not batch_ok:
            consecutive_failures += 1
            if consecutive_failures >= 5:
                print(f"    WARNING: 5 consecutive failures, stopping early.", flush=True)
                for rec in records[mb_start + MEGA_BATCH_SIZE:]:
                    results.append({"question_id": rec["question_id"], "new_prompt": rec["prompt"]})
                break
        
        if (mb_idx + 1) % 10 == 0:
            print(f"      Progress: {mb_idx+1}/{total_batches} batches", flush=True)
            # Checkpoint: apply partial results to rec_map and save
            if rec_map is not None:
                _apply_and_save_checkpoint(results, rec_map)
        time.sleep(0.5)
    return results


def _apply_and_save_checkpoint(results: List[Dict[str, Any]], rec_map: Dict[str, Any]) -> None:
    """Apply results so far and save manifest as checkpoint."""
    changed = 0
    for r in results:
        qid, new_p = r["question_id"], r["new_prompt"]
        if qid in rec_map and new_p != rec_map[qid]["prompt"]:
            rec_map[qid]["prompt"] = new_p
            changed += 1
    # Save the manifest with current progress
    all_records = list(rec_map.values())
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"      [Checkpoint saved: {changed} prompts updated]", flush=True)


def load_manifest() -> List[Dict[str, Any]]:
    records = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_manifest(records: List[Dict[str, Any]]) -> None:
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def phase_rule(records: List[Dict[str, Any]]) -> Dict[str, int]:
    stats = {"repeated_article_fixed": 0, "article_punctuation_fixed": 0,
             "synonym_replaced_records": 0, "synonym_words_replaced": 0, "total_modified": 0}
    for rec in records:
        prompt = rec["prompt"]
        modified = False
        new_prompt = fix_repeated_articles(prompt)
        if new_prompt != prompt:
            stats["repeated_article_fixed"] += 1
            prompt = new_prompt
            modified = True
        new_prompt = fix_article_before_punctuation(prompt)
        if new_prompt != prompt:
            stats["article_punctuation_fixed"] += 1
            prompt = new_prompt
            modified = True
        new_prompt, replaced = fix_rare_words_by_synonym(prompt, rec.get("dimension", ""))
        if replaced:
            stats["synonym_replaced_records"] += 1
            stats["synonym_words_replaced"] += len(replaced)
            prompt = new_prompt
            modified = True
        if modified:
            rec["prompt"] = prompt
            stats["total_modified"] += 1
    return stats


def phase_llm(records: List[Dict[str, Any]]) -> Dict[str, int]:
    stats = {"rare_word_llm_fixed": 0, "too_short_fixed": 0,
             "missing_verb_fixed": 0, "missing_camera_fixed": 0, "llm_calls": 0}
    
    from i2vcompbench.quality.prompt_rules import (
        check_word_count, check_change_verb, check_view_camera_cues,
    )
    
    needs_rare, needs_expand, needs_verb, needs_camera = [], [], [], []
    for rec in records:
        prompt, dim = rec["prompt"], rec["dimension"]
        rare = check_still_has_rare_words(prompt)
        if rare:
            rec["_rare_words"] = rare
            needs_rare.append(rec)
        if "word_count_too_short" in check_word_count(prompt, 8, 25):
            needs_expand.append(rec)
        if check_change_verb(prompt):
            needs_verb.append(rec)
        if check_view_camera_cues(prompt, dim):
            needs_camera.append(rec)
    
    print(f"\n  LLM repair targets:", flush=True)
    print(f"    Rare words: {len(needs_rare)} | Short: {len(needs_expand)} | "
          f"No verb: {len(needs_verb)} | No camera: {len(needs_camera)}", flush=True)
    
    client = create_llm_client()
    rec_map = {r["question_id"]: r for r in records}
    
    def process(items, issue_type, stat_key):
        total = len(items)
        if not total:
            return
        api_calls = (total + MEGA_BATCH_SIZE - 1) // MEGA_BATCH_SIZE
        print(f"    [{issue_type}] {total} records, ~{api_calls} API calls...", flush=True)
        results = llm_mega_batch(client, items, issue_type, rec_map=rec_map)
        stats["llm_calls"] += api_calls
        fixed = 0
        for r in results:
            qid, new_p = r["question_id"], r["new_prompt"]
            if qid in rec_map and new_p != rec_map[qid]["prompt"]:
                rec_map[qid]["prompt"] = new_p
                fixed += 1
        stats[stat_key] = fixed
        print(f"    [{issue_type}] Fixed: {fixed}/{total}", flush=True)
        # Save after each issue type
        save_manifest(list(rec_map.values()))
        print(f"    [{issue_type}] Checkpoint saved.", flush=True)
    
    process(needs_camera, "missing_camera", "missing_camera_fixed")
    process(needs_expand, "too_short", "too_short_fixed")
    process(needs_verb, "missing_verb", "missing_verb_fixed")
    
    # Process rare words (all of them - mega-batching keeps API calls manageable)
    if needs_rare:
        needs_rare.sort(key=lambda r: len(r.get("_rare_words", [])), reverse=True)
        process(needs_rare, "rare_words", "rare_word_llm_fixed")
    
    for rec in records:
        rec.pop("_rare_words", None)
    return stats


def print_comparison(records: List[Dict[str, Any]]) -> None:
    from i2vcompbench.quality.prompt_rules import check_all_rules, load_dimension_forbidden_words
    forbidden = load_dimension_forbidden_words()
    clean, issues_count = 0, 0
    issue_dist: Dict[str, int] = {}
    for rec in records:
        issues = check_all_rules(rec["prompt"], rec["dimension"],
                                forbidden.get(rec["dimension"], []), 8, 25, ZIPF_THRESHOLD)
        if issues:
            issues_count += 1
            for iss in issues:
                key = iss.split(":")[0]
                issue_dist[key] = issue_dist.get(key, 0) + 1
        else:
            clean += 1
    total = len(records)
    print(f"\n{'='*60}")
    print(f"  Prompt Rules Check Results")
    print(f"{'='*60}")
    print(f"  Total: {total} | Clean: {clean} ({clean/total*100:.1f}%) | Issues: {issues_count}")
    print(f"  Issue distribution:")
    for k, v in sorted(issue_dist.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="P1 Prompt Repair")
    parser.add_argument("--phase", choices=["rule", "llm", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])

    print("=" * 60)
    print("  P1 Prompt Quality Repair")
    print("=" * 60)
    print(f"\n  Loading {MANIFEST_PATH}...")
    records = load_manifest()
    print(f"  Loaded {len(records)} records")
    print(f"  Before: clean=729 / issues=2788 (3517 total)")

    if args.phase in ("rule", "all"):
        print(f"\n  --- Phase 1: Rule-based fixes ---")
        rule_stats = phase_rule(records)
        print(f"  Stats: articles={rule_stats['repeated_article_fixed']}, "
              f"punctuation={rule_stats['article_punctuation_fixed']}, "
              f"synonyms={rule_stats['synonym_replaced_records']} records "
              f"({rule_stats['synonym_words_replaced']} words), "
              f"total modified={rule_stats['total_modified']}")
        if not args.dry_run:
            save_manifest(records)
            print(f"  Saved to {MANIFEST_PATH}")
        print_comparison(records)

    if args.phase in ("llm", "all"):
        print(f"\n  --- Phase 2: LLM-assisted fixes ---")
        if args.dry_run:
            print("  [DRY-RUN] Skipping LLM fixes")
        else:
            if args.phase == "llm":
                records = load_manifest()
            llm_stats = phase_llm(records)
            print(f"\n  LLM Stats: rare={llm_stats['rare_word_llm_fixed']}, "
                  f"short={llm_stats['too_short_fixed']}, "
                  f"verb={llm_stats['missing_verb_fixed']}, "
                  f"camera={llm_stats['missing_camera_fixed']}, "
                  f"API calls={llm_stats['llm_calls']}")
            save_manifest(records)
            print(f"  Saved to {MANIFEST_PATH}")

    if not args.dry_run:
        print(f"\n  --- Final Verification ---")
        records = load_manifest()
        print_comparison(records)


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
