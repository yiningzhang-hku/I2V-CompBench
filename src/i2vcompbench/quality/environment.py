"""环境体检模块 — 检查依赖、资源和实验可用性。

不安装任何依赖，仅检测当前环境状态并生成报告。
"""

from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


# ============================================================
# 依赖检查列表
# ============================================================

CORE_DEPENDENCIES: list[tuple[str, str]] = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("wordfreq", "wordfreq"),
    ("opencv", "cv2"),
    ("scikit-image", "skimage"),
    ("pydantic", "pydantic"),
    ("PIL/Pillow", "PIL"),
    ("yaml", "yaml"),
    ("loguru", "loguru"),
]

MODEL_DEPENDENCIES: list[tuple[str, str]] = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("sentence-transformers", "sentence_transformers"),
]

OPTIONAL_DEPENDENCIES: list[tuple[str, str]] = [
    ("gfpgan", "gfpgan"),
    ("diffusers", "diffusers"),
]


# ============================================================
# 检查函数
# ============================================================


def check_python_version() -> dict:
    """检查Python版本。"""
    info = sys.version_info
    return {
        "version": f"{info.major}.{info.minor}.{info.micro}",
        "executable": sys.executable,
        "major": info.major,
        "minor": info.minor,
        "micro": info.micro,
        "meets_minimum": info >= (3, 10),
    }


def check_dependencies(deps: list[tuple[str, str]]) -> list[dict]:
    """检查依赖列表可导入性，返回每项状态。

    Args:
        deps: [(display_name, import_name), ...]

    Returns:
        [{"name": str, "import_name": str, "available": bool, "version": str|None}, ...]
    """
    results: list[dict] = []
    for name, import_name in deps:
        entry: dict[str, Any] = {
            "name": name,
            "import_name": import_name,
            "available": False,
            "version": None,
        }
        try:
            mod = importlib.import_module(import_name)
            entry["available"] = True
            # 尝试获取版本号
            version = getattr(mod, "__version__", None)
            if version is None:
                version = getattr(mod, "VERSION", None)
            if version is not None:
                entry["version"] = str(version)
        except ImportError:
            pass
        except Exception as e:
            entry["error"] = str(e)
        results.append(entry)
    return results


def check_data_structure(benchmark_root: Path) -> dict:
    """检查data目录结构完整性。

    检查项：
    - phase3_manifest.jsonl 存在且非空
    - question_plans.jsonl 存在且非空
    - input_assets_manifest.jsonl 存在且非空
    - first_frames/ 目录存在且含图像
    - prompts/final_prompts.jsonl 存在
    - by_dimension/ 目录存在
    """
    result: dict[str, Any] = {
        "benchmark_root": str(benchmark_root),
        "exists": benchmark_root.exists(),
        "files_present": {},
        "first_frames_count": 0,
        "candidate_count": 0,
    }

    if not benchmark_root.exists():
        logger.warning(f"Benchmark root不存在: {benchmark_root}")
        return result

    # 检查关键文件
    key_files = {
        "phase3_manifest": benchmark_root / "phase3_manifest.jsonl",
        "question_plans": benchmark_root / "question_plans.jsonl",
        "input_assets_manifest": benchmark_root / "input_assets_manifest.jsonl",
        "final_prompts": benchmark_root / "prompts" / "final_prompts.jsonl",
    }

    for key, path in key_files.items():
        present = path.exists() and path.stat().st_size > 0
        result["files_present"][key] = present

    # 检查目录
    key_dirs = {
        "first_frames": benchmark_root / "first_frames",
        "by_dimension": benchmark_root / "by_dimension",
    }
    for key, path in key_dirs.items():
        result["files_present"][key] = path.exists() and path.is_dir()

    # 统计first_frames中的图像数
    first_frames_dir = benchmark_root / "first_frames"
    if first_frames_dir.exists():
        image_exts = {".png", ".jpg", ".jpeg", ".webp"}
        count = sum(
            1
            for f in first_frames_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_exts
        )
        result["first_frames_count"] = count

    # 统计候选数（phase3_manifest行数）
    manifest_path = benchmark_root / "phase3_manifest.jsonl"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                result["candidate_count"] = sum(1 for line in f if line.strip())
        except Exception:
            pass

    return result


def check_phase1_bundle(phase1_dir: str | None) -> dict:
    """检查Phase 1 bundle可用性。

    Returns:
        {"available": bool, "reason": str|None, "details": dict|None}
    """
    if phase1_dir is None:
        return {"available": False, "reason": "path_is_null", "details": None}

    p = Path(phase1_dir)
    if not p.exists():
        return {"available": False, "reason": "path_not_found", "details": {"path": str(p)}}

    # 检查关键文件
    expected_files = [
        "prior_analysis.jsonl",
        "sampled_recipes.jsonl",
    ]
    missing = [f for f in expected_files if not (p / f).exists()]
    if missing:
        return {
            "available": False,
            "reason": "missing_files",
            "details": {"path": str(p), "missing": missing},
        }

    return {"available": True, "reason": None, "details": {"path": str(p)}}


def check_model_weights(models_config: dict | None) -> dict:
    """检查模型权重文件。

    Args:
        models_config: 包含模型权重路径的配置字典，格式为 {"model_name": "path/to/weight", ...}

    Returns:
        {"available": [str], "missing": [str]}
    """
    known_models = ["realesrgan", "swinir", "gfpgan", "clip", "dino"]

    if models_config is None:
        return {"available": [], "missing": known_models}

    available: list[str] = []
    missing: list[str] = []

    for model_name in known_models:
        weight_path = models_config.get(model_name)
        if weight_path and Path(weight_path).exists():
            available.append(model_name)
        else:
            missing.append(model_name)

    return {"available": available, "missing": missing}


def check_gpu() -> dict:
    """检查GPU可用性。

    Returns:
        {"available": bool, "device_count": int, "devices": list, "vram_gb": list}
    """
    result: dict[str, Any] = {
        "available": False,
        "device_count": 0,
        "devices": [],
        "vram_gb": [],
    }

    try:
        import torch

        if torch.cuda.is_available():
            result["available"] = True
            count = torch.cuda.device_count()
            result["device_count"] = count
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                result["devices"].append(props.name)
                vram_gb = round(props.total_mem / (1024**3), 2)
                result["vram_gb"].append(vram_gb)
    except ImportError:
        logger.debug("torch未安装，跳过GPU检查")
    except Exception as e:
        logger.debug(f"GPU检查异常: {e}")

    return result


def determine_experiment_availability(
    core_deps: list[dict],
    model_deps: list[dict],
    data_check: dict,
    phase1_check: dict,
    gpu_check: dict,
) -> dict:
    """根据资源状态判断各实验可用性。

    实验状态：
    - "core-ready": 可立即执行（无外部依赖）
    - "optional-ready": 资源就绪可执行
    - "blocked": 缺少必要资源
    - "awaiting-human": 等待人工标注

    Returns:
        每项实验的 {"status": str, "reason": str|None, "blocked_by": list|None}
    """
    # 辅助：检查某个依赖是否可用
    def _dep_available(deps_list: list[dict], import_name: str) -> bool:
        for d in deps_list:
            if d["import_name"] == import_name:
                return d["available"]
        return False

    has_gpu = gpu_check.get("available", False)
    has_torch = _dep_available(model_deps, "torch")
    has_transformers = _dep_available(model_deps, "transformers")
    has_phase1 = phase1_check.get("available", False)
    has_data = data_check.get("exists", False)

    # 检查核心依赖是否全部就绪
    core_all_available = all(d["available"] for d in core_deps)

    experiments: dict[str, dict[str, Any]] = {}

    # --- Core-ready 实验 ---
    core_ready_list = [
        "audit",
        "split",
        "prompt_rules",
        "asset_manifest",
        "asset_lineage",
    ]
    for exp in core_ready_list:
        if core_all_available and has_data:
            experiments[exp] = {"status": "core-ready", "reason": None, "blocked_by": None}
        else:
            blockers = []
            if not core_all_available:
                blockers.append("core_dependencies")
            if not has_data:
                blockers.append("data_structure")
            experiments[exp] = {
                "status": "blocked",
                "reason": "missing_core_resources",
                "blocked_by": blockers,
            }

    # --- target_repair_phase1: 需Phase 1 bundle ---
    if has_phase1 and core_all_available:
        experiments["target_repair_phase1"] = {"status": "core-ready", "reason": None, "blocked_by": None}
    else:
        blockers = []
        if not has_phase1:
            blockers.append("phase1_bundle")
        if not core_all_available:
            blockers.append("core_dependencies")
        experiments["target_repair_phase1"] = {
            "status": "blocked",
            "reason": "missing_phase1_bundle",
            "blocked_by": blockers,
        }

    # --- target_repair_vlm: 需API ---
    experiments["target_repair_vlm"] = {
        "status": "blocked",
        "reason": "requires_vlm_api",
        "blocked_by": ["vlm_api"],
    }

    # --- prompt_experiment: 需API ---
    experiments["prompt_experiment"] = {
        "status": "blocked",
        "reason": "requires_llm_api",
        "blocked_by": ["llm_api"],
    }

    # --- clarity_c0_c1: Lanczos/Unsharp 不需要GPU ---
    if core_all_available and has_data:
        experiments["clarity_c0_c1"] = {"status": "core-ready", "reason": None, "blocked_by": None}
    else:
        experiments["clarity_c0_c1"] = {
            "status": "blocked",
            "reason": "missing_core_resources",
            "blocked_by": ["core_dependencies"],
        }

    # --- clarity_c2_c3: 需GPU+权重 ---
    if has_gpu and has_torch and has_data:
        experiments["clarity_c2_c3"] = {"status": "optional-ready", "reason": None, "blocked_by": None}
    else:
        blockers = []
        if not has_torch:
            blockers.append("torch")
        if not has_gpu:
            blockers.append("gpu")
        blockers.append("realesrgan_weight")
        experiments["clarity_c2_c3"] = {
            "status": "blocked",
            "reason": "missing_gpu_and_model_weights",
            "blocked_by": blockers,
        }

    # --- clarity_c4_c7: 需GPU+权重+可选依赖 ---
    blockers_c4 = []
    if not has_torch:
        blockers_c4.append("torch")
    if not has_gpu:
        blockers_c4.append("gpu")
    if not _dep_available(core_deps + model_deps, "gfpgan"):
        blockers_c4.append("gfpgan")
    blockers_c4.append("model_weights")
    experiments["clarity_c4_c7"] = {
        "status": "blocked",
        "reason": "missing_gpu_models_and_optional_deps",
        "blocked_by": blockers_c4,
    }

    # --- aspect_d0_d4: 基础策略不需GPU ---
    if core_all_available and has_data:
        experiments["aspect_d0_d4"] = {"status": "core-ready", "reason": None, "blocked_by": None}
    else:
        experiments["aspect_d0_d4"] = {
            "status": "blocked",
            "reason": "missing_core_resources",
            "blocked_by": ["core_dependencies"],
        }

    # --- aspect_d5: Outpainting需GPU ---
    if has_gpu and has_torch:
        experiments["aspect_d5"] = {"status": "optional-ready", "reason": None, "blocked_by": None}
    else:
        blockers = []
        if not has_torch:
            blockers.append("torch")
        if not has_gpu:
            blockers.append("gpu")
        blockers.append("diffusers")
        experiments["aspect_d5"] = {
            "status": "blocked",
            "reason": "requires_gpu_and_diffusers",
            "blocked_by": blockers,
        }

    # --- subject_tier: 纯规则+词频 ---
    if core_all_available:
        experiments["subject_tier"] = {"status": "core-ready", "reason": None, "blocked_by": None}
    else:
        experiments["subject_tier"] = {
            "status": "blocked",
            "reason": "missing_core_dependencies",
            "blocked_by": ["wordfreq"],
        }

    # --- difficulty_features: 规则提取 ---
    if core_all_available and has_data:
        experiments["difficulty_features"] = {"status": "core-ready", "reason": None, "blocked_by": None}
    else:
        experiments["difficulty_features"] = {
            "status": "blocked",
            "reason": "missing_core_resources",
            "blocked_by": ["core_dependencies"],
        }

    # --- difficulty_calibration: 等待人工标注 ---
    experiments["difficulty_calibration"] = {
        "status": "awaiting-human",
        "reason": "requires_human_annotations",
        "blocked_by": ["human_difficulty_labels"],
    }

    # --- orthogonal: 需外部outcome ---
    experiments["orthogonal"] = {
        "status": "blocked",
        "reason": "requires_external_outcome_data",
        "blocked_by": ["model_evaluation_outcomes"],
    }

    # --- ablation: 需前序实验完成 ---
    experiments["ablation"] = {
        "status": "blocked",
        "reason": "requires_prior_experiments_complete",
        "blocked_by": ["clarity_experiment", "aspect_experiment", "difficulty_experiment"],
    }

    # --- final_selection: 等待人工 ---
    experiments["final_selection"] = {
        "status": "awaiting-human",
        "reason": "requires_human_review_and_decision",
        "blocked_by": ["human_final_review"],
    }

    return experiments


def run_environment_check(
    benchmark_root: Path,
    output_dir: Path,
    config: dict | None = None,
) -> dict:
    """执行完整环境体检，输出报告文件。

    输出：
    - environment_report.json — 完整体检结果
    - experiment_availability.json — 各实验可用性

    Args:
        benchmark_root: 基准数据集根目录
        output_dir: 报告输出目录
        config: 可选配置字典（含phase1_bundle_dir, model_weights等）

    Returns:
        summary dict
    """
    logger.info("开始环境体检...")
    timestamp = datetime.now().isoformat(timespec="seconds")

    # 1. Python版本
    python_info = check_python_version()
    logger.info(f"Python {python_info['version']} @ {python_info['executable']}")

    # 2. 依赖检查
    core_deps = check_dependencies(CORE_DEPENDENCIES)
    model_deps = check_dependencies(MODEL_DEPENDENCIES)
    optional_deps = check_dependencies(OPTIONAL_DEPENDENCIES)

    core_available = sum(1 for d in core_deps if d["available"])
    logger.info(f"核心依赖: {core_available}/{len(core_deps)} 可用")
    model_available = sum(1 for d in model_deps if d["available"])
    logger.info(f"模型依赖: {model_available}/{len(model_deps)} 可用")

    # 3. 数据结构
    data_check = check_data_structure(benchmark_root)
    logger.info(f"数据目录: {'存在' if data_check['exists'] else '不存在'}")
    if data_check["first_frames_count"] > 0:
        logger.info(f"  first_frames: {data_check['first_frames_count']} 张图像")
    if data_check["candidate_count"] > 0:
        logger.info(f"  候选数: {data_check['candidate_count']}")

    # 4. Phase 1 bundle
    phase1_dir = (config or {}).get("phase1_bundle_dir")
    phase1_check = check_phase1_bundle(phase1_dir)
    logger.info(f"Phase 1 bundle: {'可用' if phase1_check['available'] else phase1_check['reason']}")

    # 5. 模型权重
    models_config = (config or {}).get("model_weights")
    weights_check = check_model_weights(models_config)
    logger.info(f"模型权重: {len(weights_check['available'])} 可用, {len(weights_check['missing'])} 缺失")

    # 6. GPU
    gpu_check = check_gpu()
    if gpu_check["available"]:
        logger.info(f"GPU: {gpu_check['device_count']} 设备 — {gpu_check['devices']}")
    else:
        logger.info("GPU: 不可用")

    # 7. 实验可用性判断
    experiments = determine_experiment_availability(
        core_deps=core_deps,
        model_deps=model_deps,
        data_check=data_check,
        phase1_check=phase1_check,
        gpu_check=gpu_check,
    )

    # 8. 汇总
    status_counts: dict[str, int] = {}
    for exp_info in experiments.values():
        s = exp_info["status"]
        status_counts[s] = status_counts.get(s, 0) + 1

    # 整体状态判断
    if all(e["status"] in ("core-ready", "optional-ready") for e in experiments.values()):
        overall_status = "full"
    elif any(e["status"] == "core-ready" for e in experiments.values()):
        overall_status = "partial"
    else:
        overall_status = "minimal"

    # 构建报告
    environment_report = {
        "timestamp": timestamp,
        "python": python_info,
        "core_dependencies": core_deps,
        "model_dependencies": model_deps,
        "optional_dependencies": optional_deps,
        "data_structure": data_check,
        "phase1_bundle": phase1_check,
        "model_weights": weights_check,
        "gpu": gpu_check,
        "overall_status": overall_status,
    }

    experiment_availability = {
        "timestamp": timestamp,
        "experiments": experiments,
        "summary": {
            "core_ready": status_counts.get("core-ready", 0),
            "optional_ready": status_counts.get("optional-ready", 0),
            "blocked": status_counts.get("blocked", 0),
            "awaiting_human": status_counts.get("awaiting-human", 0),
        },
    }

    # 输出JSON
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "environment_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(environment_report, f, indent=2, ensure_ascii=False)
    logger.info(f"环境报告已写入: {report_path}")

    avail_path = output_dir / "experiment_availability.json"
    with open(avail_path, "w", encoding="utf-8") as f:
        json.dump(experiment_availability, f, indent=2, ensure_ascii=False)
    logger.info(f"实验可用性报告已写入: {avail_path}")

    # 打印摘要
    logger.info("=" * 50)
    logger.info("环境体检摘要:")
    logger.info(f"  整体状态: {overall_status}")
    logger.info(f"  core-ready: {status_counts.get('core-ready', 0)}")
    logger.info(f"  optional-ready: {status_counts.get('optional-ready', 0)}")
    logger.info(f"  blocked: {status_counts.get('blocked', 0)}")
    logger.info(f"  awaiting-human: {status_counts.get('awaiting-human', 0)}")
    logger.info("=" * 50)

    return {
        "overall_status": overall_status,
        "environment_report_path": str(report_path),
        "experiment_availability_path": str(avail_path),
        "summary": experiment_availability["summary"],
    }
