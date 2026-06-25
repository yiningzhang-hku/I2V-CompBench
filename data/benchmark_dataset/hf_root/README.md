---
license: cc-by-nc-4.0
task_categories:
  - image-to-video
  - text-to-video
language:
  - en
tags:
  - i2v
  - image-to-video
  - benchmark
  - compositional
  - evaluation
  - first-frame
  - tip-i2v
size_categories:
  - n<1K
pretty_name: I2V-CompBench
configs:
  - config_name: v2
    data_files:
      - split: attribute_binding
        path: v2/prompts/attribute_binding.txt
      - split: action_binding
        path: v2/prompts/action_binding.txt
      - split: motion_binding
        path: v2/prompts/motion_binding.txt
      - split: background_dynamics
        path: v2/prompts/background_dynamics.txt
      - split: view_transformation
        path: v2/prompts/view_transformation.txt
---

# I2V-CompBench

A compositional **image-to-video (I2V)** generation benchmark spanning 7 evaluation dimensions, with first-frame images derived from [TIP-I2V](https://huggingface.co/datasets/WenhaoWang/TIP-I2V) and refined text prompts produced by a dual VLM/LLM pipeline.

> ⚠️ **License**: CC BY-NC 4.0 (inherits from TIP-I2V). Non-commercial use only.

---

## 📦 Versions

This repository hosts **two parallel snapshots** of the same benchmark. Pick the layout that fits your tooling.

| Version | Path | Questions | Layout | Best for |
| --- | --- | ---: | --- | --- |
| **v2** ⭐ | [`v2/`](./tree/main/v2) | **650** | T2V-CompBench-style (`prompts/`, `meta_data/`, `first_frames/`) | Drop-in compatible with [T2V-CompBench](https://github.com/KaiyueSun98/T2V-CompBench) eval scripts |
| v2 (alt) | [`v2_by_dimension/`](./tree/main/v2_by_dimension) | 650 | Per-question folder (`<dim>/<qid>/{prompt.json, first_frame*.png}`) | Browsing / debugging single v2 questions |
| v1 | [`by_dimension/`](./tree/main/by_dimension) | 409 | Per-question folder (same layout as `v2_by_dimension/`) | Reproducing earlier experiments |

**Recommendation**: use **v2** for evaluation runs. `v2_by_dimension/` is the same v2 data in v1's per-folder layout for tooling that prefers per-sample directories. v1 is kept for reproducibility of earlier experiments.

---

## 🎯 Dimension coverage (v2)

| Dimension | Questions | Notes |
| --- | ---: | --- |
| attribute_binding | 121 | object color/shape/material consistency |
| action_binding | 150 | subject action faithfulness |
| motion_binding | 133 | absolute / relative / multi-motion |
| background_dynamics | 133 | scene-level dynamics |
| view_transformation | 113 | camera motion |
| spatial_composition | 0 | reserved for future release |
| interaction_reasoning | 0 | reserved for future release |
| **Total** | **650** | |

All entries pass a VLM-based VQA quality check (Qwen3-VL-30B-A3B-Instruct).

---

## 🚀 Quick start (v2, T2V-CompBench-style)

```python
from huggingface_hub import snapshot_download

local = snapshot_download(
    repo_id="YiningZ2002/I2V-CompBench",
    repo_type="dataset",
    allow_patterns="v2/*",
)
# Then read v2/prompts/<dim>.txt + v2/meta_data/<dim>.json
# and use v2/first_frames/<dim>/<id>_16x9.png as I2V model input.
```

For full schema and field reference, see [`v2/README.md`](./blob/main/v2/README.md).

---

## 🔬 Evaluation

After generating videos with your I2V model, organize them as:

```
video/
├── attribute_binding/0001.mp4 ... 0121.mp4
├── action_binding/0001.mp4 ... 0150.mp4
├── motion_binding/0001.mp4 ... 0133.mp4
├── background_dynamics/0001.mp4 ... 0133.mp4
└── view_transformation/0001.mp4 ... 0113.mp4
```

This is identical to T2V-CompBench's expected layout, so existing T2V-CompBench evaluators (MLLM-based / detection-based / tracking-based) work with minimal changes.

---

## 📚 Citation

```bibtex
@misc{i2vcompbench2026,
  title  = {I2V-CompBench: A Compositional Benchmark for Image-to-Video Generation},
  author = {YiningZ2002},
  year   = {2026},
  url    = {https://huggingface.co/datasets/YiningZ2002/I2V-CompBench}
}

@inproceedings{tip-i2v,
  title  = {TIP-I2V: A Million-Scale Real Text and Image Prompt Dataset for Image-to-Video Generation},
  author = {Wang, Wenhao and others},
  year   = {2024}
}

@inproceedings{t2v-compbench,
  title  = {T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-Video Generation},
  author = {Sun, Kaiyue and others},
  booktitle = {CVPR},
  year   = {2025}
}
```
