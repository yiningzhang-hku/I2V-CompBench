---
license: cc-by-nc-4.0
task_categories:
  - image-to-video
language:
  - en
tags:
  - i2v
  - benchmark
  - compositional
  - image-to-video
  - evaluation
size_categories:
  - n<1K
pretty_name: I2V-CompBench v2 (T2V-CompBench-style)
---

# I2V-CompBench v2 — T2V-CompBench-style layout

A compositional **image-to-video (I2V)** evaluation benchmark, organized in the same directory layout as [T2V-CompBench](https://github.com/KaiyueSun98/T2V-CompBench) for drop-in compatibility with existing evaluation tooling.

## What's new in v2 (vs v1)

- **+58% more questions** (650 vs 409)
- All 5 active dimensions ≥ 100 questions each
- Quota rebalanced: lower `multi_image` ratio for dimensions with insufficient reference assets

| Dimension | v1 | v2 | Δ |
| --- | ---: | ---: | ---: |
| attribute_binding | 68 | **121** | +53 |
| action_binding | 81 | **150** | +69 |
| motion_binding | 59 | **133** | +74 |
| background_dynamics | 107 | **133** | +26 |
| view_transformation | 94 | **113** | +19 |
| **Total** | **409** | **650** | **+241** |

`spatial_composition` and `interaction_reasoning` are reserved for future releases.

## Directory Layout (T2V-CompBench-style)

```
v2/
├── prompts/                              # one final_prompt per line, ordered by id 0001..N
│   ├── attribute_binding.txt
│   ├── action_binding.txt
│   ├── motion_binding.txt
│   ├── background_dynamics.txt
│   └── view_transformation.txt
├── meta_data/                            # evaluation-ready metadata, JSON list
│   ├── attribute_binding.json
│   ├── action_binding.json
│   ├── motion_binding.json
│   ├── background_dynamics.json
│   └── view_transformation.json
└── first_frames/                         # I2V model input images (replaces T2V's video/)
    ├── attribute_binding/
    │   ├── 0001.png        # native resolution
    │   ├── 0001_16x9.png   # 16:9 normalized (recommended for inference)
    │   └── ...
    └── ...
```

## meta_data field reference

Each entry in `meta_data/<dim>.json` contains:

| field | description |
| --- | --- |
| `id` | new id `0001..N` (matches `prompts/<dim>.txt` line and `first_frames/<dim>/<id>.png`) |
| `original_question_id` | the v1/v2 question_id from the upstream pipeline |
| `dimension` | one of the 5 active dimensions |
| `prompt` | refined I2V text prompt |
| `input_mode` | `single_image` / `multi_image` |
| `difficulty` | `easy` / `medium` / `hard` |
| `rarity` | `common` / `rare` |
| `qc_status` | `pass` (all entries pass VLM-based VQA QC) |
| `source_sample_id` | upstream TIP-I2V sample id (when applicable) |
| dimension-specific anchors | e.g. `attribute_target`, `motion_type`, `view_transformation_type` |

## Usage (T2V-CompBench-compatible)

Generate I2V outputs using `first_frames/<dim>/<id>_16x9.png` + the corresponding line from `prompts/<dim>.txt`. Organize generated videos in the `video/<dim>/<id>.mp4` layout, identical to T2V-CompBench's evaluation expectation.

## License & Attribution

Released under **CC BY-NC 4.0**. First-frame images are derived from [TIP-I2V](https://huggingface.co/datasets/WenhaoWang/TIP-I2V) (CC BY-NC 4.0); this dataset inherits the non-commercial restriction. Cite both this work and TIP-I2V if used.
