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
pretty_name: I2V-CompBench v2 (by_dimension)
---

# I2V-CompBench v2 — by_dimension layout

V2 (650 questions) organized as **one folder per question**, identical in spirit to the v1 [`by_dimension/`](../by_dimension) layout but with the full v2 data pool. Use this layout for browsing individual questions, debugging inputs, or feeding tools that expect per-sample directories.

> For evaluation pipelines compatible with [T2V-CompBench](https://github.com/KaiyueSun98/T2V-CompBench)'s `prompts/` + `meta_data/` + `video/` convention, see [`v2/`](../v2) instead.

## Counts

| Dimension | Questions |
| --- | ---: |
| attribute_binding | 121 |
| action_binding | 150 |
| motion_binding | 133 |
| background_dynamics | 133 |
| view_transformation | 113 |
| spatial_composition | 0 (reserved) |
| interaction_reasoning | 0 (reserved) |
| **Total** | **650** |

All entries pass a VLM-based VQA quality check (Qwen3-VL-30B-A3B-Instruct).

## Per-question folder layout

```
v2_by_dimension/
├── attribute_binding/
│   ├── attr_single_0001/
│   │   ├── first_frame.png       # native resolution
│   │   ├── first_frame_16x9.png  # 16:9 normalized (recommended for I2V input)
│   │   └── prompt.json           # structured prompt + metadata
│   ├── attr_single_0002/
│   └── ...
├── action_binding/
├── motion_binding/
├── background_dynamics/
└── view_transformation/
```

The folder name is the original `question_id` (e.g. `attr_single_0001`, `act_single_0042`, `motion_single_0078`), so it self-describes the dimension and `single_image` / `multi_image` mode.

## prompt.json schema (key fields)

| field | description |
| --- | --- |
| `question_id` | unique id within the dimension (matches folder name) |
| `dimension` | one of the 5 active dimensions |
| `final_prompt` | refined I2V text prompt |
| `input_mode` | `single_image` / `multi_image` |
| `difficulty` | `easy` / `medium` / `hard` |
| `rarity` | `common` / `rare` |
| `qc_status` | `pass` |
| `source_sample_id` | upstream TIP-I2V sample id (when applicable) |
| dimension-specific anchors | e.g. `attribute_target`, `motion_type`, `view_transformation_type` |

## License & Attribution

Released under **CC BY-NC 4.0**. First-frame images are derived from [TIP-I2V](https://huggingface.co/datasets/WenhaoWang/TIP-I2V) (CC BY-NC 4.0); this dataset inherits the non-commercial restriction.
