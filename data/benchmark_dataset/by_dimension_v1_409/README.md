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
pretty_name: I2V-CompBench (by_dimension)
---

# I2V-CompBench (by_dimension split)

A compositional **image-to-video (I2V)** evaluation benchmark organized by 7 evaluation dimensions. Each question bundle contains a first-frame image (native + 16:9 normalized) and a structured `prompt.json` describing the target video transformation.

## Dimensions & Counts

| Dimension | Count |
| --- | ---: |
| attribute_binding | 68 |
| action_binding | 81 |
| motion_binding | 59 |
| background_dynamics | 107 |
| view_transformation | 94 |
| spatial_composition | 0 (reserved) |
| interaction_reasoning | 0 (reserved) |
| **Total** | **409** |

## Directory Layout

```
by_dimension/
├── attribute_binding/
│   ├── attr_single_0001/
│   │   ├── first_frame.png       # native resolution
│   │   ├── first_frame_16x9.png  # normalized to 16:9
│   │   └── prompt.json           # structured prompt + metadata
│   └── ...
├── action_binding/
├── motion_binding/
├── background_dynamics/
├── view_transformation/
└── README.md
```

## prompt.json Schema (key fields)

- `question_id` — unique id within dimension
- `dimension` — one of the 7 dimensions above
- `input_mode` — `single_image` / `multi_image`
- `difficulty` — `easy` / `medium` / `hard`
- `final_prompt` — refined I2V text prompt
- `qc_status` — passes VLM-based quality check (`pass`)

## Quality Control

Each sample passes a VLM-based VQA quality check before inclusion. QC histogram of the upstream pool: `pass=409, fail=36, needs_manual_review=11`.

## Intended Use

Evaluating compositional faithfulness of I2V models along independent axes (attribute / action / motion / scene dynamics / camera transformation). Use the `first_frame_16x9.png` + `final_prompt` as model input; score generated videos against `prompt.json` ground truth.

## License & Attribution

This dataset is released under **CC BY-NC 4.0**.

First-frame images are derived from [TIP-I2V](https://huggingface.co/datasets/WenhaoWang/TIP-I2V) (CC BY-NC 4.0). This dataset inherits the non-commercial restriction. If you use this benchmark, please cite both this work and the TIP-I2V dataset.

## Citation

```bibtex
@misc{i2vcompbench2026,
  title  = {I2V-CompBench: A Compositional Benchmark for Image-to-Video Generation},
  author = {YiningZ2002},
  year   = {2026},
  url    = {https://huggingface.co/datasets/YiningZ2002/I2V-CompBench}
}
```
