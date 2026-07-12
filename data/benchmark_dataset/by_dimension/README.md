# Benchmark Dataset (by-dimension layout)

Total questions: **3517**

| Dimension | Questions |
| --- | --- |
| attribute_binding | 517 |
| action_binding | 1013 |
| motion_binding | 676 |
| spatial_composition | 0 |
| background_dynamics | 946 |
| view_transformation | 365 |
| interaction_reasoning | 0 |

## Per-question folder layout

Each `<dimension>/<question_id>/` contains:

- `prompt.json` — compact manifest (final I2V prompt + minimal evaluation metadata)
- `first_frame.png` — first frame image (single_image mode)
- `ref_1.png`, `ref_2.png`, ... — reference images (multi_image mode)

Empty dimensions have a `README.md` explaining the gap.
