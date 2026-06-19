# Benchmark Dataset Card (mode=full)

- Total samples (per-dim files): **409**
- phase3_manifest.jsonl rows: **409**
- Consistency check: **PASS**

## 1. Per-dimension counts

| dimension | total | single_image | multi_image |
| --- | --- | --- | --- |
| attribute_binding | 68 | 68 | 0 |
| action_binding | 81 | 81 | 0 |
| motion_binding | 59 | 59 | 0 |
| spatial_composition | 0 | 0 | 0 |
| background_dynamics | 107 | 107 | 0 |
| view_transformation | 94 | 94 | 0 |
| interaction_reasoning | 0 | 0 | 0 |

## 2. Quota vs actual

| bucket | target | actual | shortfall |
| --- | --- | --- | --- |
| action_binding::multi_image::easy::common | 14 | 0 | 14 |
| action_binding::multi_image::easy::rare | 4 | 0 | 4 |
| action_binding::multi_image::hard::common | 10 | 0 | 10 |
| action_binding::multi_image::hard::rare | 2 | 0 | 2 |
| action_binding::multi_image::medium::common | 24 | 0 | 24 |
| action_binding::multi_image::medium::rare | 6 | 0 | 6 |
| action_binding::single_image::easy::common | 34 | 0 | 34 |
| action_binding::single_image::easy::rare | 8 | 0 | 8 |
| action_binding::single_image::hard::common | 22 | 0 | 22 |
| action_binding::single_image::hard::rare | 6 | 0 | 6 |
| action_binding::single_image::medium::common | 56 | 0 | 56 |
| action_binding::single_image::medium::rare | 14 | 0 | 14 |
| attribute_binding::multi_image::easy::common | 19 | 0 | 19 |
| attribute_binding::multi_image::easy::rare | 5 | 0 | 5 |
| attribute_binding::multi_image::hard::common | 13 | 0 | 13 |
| attribute_binding::multi_image::hard::rare | 3 | 0 | 3 |
| attribute_binding::multi_image::medium::common | 32 | 0 | 32 |
| attribute_binding::multi_image::medium::rare | 8 | 0 | 8 |
| attribute_binding::single_image::easy::common | 29 | 0 | 29 |
| attribute_binding::single_image::easy::rare | 7 | 0 | 7 |
| attribute_binding::single_image::hard::common | 19 | 0 | 19 |
| attribute_binding::single_image::hard::rare | 5 | 0 | 5 |
| attribute_binding::single_image::medium::common | 48 | 0 | 48 |
| attribute_binding::single_image::medium::rare | 12 | 0 | 12 |
| background_dynamics::multi_image::easy::common | 10 | 0 | 10 |
| background_dynamics::multi_image::easy::rare | 2 | 0 | 2 |
| background_dynamics::multi_image::hard::common | 6 | 0 | 6 |
| background_dynamics::multi_image::hard::rare | 2 | 0 | 2 |
| background_dynamics::multi_image::medium::common | 16 | 0 | 16 |
| background_dynamics::multi_image::medium::rare | 4 | 0 | 4 |
| background_dynamics::single_image::easy::common | 38 | 0 | 38 |
| background_dynamics::single_image::easy::rare | 10 | 0 | 10 |
| background_dynamics::single_image::hard::common | 26 | 0 | 26 |
| background_dynamics::single_image::hard::rare | 6 | 0 | 6 |
| background_dynamics::single_image::medium::common | 64 | 0 | 64 |
| background_dynamics::single_image::medium::rare | 16 | 0 | 16 |
| motion_binding::type_a_absolute_single::easy::common | 24 | 24 | 0 |
| motion_binding::type_a_absolute_single::easy::rare | 6 | 6 | 0 |
| motion_binding::type_a_absolute_single::hard::common | 16 | 15 | 1 |
| motion_binding::type_a_absolute_single::hard::rare | 4 | 4 | 0 |
| motion_binding::type_a_absolute_single::medium::common | 40 | 5 | 35 |
| motion_binding::type_a_absolute_single::medium::rare | 10 | 5 | 5 |
| motion_binding::type_b_relative_single::easy::common | 17 | 0 | 17 |
| motion_binding::type_b_relative_single::easy::rare | 4 | 0 | 4 |
| motion_binding::type_b_relative_single::hard::common | 11 | 0 | 11 |
| motion_binding::type_b_relative_single::hard::rare | 3 | 0 | 3 |
| motion_binding::type_b_relative_single::medium::common | 28 | 0 | 28 |
| motion_binding::type_b_relative_single::medium::rare | 7 | 0 | 7 |
| motion_binding::type_c_multi_motion::easy::common | 7 | 0 | 7 |
| motion_binding::type_c_multi_motion::easy::rare | 2 | 0 | 2 |
| motion_binding::type_c_multi_motion::hard::common | 5 | 0 | 5 |
| motion_binding::type_c_multi_motion::hard::rare | 1 | 0 | 1 |
| motion_binding::type_c_multi_motion::medium::common | 12 | 0 | 12 |
| motion_binding::type_c_multi_motion::medium::rare | 3 | 0 | 3 |
| view_transformation::multi_image::easy::common | 2 | 0 | 2 |
| view_transformation::multi_image::easy::rare | 1 | 0 | 1 |
| view_transformation::multi_image::hard::common | 2 | 0 | 2 |
| view_transformation::multi_image::medium::common | 4 | 0 | 4 |
| view_transformation::multi_image::medium::rare | 1 | 0 | 1 |
| view_transformation::single_image::easy::common | 46 | 0 | 46 |
| view_transformation::single_image::easy::rare | 11 | 0 | 11 |
| view_transformation::single_image::hard::common | 30 | 0 | 30 |
| view_transformation::single_image::hard::rare | 8 | 0 | 8 |
| view_transformation::single_image::medium::common | 76 | 0 | 76 |
| view_transformation::single_image::medium::rare | 19 | 0 | 19 |

## 3. Contrastive pair coverage

| dimension | pairs_seen | complete(>=1O+>=1B) | missing_original | missing_baseline |
| --- | --- | --- | --- | --- |
| attribute_binding | 0 | 0 | 0 | 0 |
| motion_binding | 0 | 0 | 0 | 0 |
| spatial_composition | 0 | 0 | 0 | 0 |
| view_transformation | 0 | 0 | 0 | 0 |

## 4. Multi-image reference quality

**crop_leakage_risk**: 
**identity_visibility**: 

## 5. QC status histogram

pass=409, fail=36, needs_manual_review=11
