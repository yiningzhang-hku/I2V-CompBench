# Benchmark Dataset Card (mode=full)

- Total samples (per-dim files): **650**
- phase3_manifest.jsonl rows: **650**
- Consistency check: **PASS**

## 1. Per-dimension counts

| dimension | total | single_image | multi_image |
| --- | --- | --- | --- |
| attribute_binding | 121 | 121 | 0 |
| action_binding | 150 | 150 | 0 |
| motion_binding | 133 | 133 | 0 |
| spatial_composition | 0 | 0 | 0 |
| background_dynamics | 133 | 133 | 0 |
| view_transformation | 113 | 113 | 0 |
| interaction_reasoning | 0 | 0 | 0 |

## 2. Quota vs actual

| bucket | target | actual | shortfall |
| --- | --- | --- | --- |
| action_binding::multi_image::easy::common | 3 | 0 | 3 |
| action_binding::multi_image::easy::rare | 1 | 0 | 1 |
| action_binding::multi_image::hard::common | 2 | 0 | 2 |
| action_binding::multi_image::hard::rare | 1 | 0 | 1 |
| action_binding::multi_image::medium::common | 6 | 0 | 6 |
| action_binding::multi_image::medium::rare | 2 | 0 | 2 |
| action_binding::single_image::easy::common | 68 | 0 | 68 |
| action_binding::single_image::easy::rare | 17 | 0 | 17 |
| action_binding::single_image::hard::common | 46 | 0 | 46 |
| action_binding::single_image::hard::rare | 11 | 0 | 11 |
| action_binding::single_image::medium::common | 114 | 0 | 114 |
| action_binding::single_image::medium::rare | 29 | 0 | 29 |
| attribute_binding::multi_image::easy::common | 3 | 0 | 3 |
| attribute_binding::multi_image::easy::rare | 1 | 0 | 1 |
| attribute_binding::multi_image::hard::common | 2 | 0 | 2 |
| attribute_binding::multi_image::hard::rare | 1 | 0 | 1 |
| attribute_binding::multi_image::medium::common | 6 | 0 | 6 |
| attribute_binding::multi_image::medium::rare | 2 | 0 | 2 |
| attribute_binding::single_image::easy::common | 68 | 0 | 68 |
| attribute_binding::single_image::easy::rare | 17 | 0 | 17 |
| attribute_binding::single_image::hard::common | 46 | 0 | 46 |
| attribute_binding::single_image::hard::rare | 11 | 0 | 11 |
| attribute_binding::single_image::medium::common | 114 | 0 | 114 |
| attribute_binding::single_image::medium::rare | 29 | 0 | 29 |
| background_dynamics::multi_image::easy::common | 14 | 0 | 14 |
| background_dynamics::multi_image::easy::rare | 4 | 0 | 4 |
| background_dynamics::multi_image::hard::common | 10 | 0 | 10 |
| background_dynamics::multi_image::hard::rare | 2 | 0 | 2 |
| background_dynamics::multi_image::medium::common | 24 | 0 | 24 |
| background_dynamics::multi_image::medium::rare | 6 | 0 | 6 |
| background_dynamics::single_image::easy::common | 58 | 0 | 58 |
| background_dynamics::single_image::easy::rare | 14 | 0 | 14 |
| background_dynamics::single_image::hard::common | 38 | 0 | 38 |
| background_dynamics::single_image::hard::rare | 10 | 0 | 10 |
| background_dynamics::single_image::medium::common | 96 | 0 | 96 |
| background_dynamics::single_image::medium::rare | 24 | 0 | 24 |
| motion_binding::type_a_absolute_single::easy::common | 72 | 71 | 1 |
| motion_binding::type_a_absolute_single::easy::rare | 18 | 18 | 0 |
| motion_binding::type_a_absolute_single::hard::common | 48 | 24 | 24 |
| motion_binding::type_a_absolute_single::hard::rare | 12 | 11 | 1 |
| motion_binding::type_a_absolute_single::medium::common | 120 | 4 | 116 |
| motion_binding::type_a_absolute_single::medium::rare | 30 | 5 | 25 |
| view_transformation::multi_image::easy::common | 3 | 0 | 3 |
| view_transformation::multi_image::easy::rare | 1 | 0 | 1 |
| view_transformation::multi_image::hard::common | 2 | 0 | 2 |
| view_transformation::multi_image::hard::rare | 1 | 0 | 1 |
| view_transformation::multi_image::medium::common | 6 | 0 | 6 |
| view_transformation::multi_image::medium::rare | 2 | 0 | 2 |
| view_transformation::single_image::easy::common | 68 | 0 | 68 |
| view_transformation::single_image::easy::rare | 17 | 0 | 17 |
| view_transformation::single_image::hard::common | 46 | 0 | 46 |
| view_transformation::single_image::hard::rare | 11 | 0 | 11 |
| view_transformation::single_image::medium::common | 114 | 0 | 114 |
| view_transformation::single_image::medium::rare | 29 | 0 | 29 |

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

pass=650, fail=51, needs_manual_review=15
