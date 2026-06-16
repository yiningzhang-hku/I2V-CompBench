# Benchmark Dataset Card (mode=pilot)

- Total samples (per-dim files): **48**
- phase3_manifest.jsonl rows: **48**
- Consistency check: **PASS**

## 1. Per-dimension counts

| dimension | total | single_image | multi_image |
| --- | --- | --- | --- |
| attribute_binding | 9 | 9 | 0 |
| action_binding | 12 | 12 | 0 |
| motion_binding | 4 | 4 | 0 |
| spatial_composition | 0 | 0 | 0 |
| background_dynamics | 10 | 10 | 0 |
| view_transformation | 13 | 13 | 0 |
| interaction_reasoning | 0 | 0 | 0 |

## 2. Quota vs actual

| bucket | target | actual | shortfall |
| --- | --- | --- | --- |
| action_binding::multi_image::easy::common | 2 | 0 | 2 |
| action_binding::multi_image::hard::common | 1 | 0 | 1 |
| action_binding::multi_image::medium::common | 2 | 0 | 2 |
| action_binding::multi_image::medium::rare | 1 | 0 | 1 |
| action_binding::single_image::easy::common | 3 | 0 | 3 |
| action_binding::single_image::easy::rare | 1 | 0 | 1 |
| action_binding::single_image::hard::common | 2 | 0 | 2 |
| action_binding::single_image::hard::rare | 1 | 0 | 1 |
| action_binding::single_image::medium::common | 6 | 0 | 6 |
| action_binding::single_image::medium::rare | 1 | 0 | 1 |
| attribute_binding::multi_image::easy::common | 2 | 0 | 2 |
| attribute_binding::multi_image::hard::common | 2 | 0 | 2 |
| attribute_binding::multi_image::medium::common | 3 | 0 | 3 |
| attribute_binding::multi_image::medium::rare | 1 | 0 | 1 |
| attribute_binding::single_image::easy::common | 3 | 0 | 3 |
| attribute_binding::single_image::easy::rare | 1 | 0 | 1 |
| attribute_binding::single_image::hard::common | 2 | 0 | 2 |
| attribute_binding::single_image::medium::common | 5 | 0 | 5 |
| attribute_binding::single_image::medium::rare | 1 | 0 | 1 |
| background_dynamics::multi_image::easy::common | 1 | 0 | 1 |
| background_dynamics::multi_image::hard::common | 1 | 0 | 1 |
| background_dynamics::multi_image::medium::common | 2 | 0 | 2 |
| background_dynamics::single_image::easy::common | 4 | 0 | 4 |
| background_dynamics::single_image::easy::rare | 1 | 0 | 1 |
| background_dynamics::single_image::hard::common | 2 | 0 | 2 |
| background_dynamics::single_image::hard::rare | 1 | 0 | 1 |
| background_dynamics::single_image::medium::common | 6 | 0 | 6 |
| background_dynamics::single_image::medium::rare | 2 | 0 | 2 |
| interaction_reasoning::multi_image::easy::common | 2 | 0 | 2 |
| interaction_reasoning::multi_image::easy::rare | 1 | 0 | 1 |
| interaction_reasoning::multi_image::hard::common | 2 | 0 | 2 |
| interaction_reasoning::multi_image::medium::common | 4 | 0 | 4 |
| interaction_reasoning::multi_image::medium::rare | 1 | 0 | 1 |
| interaction_reasoning::single_image::easy::common | 2 | 0 | 2 |
| interaction_reasoning::single_image::easy::rare | 1 | 0 | 1 |
| interaction_reasoning::single_image::hard::common | 2 | 0 | 2 |
| interaction_reasoning::single_image::medium::common | 4 | 0 | 4 |
| interaction_reasoning::single_image::medium::rare | 1 | 0 | 1 |
| motion_binding::type_a_absolute_single::easy::common | 2 | 2 | 0 |
| motion_binding::type_a_absolute_single::easy::rare | 1 | 1 | 0 |
| motion_binding::type_a_absolute_single::hard::common | 2 | 0 | 2 |
| motion_binding::type_a_absolute_single::medium::common | 4 | 0 | 4 |
| motion_binding::type_a_absolute_single::medium::rare | 1 | 1 | 0 |
| motion_binding::type_b_relative_single::easy::common | 2 | 0 | 2 |
| motion_binding::type_b_relative_single::hard::common | 1 | 0 | 1 |
| motion_binding::type_b_relative_single::medium::common | 3 | 0 | 3 |
| motion_binding::type_b_relative_single::medium::rare | 1 | 0 | 1 |
| motion_binding::type_c_multi_motion::easy::common | 1 | 0 | 1 |
| motion_binding::type_c_multi_motion::hard::common | 1 | 0 | 1 |
| motion_binding::type_c_multi_motion::medium::common | 1 | 0 | 1 |
| spatial_composition::multi_image::easy::common | 4 | 0 | 4 |
| spatial_composition::multi_image::easy::rare | 1 | 0 | 1 |
| spatial_composition::multi_image::hard::common | 2 | 0 | 2 |
| spatial_composition::multi_image::hard::rare | 1 | 0 | 1 |
| spatial_composition::multi_image::medium::common | 6 | 0 | 6 |
| spatial_composition::multi_image::medium::rare | 2 | 0 | 2 |
| spatial_composition::single_image::easy::common | 1 | 0 | 1 |
| spatial_composition::single_image::hard::common | 1 | 0 | 1 |
| spatial_composition::single_image::medium::common | 2 | 0 | 2 |
| view_transformation::multi_image::medium::common | 1 | 0 | 1 |
| view_transformation::single_image::easy::common | 5 | 0 | 5 |
| view_transformation::single_image::easy::rare | 1 | 0 | 1 |
| view_transformation::single_image::hard::common | 3 | 0 | 3 |
| view_transformation::single_image::hard::rare | 1 | 0 | 1 |
| view_transformation::single_image::medium::common | 7 | 0 | 7 |
| view_transformation::single_image::medium::rare | 2 | 0 | 2 |

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

pass=48, fail=5
