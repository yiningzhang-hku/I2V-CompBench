# Benchmark Dataset Card (mode=full)

- Total samples (per-dim files): **3517**
- phase3_manifest.jsonl rows: **3517**
- Consistency check: **PASS**

## 1. Per-dimension counts

| dimension | total | single_image | multi_image |
| --- | --- | --- | --- |
| attribute_binding | 517 | 517 | 0 |
| action_binding | 1013 | 1013 | 0 |
| motion_binding | 676 | 676 | 0 |
| spatial_composition | 0 | 0 | 0 |
| background_dynamics | 946 | 946 | 0 |
| view_transformation | 365 | 365 | 0 |
| interaction_reasoning | 0 | 0 | 0 |

## 2. Quota vs actual

| bucket | target | actual | shortfall |
| --- | --- | --- | --- |
| action_binding::multi_image::easy::common | 120 | 0 | 120 |
| action_binding::multi_image::easy::rare | 30 | 0 | 30 |
| action_binding::multi_image::hard::common | 80 | 0 | 80 |
| action_binding::multi_image::hard::rare | 20 | 0 | 20 |
| action_binding::multi_image::medium::common | 200 | 0 | 200 |
| action_binding::multi_image::medium::rare | 50 | 0 | 50 |
| action_binding::single_image::easy::common | 2280 | 0 | 2280 |
| action_binding::single_image::easy::rare | 570 | 0 | 570 |
| action_binding::single_image::hard::common | 1520 | 0 | 1520 |
| action_binding::single_image::hard::rare | 380 | 0 | 380 |
| action_binding::single_image::medium::common | 3800 | 0 | 3800 |
| action_binding::single_image::medium::rare | 950 | 0 | 950 |
| attribute_binding::multi_image::easy::common | 120 | 0 | 120 |
| attribute_binding::multi_image::easy::rare | 30 | 0 | 30 |
| attribute_binding::multi_image::hard::common | 80 | 0 | 80 |
| attribute_binding::multi_image::hard::rare | 20 | 0 | 20 |
| attribute_binding::multi_image::medium::common | 200 | 0 | 200 |
| attribute_binding::multi_image::medium::rare | 50 | 0 | 50 |
| attribute_binding::single_image::easy::common | 2280 | 0 | 2280 |
| attribute_binding::single_image::easy::rare | 570 | 0 | 570 |
| attribute_binding::single_image::hard::common | 1520 | 0 | 1520 |
| attribute_binding::single_image::hard::rare | 380 | 0 | 380 |
| attribute_binding::single_image::medium::common | 3800 | 0 | 3800 |
| attribute_binding::single_image::medium::rare | 950 | 0 | 950 |
| background_dynamics::multi_image::easy::common | 480 | 0 | 480 |
| background_dynamics::multi_image::easy::rare | 120 | 0 | 120 |
| background_dynamics::multi_image::hard::common | 320 | 0 | 320 |
| background_dynamics::multi_image::hard::rare | 80 | 0 | 80 |
| background_dynamics::multi_image::medium::common | 800 | 0 | 800 |
| background_dynamics::multi_image::medium::rare | 200 | 0 | 200 |
| background_dynamics::single_image::easy::common | 1920 | 0 | 1920 |
| background_dynamics::single_image::easy::rare | 480 | 0 | 480 |
| background_dynamics::single_image::hard::common | 1280 | 0 | 1280 |
| background_dynamics::single_image::hard::rare | 320 | 0 | 320 |
| background_dynamics::single_image::medium::common | 3200 | 0 | 3200 |
| background_dynamics::single_image::medium::rare | 800 | 0 | 800 |
| motion_binding::type_a_absolute_single::easy::common | 2400 | 297 | 2103 |
| motion_binding::type_a_absolute_single::easy::rare | 600 | 275 | 325 |
| motion_binding::type_a_absolute_single::hard::common | 1600 | 36 | 1564 |
| motion_binding::type_a_absolute_single::hard::rare | 400 | 43 | 357 |
| motion_binding::type_a_absolute_single::medium::common | 4000 | 10 | 3990 |
| motion_binding::type_a_absolute_single::medium::rare | 1000 | 15 | 985 |
| view_transformation::multi_image::easy::common | 120 | 0 | 120 |
| view_transformation::multi_image::easy::rare | 30 | 0 | 30 |
| view_transformation::multi_image::hard::common | 80 | 0 | 80 |
| view_transformation::multi_image::hard::rare | 20 | 0 | 20 |
| view_transformation::multi_image::medium::common | 200 | 0 | 200 |
| view_transformation::multi_image::medium::rare | 50 | 0 | 50 |
| view_transformation::single_image::easy::common | 2280 | 0 | 2280 |
| view_transformation::single_image::easy::rare | 570 | 0 | 570 |
| view_transformation::single_image::hard::common | 1520 | 0 | 1520 |
| view_transformation::single_image::hard::rare | 380 | 0 | 380 |
| view_transformation::single_image::medium::common | 3800 | 0 | 3800 |
| view_transformation::single_image::medium::rare | 950 | 0 | 950 |

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

pass=3517, fail=471, needs_manual_review=104
