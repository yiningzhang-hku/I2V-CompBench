"""
Phase 1 模块（已合并至 i2vcompbench 包）。

模块拓扑：
    step1_manifest      — 处理原始数据生成 manifest
    step2_image_analysis  — VLM 图像结构化分析
    step3_text_analysis   — LLM 文本意图与语义分析
    step4_joint_analysis  — 联合分析 + 先验提取
    step5_pool_and_report — 先验包装与报告生成
    align_instances     — 图-文实例对齐（aligned_instances.jsonl）
    reference_bank      — 资产抽取 + assets.jsonl（P0 走 mock_geometry）
    priors_enhance      — frequency_tiers / pair_dist / multi_ref / compatibility_matrix
    recipes             — candidate_recipes.jsonl
    audit               — phase1_audit_report.md
    mock_geometry       — P0 从 position_in_frame 反推 bbox，提供裁剪工具
    patch_existing_outputs — 不重跑 VLM/LLM，靠 patch 脚本补齐新字段
"""
