"""
P2 图像清晰度增强脚本 (Task 21)
方案: C0+C1 (Lanczos放大 + Unsharp Mask锐化)

环境: 无GPU/Real-ESRGAN，使用纯CPU OpenCV方案
目标: 对所有first_frames图像（含16x9版本）进行清晰度增强
"""

import os
import sys
import json
import shutil
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


# ============== 配置 ==============
@dataclass
class EnhanceConfig:
    """清晰度增强配置"""
    # 路径
    benchmark_root: str = "data/benchmark_dataset"
    first_frames_dir: str = "data/benchmark_dataset/first_frames"
    backup_dir: str = "data/benchmark_dataset/first_frames_backup_pre_enhance"
    
    # 放大参数
    target_long_edge: int = 854  # 目标长边像素
    upscale_interpolation: int = cv2.INTER_LANCZOS4
    
    # Unsharp Mask 参数
    unsharp_kernel_size: int = 5       # 高斯核大小 (奇数)
    unsharp_sigma: float = 1.5         # 高斯模糊sigma
    unsharp_amount: float = 1.0        # 锐化强度 (0.5~2.0)
    unsharp_threshold: int = 0         # 锐化阈值（低于此差异不增强）
    
    # 自适应锐化（根据当前模糊程度调节强度）
    adaptive_sharpen: bool = True
    # Laplacian方差低于此值时用更强锐化
    blur_threshold_low: float = 5.0
    blur_threshold_mid: float = 15.0
    # 对应的锐化强度
    amount_for_very_blurry: float = 1.5
    amount_for_blurry: float = 1.0
    amount_for_sharp: float = 0.5
    
    # 输出
    output_format: str = "PNG"
    png_compression: int = 6  # 0-9, 6是默认平衡


# ============== 核心函数 ==============

def compute_laplacian_variance(image: np.ndarray) -> float:
    """计算图像Laplacian方差（清晰度指标）"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def unsharp_mask(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.5,
                 amount: float = 1.0, threshold: int = 0) -> np.ndarray:
    """
    Unsharp Mask 锐化
    
    原理: sharpened = original + amount * (original - blurred)
    
    Args:
        image: 输入BGR图像 (uint8)
        kernel_size: 高斯核大小
        sigma: 高斯模糊标准差
        amount: 锐化强度
        threshold: 最小差异阈值
    
    Returns:
        锐化后的图像 (uint8)
    """
    # 高斯模糊
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    # 计算差异（高频细节）
    if threshold > 0:
        # 带阈值版本：只锐化差异超过阈值的区域
        diff = image.astype(np.int16) - blurred.astype(np.int16)
        mask = np.abs(diff) > threshold
        sharpened = image.astype(np.float32)
        sharpened[mask] = sharpened[mask] + amount * diff[mask].astype(np.float32)
    else:
        # 标准Unsharp Mask
        sharpened = image.astype(np.float32) + amount * (
            image.astype(np.float32) - blurred.astype(np.float32)
        )
    
    # 裁剪到有效范围
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened


def upscale_if_needed(image: np.ndarray, target_long_edge: int,
                      interpolation: int = cv2.INTER_LANCZOS4) -> Tuple[np.ndarray, bool]:
    """
    如果图像长边小于目标尺寸，使用Lanczos插值放大
    
    Returns:
        (处理后图像, 是否执行了放大)
    """
    h, w = image.shape[:2]
    long_edge = max(h, w)
    
    if long_edge >= target_long_edge:
        return image, False
    
    scale = target_long_edge / long_edge
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return resized, True


def get_adaptive_amount(laplacian_var: float, config: EnhanceConfig) -> float:
    """根据当前清晰度自适应调节锐化强度"""
    if not config.adaptive_sharpen:
        return config.unsharp_amount
    
    if laplacian_var < config.blur_threshold_low:
        return config.amount_for_very_blurry
    elif laplacian_var < config.blur_threshold_mid:
        return config.amount_for_blurry
    else:
        return config.amount_for_sharp


def enhance_single_image(image_path: str, config: EnhanceConfig) -> dict:
    """
    增强单张图像
    
    Returns:
        处理结果字典
    """
    result = {
        "file": os.path.basename(image_path),
        "status": "success",
        "original_size": None,
        "final_size": None,
        "upscaled": False,
        "laplacian_before": 0.0,
        "laplacian_after": 0.0,
        "improvement_pct": 0.0,
        "amount_used": 0.0,
        "error": None,
    }
    
    try:
        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            result["status"] = "failed"
            result["error"] = "Cannot read image"
            return result
        
        h, w = image.shape[:2]
        result["original_size"] = f"{w}x{h}"
        
        # 计算增强前清晰度
        lap_before = compute_laplacian_variance(image)
        result["laplacian_before"] = lap_before
        
        # Step 1: 放大（如有需要）
        image, upscaled = upscale_if_needed(image, config.target_long_edge,
                                            config.upscale_interpolation)
        result["upscaled"] = upscaled
        
        if upscaled:
            # 放大后重新计算Laplacian（放大本身会降低清晰度）
            lap_before = compute_laplacian_variance(image)
            result["laplacian_before"] = lap_before
        
        # Step 2: 自适应Unsharp Mask锐化
        amount = get_adaptive_amount(lap_before, config)
        result["amount_used"] = amount
        
        enhanced = unsharp_mask(
            image,
            kernel_size=config.unsharp_kernel_size,
            sigma=config.unsharp_sigma,
            amount=amount,
            threshold=config.unsharp_threshold,
        )
        
        # 计算增强后清晰度
        lap_after = compute_laplacian_variance(enhanced)
        result["laplacian_after"] = lap_after
        
        # 计算提升百分比
        if lap_before > 0:
            result["improvement_pct"] = (lap_after - lap_before) / lap_before * 100
        
        # 记录最终尺寸
        fh, fw = enhanced.shape[:2]
        result["final_size"] = f"{fw}x{fh}"
        
        # 保存（覆盖原文件）
        cv2.imwrite(image_path, enhanced, [cv2.IMWRITE_PNG_COMPRESSION, config.png_compression])
        
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
    
    return result


# ============== 主流程 ==============

def create_backup(first_frames_dir: str, backup_dir: str) -> int:
    """
    创建原始图像备份
    只在backup目录不存在时执行完整备份
    
    Returns:
        备份文件数
    """
    if os.path.exists(backup_dir):
        existing = len([f for f in os.listdir(backup_dir) if f.endswith('.png')])
        print(f"[INFO] Backup directory already exists with {existing} files, skipping backup")
        return existing
    
    print(f"[INFO] Creating backup: {backup_dir}")
    os.makedirs(backup_dir, exist_ok=True)
    
    files = [f for f in os.listdir(first_frames_dir) if f.endswith('.png')]
    for f in tqdm(files, desc="Backing up"):
        src = os.path.join(first_frames_dir, f)
        dst = os.path.join(backup_dir, f)
        shutil.copy2(src, dst)
    
    print(f"[INFO] Backed up {len(files)} files")
    return len(files)


def collect_image_files(first_frames_dir: str) -> Tuple[List[str], List[str]]:
    """
    收集需要处理的图像文件
    
    Returns:
        (原始图像列表, 16x9图像列表)
    """
    all_files = sorted([f for f in os.listdir(first_frames_dir) if f.endswith('.png')])
    originals = [f for f in all_files if '_16x9' not in f]
    x16_files = [f for f in all_files if '_16x9' in f]
    return originals, x16_files


def run_enhancement(config: EnhanceConfig, skip_backup: bool = False,
                    dry_run: bool = False, limit: int = 0):
    """
    执行批量清晰度增强
    """
    first_frames_dir = os.path.join(os.getcwd(), config.first_frames_dir)
    backup_dir = os.path.join(os.getcwd(), config.backup_dir)
    
    print("=" * 60)
    print("P2 图像清晰度增强 (C0+C1: Lanczos + Unsharp Mask)")
    print("=" * 60)
    print(f"源目录: {first_frames_dir}")
    print(f"备份目录: {backup_dir}")
    print(f"目标长边: {config.target_long_edge}px")
    print(f"锐化参数: kernel={config.unsharp_kernel_size}, sigma={config.unsharp_sigma}")
    print(f"自适应锐化: {config.adaptive_sharpen}")
    print(f"  - 非常模糊 (Lap<{config.blur_threshold_low}): amount={config.amount_for_very_blurry}")
    print(f"  - 模糊 (Lap<{config.blur_threshold_mid}): amount={config.amount_for_blurry}")
    print(f"  - 清晰 (Lap>={config.blur_threshold_mid}): amount={config.amount_for_sharp}")
    print("=" * 60)
    
    # Step 1: 备份
    if not skip_backup and not dry_run:
        create_backup(first_frames_dir, backup_dir)
    
    # Step 2: 收集文件
    originals, x16_files = collect_image_files(first_frames_dir)
    all_to_process = originals + x16_files
    
    if limit > 0:
        all_to_process = all_to_process[:limit]
    
    print(f"\n[INFO] Files to process: {len(all_to_process)} "
          f"(originals={len(originals)}, 16x9={len(x16_files)})")
    
    if dry_run:
        print("[DRY RUN] Would process above files. Exiting.")
        return
    
    # Step 3: 批量处理
    results = []
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    start_time = time.time()
    
    for filename in tqdm(all_to_process, desc="Enhancing"):
        filepath = os.path.join(first_frames_dir, filename)
        result = enhance_single_image(filepath, config)
        results.append(result)
        
        if result["status"] == "success":
            success_count += 1
        elif result["status"] == "failed":
            failed_count += 1
        else:
            skipped_count += 1
    
    elapsed = time.time() - start_time
    
    # Step 4: 统计报告
    print("\n" + "=" * 60)
    print("清晰度增强完成 - 统计报告")
    print("=" * 60)
    print(f"处理时间: {elapsed:.1f}s ({elapsed/len(all_to_process)*1000:.1f}ms/张)")
    print(f"总计: {len(all_to_process)} | 成功: {success_count} | 失败: {failed_count} | 跳过: {skipped_count}")
    
    # Laplacian方差统计
    successful = [r for r in results if r["status"] == "success"]
    if successful:
        lap_before = [r["laplacian_before"] for r in successful]
        lap_after = [r["laplacian_after"] for r in successful]
        improvements = [r["improvement_pct"] for r in successful]
        
        print(f"\n--- Laplacian方差 (清晰度指标) ---")
        print(f"增强前: mean={np.mean(lap_before):.2f}, median={np.median(lap_before):.2f}, "
              f"min={np.min(lap_before):.2f}, max={np.max(lap_before):.2f}")
        print(f"增强后: mean={np.mean(lap_after):.2f}, median={np.median(lap_after):.2f}, "
              f"min={np.min(lap_after):.2f}, max={np.max(lap_after):.2f}")
        print(f"提升%:  mean={np.mean(improvements):.1f}%, median={np.median(improvements):.1f}%, "
              f"min={np.min(improvements):.1f}%, max={np.max(improvements):.1f}%")
        
        # 按类型统计
        orig_results = [r for r in successful if '_16x9' not in r['file']]
        x16_results = [r for r in successful if '_16x9' in r['file']]
        
        if orig_results:
            orig_imp = [r["improvement_pct"] for r in orig_results]
            print(f"\n  原始图像: {len(orig_results)}张, 平均提升={np.mean(orig_imp):.1f}%")
        if x16_results:
            x16_imp = [r["improvement_pct"] for r in x16_results]
            print(f"  16x9图像: {len(x16_results)}张, 平均提升={np.mean(x16_imp):.1f}%")
        
        # 自适应强度分布
        amounts = [r["amount_used"] for r in successful]
        print(f"\n--- 自适应锐化强度分布 ---")
        for amt_val in sorted(set(amounts)):
            cnt = amounts.count(amt_val)
            print(f"  amount={amt_val:.1f}: {cnt}张 ({cnt/len(amounts)*100:.1f}%)")
        
        # 验收: Laplacian均值提升>30%
        mean_improvement = np.mean(improvements)
        print(f"\n--- 验收检查 ---")
        pass_str = "PASS" if mean_improvement > 30 else "FAIL (need >30%)"
        print(f"  Laplacian variance mean improvement: {mean_improvement:.1f}% [{pass_str}]")
        damage_str = "PASS" if failed_count == 0 else f"FAIL ({failed_count} damaged)"
        print(f"  No damaged images: [{damage_str}]")
        
        # 被放大的图像统计
        upscaled_count = sum(1 for r in successful if r["upscaled"])
        print(f"  需要放大的图像: {upscaled_count}张")
    
    # 保存详细结果
    report_path = os.path.join(os.getcwd(), config.benchmark_root,
                               "quality_experiments", "clarity_enhance_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "method": "lanczos_unsharp",
            "target_long_edge": config.target_long_edge,
            "unsharp_kernel_size": config.unsharp_kernel_size,
            "unsharp_sigma": config.unsharp_sigma,
            "adaptive_sharpen": config.adaptive_sharpen,
        },
        "summary": {
            "total": len(all_to_process),
            "success": success_count,
            "failed": failed_count,
            "elapsed_seconds": round(elapsed, 1),
            "laplacian_before_mean": round(np.mean(lap_before), 2) if successful else 0,
            "laplacian_after_mean": round(np.mean(lap_after), 2) if successful else 0,
            "mean_improvement_pct": round(np.mean(improvements), 1) if successful else 0,
        },
        "details": results,
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] 详细报告已保存: {report_path}")
    
    # 输出随机抽样10张供人工检查
    if successful:
        import random
        random.seed(42)
        sample = random.sample(successful, min(10, len(successful)))
        print(f"\n--- 随机抽样10张 (供人工检查) ---")
        for s in sample:
            print(f"  {s['file']}: {s['original_size']} → Lap {s['laplacian_before']:.2f} → {s['laplacian_after']:.2f} (+{s['improvement_pct']:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="P2 图像清晰度增强")
    parser.add_argument("--dry-run", action="store_true", help="仅统计，不执行增强")
    parser.add_argument("--skip-backup", action="store_true", help="跳过备份步骤")
    parser.add_argument("--limit", type=int, default=0, help="限制处理数量(0=全部)")
    parser.add_argument("--amount", type=float, default=None, help="覆盖锐化强度")
    parser.add_argument("--no-adaptive", action="store_true", help="禁用自适应锐化")
    args = parser.parse_args()
    
    config = EnhanceConfig()
    
    if args.amount is not None:
        config.unsharp_amount = args.amount
        config.adaptive_sharpen = False
    
    if args.no_adaptive:
        config.adaptive_sharpen = False
    
    run_enhancement(config, skip_backup=args.skip_backup,
                    dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
