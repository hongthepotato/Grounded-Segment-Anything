#!/usr/bin/env python3
"""
Detailed Profiling CLI for auto-labeling: Sequential vs Batched inference.

Measures each step of the pipeline:
1. Image I/O (cv2.imread)
2. Image preprocessing (resize, normalize, to tensor)
3. Batch assembly (NestedTensor creation, padding)
4. Model forward pass (detection)
5. Post-processing (threshold, NMS, phrase extraction)
6. SAM segmentation (if masks mode)

Usage:
    python cli/profile_auto_label.py \
        --images data/raw/images/ \
        --classes "ear of bag,defect" \
        --compare-sequential
"""
import argparse
import logging
import sys
import time
import gc
import cv2
import numpy as np
import torch
from pathlib import Path
from glob import glob
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "GroundingDINO"))

from cli.utils import (
    setup_cuda_device,
    print_header,
    print_section,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TimingBreakdown:
    """Detailed timing for each step."""
    image_io: float = 0.0          # cv2.imread time
    preprocessing: float = 0.0     # Image transform/resize/normalize
    batch_assembly: float = 0.0    # NestedTensor creation, padding
    text_encoding: float = 0.0     # Tokenization + text encoder
    forward_pass: float = 0.0      # Model backbone + decoder
    postprocess: float = 0.0       # Threshold, NMS, phrase extraction
    segmentation: float = 0.0      # SAM (if enabled)
    total: float = 0.0
    
    # Counts
    num_images: int = 0
    num_batches: int = 0
    num_detections: int = 0
    
    def __add__(self, other: 'TimingBreakdown') -> 'TimingBreakdown':
        return TimingBreakdown(
            image_io=self.image_io + other.image_io,
            preprocessing=self.preprocessing + other.preprocessing,
            batch_assembly=self.batch_assembly + other.batch_assembly,
            text_encoding=self.text_encoding + other.text_encoding,
            forward_pass=self.forward_pass + other.forward_pass,
            postprocess=self.postprocess + other.postprocess,
            segmentation=self.segmentation + other.segmentation,
            total=self.total + other.total,
            num_images=self.num_images + other.num_images,
            num_batches=self.num_batches + other.num_batches,
            num_detections=self.num_detections + other.num_detections,
        )


@dataclass
class ProfileResult:
    """Complete profiling result."""
    mode: str
    batch_size: int
    timing: TimingBreakdown
    gpu_memory_peak_mb: Optional[float] = None


class Timer:
    """Simple timer for measuring elapsed time."""
    def __init__(self):
        self.start_time = None
        self.elapsed = 0.0
    
    def start(self):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.elapsed = time.perf_counter() - self.start_time
        return self.elapsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detailed profiling of auto-labeling pipeline",
    )
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--classes", type=str, required=True)
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--output-mode", choices=["boxes", "masks", "both"], default="boxes")
    parser.add_argument("--box-threshold", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--compare-sequential", action="store_true")
    return parser.parse_args()


def collect_image_paths(images_path: str, max_images: Optional[int] = None) -> List[str]:
    extensions = ["jpg", "jpeg", "png", "bmp"]
    path = Path(images_path)
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob(str(path / f"*.{ext}")))
        image_paths.extend(glob(str(path / f"*.{ext.upper()}")))
    image_paths = sorted(set(image_paths))
    if max_images:
        image_paths = image_paths[:max_images]
    return image_paths


def get_gpu_memory_mb() -> Optional[float]:
    try:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
    except:
        pass
    return None


def reset_gpu_memory():
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            gc.collect()
    except:
        pass


def profile_sequential_detailed(
    image_paths: List[str],
    class_prompts: List[str],
    config: Dict[str, Any],
    device: str
) -> ProfileResult:
    """
    Profile sequential processing with detailed step timing.
    
    Directly calls low-level functions to measure each step.
    """
    from groundingdino.util.inference import Model, preprocess_caption, predict
    from groundingdino.util.utils import get_phrases_from_posmap
    import torchvision.ops
    
    timing = TimingBreakdown()
    timer = Timer()
    reset_gpu_memory()
    
    # Load model (not timed - same for both)
    model = Model(
        model_config_path=config['dino_config'],
        model_checkpoint_path=config['dino_checkpoint'],
        device=device
    )
    
    caption = ". ".join(class_prompts)
    caption = preprocess_caption(caption)
    
    total_start = time.perf_counter()
    
    for image_path in image_paths:
        # Step 1: Image I/O
        timer.start()
        image_bgr = cv2.imread(image_path)
        timing.image_io += timer.stop()
        
        if image_bgr is None:
            continue
        
        # Step 2: Preprocessing
        timer.start()
        processed_image = Model.preprocess_image(image_bgr).to(device)
        timing.preprocessing += timer.stop()
        
        # Step 3: Text encoding (done per image in sequential)
        timer.start()
        tokenizer = model.model.tokenizer
        tokenized = tokenizer(caption)
        timing.text_encoding += timer.stop()
        
        # Step 4: Forward pass
        timer.start()
        with torch.no_grad():
            outputs = model.model(processed_image[None], captions=[caption])
        timing.forward_pass += timer.stop()
        
        # Step 5: Post-processing
        timer.start()
        pred_logits = outputs["pred_logits"].cpu().sigmoid()[0]
        pred_boxes = outputs["pred_boxes"].cpu()[0]
        
        mask = pred_logits.max(dim=1)[0] > config['box_threshold']
        logits = pred_logits[mask]
        boxes = pred_boxes[mask]
        
        # Phrase extraction
        phrases = [
            get_phrases_from_posmap(logit > config['text_threshold'], tokenized, tokenizer).replace('.', '')
            for logit in logits
        ]
        
        # Scale boxes
        source_h, source_w = image_bgr.shape[:2]
        boxes_scaled = boxes * torch.tensor([source_w, source_h, source_w, source_h])
        
        # NMS
        if len(boxes_scaled) > 0:
            nms_idx = torchvision.ops.nms(
                boxes_scaled, logits.max(dim=1)[0], config['nms_threshold']
            )
            timing.num_detections += len(nms_idx)
        
        timing.postprocess += timer.stop()
        timing.num_images += 1
    
    timing.total = time.perf_counter() - total_start
    timing.num_batches = timing.num_images  # Each image is its own "batch"
    
    return ProfileResult(
        mode="sequential",
        batch_size=1,
        timing=timing,
        gpu_memory_peak_mb=get_gpu_memory_mb()
    )


def profile_batched_detailed(
    image_paths: List[str],
    class_prompts: List[str],
    batch_size: int,
    config: Dict[str, Any],
    device: str
) -> ProfileResult:
    """
    Profile batched processing with detailed step timing.
    
    Directly calls low-level functions to measure each step.
    """
    from groundingdino.util.inference import Model, preprocess_caption
    from groundingdino.util.misc import NestedTensor
    from groundingdino.util.utils import get_phrases_from_posmap
    import torchvision.ops
    
    timing = TimingBreakdown()
    timer = Timer()
    reset_gpu_memory()
    
    # Load model
    model = Model(
        model_config_path=config['dino_config'],
        model_checkpoint_path=config['dino_checkpoint'],
        device=device
    )
    
    caption = ". ".join(class_prompts)
    caption = preprocess_caption(caption)
    
    total_start = time.perf_counter()
    
    # Process in batches
    for batch_start in range(0, len(image_paths), batch_size):
        batch_end = min(batch_start + batch_size, len(image_paths))
        batch_paths = image_paths[batch_start:batch_end]
        current_batch_size = len(batch_paths)
        
        # Step 1: Image I/O for batch
        timer.start()
        images_bgr = []
        for path in batch_paths:
            img = cv2.imread(path)
            if img is not None:
                images_bgr.append(img)
        timing.image_io += timer.stop()
        
        if not images_bgr:
            continue
        
        actual_batch_size = len(images_bgr)
        
        # Step 2: Preprocessing for batch
        timer.start()
        processed_images = [Model.preprocess_image(img) for img in images_bgr]
        timing.preprocessing += timer.stop()
        
        # Step 3: Batch assembly (NestedTensor creation with padding)
        timer.start()
        max_h = max(img.shape[1] for img in processed_images)
        max_w = max(img.shape[2] for img in processed_images)
        
        batch_tensor = torch.zeros(actual_batch_size, 3, max_h, max_w, device=device)
        batch_mask = torch.ones(actual_batch_size, max_h, max_w, dtype=torch.bool, device=device)
        
        for i, img in enumerate(processed_images):
            _, h, w = img.shape
            batch_tensor[i, :, :h, :w] = img.to(device)
            batch_mask[i, :h, :w] = False
        
        samples = NestedTensor(batch_tensor, batch_mask)
        timing.batch_assembly += timer.stop()
        
        # Step 4: Text encoding (once per batch)
        timer.start()
        tokenizer = model.model.tokenizer
        tokenized = tokenizer(caption)
        timing.text_encoding += timer.stop()
        
        # Step 5: Forward pass (batched)
        timer.start()
        with torch.no_grad():
            outputs = model.model(samples, captions=[caption] * actual_batch_size)
        timing.forward_pass += timer.stop()
        
        # Step 6: Post-processing (per image in batch)
        timer.start()
        all_pred_logits = outputs["pred_logits"].cpu().sigmoid()
        all_pred_boxes = outputs["pred_boxes"].cpu()
        
        for b in range(actual_batch_size):
            pred_logits = all_pred_logits[b]
            pred_boxes = all_pred_boxes[b]
            
            mask = pred_logits.max(dim=1)[0] > config['box_threshold']
            logits = pred_logits[mask]
            boxes = pred_boxes[mask]
            
            # Phrase extraction
            phrases = [
                get_phrases_from_posmap(logit > config['text_threshold'], tokenized, tokenizer).replace('.', '')
                for logit in logits
            ]
            
            # Scale boxes
            source_h, source_w = images_bgr[b].shape[:2]
            boxes_scaled = boxes * torch.tensor([source_w, source_h, source_w, source_h])
            
            # NMS
            if len(boxes_scaled) > 0:
                nms_idx = torchvision.ops.nms(
                    boxes_scaled, logits.max(dim=1)[0], config['nms_threshold']
                )
                timing.num_detections += len(nms_idx)
        
        timing.postprocess += timer.stop()
        timing.num_images += actual_batch_size
        timing.num_batches += 1
    
    timing.total = time.perf_counter() - total_start
    
    return ProfileResult(
        mode=f"batch_{batch_size}",
        batch_size=batch_size,
        timing=timing,
        gpu_memory_peak_mb=get_gpu_memory_mb()
    )


def print_timing_table(results: List[ProfileResult]):
    """Print detailed timing breakdown table."""
    print()
    print("=" * 120)
    print(f"{'Mode':<12} {'Batch':<6} {'Images':<7} {'I/O':<8} {'Preproc':<8} "
          f"{'BatchAsm':<9} {'TextEnc':<8} {'Forward':<9} {'PostProc':<9} "
          f"{'Total':<9} {'Per Img':<9}")
    print("-" * 120)
    
    for r in results:
        t = r.timing
        per_img = t.total / max(t.num_images, 1)
        
        print(f"{r.mode:<12} {r.batch_size:<6} {t.num_images:<7} "
              f"{t.image_io:<8.3f} {t.preprocessing:<8.3f} "
              f"{t.batch_assembly:<9.3f} {t.text_encoding:<8.3f} "
              f"{t.forward_pass:<9.3f} {t.postprocess:<9.3f} "
              f"{t.total:<9.3f} {per_img:<9.4f}")
    
    print("=" * 120)


def print_percentage_breakdown(results: List[ProfileResult]):
    """Print timing as percentage of total."""
    print()
    print_section("Time Distribution (% of total)")
    
    for r in results:
        t = r.timing
        total = t.total if t.total > 0 else 1
        
        print(f"\n{r.mode} (batch_size={r.batch_size}):")
        print(f"  Image I/O:       {t.image_io/total*100:6.1f}%  ({t.image_io:.3f}s)")
        print(f"  Preprocessing:   {t.preprocessing/total*100:6.1f}%  ({t.preprocessing:.3f}s)")
        print(f"  Batch Assembly:  {t.batch_assembly/total*100:6.1f}%  ({t.batch_assembly:.3f}s)")
        print(f"  Text Encoding:   {t.text_encoding/total*100:6.1f}%  ({t.text_encoding:.3f}s)")
        print(f"  Forward Pass:    {t.forward_pass/total*100:6.1f}%  ({t.forward_pass:.3f}s)")
        print(f"  Post-processing: {t.postprocess/total*100:6.1f}%  ({t.postprocess:.3f}s)")
        print(f"  ─────────────────────────")
        print(f"  Total:           100.0%  ({t.total:.3f}s)")
        print(f"  Per image:       {t.total/max(t.num_images,1)*1000:.1f}ms")
        print(f"  Detections:      {t.num_detections}")
        if r.gpu_memory_peak_mb:
            print(f"  GPU Memory:      {r.gpu_memory_peak_mb:.0f}MB")


def print_comparison(results: List[ProfileResult]):
    """Print comparison between sequential and batched."""
    if len(results) < 2:
        return
    
    baseline = results[0]
    
    print()
    print_section("Speedup Analysis (vs baseline)")
    
    print(f"\nBaseline: {baseline.mode}")
    print(f"  Total time: {baseline.timing.total:.3f}s")
    print()
    
    for r in results[1:]:
        speedup = baseline.timing.total / r.timing.total if r.timing.total > 0 else 0
        
        print(f"{r.mode}:")
        print(f"  Speedup:        {speedup:.2f}x")
        
        # Per-step comparison
        bt = baseline.timing
        rt = r.timing
        
        def compare(name, base_val, new_val):
            if base_val > 0:
                ratio = new_val / base_val
                return f"{ratio:.2f}x" if ratio >= 1 else f"{1/ratio:.2f}x faster"
            return "N/A"
        
        print(f"  I/O:            {compare('io', bt.image_io, rt.image_io)}")
        print(f"  Preprocessing:  {compare('pre', bt.preprocessing, rt.preprocessing)}")
        print(f"  Batch Assembly: {rt.batch_assembly:.3f}s (overhead)")
        print(f"  Text Encoding:  {compare('text', bt.text_encoding, rt.text_encoding)}")
        print(f"  Forward Pass:   {compare('fwd', bt.forward_pass, rt.forward_pass)}")
        print(f"  Post-process:   {compare('post', bt.postprocess, rt.postprocess)}")
        print()


def analyze_bottleneck(results: List[ProfileResult]):
    """Analyze where the bottleneck is."""
    print()
    print_section("Bottleneck Analysis")
    
    for r in results:
        t = r.timing
        total = t.total if t.total > 0 else 1
        
        # Find the dominant step
        steps = [
            ("Image I/O", t.image_io),
            ("Preprocessing", t.preprocessing),
            ("Batch Assembly", t.batch_assembly),
            ("Text Encoding", t.text_encoding),
            ("Forward Pass", t.forward_pass),
            ("Post-processing", t.postprocess),
        ]
        
        steps_sorted = sorted(steps, key=lambda x: -x[1])
        
        print(f"\n{r.mode}:")
        print(f"  Bottleneck: {steps_sorted[0][0]} ({steps_sorted[0][1]/total*100:.1f}%)")
        print(f"  Top 3:")
        for name, val in steps_sorted[:3]:
            print(f"    - {name}: {val:.3f}s ({val/total*100:.1f}%)")


def main():
    args = parse_args()
    
    print_header("Detailed Auto-Label Profiler")
    print("Step-by-step timing analysis")
    print()
    
    # Setup
    if args.gpu >= 0:
        setup_cuda_device(args.gpu)
        device = "cuda"
    else:
        device = "cpu"
    
    class_prompts = [c.strip() for c in args.classes.split(",")]
    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",")]
    image_paths = collect_image_paths(args.images, args.max_images)
    
    if not image_paths:
        logger.error("No images found")
        sys.exit(1)
    
    print_section("Configuration")
    print(f"  Images:        {len(image_paths)}")
    print(f"  Classes:       {class_prompts}")
    print(f"  Output mode:   {args.output_mode}")
    print(f"  Batch sizes:   {batch_sizes}")
    print(f"  Device:        {device}")
    print()
    
    config = {
        'dino_config': "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        'dino_checkpoint': "data/models/pretrained/groundingdino_swint_ogc.pth",
        'box_threshold': args.box_threshold,
        'text_threshold': args.box_threshold,
        'nms_threshold': 0.7,
    }
    
    results: List[ProfileResult] = []
    
    # Warmup
    if args.warmup > 0:
        print_section("Warmup")
        warmup_paths = image_paths[:min(5, len(image_paths))]
        for i in range(args.warmup):
            logger.info(f"Warmup run {i+1}/{args.warmup}")
            _ = profile_batched_detailed(warmup_paths, class_prompts, 4, config, device)
        print()
    
    # Sequential baseline
    if args.compare_sequential:
        print_section("Profiling: Sequential")
        logger.info("Running sequential...")
        result = profile_sequential_detailed(image_paths, class_prompts, config, device)
        results.append(result)
        logger.info(f"  Total: {result.timing.total:.3f}s ({result.timing.total/result.timing.num_images:.4f}s/img)")
    
    # Batched runs
    for batch_size in batch_sizes:
        print_section(f"Profiling: Batch size {batch_size}")
        logger.info(f"Running batch_size={batch_size}...")
        result = profile_batched_detailed(image_paths, class_prompts, batch_size, config, device)
        results.append(result)
        logger.info(f"  Total: {result.timing.total:.3f}s ({result.timing.total/result.timing.num_images:.4f}s/img)")
    
    # Results
    print()
    print_section("Detailed Timing Results (seconds)")
    print_timing_table(results)
    
    print_percentage_breakdown(results)
    
    if len(results) >= 2:
        print_comparison(results)
    
    analyze_bottleneck(results)
    
    # Final recommendation
    print()
    print_section("Conclusion")
    
    best = min(results, key=lambda r: r.timing.total)
    worst = max(results, key=lambda r: r.timing.total)
    
    print(f"  Best:  {best.mode} ({best.timing.total:.3f}s)")
    print(f"  Worst: {worst.mode} ({worst.timing.total:.3f}s)")
    
    if best.mode == "sequential" or best.batch_size == 1:
        print("\n  ⚠️  Batching is NOT faster for this workload.")
        print("  Possible reasons:")
        print("    - Variable image sizes cause excessive padding")
        print("    - Batch assembly overhead exceeds forward pass savings")
        print("    - I/O or preprocessing is the bottleneck (not forward pass)")
    else:
        speedup = worst.timing.total / best.timing.total
        print(f"\n  ✓ Batching provides {speedup:.1f}x speedup")


if __name__ == "__main__":
    main()
