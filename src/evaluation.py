"""
Evaluation Module
Compare detected segmentation against ground truth and compute metrics.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .segmentation import SegmentationResult, VideoSegment, SegmentType


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    precision: float
    recall: float
    f1_score: float
    iou: float
    accuracy: float
    
    true_positives: float
    false_positives: float
    false_negatives: float
    true_negatives: float
    
    boundary_error_avg: float
    boundary_error_std: float
    
    def to_dict(self) -> Dict:
        return {
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'iou': self.iou,
            'accuracy': self.accuracy,
            'true_positives_seconds': self.true_positives,
            'false_positives_seconds': self.false_positives,
            'false_negatives_seconds': self.false_negatives,
            'true_negatives_seconds': self.true_negatives,
            'boundary_error_avg_seconds': self.boundary_error_avg,
            'boundary_error_std_seconds': self.boundary_error_std
        }
    
    def __str__(self) -> str:
        return (
            f"Evaluation Metrics:\n"
            f"  Precision: {self.precision:.2%}\n"
            f"  Recall: {self.recall:.2%}\n"
            f"  F1 Score: {self.f1_score:.2%}\n"
            f"  IoU: {self.iou:.2%}\n"
            f"  Accuracy: {self.accuracy:.2%}\n"
            f"  Boundary Error: {self.boundary_error_avg:.2f}s ± {self.boundary_error_std:.2f}s"
        )


class SegmentationEvaluator:
    """
    Evaluates detected segmentation against ground truth.
    Computes frame-level and segment-level metrics.
    """
    
    def __init__(self, resolution: float = 0.1):
        """
        Initialize evaluator.
        
        Args:
            resolution: Time resolution in seconds for frame-level comparison
        """
        self.resolution = resolution
    
    def evaluate(self, detected: SegmentationResult, 
                 ground_truth: SegmentationResult,
                 target_type: SegmentType = SegmentType.AD) -> EvaluationMetrics:
        """
        Evaluate detected segmentation against ground truth.
        
        Args:
            detected: Detected segmentation result
            ground_truth: Ground truth segmentation result
            target_type: Segment type to evaluate (default: AD)
            
        Returns:
            EvaluationMetrics object
        """
        duration = max(detected.duration, ground_truth.duration)
        num_frames = int(duration / self.resolution) + 1
        
        detected_mask = self._create_mask(detected.segments, num_frames, target_type)
        gt_mask = self._create_mask(ground_truth.segments, num_frames, target_type)
        
        tp = np.sum(detected_mask & gt_mask) * self.resolution
        fp = np.sum(detected_mask & ~gt_mask) * self.resolution
        fn = np.sum(~detected_mask & gt_mask) * self.resolution
        tn = np.sum(~detected_mask & ~gt_mask) * self.resolution
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
        
        boundary_errors = self._compute_boundary_errors(
            detected.segments, ground_truth.segments, target_type
        )
        
        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            iou=iou,
            accuracy=accuracy,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            boundary_error_avg=np.mean(boundary_errors) if boundary_errors else 0,
            boundary_error_std=np.std(boundary_errors) if boundary_errors else 0
        )
    
    def _create_mask(self, segments: List[VideoSegment], 
                     num_frames: int, target_type: SegmentType) -> np.ndarray:
        """Create binary mask for target segment type."""
        mask = np.zeros(num_frames, dtype=bool)
        
        for seg in segments:
            if seg.segment_type == target_type:
                start_idx = int(seg.start_time / self.resolution)
                end_idx = int(seg.end_time / self.resolution)
                mask[start_idx:min(end_idx, num_frames)] = True
        
        return mask
    
    def _compute_boundary_errors(self, detected: List[VideoSegment],
                                  ground_truth: List[VideoSegment],
                                  target_type: SegmentType) -> List[float]:
        """Compute boundary errors between detected and ground truth segments."""
        detected_boundaries = []
        for seg in detected:
            if seg.segment_type == target_type:
                detected_boundaries.extend([seg.start_time, seg.end_time])
        
        gt_boundaries = []
        for seg in ground_truth:
            if seg.segment_type == target_type:
                gt_boundaries.extend([seg.start_time, seg.end_time])
        
        if not detected_boundaries or not gt_boundaries:
            return []
        
        errors = []
        for det_b in detected_boundaries:
            min_error = min(abs(det_b - gt_b) for gt_b in gt_boundaries)
            errors.append(min_error)
        
        return errors
    
    def evaluate_all_types(self, detected: SegmentationResult,
                          ground_truth: SegmentationResult) -> Dict[str, EvaluationMetrics]:
        """Evaluate all segment types."""
        results = {}
        
        all_types = set()
        for seg in detected.segments + ground_truth.segments:
            all_types.add(seg.segment_type)
        
        for seg_type in all_types:
            results[seg_type.value] = self.evaluate(detected, ground_truth, seg_type)
        
        return results


def evaluate_video(detected_path: str, ground_truth_path: str,
                   target_type: str = 'ad') -> EvaluationMetrics:
    """
    Evaluate a single video's segmentation.
    
    Args:
        detected_path: Path to detected segmentation JSON
        ground_truth_path: Path to ground truth info JSON
        target_type: Segment type to evaluate
        
    Returns:
        EvaluationMetrics object
    """
    detected = SegmentationResult.load(detected_path)
    
    from .segmentation import GroundTruthSegmenter
    gt_segmenter = GroundTruthSegmenter(detected.video_path, ground_truth_path)
    ground_truth = gt_segmenter.get_segmentation()
    
    evaluator = SegmentationEvaluator()
    return evaluator.evaluate(detected, ground_truth, SegmentType(target_type))


def batch_evaluate(detected_dir: str, ground_truth_dir: str,
                   output_path: Optional[str] = None) -> Dict:
    """
    Batch evaluate multiple videos.
    
    Args:
        detected_dir: Directory containing detected segmentation JSONs
        ground_truth_dir: Directory containing ground truth info JSONs
        output_path: Optional path to save results
        
    Returns:
        Dictionary with per-video and aggregate metrics
    """
    detected_dir = Path(detected_dir)
    ground_truth_dir = Path(ground_truth_dir)
    
    results = {
        'per_video': {},
        'aggregate': {}
    }
    
    all_metrics = []
    
    for det_file in detected_dir.glob('*.segmentation.json'):
        video_name = det_file.stem.replace('.segmentation', '')
        gt_file = ground_truth_dir / f'{video_name}.json'
        
        if not gt_file.exists():
            print(f"Warning: No ground truth found for {video_name}")
            continue
        
        try:
            metrics = evaluate_video(str(det_file), str(gt_file))
            results['per_video'][video_name] = metrics.to_dict()
            all_metrics.append(metrics)
            print(f"{video_name}: F1={metrics.f1_score:.2%}, IoU={metrics.iou:.2%}")
        except Exception as e:
            print(f"Error evaluating {video_name}: {e}")
    
    if all_metrics:
        results['aggregate'] = {
            'precision': np.mean([m.precision for m in all_metrics]),
            'recall': np.mean([m.recall for m in all_metrics]),
            'f1_score': np.mean([m.f1_score for m in all_metrics]),
            'iou': np.mean([m.iou for m in all_metrics]),
            'accuracy': np.mean([m.accuracy for m in all_metrics]),
            'num_videos': len(all_metrics)
        }
        
        print(f"\nAggregate Results ({len(all_metrics)} videos):")
        print(f"  Precision: {results['aggregate']['precision']:.2%}")
        print(f"  Recall: {results['aggregate']['recall']:.2%}")
        print(f"  F1 Score: {results['aggregate']['f1_score']:.2%}")
        print(f"  IoU: {results['aggregate']['iou']:.2%}")
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return results


def generate_confusion_matrix_plot(detected: SegmentationResult,
                                   ground_truth: SegmentationResult,
                                   output_path: str):
    """Generate confusion matrix visualization."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib and seaborn required for plotting")
        return
    
    evaluator = SegmentationEvaluator()
    
    duration = max(detected.duration, ground_truth.duration)
    num_frames = int(duration / evaluator.resolution) + 1
    
    det_content = evaluator._create_mask(detected.segments, num_frames, SegmentType.CORE_CONTENT)
    det_ad = evaluator._create_mask(detected.segments, num_frames, SegmentType.AD)
    gt_content = evaluator._create_mask(ground_truth.segments, num_frames, SegmentType.CORE_CONTENT)
    gt_ad = evaluator._create_mask(ground_truth.segments, num_frames, SegmentType.AD)
    
    confusion = np.zeros((2, 2))
    confusion[0, 0] = np.sum(det_content & gt_content)
    confusion[0, 1] = np.sum(det_content & gt_ad)
    confusion[1, 0] = np.sum(det_ad & gt_content)
    confusion[1, 1] = np.sum(det_ad & gt_ad)
    
    confusion = confusion * evaluator.resolution
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=['Content', 'Ad'],
                yticklabels=['Content', 'Ad'],
                ax=ax)
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Detected')
    ax.set_title('Confusion Matrix (seconds)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Confusion matrix saved to: {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 3:
        detected_path = sys.argv[1]
        gt_path = sys.argv[2]
        
        metrics = evaluate_video(detected_path, gt_path)
        print(metrics)
