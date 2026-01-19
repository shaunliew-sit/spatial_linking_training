"""
Evaluation Metrics for HOI Detection.

Ported from hoi-benchmarks for consistent evaluation.

Provides:
- Grounding metrics: AR (Average Recall) at various IoU thresholds
- Referring metrics: BERTScore for action description matching

Reference:
- hoi-benchmarks/recalculate_ground_metrics.py
- hoi-benchmarks/calculate_bertscore.py
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# IoU AND BOX UTILITIES
# =============================================================================

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate IoU between two boxes in [x1, y1, x2, y2] format.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def get_box_area(box: List[float]) -> float:
    """Calculate box area from [x1, y1, x2, y2] format."""
    return (box[2] - box[0]) * (box[3] - box[1])


def categorize_pair_by_size(
    gt_pair: Dict, 
    area_small: float = 1024, 
    area_medium: float = 9216
) -> str:
    """
    Categorize a ground truth pair by object size.
    
    COCO-style size categories:
    - small: area < 32² = 1024
    - medium: 32² <= area < 96² = 9216
    - large: area >= 96²
    
    Args:
        gt_pair: Dict with 'object_bbox' or 'object_box' key
        area_small: Threshold for small objects
        area_medium: Threshold for medium objects
        
    Returns:
        Size category: 'small', 'medium', or 'large'
    """
    object_box = gt_pair.get('object_bbox', gt_pair.get('object_box', [0, 0, 0, 0]))
    object_area = get_box_area(object_box)
    
    if object_area < area_small:
        return 'small'
    elif object_area < area_medium:
        return 'medium'
    else:
        return 'large'


# =============================================================================
# GROUNDING METRICS
# =============================================================================

def match_pairs_greedy(
    pred_pairs: List[Dict],
    gt_pairs: List[Dict],
    iou_threshold: float = 0.5
) -> Tuple[List[Tuple], set, set]:
    """
    Match predicted pairs to ground truth pairs using greedy matching.
    
    A pair matches if BOTH person and object boxes have IoU >= threshold.
    
    Args:
        pred_pairs: List of predicted pairs with 'person_box' and 'object_box'
        gt_pairs: List of ground truth pairs
        iou_threshold: IoU threshold for matching
        
    Returns:
        Tuple of (matches, matched_preds, matched_gts)
    """
    matches = []
    matched_preds = set()
    matched_gts = set()
    
    # Build score matrix
    scores = []
    for pred_idx, pred_pair in enumerate(pred_pairs):
        pred_person = pred_pair.get('person_bbox', pred_pair.get('person_box', []))
        pred_object = pred_pair.get('object_bbox', pred_pair.get('object_box', []))
        
        if not pred_person or not pred_object:
            continue
        
        for gt_idx, gt_pair in enumerate(gt_pairs):
            gt_person = gt_pair.get('person_bbox', gt_pair.get('person_box', []))
            gt_object = gt_pair.get('object_bbox', gt_pair.get('object_box', []))
            
            if not gt_person or not gt_object:
                continue
            
            person_iou = calculate_iou(pred_person, gt_person)
            object_iou = calculate_iou(pred_object, gt_object)
            
            if person_iou >= iou_threshold and object_iou >= iou_threshold:
                avg_iou = (person_iou + object_iou) / 2.0
                scores.append((avg_iou, pred_idx, gt_idx, person_iou, object_iou))
    
    # Greedy matching by score
    scores.sort(reverse=True)
    
    for avg_iou, pred_idx, gt_idx, person_iou, object_iou in scores:
        if pred_idx not in matched_preds and gt_idx not in matched_gts:
            matches.append((pred_idx, gt_idx, person_iou, object_iou))
            matched_preds.add(pred_idx)
            matched_gts.add(gt_idx)
    
    return matches, matched_preds, matched_gts


def compute_grounding_metrics(results: List[Dict]) -> Dict[str, float]:
    """
    Compute COCO-style Average Recall metrics for grounding task.
    
    Metrics computed:
    - AR: Average Recall over IoU 0.5:0.95
    - AR@0.5: Recall at IoU 0.5
    - AR@0.75: Recall at IoU 0.75
    - ARs: AR for small objects
    - ARm: AR for medium objects
    - ARl: AR for large objects
    
    Args:
        results: List of result dicts with 'predicted_pairs' and 'gt_pairs'
        
    Returns:
        Dict of metric names to values
    """
    metrics = {}
    
    # COCO area thresholds (in 1000x1000 normalized space, need to adjust)
    AREA_SMALL = 32 ** 2
    AREA_MEDIUM = 96 ** 2
    
    # Extract predictions and ground truths
    predictions = []
    ground_truths = []
    
    for r in results:
        pred_pairs = r.get('predicted_pairs', r.get('pairs', []))
        gt_pairs = r.get('gt_pairs', [])
        predictions.append({'pairs': pred_pairs})
        ground_truths.append({'pairs': gt_pairs})
    
    # Compute AR at specific thresholds
    for iou_thresh in [0.5, 0.75]:
        total_recall = 0.0
        total_samples = 0
        
        for pred, gt in zip(predictions, ground_truths):
            pred_pairs = pred.get('pairs', [])
            gt_pairs = gt.get('pairs', [])
            
            if len(gt_pairs) == 0:
                continue
            
            matches, _, matched_gts = match_pairs_greedy(pred_pairs, gt_pairs, iou_thresh)
            
            recall = len(matched_gts) / len(gt_pairs)
            total_recall += recall
            total_samples += 1
        
        avg_recall = total_recall / total_samples if total_samples > 0 else 0.0
        metrics[f'AR@{iou_thresh}'] = avg_recall
    
    # Compute AR over range [0.5, 0.95] with size-based tracking
    iou_range = np.arange(0.5, 1.0, 0.05)
    ar_scores = []
    
    recalls_small = []
    recalls_medium = []
    recalls_large = []
    
    for iou_thresh in iou_range:
        total_recall = 0.0
        total_samples = 0
        
        tp_small, fn_small = 0, 0
        tp_medium, fn_medium = 0, 0
        tp_large, fn_large = 0, 0
        
        for pred, gt in zip(predictions, ground_truths):
            pred_pairs = pred.get('pairs', [])
            gt_pairs = gt.get('pairs', [])
            
            if len(gt_pairs) == 0:
                continue
            
            matches, _, matched_gts = match_pairs_greedy(pred_pairs, gt_pairs, iou_thresh)
            
            recall = len(matched_gts) / len(gt_pairs)
            total_recall += recall
            total_samples += 1
            
            # Track size-specific metrics
            for gt_idx, gt_pair in enumerate(gt_pairs):
                size_cat = categorize_pair_by_size(gt_pair, AREA_SMALL, AREA_MEDIUM)
                matched = gt_idx in matched_gts
                
                if size_cat == 'small':
                    if matched:
                        tp_small += 1
                    else:
                        fn_small += 1
                elif size_cat == 'medium':
                    if matched:
                        tp_medium += 1
                    else:
                        fn_medium += 1
                else:
                    if matched:
                        tp_large += 1
                    else:
                        fn_large += 1
        
        avg_recall = total_recall / total_samples if total_samples > 0 else 0.0
        ar_scores.append(avg_recall)
        
        recall_small = tp_small / (tp_small + fn_small) if (tp_small + fn_small) > 0 else 0.0
        recall_medium = tp_medium / (tp_medium + fn_medium) if (tp_medium + fn_medium) > 0 else 0.0
        recall_large = tp_large / (tp_large + fn_large) if (tp_large + fn_large) > 0 else 0.0
        
        recalls_small.append(recall_small)
        recalls_medium.append(recall_medium)
        recalls_large.append(recall_large)
    
    metrics['AR'] = float(np.mean(ar_scores))
    metrics['ARs'] = float(np.mean(recalls_small)) if recalls_small else 0.0
    metrics['ARm'] = float(np.mean(recalls_medium)) if recalls_medium else 0.0
    metrics['ARl'] = float(np.mean(recalls_large)) if recalls_large else 0.0
    
    return metrics


def evaluate_grounding(
    predictions: List[Dict], 
    ground_truths: List[Dict]
) -> Dict[str, float]:
    """
    Evaluate grounding predictions against ground truths.
    
    Convenience wrapper that formats inputs for compute_grounding_metrics.
    
    Args:
        predictions: List of prediction dicts with 'pairs' key
        ground_truths: List of ground truth dicts with 'pairs' key
        
    Returns:
        Dict of metrics
    """
    results = []
    for pred, gt in zip(predictions, ground_truths):
        results.append({
            'predicted_pairs': pred.get('pairs', pred),
            'gt_pairs': gt.get('pairs', gt)
        })
    
    return compute_grounding_metrics(results)


# =============================================================================
# REFERRING METRICS (BERTScore)
# =============================================================================

def clean_text(text: str) -> str:
    """
    Clean prediction/reference text for more accurate BERTScore calculation.
    
    Handles:
    - Markdown bold: **text** -> text
    - Markdown italic: *text* -> text
    - Extra whitespace and newlines
    - Empty or None values
    """
    import re
    
    if text is None:
        return ""
    
    text = str(text)
    
    # Remove markdown bold
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    
    # Remove markdown italic
    text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'\1', text)
    
    # Remove markdown headers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    # Remove code blocks
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def evaluate_referring(
    predictions: List[str],
    references: List[str],
    model_type: str = "roberta-large",
    batch_size: int = 64,
    device: Optional[str] = None,
    clean: bool = True,
) -> Dict[str, float]:
    """
    Evaluate referring predictions using BERTScore.
    
    Args:
        predictions: List of predicted action strings
        references: List of ground truth action strings
        model_type: BERTScore model (roberta-large, deberta-v2-xxlarge-mnli)
        batch_size: Batch size for BERTScore computation
        device: Device to use (None for auto)
        clean: Whether to clean text before scoring
        
    Returns:
        Dict with precision, recall, f1 (mean values)
    """
    try:
        from bert_score import score as bert_score
    except ImportError:
        logger.error("bert_score not installed. Install with: pip install bert_score")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    if len(predictions) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # Clean text if requested
    if clean:
        predictions = [clean_text(p) for p in predictions]
        references = [clean_text(r) for r in references]
    
    # Compute BERTScore
    P, R, F1 = bert_score(
        predictions,
        references,
        model_type=model_type,
        batch_size=batch_size,
        device=device,
        rescale_with_baseline=True,
        lang="en",
        verbose=False,
    )
    
    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F1.mean()),
        "precision_std": float(P.std()),
        "recall_std": float(R.std()),
        "f1_std": float(F1.std()),
    }


def evaluate_referring_simple(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Simple exact match evaluation for referring task.
    
    Useful for quick evaluation without BERTScore dependency.
    """
    if len(predictions) == 0:
        return {"exact_match": 0.0, "verb_match": 0.0}
    
    exact_matches = 0
    verb_matches = 0
    
    for pred, ref in zip(predictions, references):
        pred_clean = clean_text(pred).lower()
        ref_clean = clean_text(ref).lower()
        
        # Exact match
        if pred_clean == ref_clean:
            exact_matches += 1
        
        # Verb match (first word)
        pred_verb = pred_clean.split()[0] if pred_clean else ""
        ref_verb = ref_clean.split()[0] if ref_clean else ""
        
        if pred_verb == ref_verb:
            verb_matches += 1
    
    return {
        "exact_match": exact_matches / len(predictions),
        "verb_match": verb_matches / len(predictions),
    }


# =============================================================================
# COMBINED EVALUATION
# =============================================================================

def evaluate_hoi_results(
    results: List[Dict],
    task_type: str = "both",
) -> Dict[str, float]:
    """
    Evaluate HOI detection results for both grounding and referring tasks.
    
    Args:
        results: List of result dicts
        task_type: "grounding", "referring", or "both"
        
    Returns:
        Dict of all metrics
    """
    metrics = {}
    
    if task_type in ["grounding", "both"]:
        grounding_results = [r for r in results if r.get("task_type") == "grounding"]
        if grounding_results:
            grounding_metrics = compute_grounding_metrics(grounding_results)
            metrics.update({f"grounding/{k}": v for k, v in grounding_metrics.items()})
    
    if task_type in ["referring", "both"]:
        referring_results = [r for r in results if r.get("task_type") == "referring"]
        if referring_results:
            predictions = [r.get("prediction", "") for r in referring_results]
            references = [r.get("ground_truth", "") for r in referring_results]
            
            if predictions and references:
                # Try BERTScore first
                try:
                    referring_metrics = evaluate_referring(predictions, references)
                except Exception as e:
                    logger.warning(f"BERTScore failed, using simple metrics: {e}")
                    referring_metrics = evaluate_referring_simple(predictions, references)
                
                metrics.update({f"referring/{k}": v for k, v in referring_metrics.items()})
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """Pretty print metrics."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  {key:<30} {value:>10.4f}")
        else:
            print(f"  {key:<30} {value:>10}")
    
    print("=" * 60)
