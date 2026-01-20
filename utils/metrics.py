"""
Evaluation Metrics for HOI Detection.

Ported from hoi-benchmarks for consistent evaluation.

Provides:
- Grounding metrics: AR (Average Recall) at various IoU thresholds
- Referring metrics: METEOR, CIDEr, BLEU, ROUGE-L (COCO caption metrics)

Reference:
- hoi-benchmarks/eval_swig_ground_qwen3vl.py
- hoi-benchmarks/eval_swig_action_referring_qwen3vl.py
"""

import os
import json
import tempfile
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


def normalize_pair_format(pair: Union[Dict, List]) -> Dict:
    """
    Normalize pair format to standard dict with 'person_box' and 'object_box'.
    
    Handles:
    - Dict with 'person_box'/'object_box'
    - Dict with 'person_bbox'/'object_bbox'
    - List format (shouldn't happen but handle gracefully)
    """
    if isinstance(pair, dict):
        return {
            'person_box': pair.get('person_box', pair.get('person_bbox', [])),
            'object_box': pair.get('object_box', pair.get('object_bbox', []))
        }
    elif isinstance(pair, list):
        # Assume it's a flat list or already in some format
        return {'person_box': [], 'object_box': []}
    return {'person_box': [], 'object_box': []}


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
        
        # Normalize to list of dicts
        if isinstance(pred_pairs, list):
            pred_pairs = [normalize_pair_format(p) for p in pred_pairs]
        
        if isinstance(gt_pairs, list):
            gt_pairs = [normalize_pair_format(p) for p in gt_pairs]
        
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
    predictions: List[Union[Dict, List]], 
    ground_truths: List[Union[Dict, List]]
) -> Dict[str, float]:
    """
    Evaluate grounding predictions against ground truths.
    
    Convenience wrapper that formats inputs for compute_grounding_metrics.
    
    Args:
        predictions: List of prediction dicts/lists with pairs
        ground_truths: List of ground truth dicts/lists with pairs
        
    Returns:
        Dict of metrics
    """
    results = []
    for pred, gt in zip(predictions, ground_truths):
        # Handle various input formats
        if isinstance(pred, dict):
            pred_pairs = pred.get('pairs', [])
        elif isinstance(pred, list):
            pred_pairs = pred
        else:
            pred_pairs = []
        
        if isinstance(gt, dict):
            gt_pairs = gt.get('pairs', [])
        elif isinstance(gt, list):
            gt_pairs = gt
        else:
            gt_pairs = []
        
        results.append({
            'predicted_pairs': pred_pairs,
            'gt_pairs': gt_pairs
        })
    
    return compute_grounding_metrics(results)


# =============================================================================
# REFERRING METRICS (COCO Caption Metrics)
# =============================================================================

def clean_text(text: str) -> str:
    """
    Clean prediction/reference text for evaluation.
    
    Handles:
    - Markdown bold: **text** -> text
    - Markdown italic: *text* -> text
    - Extra whitespace and newlines
    - Empty or None values
    - Common prefixes from model outputs
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
    
    # Remove common prefixes
    prefixes_to_remove = [
        "the person is ",
        "person is ",
        "they are ",
        "action: ",
        "answer: ",
        "output: ",
    ]
    
    text_lower = text.lower()
    for prefix in prefixes_to_remove:
        if text_lower.startswith(prefix):
            text = text[len(prefix):].strip()
            text_lower = text.lower()
    
    # Remove trailing punctuation
    text = text.rstrip('.!?,;:')
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip().lower()


def evaluate_referring_nltk(
    predictions: List[str],
    references: List[str],
    clean: bool = True,
    compute_bertscore: bool = True,
) -> Dict[str, float]:
    """
    Evaluate referring predictions using NLTK metrics (no Java required).
    
    Computes:
    - METEOR (via NLTK - doesn't need Java)
    - CIDEr (via pycocoevalcap - doesn't need Java)
    - BLEU-1, BLEU-2, BLEU-3, BLEU-4
    - BERTScore (precision, recall, F1)
    - Exact match
    - Verb match (first word)
    - Word overlap (Jaccard similarity)
    
    This is a faster alternative to evaluate_referring_coco that doesn't require Java.
    """
    import numpy as np
    
    if len(predictions) == 0:
        return {
            "METEOR": 0.0, "CIDEr": 0.0, "BLEU_1": 0.0, "BLEU_2": 0.0, "BLEU_3": 0.0, "BLEU_4": 0.0,
            "exact_match": 0.0, "verb_match": 0.0, "word_overlap": 0.0,
            "bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0
        }
    
    # Clean text if requested
    if clean:
        predictions = [clean_text(p) for p in predictions]
        references = [clean_text(r) for r in references]
    
    metrics = {}
    
    # Try NLTK METEOR (doesn't require Java)
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        
        # Download required data (silent)
        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except:
            pass
        
        meteor_scores = []
        for pred, ref in zip(predictions, references):
            if pred and ref:
                try:
                    score = meteor_score([ref.split()], pred.split())
                    meteor_scores.append(score)
                except:
                    meteor_scores.append(0.0)
            else:
                meteor_scores.append(0.0)
        
        metrics['METEOR'] = float(np.mean(meteor_scores)) if meteor_scores else 0.0
        
    except ImportError:
        logger.warning("NLTK not available for METEOR. Install with: pip install nltk")
        metrics['METEOR'] = 0.0
    
    # BLEU scores using NLTK
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        smoothie = SmoothingFunction().method1
        
        for n in [1, 2, 3, 4]:
            bleu_scores = []
            for pred, ref in zip(predictions, references):
                if pred and ref:
                    try:
                        weights = tuple([1.0/n] * n + [0.0] * (4-n))
                        score = sentence_bleu([ref.split()], pred.split(), weights=weights, smoothing_function=smoothie)
                        bleu_scores.append(score)
                    except:
                        bleu_scores.append(0.0)
                else:
                    bleu_scores.append(0.0)
            
            metrics[f'BLEU_{n}'] = float(np.mean(bleu_scores)) if bleu_scores else 0.0
            
    except ImportError:
        logger.warning("NLTK not available for BLEU. Install with: pip install nltk")
        for n in [1, 2, 3, 4]:
            metrics[f'BLEU_{n}'] = 0.0
    
    # CIDEr score using pycocoevalcap (doesn't require Java)
    try:
        from pycocoevalcap.cider.cider import Cider
        
        # Format for pycocoevalcap: {id: [caption]}
        gts = {i: [ref] for i, ref in enumerate(references) if ref}
        res = {i: [pred] for i, pred in enumerate(predictions) if pred and i in gts}
        
        if gts and res:
            cider_scorer = Cider()
            cider_score, _ = cider_scorer.compute_score(gts, res)
            metrics['CIDEr'] = float(cider_score)
        else:
            metrics['CIDEr'] = 0.0
            
    except ImportError:
        logger.warning("pycocoevalcap not available for CIDEr. Install with: pip install pycocoevalcap")
        metrics['CIDEr'] = 0.0
    except Exception as e:
        logger.warning(f"CIDEr computation failed: {e}")
        metrics['CIDEr'] = 0.0
    
    # Exact match
    exact_matches = sum(1 for p, r in zip(predictions, references) if p == r)
    metrics['exact_match'] = exact_matches / len(predictions) if len(predictions) > 0 else 0.0
    
    # Verb match (first word match)
    verb_matches = 0
    for pred, ref in zip(predictions, references):
        pred_words = pred.split() if pred else []
        ref_words = ref.split() if ref else []
        if pred_words and ref_words and pred_words[0] == ref_words[0]:
            verb_matches += 1
    metrics['verb_match'] = verb_matches / len(predictions) if len(predictions) > 0 else 0.0
    
    # Word overlap (Jaccard similarity)
    overlaps = []
    for pred, ref in zip(predictions, references):
        pred_words = set(pred.split()) if pred else set()
        ref_words = set(ref.split()) if ref else set()
        if pred_words or ref_words:
            intersection = len(pred_words & ref_words)
            union = len(pred_words | ref_words)
            overlaps.append(intersection / union if union > 0 else 0.0)
        else:
            overlaps.append(0.0)
    metrics['word_overlap'] = float(np.mean(overlaps)) if overlaps else 0.0
    
    # BERTScore computation
    if compute_bertscore:
        try:
            from bert_score import score as bert_score
            
            # Filter out empty predictions/references for BERTScore
            valid_pairs = [(p, r) for p, r in zip(predictions, references) if p and r]
            
            if valid_pairs:
                preds_valid, refs_valid = zip(*valid_pairs)
                
                # Compute BERTScore (uses roberta-large by default)
                P, R, F1 = bert_score(
                    list(preds_valid), 
                    list(refs_valid),
                    model_type="microsoft/deberta-v2-xxlarge-mnli",
                    lang="en",
                    batch_size=32,
                    rescale_with_baseline=True,
                    verbose=False
                )
                
                metrics['bertscore_precision'] = float(P.mean())
                metrics['bertscore_recall'] = float(R.mean())
                metrics['bertscore_f1'] = float(F1.mean())
            else:
                metrics['bertscore_precision'] = 0.0
                metrics['bertscore_recall'] = 0.0
                metrics['bertscore_f1'] = 0.0
                
        except ImportError:
            logger.warning("bert_score not available. Install with: pip install bert-score")
            metrics['bertscore_precision'] = 0.0
            metrics['bertscore_recall'] = 0.0
            metrics['bertscore_f1'] = 0.0
        except Exception as e:
            logger.warning(f"BERTScore computation failed: {e}")
            metrics['bertscore_precision'] = 0.0
            metrics['bertscore_recall'] = 0.0
            metrics['bertscore_f1'] = 0.0
    
    return metrics


def evaluate_referring_coco(
    predictions: List[str],
    references: List[str],
    clean: bool = True,
) -> Dict[str, float]:
    """
    Evaluate referring predictions using COCO caption metrics.
    
    Uses pycocoevalcap to compute:
    - BLEU-1, BLEU-2, BLEU-3, BLEU-4
    - METEOR
    - ROUGE-L
    - CIDEr
    
    Args:
        predictions: List of predicted action strings
        references: List of ground truth action strings
        clean: Whether to clean text before scoring
        
    Returns:
        Dict with all metrics
    """
    try:
        from pycocotools.coco import COCO
        from pycocoevalcap.eval import COCOEvalCap
    except ImportError:
        logger.warning("pycocoevalcap not installed. Install with: pip install pycocoevalcap")
        return evaluate_referring_simple(predictions, references)
    
    if len(predictions) == 0:
        return {
            "BLEU_1": 0.0, "BLEU_2": 0.0, "BLEU_3": 0.0, "BLEU_4": 0.0,
            "METEOR": 0.0, "ROUGE_L": 0.0, "CIDEr": 0.0, "exact_match": 0.0
        }
    
    # Clean text if requested
    if clean:
        predictions = [clean_text(p) for p in predictions]
        references = [clean_text(r) for r in references]
    
    # Create ground truth annotations in COCO format
    annotations = []
    images_info = []
    for idx, ref in enumerate(references):
        images_info.append({'id': idx})
        annotations.append({
            'image_id': idx,
            'caption': ref,
            'id': idx
        })
    
    # Create predictions in COCO format
    pred_annotations = []
    for idx, pred in enumerate(predictions):
        pred_annotations.append({
            'image_id': idx,
            'caption': pred
        })
    
    # Create temporary files for COCO evaluation
    gt_coco_format = {
        'info': {
            'description': 'HOI Action Referring Ground Truth',
            'version': '1.0',
            'year': 2025
        },
        'licenses': [{'id': 1, 'name': 'Unknown', 'url': ''}],
        'images': images_info,
        'annotations': annotations,
        'type': 'captions'
    }
    
    # Write to temp files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(gt_coco_format, f)
        gt_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(pred_annotations, f)
        pred_file = f.name
    
    try:
        # Run COCO evaluation
        # Suppress COCO output
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        coco = COCO(gt_file)
        coco_result = coco.loadRes(pred_file)
        coco_eval = COCOEvalCap(coco, coco_result)
        coco_eval.evaluate()
        
        sys.stdout = old_stdout
        
        # Extract metrics
        metrics = {}
        for metric, score in coco_eval.eval.items():
            metrics[metric] = float(score)
        
        # Add exact match accuracy
        exact_matches = sum(1 for p, r in zip(predictions, references) if p == r)
        metrics['exact_match'] = exact_matches / len(predictions) if len(predictions) > 0 else 0.0
        
    except Exception as e:
        logger.warning(f"COCO evaluation failed: {e}. Using simple metrics.")
        metrics = evaluate_referring_simple(predictions, references)
    finally:
        # Clean up temp files
        try:
            os.unlink(gt_file)
            os.unlink(pred_file)
        except:
            pass
    
    return metrics


def evaluate_referring(
    predictions: List[str],
    references: List[str],
    model_type: str = "roberta-large",
    batch_size: int = 64,
    device: Optional[str] = None,
    clean: bool = True,
) -> Dict[str, float]:
    """
    Evaluate referring predictions using BERTScore (legacy).
    
    NOTE: This is kept for backwards compatibility.
    Use evaluate_referring_coco() for METEOR, CIDEr, BLEU, ROUGE-L metrics.
    
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
        logger.warning("bert_score not installed. Using COCO metrics instead.")
        return evaluate_referring_coco(predictions, references, clean)
    
    if len(predictions) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # Clean text if requested
    if clean:
        predictions_clean = [clean_text(p) for p in predictions]
        references_clean = [clean_text(r) for r in references]
    else:
        predictions_clean = predictions
        references_clean = references
    
    # Compute BERTScore
    P, R, F1 = bert_score(
        predictions_clean,
        references_clean,
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
    Simple evaluation for referring task without external dependencies.
    
    Metrics:
    - exact_match: Exact string match
    - verb_match: First word (verb) matches
    - word_overlap: Jaccard similarity of words
    - non_empty: Percentage of non-empty predictions
    """
    if len(predictions) == 0:
        return {"exact_match": 0.0, "verb_match": 0.0, "word_overlap": 0.0, "non_empty": 0.0}
    
    exact_matches = 0
    verb_matches = 0
    word_overlaps = []
    non_empty = 0
    
    for pred, ref in zip(predictions, references):
        pred_clean = clean_text(pred).lower()
        ref_clean = clean_text(ref).lower()
        
        # Track non-empty predictions
        if pred_clean:
            non_empty += 1
        
        # Exact match
        if pred_clean == ref_clean:
            exact_matches += 1
        
        # Verb match (first word)
        pred_words = pred_clean.split() if pred_clean else []
        ref_words = ref_clean.split() if ref_clean else []
        
        pred_verb = pred_words[0] if pred_words else ""
        ref_verb = ref_words[0] if ref_words else ""
        
        if pred_verb and ref_verb and pred_verb == ref_verb:
            verb_matches += 1
        
        # Word overlap (Jaccard similarity)
        if pred_words and ref_words:
            pred_set = set(pred_words)
            ref_set = set(ref_words)
            intersection = len(pred_set & ref_set)
            union = len(pred_set | ref_set)
            word_overlaps.append(intersection / union if union > 0 else 0.0)
        else:
            word_overlaps.append(0.0)
    
    return {
        "exact_match": exact_matches / len(predictions),
        "verb_match": verb_matches / len(predictions),
        "word_overlap": sum(word_overlaps) / len(word_overlaps) if word_overlaps else 0.0,
        "non_empty": non_empty / len(predictions),
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
                # Use COCO metrics
                try:
                    referring_metrics = evaluate_referring_coco(predictions, references)
                except Exception as e:
                    logger.warning(f"COCO evaluation failed, using simple metrics: {e}")
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
