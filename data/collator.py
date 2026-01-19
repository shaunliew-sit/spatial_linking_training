"""
Custom Data Collator for HOI Spatial Linking Training.

Handles batching of multimodal data including:
- Text messages (with chat template)
- Images (single or multiple per sample)
- refer_boxes for spatial linking

Compatible with TRL SFTTrainer.
"""

import torch
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class HOISpatialCollator:
    """
    Custom collator for spatial linking training with Qwen3-VL.
    
    Handles:
    - Loading and processing images
    - Applying chat templates to messages
    - Tokenizing with the processor
    - Creating labels with proper masking
    - Padding refer_boxes for batching
    
    Args:
        processor: Qwen3-VL processor (AutoProcessor)
        max_length: Maximum sequence length (default: 2048)
        image_base_dir: Base directory for resolving relative image paths
        padding: Padding strategy ('max_length' or 'longest')
    """
    
    def __init__(
        self,
        processor,
        max_length: int = 2048,
        image_base_dir: Optional[str] = None,
        padding: str = "longest",
    ):
        self.processor = processor
        self.max_length = max_length
        self.image_base_dir = Path(image_base_dir) if image_base_dir else None
        self.padding = padding
        
        # Get special token IDs
        self.pad_token_id = processor.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = processor.tokenizer.eos_token_id
    
    def _load_image(self, image_path: Union[str, Path]) -> Optional[Image.Image]:
        """Load an image from path."""
        try:
            path = Path(image_path)
            
            # Try with base directory if path doesn't exist
            if not path.exists() and self.image_base_dir:
                path = self.image_base_dir / image_path
            
            if path.exists():
                return Image.open(path).convert("RGB")
            else:
                logger.warning(f"Image not found: {image_path}")
                return None
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None
    
    def _load_images(self, image_paths: List[str]) -> List[Image.Image]:
        """Load multiple images, filtering out failed loads."""
        images = []
        for path in image_paths:
            img = self._load_image(path)
            if img is not None:
                images.append(img)
        return images
    
    def _apply_chat_template(self, messages: List[Dict]) -> str:
        """Apply chat template to messages."""
        try:
            # Use processor's tokenizer chat template
            text = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            return text
        except Exception as e:
            # Fallback: simple concatenation
            logger.warning(f"Chat template failed, using fallback: {e}")
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"<|{role}|>\n{content}")
            return "\n".join(parts)
    
    def _pad_boxes(
        self, 
        boxes_list: List[Optional[List[List[int]]]], 
        max_boxes: int = 6
    ) -> torch.Tensor:
        """
        Pad refer_boxes to a fixed size for batching.
        
        Args:
            boxes_list: List of box lists, each [N, 4]
            max_boxes: Maximum number of boxes to keep
            
        Returns:
            Padded tensor of shape [batch, max_boxes, 4]
        """
        batch_size = len(boxes_list)
        padded = torch.zeros(batch_size, max_boxes, 4, dtype=torch.float32)
        
        for i, boxes in enumerate(boxes_list):
            if boxes is None:
                continue
            
            # Convert to tensor if needed
            if isinstance(boxes, list):
                boxes = torch.tensor(boxes, dtype=torch.float32)
            
            # Normalize to 0-1 if in 0-1000 format
            if boxes.max() > 1.0:
                boxes = boxes / 1000.0
            
            # Truncate or pad
            num_boxes = min(boxes.shape[0], max_boxes)
            padded[i, :num_boxes] = boxes[:num_boxes]
        
        return padded
    
    def _create_box_mask(
        self, 
        boxes_list: List[Optional[List[List[int]]]], 
        max_boxes: int = 6
    ) -> torch.Tensor:
        """Create mask indicating which boxes are valid (not padding)."""
        batch_size = len(boxes_list)
        mask = torch.zeros(batch_size, max_boxes, dtype=torch.bool)
        
        for i, boxes in enumerate(boxes_list):
            if boxes is not None:
                num_boxes = min(len(boxes), max_boxes)
                mask[i, :num_boxes] = True
        
        return mask
    
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.
        
        Each example should have:
        - messages: List of message dicts with 'role' and 'content'
        - images: List of image paths
        - refer_boxes (optional): List of [x1, y1, x2, y2] boxes
        
        Returns:
            Batch dict with input_ids, attention_mask, labels, pixel_values, etc.
        """
        batch_texts = []
        batch_images = []
        batch_boxes = []
        
        for ex in examples:
            # Get messages and apply chat template
            messages = ex.get("messages", [])
            text = self._apply_chat_template(messages)
            batch_texts.append(text)
            
            # Load images
            image_paths = ex.get("images", [])
            images = self._load_images(image_paths)
            batch_images.append(images)
            
            # Get refer_boxes
            boxes = ex.get("refer_boxes", None)
            batch_boxes.append(boxes)
        
        # Process with Qwen3-VL processor
        # Handle cases where some samples might have no images
        try:
            # Flatten images for processor (it expects list of images)
            all_images = []
            image_counts = []
            for imgs in batch_images:
                image_counts.append(len(imgs))
                all_images.extend(imgs)
            
            if all_images:
                batch = self.processor(
                    text=batch_texts,
                    images=all_images if all_images else None,
                    return_tensors="pt",
                    padding=self.padding,
                    truncation=True,
                    max_length=self.max_length,
                )
            else:
                # Text-only batch
                batch = self.processor.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=self.padding,
                    truncation=True,
                    max_length=self.max_length,
                )
        except Exception as e:
            logger.error(f"Processor failed: {e}")
            # Fallback to text-only
            batch = self.processor.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=self.padding,
                truncation=True,
                max_length=self.max_length,
            )
        
        # Create labels (same as input_ids, but mask padding)
        labels = batch["input_ids"].clone()
        labels[labels == self.pad_token_id] = -100
        batch["labels"] = labels
        
        # Add refer_boxes as list of tensors (model expects List[Optional[torch.Tensor]])
        if any(b is not None for b in batch_boxes):
            # Convert to list of normalized tensors
            refer_boxes_list = []
            for boxes in batch_boxes:
                if boxes is None:
                    refer_boxes_list.append(None)
                else:
                    # Convert to tensor if needed
                    if isinstance(boxes, list):
                        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
                    else:
                        boxes_tensor = boxes.float()
                    
                    # Normalize to 0-1 if in 0-1000 format
                    if boxes_tensor.max() > 1.0:
                        boxes_tensor = boxes_tensor / 1000.0
                    
                    refer_boxes_list.append(boxes_tensor)
            
            batch["refer_boxes"] = refer_boxes_list
        
        return batch


class HOISimpleCollator:
    """
    Simplified collator for when images are pre-processed or for text-only training.
    
    Use this when:
    - Images are already processed in the dataset
    - You want to skip image loading for faster debugging
    """
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        padding: str = "longest",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate text-only examples."""
        texts = []
        
        for ex in examples:
            messages = ex.get("messages", [])
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
        )
        
        labels = batch["input_ids"].clone()
        labels[labels == self.pad_token_id] = -100
        batch["labels"] = labels
        
        return batch
