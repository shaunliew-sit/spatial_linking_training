"""
Spatial Linking Interaction Model for Qwen3-VL.

Extends Qwen3-VL with spatial linking for 3-region representation:
- Person box
- Object box  
- Interaction box (union of person + object)

The spatial linking module enhances <|box_end|> tokens with cross-attention
to image patches within their corresponding bounding boxes.

Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modular_qwen3_vl.py
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

# Import Qwen3VL classes (correct for Qwen3-VL-8B-Instruct)
try:
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLConfig
except ImportError:
    # Fallback for older transformers versions
    from transformers import Qwen2VLForConditionalGeneration as Qwen3VLForConditionalGeneration
    from transformers import Qwen2VLConfig as Qwen3VLConfig
    import warnings
    warnings.warn(
        "Qwen3VL classes not found. Using Qwen2VL as fallback. "
        "Please upgrade transformers: pip install transformers>=4.45.0"
    )

from .spatial_linking import SpatialLinkingModule


def compute_union_box(boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute the union (bounding box) of multiple boxes.
    
    Args:
        boxes: Tensor of shape [N, 4] with boxes in [x1, y1, x2, y2] format
        
    Returns:
        Union box of shape [1, 4]
    """
    if boxes.shape[0] == 0:
        return torch.zeros(1, 4, dtype=boxes.dtype, device=boxes.device)
    
    x1 = boxes[:, 0].min()
    y1 = boxes[:, 1].min()
    x2 = boxes[:, 2].max()
    y2 = boxes[:, 3].max()
    
    return torch.tensor([[x1, y1, x2, y2]], dtype=boxes.dtype, device=boxes.device)


def compute_interaction_box(
    person_box: Union[torch.Tensor, List[float]], 
    object_box: Union[torch.Tensor, List[float]]
) -> torch.Tensor:
    """
    Compute the interaction box as the union of person and object boxes.
    
    The interaction box captures the spatial context where the interaction occurs,
    encompassing both entities and the space between them.
    
    Args:
        person_box: Person bounding box [x1, y1, x2, y2]
        object_box: Object bounding box [x1, y1, x2, y2]
        
    Returns:
        Interaction box [x1, y1, x2, y2] as tensor of shape [4]
    """
    # Convert to tensor if needed
    if isinstance(person_box, list):
        person_box = torch.tensor(person_box)
    if isinstance(object_box, list):
        object_box = torch.tensor(object_box)
    
    # Ensure 2D tensors
    if person_box.dim() == 1:
        person_box = person_box.unsqueeze(0)
    if object_box.dim() == 1:
        object_box = object_box.unsqueeze(0)
    
    # Stack and compute union
    boxes = torch.cat([person_box, object_box], dim=0)
    union = compute_union_box(boxes)
    
    return union.squeeze(0)  # Return [4] tensor


class SpatialLinkingInteractionModel(Qwen3VLForConditionalGeneration):
    """
    Extends Qwen3-VL with spatial linking for 3-region representation.
    
    This model enhances the base Qwen3-VL with:
    1. Spatial linking module that links <|box_end|> tokens to image patches
    2. Support for 3-region representation (person, object, interaction)
    3. Automatic interaction box computation from person + object boxes
    
    The spatial linking module is the ONLY trainable component during fine-tuning,
    while the vision encoder and LLM remain frozen (optionally with LoRA).
    
    Key Differences from Qwen2VL:
    - Config uses nested text_config and vision_config
    - Vision encoder returns (image_embeds, deepstack_image_embeds)
    - DeepStack: Visual features injected into early LLM layers
    
    Args:
        config: Qwen3VLConfig configuration
    """
    
    # Qwen3-VL special token IDs
    BOX_START_TOKEN_ID = 151648  # <|box_start|>
    BOX_END_TOKEN_ID = 151649    # <|box_end|>
    IMAGE_TOKEN_ID = 151655      # <|image_pad|>
    
    def __init__(self, config: Qwen3VLConfig):
        super().__init__(config)
        
        # Get hidden size from config (Qwen3VL uses nested config)
        # Try text_config first (Qwen3VL), fallback to direct access (Qwen2VL)
        if hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
            hidden_size = config.text_config.hidden_size
        else:
            hidden_size = getattr(config, 'hidden_size', 3584)
        
        # Create Spatial Linking Module
        self.spatial_linking = SpatialLinkingModule(
            hidden_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            use_refine_mlp=True
        )
        
        # Token IDs for box tokens
        self.box_end_token_id = self.BOX_END_TOKEN_ID
        self.box_start_token_id = self.BOX_START_TOKEN_ID
        self.image_token_id = self.IMAGE_TOKEN_ID
        
        # Flag for using interaction token
        self.use_interaction_token = True
        
        # Store attention weights for visualization
        self._spatial_attention_weights = None
    
    def set_box_token_ids(self, tokenizer):
        """Set box token IDs from tokenizer (for safety)."""
        try:
            self.box_end_token_id = tokenizer.convert_tokens_to_ids("<|box_end|>")
            self.box_start_token_id = tokenizer.convert_tokens_to_ids("<|box_start|>")
            # Also try to get image token ID
            img_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
            if img_token_id is not None:
                self.image_token_id = img_token_id
        except Exception:
            # Fall back to default IDs
            pass
    
    def augment_boxes_with_interaction(
        self,
        refer_boxes: List[Optional[torch.Tensor]],
    ) -> List[Optional[torch.Tensor]]:
        """
        Augment refer_boxes with interaction boxes.
        
        For each pair of boxes (person, object), compute and append the
        interaction box (union).
        
        Args:
            refer_boxes: List of [N, 4] tensors with person/object boxes
            
        Returns:
            Augmented list with interaction boxes added
        """
        augmented = []
        
        for boxes in refer_boxes:
            if boxes is None or boxes.shape[0] < 2:
                augmented.append(boxes)
                continue
            
            # Compute interaction boxes for each pair
            new_boxes_list = []
            for i in range(0, boxes.shape[0], 2):
                if i + 1 < boxes.shape[0]:
                    person_box = boxes[i]
                    object_box = boxes[i + 1]
                    interaction_box = compute_interaction_box(person_box, object_box)
                    
                    # Add person, object, interaction
                    new_boxes_list.extend([person_box, object_box, interaction_box])
                else:
                    # Odd number of boxes - just add the last one
                    new_boxes_list.append(boxes[i])
            
            if new_boxes_list:
                augmented.append(torch.stack(new_boxes_list))
            else:
                augmented.append(boxes)
        
        return augmented
    
    def _get_image_mask(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create a mask indicating where image patches are in the sequence.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            
        Returns:
            Boolean mask [batch, seq_len]
        """
        return input_ids == self.image_token_id
    
    def get_spatial_attention_weights(self) -> Optional[List[torch.Tensor]]:
        """
        Get the attention weights from the last spatial linking forward pass.
        
        Returns:
            List of attention weight tensors, one per box, or None if not available.
            Each tensor has shape [num_heads, 1, num_patches].
        """
        return self._spatial_attention_weights
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        refer_boxes: Optional[List[Optional[torch.Tensor]]] = None,
        output_spatial_attentions: bool = False,
        **kwargs,
    ):
        """
        Forward pass with spatial linking.
        
        This extends the base Qwen3VL forward pass by:
        1. Computing input embeddings
        2. Merging image embeddings into the sequence
        3. Applying spatial linking to enhance <|box_end|> tokens
        4. Running the LLM forward pass
        
        Args:
            refer_boxes: List of [N, 4] tensors with bounding boxes per batch item.
                        Boxes should be normalized to [0, 1] range or [0, 1000] range.
                        If use_interaction_token=True, interaction boxes are auto-computed.
            output_spatial_attentions: If True, store spatial linking attention weights
                                       for later retrieval via get_spatial_attention_weights()
            ... (other args same as Qwen3VLForConditionalGeneration)
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Clear previous attention weights
        self._spatial_attention_weights = None

        # If we have inputs_embeds, skip embedding creation
        if inputs_embeds is None and input_ids is not None:
            # Get text embeddings
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            
            # Process and merge image embeddings if provided
            if pixel_values is not None and image_grid_thw is not None:
                # Get image features from vision encoder
                # Qwen3VL returns (image_embeds, deepstack_image_embeds)
                vision_output = self.model.get_image_features(pixel_values, image_grid_thw)
                
                # Handle both Qwen3VL (tuple) and Qwen2VL (single tensor) outputs
                if isinstance(vision_output, tuple):
                    image_embeds_list, deepstack_embeds = vision_output
                    # Concatenate image embeddings if it's a list
                    if isinstance(image_embeds_list, (list, tuple)):
                        image_embeds = torch.cat(image_embeds_list, dim=0)
                    else:
                        image_embeds = image_embeds_list
                else:
                    image_embeds = vision_output
                
                # Merge image embeddings into sequence
                image_mask = self._get_image_mask(input_ids)
                
                if image_mask.any():
                    # Flatten image embeddings if needed
                    if image_embeds.dim() == 3:
                        image_embeds_flat = image_embeds.reshape(-1, image_embeds.shape[-1])
                    else:
                        image_embeds_flat = image_embeds
                    
                    # Replace placeholders with image embeddings
                    inputs_embeds = inputs_embeds.clone()
                    inputs_embeds[image_mask] = image_embeds_flat.to(inputs_embeds.dtype)
                
                # Apply spatial linking if refer_boxes provided
                if refer_boxes is not None:
                    # Normalize boxes if needed (assume 0-1000 format, convert to 0-1)
                    norm_refer_boxes = []
                    for boxes in refer_boxes:
                        if boxes is not None:
                            boxes_tensor = boxes.to(inputs_embeds.device)
                            # Check if normalization needed (values > 1 suggest 0-1000 format)
                            if boxes_tensor.max() > 1.0:
                                boxes_tensor = boxes_tensor.float() / 1000.0
                            norm_refer_boxes.append(boxes_tensor)
                        else:
                            norm_refer_boxes.append(None)
                    
                    # Add interaction boxes if enabled and boxes don't already include them
                    # (boxes from collator already have 3 boxes: person, object, interaction)
                    if self.use_interaction_token:
                        # Only augment if boxes come in pairs (no interaction box yet)
                        should_augment = all(
                            b is None or b.shape[0] % 3 != 0 
                            for b in norm_refer_boxes if b is not None
                        )
                        if should_augment:
                            norm_refer_boxes = self.augment_boxes_with_interaction(norm_refer_boxes)
                    
                    # Apply spatial linking
                    spatial_output = self.spatial_linking(
                        inputs_embeds=inputs_embeds,
                        input_ids=input_ids,
                        image_mask=image_mask,
                        refer_boxes=norm_refer_boxes,
                        image_grid_thw=image_grid_thw,
                        box_end_token_id=self.box_end_token_id,
                        output_attentions=output_spatial_attentions,
                    )
                    
                    # Handle output based on whether attention weights are returned
                    if output_spatial_attentions and isinstance(spatial_output, tuple):
                        inputs_embeds, self._spatial_attention_weights = spatial_output
                    else:
                        inputs_embeds = spatial_output
        
        # Call parent forward with embeddings
        return super().forward(
            input_ids=None,  # Use inputs_embeds instead
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=None,  # Already processed
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            **kwargs,
        )
    
    def freeze_base_model(self, freeze_vision: bool = True, freeze_llm: bool = True):
        """
        Freeze base model components, keeping only spatial linking trainable.
        
        Args:
            freeze_vision: Whether to freeze vision encoder
            freeze_llm: Whether to freeze LLM (if using LoRA, set to False)
        """
        # Freeze vision encoder (Qwen3VL uses 'visual' or 'model.visual')
        if freeze_vision:
            if hasattr(self, 'visual'):
                for param in self.visual.parameters():
                    param.requires_grad = False
            elif hasattr(self.model, 'visual'):
                for param in self.model.visual.parameters():
                    param.requires_grad = False
        
        # Freeze LLM (Qwen3VL uses 'model.language_model' or 'model')
        if freeze_llm:
            if hasattr(self.model, 'language_model'):
                for param in self.model.language_model.parameters():
                    param.requires_grad = False
            elif hasattr(self, 'model'):
                for param in self.model.parameters():
                    param.requires_grad = False
        
        # Ensure spatial linking is trainable
        for param in self.spatial_linking.parameters():
            param.requires_grad = True
    
    def get_trainable_params_info(self) -> dict:
        """Get information about trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        spatial_linking_params = self.spatial_linking.get_trainable_params()
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "spatial_linking_params": spatial_linking_params,
            "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
        }
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load pretrained model and initialize spatial linking module."""
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return model
