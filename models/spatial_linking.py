"""
Spatial Linking Module for Qwen3-VL.

Links <|box_end|> tokens to original image patch embeddings within bounding boxes
via cross-attention. The result is ADDED (not replaced) to preserve token semantics.

Key Features:
- Preserves spatial position information (patches have position embeddings)
- Preserves token semantics via residual addition
- Trainable params: ~10M (cross-attn + MLP)
- Supports attention weight output for visualization
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union


class SpatialLinkingModule(nn.Module):
    """
    Links <|box_end|> tokens to original image patch embeddings within bounding boxes.
    
    Cross-attention where:
    - Query: <|box_end|> token embedding
    - Key/Value: Original image patch embeddings that fall within the bbox
    
    Result is ADDED to (not replacing) the <|box_end|> embedding.
    
    Args:
        hidden_dim: Must match LLM hidden size (3584 for Qwen3-VL-8B)
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_refine_mlp: Whether to use MLP refinement after cross-attention
    """
    
    def __init__(
        self, 
        hidden_dim: int = 3584,  # Qwen3-VL-8B hidden_size
        num_heads: int = 8, 
        dropout: float = 0.1,
        use_refine_mlp: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_refine_mlp = use_refine_mlp
        
        # Cross-attention: Query from <|box_end|>, Key/Value from image patches
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization for residual connection
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Optional MLP to refine the linked representation
        if use_refine_mlp:
            self.refine_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout)
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        nn.init.xavier_uniform_(self.cross_attn.in_proj_weight)
        nn.init.xavier_uniform_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.in_proj_bias)
        nn.init.zeros_(self.cross_attn.out_proj.bias)
        
        if self.use_refine_mlp:
            for module in self.refine_mlp:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
    
    def get_patches_in_bbox(
        self, 
        bbox: torch.Tensor,           # [4] normalized 0-1: [x1, y1, x2, y2]
        image_grid_thw: torch.Tensor, # [3] = (T, H, W) - grid dimensions
        image_start_idx: int,         # Where image patches start in sequence
        device: torch.device,
    ) -> torch.Tensor:
        """
        Returns indices of patches that fall within the bounding box.
        
        Patches are laid out in row-major order:
        index = t * H * W + h * W + w + image_start_idx
        
        Args:
            bbox: Normalized bounding box [x1, y1, x2, y2] in range [0, 1]
            image_grid_thw: Grid dimensions (T, H, W)
            image_start_idx: Start index of image patches in sequence
            device: Target device
            
        Returns:
            Tensor of patch indices within the bbox
        """
        T, H, W = image_grid_thw.tolist()
        x1, y1, x2, y2 = bbox.tolist()
        
        # Convert normalized coords to patch grid indices
        patch_x1 = int(x1 * W)
        patch_x2 = int(x2 * W) 
        patch_y1 = int(y1 * H)
        patch_y2 = int(y2 * H)
        
        # Clamp to valid range
        patch_x1 = max(0, min(patch_x1, W - 1))
        patch_x2 = max(patch_x1 + 1, min(patch_x2, W))
        patch_y1 = max(0, min(patch_y1, H - 1))
        patch_y2 = max(patch_y1 + 1, min(patch_y2, H))
        
        # Generate indices for patches in the bbox (row-major order)
        indices = []
        for t in range(int(T)):
            for h in range(patch_y1, patch_y2):
                for w in range(patch_x1, patch_x2):
                    patch_idx = t * H * W + h * W + w
                    indices.append(image_start_idx + patch_idx)
        
        # Fallback for very small boxes - use center patch
        if len(indices) == 0:
            center_h = (patch_y1 + patch_y2) // 2
            center_w = (patch_x1 + patch_x2) // 2
            for t in range(int(T)):
                indices.append(image_start_idx + t * H * W + center_h * W + center_w)
        
        return torch.tensor(indices, dtype=torch.long, device=device)
    
    def forward(
        self,
        inputs_embeds: torch.Tensor,      # [batch, seq_len, hidden_dim]
        input_ids: torch.Tensor,          # [batch, seq_len]
        image_mask: torch.Tensor,         # [batch, seq_len] - where image patches are
        refer_boxes: List[Optional[torch.Tensor]],  # List of [N, 4] normalized bboxes per batch
        image_grid_thw: torch.Tensor,     # [num_images, 3] or [3] for single image
        box_end_token_id: int,
        output_attentions: bool = False,  # NEW: Whether to return attention weights
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[dict]]]:
        """
        Apply spatial linking to enhance <|box_end|> tokens with region features.
        
        For each <|box_end|> token:
        1. Find its corresponding bbox
        2. Identify image patches within that bbox
        3. Cross-attend from <|box_end|> to those patches
        4. ADD the result to the <|box_end|> embedding (preserves semantics)
        
        Args:
            inputs_embeds: Input embeddings [batch, seq_len, hidden_dim]
            input_ids: Token IDs [batch, seq_len]
            image_mask: Boolean mask for image patch positions [batch, seq_len]
            refer_boxes: List of bounding boxes per batch item, each [N, 4] normalized 0-1
            image_grid_thw: Image grid dimensions [num_images, 3] or [3]
            box_end_token_id: Token ID for <|box_end|>
            output_attentions: If True, return attention weights for visualization
            
        Returns:
            If output_attentions=False:
                Enhanced embeddings with spatial linking applied to <|box_end|> tokens
            If output_attentions=True:
                Tuple of (enhanced_embeddings, attention_info_list)
                where attention_info_list contains dicts with:
                - 'attention_weights': [num_heads, 1, num_patches]
                - 'bbox': [4] the bounding box
                - 'patch_indices': indices of patches attended to
                - 'box_type': 'person', 'object', or 'interaction'
        """
        batch_size, seq_len, hidden_dim = inputs_embeds.shape
        outputs = inputs_embeds.clone()
        
        # Store attention info if requested
        all_attention_info = [] if output_attentions else None
        
        # Handle single image grid dimensions
        if image_grid_thw.dim() == 1:
            image_grid_thw = image_grid_thw.unsqueeze(0)
        
        for b in range(batch_size):
            batch_attention_info = []
            
            # Find <|box_end|> positions
            box_end_positions = (input_ids[b] == box_end_token_id).nonzero(as_tuple=True)[0]
            
            if len(box_end_positions) == 0 or refer_boxes[b] is None:
                if output_attentions:
                    all_attention_info.append(batch_attention_info)
                continue
            
            # Find where image patches start in sequence
            image_positions = image_mask[b].nonzero(as_tuple=True)[0]
            if len(image_positions) == 0:
                if output_attentions:
                    all_attention_info.append(batch_attention_info)
                continue
            image_start_idx = image_positions[0].item()
            
            # Get grid dimensions for this batch item (assume first image)
            grid_thw = image_grid_thw[0] if image_grid_thw.shape[0] > 0 else image_grid_thw
            
            # Process each <|box_end|> with its corresponding bbox
            num_boxes = min(len(box_end_positions), refer_boxes[b].shape[0])
            
            for i in range(num_boxes):
                box_end_pos = box_end_positions[i].item()
                bbox = refer_boxes[b][i]  # [4] normalized 0-1
                
                # Determine box type for visualization
                box_type = self._get_box_type(i)
                
                # Get indices of patches within this bbox
                patch_indices = self.get_patches_in_bbox(
                    bbox, grid_thw, image_start_idx, inputs_embeds.device
                )
                
                # Filter valid indices
                valid_mask = (patch_indices >= 0) & (patch_indices < seq_len)
                patch_indices = patch_indices[valid_mask]
                
                if len(patch_indices) == 0:
                    continue
                
                # Get ORIGINAL patch embeddings (they have position info!)
                patch_embeds = inputs_embeds[b, patch_indices, :]  # [num_patches, hidden_dim]
                
                # Get <|box_end|> embedding
                box_end_embed = inputs_embeds[b, box_end_pos, :].unsqueeze(0)  # [1, hidden_dim]
                
                # Cross-attention: <|box_end|> (Query) → patches (Key/Value)
                linked_embed, attn_weights = self.cross_attn(
                    query=box_end_embed.unsqueeze(0),    # [1, 1, hidden_dim]
                    key=patch_embeds.unsqueeze(0),       # [1, num_patches, hidden_dim]
                    value=patch_embeds.unsqueeze(0),     # [1, num_patches, hidden_dim]
                    need_weights=True,                   # Always compute weights
                    average_attn_weights=False           # Keep per-head weights
                )
                linked_embed = linked_embed.squeeze(0).squeeze(0)  # [hidden_dim]
                
                # Store attention info if requested
                if output_attentions:
                    batch_attention_info.append({
                        'attention_weights': attn_weights.detach().cpu(),  # [1, num_heads, 1, num_patches]
                        'bbox': bbox.detach().cpu(),
                        'patch_indices': patch_indices.detach().cpu(),
                        'box_type': box_type,
                        'box_end_position': box_end_pos,
                        'grid_thw': grid_thw.detach().cpu(),
                    })
                
                # RESIDUAL CONNECTION: ADD (not replace!)
                combined = box_end_embed.squeeze(0) + linked_embed
                combined = self.layer_norm(combined)
                
                # Optional refinement MLP
                if self.use_refine_mlp:
                    combined = combined + self.refine_mlp(combined)
                
                # Update the <|box_end|> embedding
                outputs[b, box_end_pos, :] = combined
            
            if output_attentions:
                all_attention_info.append(batch_attention_info)
        
        if output_attentions:
            return outputs, all_attention_info
        return outputs
    
    def _get_box_type(self, box_index: int) -> str:
        """
        Determine the type of box based on its index.
        
        With interaction tokens enabled, boxes come in groups of 3:
        - Index 0, 3, 6, ... = person
        - Index 1, 4, 7, ... = object
        - Index 2, 5, 8, ... = interaction
        
        Args:
            box_index: Index of the box
            
        Returns:
            Box type string: 'person', 'object', or 'interaction'
        """
        position_in_group = box_index % 3
        if position_in_group == 0:
            return 'person'
        elif position_in_group == 1:
            return 'object'
        else:
            return 'interaction'
    
    def get_trainable_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
