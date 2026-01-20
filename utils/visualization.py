"""
Attention Visualization Utilities for Spatial Linking.

Provides functions to visualize cross-attention weights from the SpatialLinkingModule,
showing how <|box_end|> tokens attend to image patches within bounding boxes.

Features:
- Attention heatmap overlay on original images
- Multi-head attention visualization
- Comparison visualization for person/object/interaction boxes
- Batch visualization support
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import warnings

# Try to import visualization dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not installed. Visualization functions will not work.")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    warnings.warn("PIL not installed. Image loading will not work.")


def create_attention_heatmap(
    attention_weights: torch.Tensor,
    grid_thw: torch.Tensor,
    patch_indices: torch.Tensor,
    image_start_idx: int = 0,
    head_reduction: str = 'mean',
) -> np.ndarray:
    """
    Convert attention weights to a 2D heatmap matching the image grid.
    
    Args:
        attention_weights: Attention weights [1, num_heads, 1, num_patches]
        grid_thw: Grid dimensions (T, H, W)
        patch_indices: Indices of patches that were attended to
        image_start_idx: Start index of image patches in sequence
        head_reduction: How to reduce across heads ('mean', 'max', or int for specific head)
        
    Returns:
        2D numpy array heatmap of shape [H, W]
    """
    T, H, W = grid_thw.tolist() if torch.is_tensor(grid_thw) else grid_thw
    
    # Handle different attention tensor shapes
    attn = attention_weights
    
    if attn.dim() == 4:
        # [1, num_heads, 1, num_patches] -> [num_heads, num_patches]
        attn = attn.squeeze(0).squeeze(1)
    elif attn.dim() == 3:
        # [num_heads, 1, num_patches] or [1, num_heads, num_patches]
        if attn.shape[1] == 1:
            attn = attn.squeeze(1)  # [num_heads, num_patches]
        else:
            attn = attn.squeeze(0)  # [num_heads, num_patches]
    # Now attn should be [num_heads, num_patches] or [num_patches]
    
    # Reduce across heads if needed
    if attn.dim() == 2:
        if head_reduction == 'mean':
            attn = attn.mean(dim=0)  # [num_patches]
        elif head_reduction == 'max':
            attn = attn.max(dim=0).values  # [num_patches]
        elif isinstance(head_reduction, int):
            attn = attn[head_reduction]  # [num_patches]
        else:
            attn = attn.mean(dim=0)
    # attn is now [num_patches]
    
    # Convert to numpy (handle bfloat16 which doesn't support direct numpy conversion)
    if torch.is_tensor(attn):
        attn = attn.float().cpu().numpy()
    patch_indices = patch_indices.cpu().numpy() if torch.is_tensor(patch_indices) else patch_indices
    
    # Create full heatmap
    heatmap = np.zeros((int(H), int(W)), dtype=np.float32)
    
    for i, patch_idx in enumerate(patch_indices):
        # Convert linear index back to 2D position
        relative_idx = patch_idx - image_start_idx
        h = (relative_idx // W) % H
        w = relative_idx % W
        if 0 <= h < H and 0 <= w < W and i < len(attn):
            heatmap[int(h), int(w)] = attn[i]
    
    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap


def visualize_spatial_attention(
    image: Union[str, Path, 'Image.Image', np.ndarray],
    attention_info: Dict,
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    alpha: float = 0.6,
    cmap: str = 'hot',
    show_bbox: bool = True,
    bbox_color: str = 'lime',
    bbox_linewidth: int = 2,
) -> Optional['plt.Figure']:
    """
    Visualize attention weights overlaid on the original image.
    
    Args:
        image: Input image (path, PIL Image, or numpy array)
        attention_info: Dict containing:
            - 'attention_weights': [1, num_heads, 1, num_patches]
            - 'bbox': [4] normalized bounding box
            - 'patch_indices': indices of patches
            - 'box_type': 'person', 'object', or 'interaction'
            - 'grid_thw': grid dimensions
        output_path: If provided, save figure to this path
        title: Optional title for the plot
        figsize: Figure size
        alpha: Transparency of heatmap overlay
        cmap: Colormap for heatmap
        show_bbox: Whether to draw the bounding box
        bbox_color: Color of bounding box
        bbox_linewidth: Width of bounding box line
        
    Returns:
        matplotlib Figure object if output_path is None, else None
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")
    
    # Load image if needed
    if isinstance(image, (str, Path)):
        if not HAS_PIL:
            raise ImportError("PIL is required for loading images. Install with: pip install Pillow")
        img = Image.open(image).convert('RGB')
        img_array = np.array(img)
    elif HAS_PIL and isinstance(image, Image.Image):
        img_array = np.array(image.convert('RGB'))
    else:
        img_array = image
    
    H_img, W_img = img_array.shape[:2]
    
    # Create heatmap
    heatmap = create_attention_heatmap(
        attention_weights=attention_info['attention_weights'],
        grid_thw=attention_info['grid_thw'],
        patch_indices=attention_info['patch_indices'],
        image_start_idx=0,
    )
    
    # Resize heatmap to image size
    from scipy.ndimage import zoom as scipy_zoom
    zoom_h = H_img / heatmap.shape[0]
    zoom_w = W_img / heatmap.shape[1]
    heatmap_resized = scipy_zoom(heatmap, (zoom_h, zoom_w), order=1)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Show image
    ax.imshow(img_array)
    
    # Overlay heatmap
    im = ax.imshow(heatmap_resized, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    
    # Draw bounding box
    if show_bbox and 'bbox' in attention_info:
        bbox = attention_info['bbox']
        if torch.is_tensor(bbox):
            bbox = bbox.numpy()
        x1, y1, x2, y2 = bbox * np.array([W_img, H_img, W_img, H_img])
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=bbox_linewidth,
            edgecolor=bbox_color,
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add box type label
        box_type = attention_info.get('box_type', 'unknown')
        ax.text(
            x1, y1 - 5,
            box_type.capitalize(),
            color=bbox_color,
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5)
        )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', fontsize=10)
    
    # Set title
    if title:
        ax.set_title(title, fontsize=14)
    else:
        box_type = attention_info.get('box_type', 'region')
        ax.set_title(f'Spatial Linking Attention - {box_type.capitalize()} Box', fontsize=14)
    
    ax.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig


def visualize_multi_region_attention(
    image: Union[str, Path, 'Image.Image', np.ndarray],
    attention_info_list: List[Dict],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (16, 5),
    alpha: float = 0.6,
    cmap: str = 'hot',
) -> Optional['plt.Figure']:
    """
    Visualize attention for multiple regions (person, object, interaction) side by side.
    
    Args:
        image: Input image
        attention_info_list: List of attention info dicts for each region
        output_path: If provided, save figure to this path
        figsize: Figure size
        alpha: Transparency of heatmap overlay
        cmap: Colormap for heatmap
        
    Returns:
        matplotlib Figure object if output_path is None, else None
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")
    
    # Handle empty attention list
    if not attention_info_list or len(attention_info_list) == 0:
        warnings.warn("No attention info provided for visualization")
        return None
    
    # Load image if needed
    if isinstance(image, (str, Path)):
        if not HAS_PIL:
            raise ImportError("PIL is required for loading images")
        img = Image.open(image).convert('RGB')
        img_array = np.array(img)
    elif HAS_PIL and isinstance(image, Image.Image):
        img_array = np.array(image.convert('RGB'))
    else:
        img_array = image
    
    H_img, W_img = img_array.shape[:2]
    
    # Create figure with subplots
    n_regions = len(attention_info_list)
    fig, axes = plt.subplots(1, n_regions + 1, figsize=figsize)
    
    # Ensure axes is always iterable
    if n_regions == 0:
        plt.close(fig)
        return None
    
    # Box type colors
    box_colors = {
        'person': 'cyan',
        'object': 'yellow',
        'interaction': 'lime'
    }
    
    # First subplot: original image with all boxes
    axes[0].imshow(img_array)
    axes[0].set_title('Original Image with Boxes', fontsize=12)
    
    for info in attention_info_list:
        if 'bbox' in info:
            bbox = info['bbox']
            if torch.is_tensor(bbox):
                bbox = bbox.numpy()
            x1, y1, x2, y2 = bbox * np.array([W_img, H_img, W_img, H_img])
            box_type = info.get('box_type', 'unknown')
            color = box_colors.get(box_type, 'white')
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor='none',
                linestyle='--' if box_type == 'interaction' else '-'
            )
            axes[0].add_patch(rect)
    axes[0].axis('off')
    
    # Subsequent subplots: each region's attention
    from scipy.ndimage import zoom as scipy_zoom
    
    for i, info in enumerate(attention_info_list):
        ax = axes[i + 1]
        
        # Create heatmap
        heatmap = create_attention_heatmap(
            attention_weights=info['attention_weights'],
            grid_thw=info['grid_thw'],
            patch_indices=info['patch_indices'],
        )
        
        # Resize heatmap
        zoom_h = H_img / heatmap.shape[0]
        zoom_w = W_img / heatmap.shape[1]
        heatmap_resized = scipy_zoom(heatmap, (zoom_h, zoom_w), order=1)
        
        # Show image and overlay
        ax.imshow(img_array)
        im = ax.imshow(heatmap_resized, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
        
        # Draw bounding box
        if 'bbox' in info:
            bbox = info['bbox']
            if torch.is_tensor(bbox):
                bbox = bbox.numpy()
            x1, y1, x2, y2 = bbox * np.array([W_img, H_img, W_img, H_img])
            box_type = info.get('box_type', 'unknown')
            color = box_colors.get(box_type, 'white')
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.set_title(f'{box_type.capitalize()} Box Attention', fontsize=12)
        
        ax.axis('off')
    
    # Add shared colorbar
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label='Attention Weight')
    
    plt.suptitle('Spatial Linking Cross-Attention Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig


def visualize_attention_heads(
    image: Union[str, Path, 'Image.Image', np.ndarray],
    attention_info: Dict,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (16, 8),
    alpha: float = 0.7,
    cmap: str = 'hot',
) -> Optional['plt.Figure']:
    """
    Visualize attention patterns for each attention head separately.
    
    Args:
        image: Input image
        attention_info: Attention info dict
        output_path: If provided, save figure to this path
        figsize: Figure size
        alpha: Transparency of heatmap overlay
        cmap: Colormap for heatmap
        
    Returns:
        matplotlib Figure object if output_path is None, else None
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")
    
    # Load image
    if isinstance(image, (str, Path)):
        if not HAS_PIL:
            raise ImportError("PIL is required for loading images")
        img = Image.open(image).convert('RGB')
        img_array = np.array(img)
    elif HAS_PIL and isinstance(image, Image.Image):
        img_array = np.array(image.convert('RGB'))
    else:
        img_array = image
    
    H_img, W_img = img_array.shape[:2]
    
    # Get number of heads
    attn_weights = attention_info['attention_weights']
    num_heads = attn_weights.shape[1]
    
    # Calculate grid layout
    n_cols = 4
    n_rows = (num_heads + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    from scipy.ndimage import zoom as scipy_zoom
    
    for head_idx in range(num_heads):
        ax = axes[head_idx]
        
        # Create heatmap for this head
        heatmap = create_attention_heatmap(
            attention_weights=attn_weights,
            grid_thw=attention_info['grid_thw'],
            patch_indices=attention_info['patch_indices'],
            head_reduction=head_idx,
        )
        
        # Resize heatmap
        zoom_h = H_img / heatmap.shape[0]
        zoom_w = W_img / heatmap.shape[1]
        heatmap_resized = scipy_zoom(heatmap, (zoom_h, zoom_w), order=1)
        
        # Show image and overlay
        ax.imshow(img_array)
        ax.imshow(heatmap_resized, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
        ax.set_title(f'Head {head_idx + 1}', fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_heads, len(axes)):
        axes[idx].axis('off')
    
    box_type = attention_info.get('box_type', 'region')
    plt.suptitle(f'Per-Head Attention - {box_type.capitalize()} Box', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig


def save_attention_batch(
    images: List[Union[str, Path]],
    batch_attention_info: List[List[Dict]],
    output_dir: Union[str, Path],
    prefix: str = 'attention',
) -> List[Path]:
    """
    Save attention visualizations for a batch of samples.
    
    Args:
        images: List of image paths
        batch_attention_info: List of attention info lists (one per sample)
        output_dir: Directory to save visualizations
        prefix: Filename prefix
        
    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for i, (image_path, attention_list) in enumerate(zip(images, batch_attention_info)):
        if len(attention_list) == 0:
            continue
        
        # Multi-region visualization
        output_path = output_dir / f'{prefix}_sample_{i:04d}_multi.png'
        visualize_multi_region_attention(
            image=image_path,
            attention_info_list=attention_list,
            output_path=output_path,
        )
        saved_paths.append(output_path)
        
        # Individual region visualizations
        for j, info in enumerate(attention_list):
            box_type = info.get('box_type', f'region_{j}')
            output_path = output_dir / f'{prefix}_sample_{i:04d}_{box_type}.png'
            visualize_spatial_attention(
                image=image_path,
                attention_info=info,
                output_path=output_path,
            )
            saved_paths.append(output_path)
    
    return saved_paths


def log_attention_to_wandb(
    image: Union[str, Path, 'Image.Image', np.ndarray],
    attention_info_list: List[Dict],
    step: int,
    prefix: str = 'spatial_attention',
):
    """
    Log attention visualizations to Weights & Biases.
    
    Args:
        image: Input image
        attention_info_list: List of attention info dicts
        step: Training step
        prefix: Logging prefix
    """
    try:
        import wandb
    except ImportError:
        warnings.warn("wandb not installed. Skipping attention logging.")
        return
    
    if not HAS_MATPLOTLIB:
        warnings.warn("matplotlib not installed. Skipping attention logging.")
        return
    
    # Create multi-region visualization
    fig = visualize_multi_region_attention(
        image=image,
        attention_info_list=attention_info_list,
    )
    
    if fig is not None:
        wandb.log({f'{prefix}/multi_region': wandb.Image(fig)}, step=step)
        plt.close(fig)
    
    # Log individual regions
    for info in attention_info_list:
        box_type = info.get('box_type', 'region')
        fig = visualize_spatial_attention(
            image=image,
            attention_info=info,
        )
        if fig is not None:
            wandb.log({f'{prefix}/{box_type}': wandb.Image(fig)}, step=step)
            plt.close(fig)
