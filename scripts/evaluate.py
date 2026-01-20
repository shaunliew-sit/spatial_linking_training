#!/usr/bin/env python3
"""
Evaluation Script for Spatial Linking HOI Model.

Evaluates trained models on HOI grounding and referring tasks
using metrics from hoi-benchmarks:
- Referring: METEOR, CIDEr, BLEU, ROUGE-L (COCO caption metrics)
- Grounding: AR (Average Recall) at various IoU thresholds

Usage:
    # Evaluate referring task
    python scripts/evaluate.py \
        --model_path outputs/spatial_linking_sft_dgx/run_XXXX \
        --test_file /dataset/benchmarks_simplified/hico_action_referring_test_simplified.json \
        --task_type referring \
        --verbose

    # Evaluate grounding task
    python scripts/evaluate.py \
        --model_path outputs/spatial_linking_sft_dgx/run_XXXX \
        --test_file /dataset/benchmarks_simplified/hico_ground_test_simplified.json \
        --task_type grounding \
        --verbose

    # With attention visualization
    python scripts/evaluate.py \
        --model_path outputs/spatial_linking_sft_dgx/run_XXXX \
        --test_file /dataset/benchmarks_simplified/hico_action_referring_test_simplified.json \
        --task_type referring \
        --visualize_attention \
        --viz_output_dir ./attention_viz
"""

import os
import sys
import json
import re
import argparse
import logging
import base64
import io
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from tqdm import tqdm

import torch
import requests
from transformers import AutoProcessor
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.spatial_model import SpatialLinkingInteractionModel, compute_interaction_box

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# SYSTEM PROMPT (same as training)
# =============================================================================

SYSTEM_PROMPT = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name":"zoom_in","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox_2d). Coordinates use 1000x1000 normalized format.","parameters":{"properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2] in 1000x1000 normalized format, where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"target_image":{"type":"number","description":"The index of the image to zoom in on. Use 1 for the main image."}},"required":["bbox_2d", "target_image"], "type":"object"},"args_format": "Format the arguments as a JSON object."}}
</tools>

For the function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""


# =============================================================================
# ZOOM IN TOOL (Real Tool for Multi-turn Evaluation)
# =============================================================================

class ZoomInTool:
    """
    Real zoom_in tool for HOI evaluation.
    Handles 1000x1000 normalized coordinates matching Qwen3-VL training format.
    Manages trajectory state across multi-turn conversations.
    Tracks zoom transforms for coordinate conversion back to original image space.
    """
    
    def __init__(self, padding: Tuple[float, float] = (0.1, 0.1)):
        """
        Args:
            padding: Fractional padding to add around crops (x_pad, y_pad)
        """
        self.env_cache: Dict[str, Dict] = {}
        self.padding = padding
    
    def _load_env(self, trajectory_id: str) -> Dict:
        """Load or create environment for trajectory."""
        if trajectory_id not in self.env_cache:
            self.env_cache[trajectory_id] = {
                'images': [],
                'turns': 0,
                'zoom_transforms': [],  # List of (x_offset, y_offset, scale_x, scale_y) for each zoom
                'original_size': None,  # (width, height) of original image
            }
        return self.env_cache[trajectory_id]
    
    def init_trajectory(self, trajectory_id: str, image: Image.Image):
        """Initialize a new trajectory with the original image."""
        env = self._load_env(trajectory_id)
        if isinstance(image, Image.Image):
            env['images'] = [image.copy()]
            env['original_size'] = image.size  # (width, height)
        else:
            img = Image.open(image).convert('RGB')
            env['images'] = [img]
            env['original_size'] = img.size
        env['zoom_transforms'] = []  # Reset transforms
    
    def delete_trajectory(self, trajectory_id: str):
        """Cleanup trajectory state."""
        self.env_cache.pop(trajectory_id, None)
    
    def get_cumulative_transform(self, trajectory_id: str) -> Tuple[float, float, float, float]:
        """
        Get cumulative transform from current view to original image coordinates.
        
        Returns:
            (x_offset, y_offset, scale_x, scale_y) where:
            - original_x = (current_x / 1000 * scale_x + x_offset) * 1000
            - original_y = (current_y / 1000 * scale_y + y_offset) * 1000
        """
        env = self._load_env(trajectory_id)
        transforms = env.get('zoom_transforms', [])
        
        if not transforms:
            return (0.0, 0.0, 1.0, 1.0)
        
        # Compose all transforms
        x_off, y_off, scale_x, scale_y = 0.0, 0.0, 1.0, 1.0
        
        for tx, ty, sx, sy in transforms:
            # Apply this transform on top of accumulated transform
            x_off = x_off + tx * scale_x
            y_off = y_off + ty * scale_y
            scale_x = scale_x * sx
            scale_y = scale_y * sy
        
        return (x_off, y_off, scale_x, scale_y)
    
    def transform_box_to_original(self, trajectory_id: str, box: List[float]) -> List[int]:
        """
        Transform a box from current view coordinates to original image coordinates.
        
        Args:
            trajectory_id: Trajectory ID
            box: Box in current view's 1000x1000 coordinates [x1, y1, x2, y2]
            
        Returns:
            Box in original image's 1000x1000 coordinates
        """
        x_off, y_off, scale_x, scale_y = self.get_cumulative_transform(trajectory_id)
        
        x1, y1, x2, y2 = [float(c) / 1000.0 for c in box]
        
        # Transform back to original coordinates
        orig_x1 = (x1 * scale_x + x_off) * 1000
        orig_y1 = (y1 * scale_y + y_off) * 1000
        orig_x2 = (x2 * scale_x + x_off) * 1000
        orig_y2 = (y2 * scale_y + y_off) * 1000
        
        return [int(orig_x1), int(orig_y1), int(orig_x2), int(orig_y2)]
    
    def parse_tool_call(self, response: str) -> Tuple[Optional[Dict], bool]:
        """Parse <tool_call>...</tool_call> from model response."""
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        match = re.search(pattern, response, re.DOTALL)
        if not match:
            return None, False
        try:
            call = json.loads(match.group(1))
            if call.get('name') == 'zoom_in':
                return call, True
            return None, False
        except json.JSONDecodeError:
            return None, False
    
    def execute(
        self,
        trajectory_id: str,
        response: str
    ) -> Tuple[Optional[Image.Image], str, bool]:
        """
        Execute zoom_in based on model response.
        
        Args:
            trajectory_id: Unique conversation ID
            response: Full model response containing <tool_call>
            
        Returns:
            (cropped_image, observation_text, is_valid)
            - cropped_image: PIL Image if successful, None otherwise
            - observation_text: Description of result
            - is_valid: Whether execution succeeded
        """
        env = self._load_env(trajectory_id)
        
        # Parse tool call
        parsed, valid = self.parse_tool_call(response)
        if not valid:
            return None, "No valid zoom_in tool call found.", False
        
        args = parsed.get('arguments', {})
        bbox_2d = args.get('bbox_2d', args.get('bbox', []))
        target_image = args.get('target_image', 1)
        
        # Validate
        try:
            target_image = int(target_image)
        except (ValueError, TypeError):
            target_image = 1
        
        if not isinstance(bbox_2d, list) or len(bbox_2d) != 4:
            return None, "Invalid bbox_2d: expected [x1, y1, x2, y2]", False
        
        if target_image < 1 or target_image > len(env['images']):
            return None, f"Invalid target_image: {target_image}. Have {len(env['images'])} images.", False
        
        # Get source image
        image = env['images'][target_image - 1]
        img_w, img_h = image.size
        
        # Convert 1000x1000 normalized to pixel coordinates
        x1, y1, x2, y2 = [float(c) for c in bbox_2d]
        
        # Handle 1000-scale normalization (training format)
        x1_norm = x1 / 1000.0
        y1_norm = y1 / 1000.0
        x2_norm = x2 / 1000.0
        y2_norm = y2 / 1000.0
        
        # Add padding
        pad_x, pad_y = self.padding
        x1_norm = max(0, x1_norm - pad_x)
        y1_norm = max(0, y1_norm - pad_y)
        x2_norm = min(1, x2_norm + pad_x)
        y2_norm = min(1, y2_norm + pad_y)
        
        # Convert to pixels
        px1 = int(x1_norm * img_w)
        py1 = int(y1_norm * img_h)
        px2 = int(x2_norm * img_w)
        py2 = int(y2_norm * img_h)
        
        # Validate crop region
        if px2 <= px1 or py2 <= py1:
            return None, f"Invalid crop region after conversion: [{px1},{py1},{px2},{py2}]", False
        
        # Crop
        try:
            cropped = image.crop((px1, py1, px2, py2))
            if cropped.mode != 'RGB':
                cropped = cropped.convert('RGB')
            
            # Track the zoom transform for coordinate conversion
            # The transform maps from cropped image coords (0-1) to source image coords (0-1)
            # cropped coord * scale + offset = source coord
            crop_w_px = px2 - px1
            crop_h_px = py2 - py1
            
            # Offset in normalized coords (relative to current view, not original)
            # We need to track cumulative transforms
            x_offset = px1 / img_w  # offset in source image normalized coords
            y_offset = py1 / img_h
            scale_x = crop_w_px / img_w  # scale factor
            scale_y = crop_h_px / img_h
            
            env['zoom_transforms'].append((x_offset, y_offset, scale_x, scale_y))
            
            # Add to trajectory
            env['images'].append(cropped)
            env['turns'] += 1
            
            crop_w, crop_h = cropped.size
            obs = f"Zoomed into region {[int(c) for c in bbox_2d]}. Now viewing {crop_w}x{crop_h} cropped area."
            
            return cropped, obs, True
            
        except Exception as e:
            return None, f"Crop failed: {str(e)}", False
    
    def get_all_images(self, trajectory_id: str) -> List[Image.Image]:
        """Get all images accumulated in trajectory."""
        env = self._load_env(trajectory_id)
        return env.get('images', [])


# =============================================================================
# VLLM CLIENT FUNCTIONS
# =============================================================================

def encode_image_to_base64(image: Image.Image, max_size: int = 1024, quality: int = 85) -> str:
    """
    Encode PIL Image to base64 string for vLLM API.
    
    Args:
        image: PIL Image to encode
        max_size: Maximum dimension (will resize if larger)
        quality: JPEG quality (1-100)
    
    Returns:
        Base64 encoded image string with data URI prefix
    """
    # Resize if too large
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Encode to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return f"data:image/jpeg;base64,{img_b64}"


def format_messages_for_vllm(
    messages: List[Dict], 
    images: List[Image.Image]
) -> List[Dict]:
    """
    Format messages with images for vLLM's OpenAI-compatible API.
    
    Args:
        messages: List of message dicts with role and content
        images: List of PIL Images to embed
    
    Returns:
        Formatted messages ready for vLLM API
    """
    formatted = []
    image_idx = 0
    
    for msg in messages:
        role = msg['role']
        content = msg['content']
        
        if role == 'system':
            formatted.append({'role': 'system', 'content': content})
        
        elif role == 'assistant':
            formatted.append({'role': 'assistant', 'content': content})
        
        elif role == 'user':
            if isinstance(content, list):
                # Content has structured parts (text + images)
                parts = []
                for item in content:
                    if item.get('type') == 'image' and image_idx < len(images):
                        img_b64 = encode_image_to_base64(images[image_idx])
                        parts.append({
                            'type': 'image_url',
                            'image_url': {'url': img_b64}
                        })
                        image_idx += 1
                    elif item.get('type') == 'text':
                        parts.append({'type': 'text', 'text': item['text']})
                formatted.append({'role': 'user', 'content': parts})
            else:
                # Plain text content
                formatted.append({'role': 'user', 'content': content})
    
    return formatted


def call_vllm(
    messages: List[Dict],
    images: List[Image.Image],
    endpoint: str,
    model_name: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    timeout: int = 120,
) -> str:
    """
    Call vLLM server with OpenAI-compatible API.
    
    Args:
        messages: Conversation messages
        images: Images to include
        endpoint: vLLM server endpoint (e.g., http://localhost:8000/v1)
        model_name: Model name registered in vLLM
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling
        timeout: Request timeout in seconds
    
    Returns:
        Generated text response
    
    Raises:
        Exception: If API call fails
    """
    # Format messages with images
    formatted_messages = format_messages_for_vllm(messages, images)
    
    # Build request payload
    payload = {
        'model': model_name,
        'messages': formatted_messages,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'top_p': top_p,
    }
    
    # Make API request
    url = f"{endpoint.rstrip('/')}/chat/completions"
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=timeout
        )
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
        
    except requests.exceptions.Timeout:
        raise Exception(f"vLLM request timed out after {timeout}s")
    except requests.exceptions.RequestException as e:
        raise Exception(f"vLLM API error: {str(e)}")
    except (KeyError, IndexError) as e:
        raise Exception(f"Invalid vLLM response format: {str(e)}")


def check_vllm_health(endpoint: str, model_name: str) -> bool:
    """
    Check if vLLM server is healthy and model is loaded.
    
    Args:
        endpoint: vLLM server endpoint
        model_name: Expected model name
    
    Returns:
        True if server is healthy and model is available
    """
    try:
        url = f"{endpoint.rstrip('/')}/models"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        models = response.json()
        available_models = [m['id'] for m in models.get('data', [])]
        
        if model_name in available_models:
            return True
        else:
            logger.warning(f"Model '{model_name}' not found. Available: {available_models}")
            return False
            
    except Exception as e:
        logger.error(f"vLLM health check failed: {e}")
        return False


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(model_path: str, device: str = "cuda", model_type: str = "auto"):
    """
    Load trained model and processor.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
        model_type: Type of model to load:
            - "auto": Auto-detect based on checkpoint contents
            - "spatial_linking": SpatialLinkingInteractionModel with spatial linking module
            - "base": Standard Qwen3-VL with LoRA adapters (LLaMA Factory style)
    
    Returns:
        Tuple of (model, processor)
    """
    import warnings
    import os
    from peft import PeftModel
    from transformers import Qwen2_5_VLForConditionalGeneration
    
    logger.info(f"Loading model from {model_path}")
    
    # Suppress known deprecation and compatibility warnings
    warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*Unexpected keyword arguments.*LoraConfig.*")
    warnings.filterwarnings("ignore", message=".*Some weights.*were not initialized.*")
    
    # Auto-detect model type
    if model_type == "auto":
        spatial_linking_path = os.path.join(model_path, "spatial_linking.pt")
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        
        if os.path.exists(spatial_linking_path):
            model_type = "spatial_linking"
            logger.info("Auto-detected model type: spatial_linking (found spatial_linking.pt)")
        elif os.path.exists(adapter_config_path):
            model_type = "base"
            logger.info("Auto-detected model type: base (LoRA adapter, no spatial_linking)")
        else:
            # Default to spatial_linking for backwards compatibility
            model_type = "spatial_linking"
            logger.info("Auto-detected model type: spatial_linking (default)")
    
    # Try to load processor from model path, fallback to base model
    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
    except Exception:
        logger.info("Loading processor from base model")
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            trust_remote_code=True,
        )
    
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    if model_type == "spatial_linking":
        # Load SpatialLinkingInteractionModel
        logger.info("Loading SpatialLinkingInteractionModel...")
        model = SpatialLinkingInteractionModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        
        # Load spatial linking weights if available
        spatial_linking_path = os.path.join(model_path, "spatial_linking.pt")
        if os.path.exists(spatial_linking_path):
            logger.info(f"Loading spatial linking weights from {spatial_linking_path}")
            spatial_linking_state = torch.load(spatial_linking_path, map_location="cpu")
            model.spatial_linking.load_state_dict(spatial_linking_state)
            logger.info(f"  Loaded keys: {list(spatial_linking_state.keys())}")
        else:
            logger.warning(f"Spatial linking weights not found at {spatial_linking_path}")
            logger.warning("  The spatial_linking module will use random initialization!")
        
        # Set box token IDs
        model.set_box_token_ids(processor.tokenizer)
        
    else:
        # Load base Qwen3-VL with LoRA adapters (LLaMA Factory style)
        logger.info("Loading base Qwen3-VL with LoRA adapters...")
        
        # Read adapter config to get base model
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen3-VL-8B-Instruct")
        else:
            base_model_name = "Qwen/Qwen3-VL-8B-Instruct"
        
        logger.info(f"  Base model: {base_model_name}")
        
        # Try Qwen3-VL first, fallback to Qwen2.5-VL
        try:
            from transformers import Qwen3VLForConditionalGeneration
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
            )
        except ImportError:
            # Fallback for older transformers versions
            from transformers import AutoModelForCausalLM
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
            )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, model_path)
        logger.info("  LoRA adapter loaded successfully")
    
    model.eval()
    
    return model, processor


def load_test_data(test_file: str) -> List[Dict]:
    """Load test data from JSON file."""
    with open(test_file, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} test samples")
    return data


def get_image_path(file_name: str, image_base_dir: str) -> str:
    """Get full image path."""
    # Try direct path first
    direct_path = Path(image_base_dir) / file_name
    if direct_path.exists():
        return str(direct_path)
    
    # HICO-DET paths
    if "HICO" in file_name or "hico" in str(image_base_dir).lower():
        for subdir in ["train2015", "test2015"]:
            path = Path(image_base_dir) / "hico_20160224_det" / "images" / subdir / file_name
            if path.exists():
                return str(path)
    
    # SWIG paths
    swig_path = Path(image_base_dir) / "swig_hoi" / "images" / file_name
    if swig_path.exists():
        return str(swig_path)
    
    return str(direct_path)


# =============================================================================
# PROMPT CONSTRUCTION (matching training format)
# =============================================================================

def create_referring_prompt(person_box: List[int], object_box: List[int]) -> str:
    """Create referring task prompt matching training format (text-only part)."""
    return (
        f"Question: What action is the person performing with the object?\n"
        f"The person is located at {person_box} and the object is at {object_box}.\n"
        f"Respond with ONLY the action phrase in format: \"{{verb}} {{object}}\" "
        f"(e.g., \"riding bicycle\", \"holding cup\"). Use base verb form, no articles.\n"
        f"Think in the mind first, and then decide whether to call tools one or more times "
        f"OR provide final answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> "
        f"<tool_call>...</tool_call> (if any tools needed) OR <answer>...</answer> (if no tools needed)."
    )


def create_referring_prompt_with_image(person_box: List[int], object_box: List[int], image) -> List[dict]:
    """Create referring task prompt with image as structured content for Qwen3-VL."""
    text_content = create_referring_prompt(person_box, object_box)
    return [
        {"type": "image", "image": image},
        {"type": "text", "text": text_content}
    ]


def create_grounding_prompt(action: str, object_category: str) -> str:
    """Create grounding task prompt matching training format (text-only part)."""
    return (
        f"Question: Locate every person who is {action} {object_category} "
        f"and the {object_category} they interact with.\n"
        f"For each person-object pair, output bbox coordinates in JSON format like: "
        f"{{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"description\"}}. "
        f"Coordinates should be in 1000x1000 normalized format.\n"
        f"Think in the mind first, and then decide whether to call tools one or more times "
        f"OR provide final answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> "
        f"<tool_call>...</tool_call> (if any tools needed) OR <answer>...</answer> (if no tools needed)."
    )


def create_grounding_prompt_with_image(action: str, object_category: str, image) -> List[dict]:
    """Create grounding task prompt with image as structured content for Qwen3-VL."""
    text_content = create_grounding_prompt(action, object_category)
    return [
        {"type": "image", "image": image},
        {"type": "text", "text": text_content}
    ]


def normalize_bbox_to_1000(bbox: List, width: int, height: int) -> List[int]:
    """
    Convert pixel coordinates to 1000x1000 normalized format.
    
    This matches the training data format where all coordinates are
    normalized to a 1000x1000 grid regardless of actual image size.
    
    Args:
        bbox: Bounding box in pixel coordinates [x1, y1, x2, y2]
        width: Image width in pixels
        height: Image height in pixels
    
    Returns:
        Bounding box in 1000x1000 normalized format
    """
    x1, y1, x2, y2 = bbox
    return [
        int(x1 * 1000 / width),
        int(y1 * 1000 / height),
        int(x2 * 1000 / width),
        int(y2 * 1000 / height)
    ]


def execute_mock_tool(tool_name: str, arguments: Dict, image: Image.Image) -> Tuple[str, Image.Image]:
    """
    Execute a mock tool call and return the result.
    
    For zoom_in: crops the image to the specified bbox and returns a description.
    """
    if tool_name == "zoom_in":
        bbox = arguments.get("bbox_2d", arguments.get("bbox", []))
        target = arguments.get("target_image", 1)
        
        if len(bbox) == 4:
            # Convert from 1000x1000 normalized to pixel coordinates
            w, h = image.size
            x1 = int(bbox[0] * w / 1000)
            y1 = int(bbox[1] * h / 1000)
            x2 = int(bbox[2] * w / 1000)
            y2 = int(bbox[3] * h / 1000)
            
            # Ensure valid crop region
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                cropped = image.crop((x1, y1, x2, y2))
                result = f"Zoomed into region {bbox}. Now viewing a {cropped.size[0]}x{cropped.size[1]} cropped area."
                return result, cropped
        
        return "Invalid bbox format. Expected [x1, y1, x2, y2].", image
    
    return f"Unknown tool: {tool_name}", image


def run_agent_loop(
    model,
    processor,
    image: Image.Image,
    initial_prompt: str,
    system_prompt: str,
    device: str,
    max_turns: int = 3,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """
    Run multi-turn agent loop with tool execution.
    
    Matches verl-tool approach: accumulate ALL images and pass them together.
    Each turn's image is embedded in the message content, and all images
    are passed to the processor in order.
    
    Returns:
        Dict with keys: final_response, all_responses, tool_calls, num_turns
    """
    # Accumulate all images across turns
    all_images = [image.copy()]
    current_image = image.copy()
    all_responses = []
    tool_calls = []
    
    # Build conversation messages (will grow each turn)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": initial_prompt}
            ]
        }
    ]
    
    final_response = ""
    
    for turn in range(max_turns):
        # Apply chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process inputs with ALL accumulated images
        inputs = processor(
            text=[text],
            images=all_images,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        
        # Decode response (only new tokens)
        generated = processor.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        all_responses.append(generated)
        
        # Parse response
        components = parse_response_components(generated)
        
        # Check if model requested tool call
        if components['has_tool_call'] and components['tool_calls']:
            # Execute first tool call
            tc = components['tool_calls'][0]
            if isinstance(tc, dict) and 'name' in tc:
                tool_name = tc['name']
                arguments = tc.get('arguments', {})
                
                # Execute the tool (e.g., zoom_in crops the image)
                tool_result, new_image = execute_mock_tool(tool_name, arguments, current_image)
                current_image = new_image
                
                # Add the new image to accumulated images
                all_images.append(new_image)
                
                tool_calls.append({
                    'turn': turn + 1,
                    'name': tool_name,
                    'arguments': arguments,
                    'result': tool_result
                })
                
                # Add assistant response to messages
                messages.append({
                    "role": "assistant",
                    "content": generated
                })
                
                # Add tool result as new user message with zoomed image
                follow_up_prompt = (
                    "Think in the mind first, and then decide whether to call tools one or more times "
                    "OR provide final answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> "
                    "<tool_call>...</tool_call> (if any tools needed) OR <answer>...</answer> (if no tools needed)."
                )
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": new_image},
                        {"type": "text", "text": follow_up_prompt}
                    ]
                })
                
                # Continue to next turn
                continue
        
        # No tool call - this is the final response
        final_response = generated
        break
    
    # If we exhausted turns without final answer, use last response
    if not final_response and all_responses:
        final_response = all_responses[-1]
    
    return {
        'final_response': final_response,
        'all_responses': all_responses,
        'tool_calls': tool_calls,
        'num_turns': len(all_responses),
        'final_answer': parse_response_components(final_response).get('answer', '') if final_response else ''
    }


def run_agent_loop_vllm(
    image: Image.Image,
    initial_prompt: str,
    system_prompt: str,
    vllm_endpoint: str,
    model_name: str,
    zoom_tool: ZoomInTool,
    max_turns: int = 3,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Run multi-turn agent loop with vLLM server and real ZoomInTool.
    
    Args:
        image: Input image
        initial_prompt: Initial user prompt
        system_prompt: System prompt with tool definitions
        vllm_endpoint: vLLM server endpoint (e.g., http://localhost:8000/v1)
        model_name: Model name in vLLM
        zoom_tool: ZoomInTool instance for real tool execution
        max_turns: Maximum conversation turns
        max_tokens: Max tokens per generation
        temperature: Sampling temperature
    
    Returns:
        Dict with keys: final_response, all_responses, tool_calls, num_turns, final_answer
    """
    # Create unique trajectory ID
    trajectory_id = str(uuid.uuid4())
    
    # Initialize tool with original image
    zoom_tool.init_trajectory(trajectory_id, image)
    
    all_responses = []
    tool_calls = []
    final_response = ""
    
    # Build conversation messages
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": initial_prompt}
            ]
        }
    ]
    
    for turn in range(max_turns):
        # Get all accumulated images from tool
        all_images = zoom_tool.get_all_images(trajectory_id)
        
        try:
            # Call vLLM
            generated = call_vllm(
                messages=messages,
                images=all_images,
                endpoint=vllm_endpoint,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(f"vLLM call failed on turn {turn + 1}: {e}")
            break
        
        all_responses.append(generated)
        
        # Parse response
        components = parse_response_components(generated)
        
        # Check if model requested tool call
        if components['has_tool_call'] and components['tool_calls']:
            # Execute real tool
            cropped_image, obs_text, valid = zoom_tool.execute(trajectory_id, generated)
            
            if valid and cropped_image is not None:
                tc = components['tool_calls'][0]
                tool_calls.append({
                    'turn': turn + 1,
                    'name': tc.get('name', 'zoom_in'),
                    'arguments': tc.get('arguments', {}),
                    'result': obs_text,
                    'valid': True
                })
                
                # Add assistant response to messages
                messages.append({
                    "role": "assistant",
                    "content": generated
                })
                
                # Add tool result with zoomed image
                follow_up_prompt = (
                    f"{obs_text}\n"
                    "Think in the mind first, and then decide whether to call tools one or more times "
                    "OR provide final answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> "
                    "<tool_call>...</tool_call> (if any tools needed) OR <answer>...</answer> (if no tools needed)."
                )
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": cropped_image},
                        {"type": "text", "text": follow_up_prompt}
                    ]
                })
                
                # Continue to next turn
                continue
            else:
                # Tool execution failed, treat as final response
                logger.debug(f"Tool execution failed: {obs_text}")
        
        # No tool call or tool failed - this is the final response
        final_response = generated
        break
    
    # Get zoom transform before cleanup (for coordinate conversion)
    zoom_transform = zoom_tool.get_cumulative_transform(trajectory_id)
    had_zoom = len(tool_calls) > 0
    
    # Cleanup trajectory
    zoom_tool.delete_trajectory(trajectory_id)
    
    # If we exhausted turns without final answer, use last response
    if not final_response and all_responses:
        final_response = all_responses[-1]
    
    return {
        'final_response': final_response,
        'all_responses': all_responses,
        'tool_calls': tool_calls,
        'num_turns': len(all_responses),
        'final_answer': parse_response_components(final_response).get('answer', '') if final_response else '',
        'zoom_transform': zoom_transform,  # (x_off, y_off, scale_x, scale_y)
        'had_zoom': had_zoom,
    }


def parse_response_components(generated_text: str) -> Dict[str, Any]:
    """
    Parse all components from generated text following the CoF format.
    
    Expected format from training:
        <think>reasoning...</think>
        <tool_call>{"name": "zoom_in", "arguments": {...}}</tool_call>
        OR
        <answer>final answer</answer>
    
    Returns:
        Dict with keys:
        - 'thinking': str - content of <think> tags
        - 'tool_calls': List[Dict] - parsed tool calls
        - 'answer': str - content of <answer> tags
        - 'has_tool_call': bool - whether tool_call was requested
        - 'raw_text': str - original text
    """
    result = {
        'thinking': '',
        'tool_calls': [],
        'answer': '',
        'has_tool_call': False,
        'raw_text': generated_text
    }
    
    # Extract thinking content (may appear multiple times in multi-turn)
    think_matches = re.findall(r'<think>(.*?)</think>', generated_text, re.DOTALL)
    if think_matches:
        result['thinking'] = ' '.join([t.strip() for t in think_matches])
    
    # Extract tool calls
    tool_call_matches = re.findall(r'<tool_call>(.*?)</tool_call>', generated_text, re.DOTALL)
    if tool_call_matches:
        result['has_tool_call'] = True
        for tc in tool_call_matches:
            try:
                # Parse JSON from tool call
                tc_clean = tc.strip()
                parsed = json.loads(tc_clean)
                result['tool_calls'].append(parsed)
            except json.JSONDecodeError:
                # Store as string if JSON parsing fails
                result['tool_calls'].append({'raw': tc_clean})
    
    # Extract answer content
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', generated_text, re.DOTALL)
    if answer_match:
        result['answer'] = answer_match.group(1).strip()
    
    return result


def parse_answer(generated_text: str) -> str:
    """Parse answer from generated text with <think>...<answer> format."""
    components = parse_response_components(generated_text)
    
    # If we have an explicit answer, return it
    if components['answer']:
        return components['answer']
    
    # If model requested tool_call, try to extract action hint from thinking
    if components['has_tool_call']:
        thinking = components['thinking']
        if thinking:
            # Look for action mentions in thinking like "the action is X"
            action_patterns = [
                r'the action (?:is|appears to be|seems to be)\s+(\w+(?:\s+\w+)?)',
                r'action is\s+(\w+ing)',
                r'they are\s+(\w+ing\s+\w+)',
            ]
            for pattern in action_patterns:
                action_match = re.search(pattern, thinking, re.IGNORECASE)
                if action_match:
                    return action_match.group(1).strip()
        # Tool call requested but no answer yet - this is expected behavior
        return "[TOOL_CALL_PENDING]"
    
    # Try to extract after last </think>
    if '</think>' in generated_text:
        parts = generated_text.split('</think>')
        after_think = parts[-1].strip()
        # Make sure it's not a tool call or empty
        if after_think and not after_think.startswith('<'):
            return after_think
    
    # Fallback: look for common action patterns in the text
    action_patterns = [
        r'(?:person is|they are|action is)\s+(\w+ing\s+\w+)',
        r'(\w+ing\s+(?:the\s+)?\w+)',
    ]
    for pattern in action_patterns:
        match = re.search(pattern, generated_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Last resort: return empty
    return ""


def format_response_for_display(generated_text: str, max_len: int = 500) -> str:
    """Format response for verbose display, showing all components clearly."""
    components = parse_response_components(generated_text)
    
    lines = []
    
    # Show thinking (truncated if needed)
    if components['thinking']:
        thinking = components['thinking']
        if len(thinking) > max_len:
            thinking = thinking[:max_len] + "..."
        lines.append(f"  <think>: {thinking}")
    
    # Show tool calls
    if components['tool_calls']:
        for i, tc in enumerate(components['tool_calls']):
            if isinstance(tc, dict) and 'name' in tc:
                args = tc.get('arguments', {})
                lines.append(f"  <tool_call> [{i+1}]: {tc['name']}({json.dumps(args)})")
            else:
                lines.append(f"  <tool_call> [{i+1}]: {tc}")
    
    # Show answer
    if components['answer']:
        lines.append(f"  <answer>: {components['answer']}")
    elif components['has_tool_call']:
        lines.append(f"  <answer>: [Pending - model requested tool execution]")
    else:
        lines.append(f"  <answer>: [None extracted]")
    
    return '\n'.join(lines) if lines else "  [Empty response]"


def parse_box_predictions(text: str, img_shape: Tuple[int, int] = (1000, 1000)) -> List[Dict]:
    """
    Parse box predictions from generated text.
    
    Handles multiple JSON formats:
    - [{"person_bbox": [...], "object_bbox": [...]}, ...]
    - [{"bbox_2d": [...], "label": "person"}, {"bbox_2d": [...], "label": "object"}, ...]
    - [{"pair_id": 1, "person_bbox": [...], "object_bbox": [...]}, ...]
    
    Args:
        text: Generated text containing JSON
        img_shape: (height, width) for coordinate conversion
        
    Returns:
        List of dicts with 'person_box' and 'object_box' keys
    """
    pairs = []
    h, w = img_shape
    
    # Extract answer content first
    answer = parse_answer(text)
    
    # Clean markdown code fences if present
    cleaned_text = answer.strip()
    if cleaned_text.startswith("```"):
        lines = cleaned_text.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned_text = '\n'.join(lines)
    
    # Try to parse JSON
    try:
        # Find JSON array
        match = re.search(r'\[[\s\S]*\]', cleaned_text)
        if match:
            parsed = json.loads(match.group())
            
            if isinstance(parsed, list):
                # Check format type
                if len(parsed) > 0:
                    first_item = parsed[0]
                    
                    # Format 1: person_bbox/object_bbox pairs
                    if isinstance(first_item, dict) and ('person_bbox' in first_item or 'person_box' in first_item):
                        for det in parsed:
                            if not isinstance(det, dict):
                                continue
                            person_bbox = det.get('person_bbox', det.get('person_box', []))
                            object_bbox = det.get('object_bbox', det.get('object_box', []))
                            
                            if len(person_bbox) == 4 and len(object_bbox) == 4:
                                pairs.append({
                                    'person_box': person_bbox,
                                    'object_box': object_bbox
                                })
                    
                    # Format 2: bbox_2d with labels (alternating person/object)
                    elif isinstance(first_item, dict) and 'bbox_2d' in first_item:
                        for i in range(0, len(parsed) - 1, 2):
                            person_det = parsed[i]
                            object_det = parsed[i + 1] if i + 1 < len(parsed) else None
                            
                            if object_det is None:
                                continue
                            
                            person_bbox = person_det.get('bbox_2d', [])
                            object_bbox = object_det.get('bbox_2d', [])
                            
                            if len(person_bbox) == 4 and len(object_bbox) == 4:
                                pairs.append({
                                    'person_box': person_bbox,
                                    'object_box': object_bbox
                                })
    except (json.JSONDecodeError, Exception) as e:
        logger.debug(f"JSON parsing failed: {e}")
    
    return pairs


def transform_box_with_zoom(box: List[float], zoom_transform: Tuple[float, float, float, float]) -> List[int]:
    """
    Transform a box from zoomed view coordinates back to original image coordinates.
    
    Args:
        box: Box in current view's 1000x1000 coordinates [x1, y1, x2, y2]
        zoom_transform: (x_offset, y_offset, scale_x, scale_y) from ZoomInTool
        
    Returns:
        Box in original image's 1000x1000 coordinates
    """
    x_off, y_off, scale_x, scale_y = zoom_transform
    
    x1, y1, x2, y2 = [float(c) / 1000.0 for c in box]
    
    # Transform back to original coordinates
    orig_x1 = (x1 * scale_x + x_off) * 1000
    orig_y1 = (y1 * scale_y + y_off) * 1000
    orig_x2 = (x2 * scale_x + x_off) * 1000
    orig_y2 = (y2 * scale_y + y_off) * 1000
    
    return [int(orig_x1), int(orig_y1), int(orig_x2), int(orig_y2)]


def transform_pairs_with_zoom(pairs: List[Dict], zoom_transform: Tuple[float, float, float, float]) -> List[Dict]:
    """
    Transform all pairs from zoomed view coordinates to original image coordinates.
    
    Args:
        pairs: List of pairs with 'person_box' and 'object_box'
        zoom_transform: (x_offset, y_offset, scale_x, scale_y)
        
    Returns:
        Transformed pairs
    """
    transformed = []
    for pair in pairs:
        person_box = pair.get('person_box', [])
        object_box = pair.get('object_box', [])
        
        if len(person_box) == 4 and len(object_box) == 4:
            transformed.append({
                'person_box': transform_box_with_zoom(person_box, zoom_transform),
                'object_box': transform_box_with_zoom(object_box, zoom_transform),
            })
    
    return transformed


def clean_action_response(response_text: str) -> str:
    """Clean action response to extract action phrase."""
    response = response_text.strip()
    
    # Remove common prefixes
    prefixes_to_remove = [
        "the person is ",
        "person is ",
        "they are ",
        "action: ",
        "answer: ",
        "output: ",
    ]
    
    response_lower = response.lower()
    for prefix in prefixes_to_remove:
        if response_lower.startswith(prefix):
            response = response[len(prefix):].strip()
            response_lower = response.lower()
    
    # Remove trailing punctuation
    response = response.rstrip('.!?,;:')
    
    # Convert to lowercase
    response = response.lower().strip()
    
    return response


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_referring_task(
    model,
    processor,
    test_data: List[Dict],
    image_base_dir: str,
    max_samples: Optional[int] = None,
    visualize_attention: bool = False,
    viz_output_dir: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[Dict, List[Dict]]:
    """Evaluate on referring task (action prediction)."""
    
    from utils.metrics import evaluate_referring_nltk, evaluate_referring_simple
    
    predictions = []
    references = []
    detailed_results = []
    
    samples = test_data[:max_samples] if max_samples else test_data
    device = next(model.parameters()).device
    
    # Setup visualization if needed
    if visualize_attention and viz_output_dir:
        Path(viz_output_dir).mkdir(parents=True, exist_ok=True)
        try:
            from utils.visualization import visualize_multi_region_attention
            has_viz = True
        except ImportError:
            logger.warning("Visualization not available, skipping attention viz")
            has_viz = False
    else:
        has_viz = False
    
    viz_count = 0  # Track saved visualizations
    
    for idx, sample in enumerate(tqdm(samples, desc="Evaluating referring")):
        try:
            # Get data
            file_name = sample["file_name"]
            boxes = sample.get("boxes_1000", sample.get("boxes", []))
            person_idx = sample.get("person_box_idx", 0)
            object_idx = sample.get("object_box_idx", 1)
            gt_action = sample.get("gt_action", sample.get("response", ""))
            
            if len(boxes) < 2:
                continue
            
            person_box_raw = boxes[person_idx]
            object_box_raw = boxes[object_idx]
            
            # Load image first to get dimensions for normalization
            image_path = get_image_path(file_name, image_base_dir)
            if not Path(image_path).exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            image = Image.open(image_path).convert("RGB")
            img_width, img_height = image.size
            
            # Normalize bounding boxes to 1000x1000 format (matching training data)
            person_box = normalize_bbox_to_1000(person_box_raw, img_width, img_height)
            object_box = normalize_bbox_to_1000(object_box_raw, img_width, img_height)
            
            # Compute interaction box in normalized coordinates
            interaction_box = [
                min(person_box[0], object_box[0]),
                min(person_box[1], object_box[1]),
                max(person_box[2], object_box[2]),
                max(person_box[3], object_box[3])
            ]
            
            # Create prompt with structured image content for Qwen3-VL
            user_prompt = create_referring_prompt(person_box, object_box)
            user_content = create_referring_prompt_with_image(person_box, object_box, image)
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
            
            # Use processor's apply_chat_template which handles images properly
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process inputs - images already handled by apply_chat_template
            inputs = processor(
                text=[text], 
                images=[image], 
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Prepare refer_boxes for spatial linking
            refer_boxes_tensor = torch.tensor(
                [person_box, object_box, interaction_box], 
                dtype=torch.float32
            ).unsqueeze(0)  # [1, 3, 4]
            
            # Generate with spatial attention output if visualizing
            with torch.no_grad():
                # For visualization, we need to do a forward pass first
                if has_viz and idx < 10:  # Only visualize first 10
                    try:
                        # Forward pass to get attention weights
                        _ = model(
                            **inputs,
                            refer_boxes=[refer_boxes_tensor.squeeze(0).to(device)],
                            output_spatial_attentions=True,
                        )
                        attention_info = model.get_spatial_attention_weights()
                        
                        if attention_info and len(attention_info) > 0:
                            # Format attention info for visualization
                            attention_info_list = format_attention_for_viz(
                                attention_info, 
                                [person_box, object_box, interaction_box],
                                img_width, img_height,
                                inputs.get('image_grid_thw')
                            )
                            
                            if attention_info_list:
                                viz_path = Path(viz_output_dir) / f"attention_{idx:04d}.png"
                                visualize_multi_region_attention(
                                    image=image,
                                    attention_info_list=attention_info_list,
                                    output_path=str(viz_path),
                                )
                                viz_count += 1
                    except Exception as e:
                        logger.debug(f"Visualization failed for sample {idx}: {e}")
                
                # Generate response (single turn first)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            
            # Decode only the NEW tokens (not the full sequence)
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            generated = processor.decode(generated_ids, skip_special_tokens=True)
            
            # Check if model requested tool call - if so, run agent loop
            first_components = parse_response_components(generated)
            all_tool_calls = []
            
            if first_components['has_tool_call'] and first_components['tool_calls']:
                # Run multi-turn with tool execution
                agent_result = run_agent_loop(
                    model, processor, image, user_prompt, SYSTEM_PROMPT,
                    device, max_turns=3, max_new_tokens=256
                )
                generated = agent_result['final_response']
                all_tool_calls = agent_result['tool_calls']
            
            answer = parse_answer(generated)
            pred = clean_action_response(answer)
            
            predictions.append(pred)
            references.append(gt_action)
            
            # Parse response components for detailed logging
            components = parse_response_components(generated)
            
            detailed_results.append({
                "file_name": file_name,
                "person_box": person_box,
                "object_box": object_box,
                "gt_action": gt_action,
                "predicted": pred,
                "raw_answer": answer,
                "full_response": generated,
                # Parsed components
                "thinking": components['thinking'],
                "tool_calls": all_tool_calls if all_tool_calls else components['tool_calls'],
                "has_tool_call": components['has_tool_call'] or len(all_tool_calls) > 0,
                "answer_tag": components['answer'],
                "num_turns": len(all_tool_calls) + 1 if all_tool_calls else 1,
            })
            
            # Verbose output with full component breakdown
            if verbose and idx < 20:
                print(f"\n{'='*60}")
                print(f"[Sample {idx+1}] {file_name}")
                print(f"{'='*60}")
                print(f"  Person box: {person_box}")
                print(f"  Object box: {object_box}")
                print(f"  GT action: {gt_action}")
                print(f"\n  Model Response:")
                print(format_response_for_display(generated))
                
                # Show tool execution details if any
                if all_tool_calls:
                    print(f"\n  Tool Execution:")
                    print(f"    Turns: {len(all_tool_calls) + 1}")
                    for tc in all_tool_calls:
                        print(f"    - Turn {tc['turn']}: {tc['name']}({tc.get('arguments', {})})")
                        print(f"      Result: {tc['result'][:80]}...")
                
                print(f"\n  Final Prediction: {pred if pred else '[EMPTY]'}")
                
                # Show match status
                if pred and gt_action:
                    is_exact = pred.lower().strip() == gt_action.lower().strip()
                    is_partial = any(w in pred.lower() for w in gt_action.lower().split())
                    status = "✓ EXACT" if is_exact else ("~ PARTIAL" if is_partial else "✗ MISMATCH")
                    print(f"  Match Status: {status}")
            
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            continue
    
    # Compute metrics
    logger.info(f"Computing COCO caption metrics for {len(predictions)} samples...")
    
    try:
        # Use NLTK-based metrics (doesn't require Java)
        metrics = evaluate_referring_nltk(predictions, references)
    except Exception as e:
        logger.warning(f"NLTK evaluation failed: {e}, using simple metrics")
        metrics = evaluate_referring_simple(predictions, references)
    
    metrics["num_samples"] = len(predictions)
    
    if has_viz:
        logger.info(f"Saved {viz_count} attention visualizations to {viz_output_dir}")
    
    return metrics, detailed_results


def evaluate_grounding_task(
    model,
    processor,
    test_data: List[Dict],
    image_base_dir: str,
    max_samples: Optional[int] = None,
    visualize_attention: bool = False,
    viz_output_dir: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[Dict, List[Dict]]:
    """Evaluate on grounding task (box prediction)."""
    
    from utils.metrics import evaluate_grounding
    
    results = []
    detailed_results = []
    
    samples = test_data[:max_samples] if max_samples else test_data
    device = next(model.parameters()).device
    
    for idx, sample in enumerate(tqdm(samples, desc="Evaluating grounding")):
        try:
            file_name = sample["file_name"]
            action = sample["action"]
            object_category = sample["object_category"]
            boxes = sample.get("boxes_1000", sample.get("boxes", []))
            num_pairs = sample.get("num_pairs", 1)
            gt_box_inds = sample.get("gt_box_inds", list(range(num_pairs * 2)))
            img_height = sample.get("height", 1000)
            img_width = sample.get("width", 1000)
            
            # Build ground truth pairs from simplified format
            # Normalize boxes to 1000x1000 format to match model output
            gt_pairs = []
            for i in range(num_pairs):
                person_idx = gt_box_inds[i * 2]
                object_idx = gt_box_inds[i * 2 + 1]
                
                if person_idx < len(boxes) and object_idx < len(boxes):
                    person_box_raw = boxes[person_idx]
                    object_box_raw = boxes[object_idx]
                    
                    # Normalize to 1000x1000 format
                    person_box = normalize_bbox_to_1000(person_box_raw, img_width, img_height)
                    object_box = normalize_bbox_to_1000(object_box_raw, img_width, img_height)
                    
                    gt_pairs.append({
                        "person_box": person_box,
                        "object_box": object_box
                    })
            
            if not gt_pairs:
                continue
            
            # Load image
            image_path = get_image_path(file_name, image_base_dir)
            if not Path(image_path).exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            image = Image.open(image_path).convert("RGB")
            
            # Create prompt with structured image content for Qwen3-VL
            user_prompt = create_grounding_prompt(action, object_category)
            user_content = create_grounding_prompt_with_image(action, object_category, image)
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
            
            # Use processor's apply_chat_template which handles images properly
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = processor(
                text=[text], 
                images=[image], 
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            
            # Decode only the newly generated tokens (after the input prompt)
            generated = processor.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            pred_pairs = parse_box_predictions(generated, (img_height, img_width))
            
            results.append({
                "predicted_pairs": pred_pairs,
                "gt_pairs": gt_pairs,
            })
            
            # Parse response components for detailed logging
            components = parse_response_components(generated)
            
            detailed_results.append({
                "file_name": file_name,
                "action": action,
                "object_category": object_category,
                "gt_pairs": gt_pairs,
                "predicted_pairs": pred_pairs,
                "num_pred": len(pred_pairs),
                "num_gt": len(gt_pairs),
                "full_response": generated,
                # Parsed components
                "thinking": components['thinking'],
                "tool_calls": components['tool_calls'],
                "has_tool_call": components['has_tool_call'],
                "answer_tag": components['answer'],
            })
            
            # Verbose output with full component breakdown
            if verbose and idx < 20:
                print(f"\n{'='*60}")
                print(f"[Sample {idx+1}] {file_name}")
                print(f"{'='*60}")
                print(f"  Action: {action} {object_category}")
                print(f"  GT pairs: {len(gt_pairs)}")
                
                print(f"\n  Model Response:")
                print(format_response_for_display(generated))
                
                print(f"\n  Predicted pairs: {len(pred_pairs)}")
                for i, pair in enumerate(pred_pairs[:3]):
                    print(f"    Pair {i+1}: Person {pair['person_box']}, Object {pair['object_box']}")
            
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            continue
    
    # Compute metrics
    logger.info(f"Computing grounding metrics for {len(results)} samples...")
    metrics = evaluate_grounding(
        [r["predicted_pairs"] for r in results],
        [r["gt_pairs"] for r in results],
    )
    
    metrics["num_samples"] = len(results)
    
    return metrics, detailed_results


# =============================================================================
# VLLM EVALUATION FUNCTIONS
# =============================================================================

def evaluate_referring_task_vllm(
    test_data: List[Dict],
    image_base_dir: str,
    vllm_endpoint: str,
    model_name: str,
    max_samples: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Dict, List[Dict]]:
    """
    Evaluate on referring task using vLLM server with real tool calling.
    
    Args:
        test_data: Test dataset
        image_base_dir: Base directory for images
        vllm_endpoint: vLLM server endpoint
        model_name: Model name in vLLM
        max_samples: Maximum samples to evaluate
        verbose: Show detailed output
    
    Returns:
        Tuple of (metrics dict, detailed results list)
    """
    from utils.metrics import evaluate_referring_nltk, evaluate_referring_simple
    
    predictions = []
    references = []
    detailed_results = []
    
    samples = test_data[:max_samples] if max_samples else test_data
    
    # Create single ZoomInTool instance for all evaluations
    zoom_tool = ZoomInTool(padding=(0.1, 0.1))
    
    # Check vLLM health
    if not check_vllm_health(vllm_endpoint, model_name):
        raise RuntimeError(f"vLLM server not healthy or model '{model_name}' not available")
    
    logger.info(f"Starting vLLM evaluation with {len(samples)} samples")
    
    for idx, sample in enumerate(tqdm(samples, desc="Evaluating referring (vLLM)")):
        try:
            # Get data
            file_name = sample["file_name"]
            boxes = sample.get("boxes_1000", sample.get("boxes", []))
            person_idx = sample.get("person_box_idx", 0)
            object_idx = sample.get("object_box_idx", 1)
            gt_action = sample.get("gt_action", sample.get("response", ""))
            
            if len(boxes) < 2:
                continue
            
            person_box_raw = boxes[person_idx]
            object_box_raw = boxes[object_idx]
            
            # Load image
            image_path = get_image_path(file_name, image_base_dir)
            if not Path(image_path).exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            image = Image.open(image_path).convert("RGB")
            img_width, img_height = image.size
            
            # Normalize bounding boxes to 1000x1000 format
            person_box = normalize_bbox_to_1000(person_box_raw, img_width, img_height)
            object_box = normalize_bbox_to_1000(object_box_raw, img_width, img_height)
            
            # Create prompt
            user_prompt = create_referring_prompt(person_box, object_box)
            
            # Run agent loop with vLLM
            agent_result = run_agent_loop_vllm(
                image=image,
                initial_prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT,
                vllm_endpoint=vllm_endpoint,
                model_name=model_name,
                zoom_tool=zoom_tool,
                max_turns=3,
                max_tokens=256,
            )
            
            generated = agent_result['final_response']
            all_tool_calls = agent_result['tool_calls']
            
            answer = parse_answer(generated)
            pred = clean_action_response(answer)
            
            predictions.append(pred)
            references.append(gt_action)
            
            # Parse response components
            components = parse_response_components(generated)
            
            detailed_results.append({
                "file_name": file_name,
                "person_box": person_box,
                "object_box": object_box,
                "gt_action": gt_action,
                "predicted": pred,
                "raw_answer": answer,
                "full_response": generated,
                "all_responses": agent_result.get('all_responses', [generated]),  # All turn responses
                "thinking": components['thinking'],
                "tool_calls": all_tool_calls if all_tool_calls else components['tool_calls'],
                "has_tool_call": len(all_tool_calls) > 0 or components['has_tool_call'],
                "answer_tag": components['answer'],
                "num_turns": agent_result['num_turns'],
            })
            
            # Verbose output
            if verbose and idx < 20:
                print(f"\n{'='*60}")
                print(f"[Sample {idx+1}] {file_name}")
                print(f"{'='*60}")
                print(f"  Person box: {person_box}")
                print(f"  Object box: {object_box}")
                print(f"  GT action: {gt_action}")
                
                # Show all responses (for multi-turn debugging)
                print(f"\n  Model Responses ({agent_result['num_turns']} turns):")
                for turn_idx, resp in enumerate(agent_result.get('all_responses', [generated])):
                    print(f"    --- Turn {turn_idx + 1} ---")
                    # Show full response for debugging
                    print(f"    {resp[:600]}{'...' if len(resp) > 600 else ''}")
                
                # Show tool calls with full details
                if all_tool_calls:
                    print(f"\n  Tool Execution Details:")
                    for tc in all_tool_calls:
                        print(f"    Turn {tc['turn']}: {tc['name']}")
                        print(f"      Args: {tc.get('arguments', {})}")
                        print(f"      Result: {tc.get('result', 'N/A')}")
                        print(f"      Valid: {tc.get('valid', 'N/A')}")
                
                print(f"\n  Final Prediction: {pred if pred else '[EMPTY]'}")
                
                if pred and gt_action:
                    is_exact = pred.lower().strip() == gt_action.lower().strip()
                    is_partial = any(w in pred.lower() for w in gt_action.lower().split())
                    status = "✓ EXACT" if is_exact else ("~ PARTIAL" if is_partial else "✗ MISMATCH")
                    print(f"  Match Status: {status}")
            
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            continue
    
    # Compute metrics
    logger.info(f"Computing COCO caption metrics for {len(predictions)} samples...")
    
    try:
        metrics = evaluate_referring_nltk(predictions, references)
    except Exception as e:
        logger.warning(f"NLTK evaluation failed: {e}, using simple metrics")
        metrics = evaluate_referring_simple(predictions, references)
    
    metrics["num_samples"] = len(predictions)
    
    return metrics, detailed_results


def evaluate_grounding_task_vllm(
    test_data: List[Dict],
    image_base_dir: str,
    vllm_endpoint: str,
    model_name: str,
    max_samples: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Dict, List[Dict]]:
    """
    Evaluate on grounding task using vLLM server with real tool calling.
    
    Args:
        test_data: Test dataset
        image_base_dir: Base directory for images
        vllm_endpoint: vLLM server endpoint
        model_name: Model name in vLLM
        max_samples: Maximum samples to evaluate
        verbose: Show detailed output
    
    Returns:
        Tuple of (metrics dict, detailed results list)
    """
    from utils.metrics import evaluate_grounding
    
    results = []
    detailed_results = []
    
    samples = test_data[:max_samples] if max_samples else test_data
    
    # Create single ZoomInTool instance
    zoom_tool = ZoomInTool(padding=(0.1, 0.1))
    
    # Check vLLM health
    if not check_vllm_health(vllm_endpoint, model_name):
        raise RuntimeError(f"vLLM server not healthy or model '{model_name}' not available")
    
    logger.info(f"Starting vLLM grounding evaluation with {len(samples)} samples")
    
    for idx, sample in enumerate(tqdm(samples, desc="Evaluating grounding (vLLM)")):
        try:
            file_name = sample["file_name"]
            action = sample["action"]
            object_category = sample["object_category"]
            boxes = sample.get("boxes_1000", sample.get("boxes", []))
            num_pairs = sample.get("num_pairs", 1)
            gt_box_inds = sample.get("gt_box_inds", list(range(num_pairs * 2)))
            img_height = sample.get("height", 1000)
            img_width = sample.get("width", 1000)
            
            # Build ground truth pairs
            gt_pairs = []
            for i in range(num_pairs):
                person_idx = gt_box_inds[i * 2]
                object_idx = gt_box_inds[i * 2 + 1]
                
                if person_idx < len(boxes) and object_idx < len(boxes):
                    person_box_raw = boxes[person_idx]
                    object_box_raw = boxes[object_idx]
                    
                    person_box = normalize_bbox_to_1000(person_box_raw, img_width, img_height)
                    object_box = normalize_bbox_to_1000(object_box_raw, img_width, img_height)
                    
                    gt_pairs.append({
                        "person_box": person_box,
                        "object_box": object_box
                    })
            
            if not gt_pairs:
                continue
            
            # Load image
            image_path = get_image_path(file_name, image_base_dir)
            if not Path(image_path).exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            image = Image.open(image_path).convert("RGB")
            
            # Create prompt
            user_prompt = create_grounding_prompt(action, object_category)
            
            # Run agent loop with vLLM
            agent_result = run_agent_loop_vllm(
                image=image,
                initial_prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT,
                vllm_endpoint=vllm_endpoint,
                model_name=model_name,
                zoom_tool=zoom_tool,
                max_turns=3,
                max_tokens=500,
            )
            
            generated = agent_result['final_response']
            all_tool_calls = agent_result['tool_calls']
            zoom_transform = agent_result.get('zoom_transform', (0.0, 0.0, 1.0, 1.0))
            had_zoom = agent_result.get('had_zoom', False)
            
            # Parse predicted boxes
            pred_pairs_raw = parse_box_predictions(generated, (img_height, img_width))
            
            # Transform boxes back to original image coordinates if there was a zoom
            if had_zoom and pred_pairs_raw:
                pred_pairs = transform_pairs_with_zoom(pred_pairs_raw, zoom_transform)
            else:
                pred_pairs = pred_pairs_raw
            
            results.append({
                "predicted_pairs": pred_pairs,
                "gt_pairs": gt_pairs,
            })
            
            # Parse response components
            components = parse_response_components(generated)
            
            detailed_results.append({
                "file_name": file_name,
                "action": action,
                "object_category": object_category,
                "img_width": img_width,
                "img_height": img_height,
                "gt_pairs": gt_pairs,
                "predicted_pairs": pred_pairs,
                "pred_pairs_raw": pred_pairs_raw if had_zoom else None,  # Store raw for debugging
                "zoom_transform": zoom_transform if had_zoom else None,
                "num_pred": len(pred_pairs),
                "num_gt": len(gt_pairs),
                "full_response": generated,
                "all_responses": agent_result.get('all_responses', [generated]),  # All turn responses
                "thinking": components['thinking'],
                "tool_calls": all_tool_calls if all_tool_calls else components['tool_calls'],
                "has_tool_call": len(all_tool_calls) > 0 or components['has_tool_call'],
                "answer_tag": components['answer'],
                "num_turns": agent_result['num_turns'],
            })
            
            # Verbose output
            if verbose and idx < 20:
                print(f"\n{'='*60}")
                print(f"[Sample {idx+1}] {file_name}")
                print(f"{'='*60}")
                print(f"  Action: {action} {object_category}")
                print(f"  Image size: {img_width}x{img_height}")
                
                # Show GT pairs with boxes
                print(f"\n  Ground Truth ({len(gt_pairs)} pairs):")
                for i, gt in enumerate(gt_pairs):
                    print(f"    GT {i+1}: Person {gt['person_box']}, Object {gt['object_box']}")
                
                # Show all responses (for multi-turn debugging)
                print(f"\n  Model Responses ({agent_result['num_turns']} turns):")
                for turn_idx, resp in enumerate(agent_result.get('all_responses', [generated])):
                    print(f"    --- Turn {turn_idx + 1} ---")
                    # Show full response for debugging
                    print(f"    {resp[:500]}{'...' if len(resp) > 500 else ''}")
                
                # Show tool calls with details
                if all_tool_calls:
                    print(f"\n  Tool Execution Details:")
                    for tc in all_tool_calls:
                        print(f"    Turn {tc['turn']}: {tc['name']}")
                        print(f"      Args: {tc.get('arguments', {})}")
                        print(f"      Result: {tc.get('result', 'N/A')}")
                        print(f"      Valid: {tc.get('valid', 'N/A')}")
                
                # Show predicted pairs (with transformation info if applicable)
                print(f"\n  Predictions ({len(pred_pairs)} pairs):")
                if had_zoom and pred_pairs_raw:
                    print(f"    [Coordinates transformed from zoomed view to original]")
                    print(f"    Zoom transform: offset=({zoom_transform[0]:.3f}, {zoom_transform[1]:.3f}), scale=({zoom_transform[2]:.3f}, {zoom_transform[3]:.3f})")
                    for i, (raw, transformed) in enumerate(zip(pred_pairs_raw[:3], pred_pairs[:3])):
                        print(f"    Pred {i+1} (raw):    Person {raw['person_box']}, Object {raw['object_box']}")
                        print(f"    Pred {i+1} (transf): Person {transformed['person_box']}, Object {transformed['object_box']}")
                else:
                    for i, pair in enumerate(pred_pairs[:5]):
                        print(f"    Pred {i+1}: Person {pair['person_box']}, Object {pair['object_box']}")
                
                # Compare GT vs Pred boxes with IoU
                if pred_pairs and gt_pairs:
                    print(f"\n  IoU Comparison (first pred vs first GT):")
                    from utils.metrics import calculate_iou
                    p_pred = pred_pairs[0]['person_box']
                    o_pred = pred_pairs[0]['object_box']
                    p_gt = gt_pairs[0]['person_box']
                    o_gt = gt_pairs[0]['object_box']
                    person_iou = calculate_iou(p_pred, p_gt)
                    object_iou = calculate_iou(o_pred, o_gt)
                    print(f"    Person IoU: {person_iou:.4f} (pred {p_pred} vs gt {p_gt})")
                    print(f"    Object IoU: {object_iou:.4f} (pred {o_pred} vs gt {o_gt})")
                    
                    # Show if it would match at IoU 0.5
                    if person_iou >= 0.5 and object_iou >= 0.5:
                        print(f"    --> MATCH at IoU 0.5 threshold!")
                    else:
                        print(f"    --> NO MATCH (need both >= 0.5)")
            
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            continue
    
    # Compute metrics
    logger.info(f"Computing grounding metrics for {len(results)} samples...")
    metrics = evaluate_grounding(
        [r["predicted_pairs"] for r in results],
        [r["gt_pairs"] for r in results],
    )
    
    metrics["num_samples"] = len(results)
    
    return metrics, detailed_results


def format_attention_for_viz(
    attention_weights, 
    boxes: List[List[float]], 
    img_width: int, 
    img_height: int,
    image_grid_thw: Optional[torch.Tensor] = None
) -> List[Dict]:
    """
    Format attention weights for visualization.
    
    Args:
        attention_weights: Raw attention weights from model
        boxes: List of boxes [person, object, interaction]
        img_width: Image width
        img_height: Image height
        image_grid_thw: Grid dimensions from processor
        
    Returns:
        List of attention info dicts for visualization
    """
    if attention_weights is None or len(attention_weights) == 0:
        return []
    
    box_types = ['person', 'object', 'interaction']
    attention_info_list = []
    
    # Get grid dimensions
    if image_grid_thw is not None and image_grid_thw.numel() >= 3:
        grid_thw = image_grid_thw[0].tolist() if image_grid_thw.dim() > 1 else image_grid_thw.tolist()
    else:
        # Estimate grid size (Qwen3VL uses 14x14 patches)
        patch_size = 14
        grid_thw = [1, img_height // patch_size, img_width // patch_size]
    
    for i, (box, box_type) in enumerate(zip(boxes[:len(attention_weights)], box_types)):
        if i >= len(attention_weights):
            break
        
        attn = attention_weights[i]
        if attn is None:
            continue
        
        # Ensure attention is on CPU
        if torch.is_tensor(attn):
            attn = attn.detach().cpu()
        
        # Normalize box to [0, 1]
        norm_box = [
            box[0] / 1000.0,
            box[1] / 1000.0,
            box[2] / 1000.0,
            box[3] / 1000.0,
        ]
        
        # Create patch indices (dummy for now - actual implementation needs image_start_idx)
        num_patches = int(grid_thw[1] * grid_thw[2])
        patch_indices = torch.arange(num_patches)
        
        attention_info_list.append({
            'attention_weights': attn.unsqueeze(0) if attn.dim() == 3 else attn,
            'bbox': torch.tensor(norm_box),
            'box_type': box_type,
            'grid_thw': torch.tensor(grid_thw),
            'patch_indices': patch_indices,
        })
    
    return attention_info_list


def print_metrics(metrics: Dict, title: str = "Evaluation Results"):
    """Print metrics in a nice format."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)
    print(f"{'Metric':<20} {'Value':>10}  {'Description':<35}")
    print("-" * 70)
    
    # Define metric descriptions
    descriptions = {
        'METEOR': 'Semantic similarity',
        'CIDEr': 'Corpus consensus',
        'BLEU_1': 'Unigram overlap',
        'BLEU_2': 'Bigram overlap',
        'BLEU_3': 'Trigram overlap',
        'BLEU_4': '4-gram overlap',
        'ROUGE_L': 'Longest common subsequence',
        'exact_match': 'Exact string match',
        'verb_match': 'First word (verb) match',
        'word_overlap': 'Jaccard word similarity',
        'non_empty': 'Non-empty prediction rate',
        'bertscore_precision': 'BERTScore Precision',
        'bertscore_recall': 'BERTScore Recall',
        'bertscore_f1': 'BERTScore F1',
        'AR': 'Average Recall @ IoU=0.50:0.95',
        'AR@0.5': 'Average Recall @ IoU=0.50',
        'AR@0.75': 'Average Recall @ IoU=0.75',
        'ARs': 'AR for small objects (area < 32^2)',
        'ARm': 'AR for medium objects',
        'ARl': 'AR for large objects (area > 96^2)',
        'num_samples': 'Total samples evaluated',
    }
    
    # Metrics to display as percentages
    percentage_metrics = [
        'METEOR', 'CIDEr', 'BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4', 
        'ROUGE_L', 'exact_match', 'verb_match', 'word_overlap', 'non_empty',
        'bertscore_precision', 'bertscore_recall', 'bertscore_f1',
        'AR', 'AR@0.5', 'AR@0.75', 'ARs', 'ARm', 'ARl'
    ]
    
    for key, value in metrics.items():
        desc = descriptions.get(key, '')
        if isinstance(value, float):
            if key in percentage_metrics:
                print(f"{key:<20} {value*100:>9.2f}%  {desc:<35}")
            else:
                print(f"{key:<20} {value:>10.4f}  {desc:<35}")
        else:
            print(f"{key:<20} {value:>10}  {desc:<35}")
    
    print("=" * 70 + "\n")


def print_response_statistics(detailed_results: List[Dict]):
    """Print statistics about model response patterns (tool calls, answers, etc.)."""
    if not detailed_results:
        return
    
    total = len(detailed_results)
    has_thinking = sum(1 for r in detailed_results if r.get('thinking'))
    has_tool_call = sum(1 for r in detailed_results if r.get('has_tool_call'))
    has_answer = sum(1 for r in detailed_results if r.get('answer_tag'))
    empty_pred = sum(1 for r in detailed_results if not r.get('predicted'))
    
    print("=" * 70)
    print(" Response Pattern Statistics")
    print("=" * 70)
    print(f"  Total samples:           {total}")
    print(f"  Has <think> tag:         {has_thinking:>5} ({100*has_thinking/total:.1f}%)")
    print(f"  Has <tool_call> tag:     {has_tool_call:>5} ({100*has_tool_call/total:.1f}%)")
    print(f"  Has <answer> tag:        {has_answer:>5} ({100*has_answer/total:.1f}%)")
    print(f"  Empty predictions:       {empty_pred:>5} ({100*empty_pred/total:.1f}%)")
    print("-" * 70)
    
    # Tool call breakdown
    if has_tool_call > 0:
        tool_names = {}
        for r in detailed_results:
            for tc in r.get('tool_calls', []):
                if isinstance(tc, dict):
                    name = tc.get('name', 'unknown')
                    tool_names[name] = tool_names.get(name, 0) + 1
        
        if tool_names:
            print("  Tool calls breakdown:")
            for name, count in sorted(tool_names.items(), key=lambda x: -x[1]):
                print(f"    - {name}: {count}")
    
    print("=" * 70 + "\n")


def save_results(
    metrics: Dict,
    detailed_results: List[Dict],
    output_dir: str,
    task_type: str,
    save_detailed: bool = True,
):
    """Save evaluation results to files with full detailed logging."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics
    metrics_file = output_path / f"{task_type}_metrics_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")
    
    # Save detailed results (JSON)
    if save_detailed:
        detailed_file = output_path / f"{task_type}_detailed_{timestamp}.json"
        with open(detailed_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        logger.info(f"Detailed results saved to {detailed_file}")
    
    # Save comprehensive log with full details
    log_file = output_path / f"{task_type}_log_{timestamp}.txt"
    with open(log_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"EVALUATION LOG - {task_type.upper()}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total samples: {len(detailed_results)}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write metrics summary
        f.write("METRICS SUMMARY:\n")
        f.write("-" * 40 + "\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Write detailed per-sample logs
        f.write("=" * 80 + "\n")
        f.write("DETAILED SAMPLE LOGS:\n")
        f.write("=" * 80 + "\n\n")
        
        for i, result in enumerate(detailed_results):
            f.write(f"{'='*80}\n")
            f.write(f"[SAMPLE {i+1}] {result.get('file_name', 'N/A')}\n")
            f.write(f"{'='*80}\n\n")
            
            # Task-specific info
            if task_type == 'referring':
                f.write(f"INPUT:\n")
                f.write(f"  Person box: {result.get('person_box', 'N/A')}\n")
                f.write(f"  Object box: {result.get('object_box', 'N/A')}\n")
                f.write(f"  Ground Truth Action: {result.get('gt_action', 'N/A')}\n\n")
            else:  # grounding
                f.write(f"INPUT:\n")
                f.write(f"  Action: {result.get('action', 'N/A')} {result.get('object_category', '')}\n")
                f.write(f"  Image size: {result.get('img_width', 'N/A')}x{result.get('img_height', 'N/A')}\n")
                f.write(f"\n  Ground Truth Pairs ({len(result.get('gt_pairs', []))}):\n")
                for j, gt in enumerate(result.get('gt_pairs', [])):
                    f.write(f"    GT {j+1}: Person {gt.get('person_box', 'N/A')}, Object {gt.get('object_box', 'N/A')}\n")
                f.write("\n")
            
            # Number of turns
            num_turns = result.get('num_turns', 1)
            all_responses = result.get('all_responses', [result.get('full_response', '')])
            tool_calls = result.get('tool_calls', [])
            
            f.write(f"CONVERSATION ({num_turns} turn{'s' if num_turns > 1 else ''}):\n")
            f.write("-" * 40 + "\n")
            
            # Show each turn's response
            for turn_idx, response in enumerate(all_responses):
                f.write(f"\n  --- TURN {turn_idx + 1} ---\n")
                f.write(f"  {'-'*60}\n")
                for line in response.split('\n'):
                    f.write(f"  {line}\n")
                f.write(f"  {'-'*60}\n")
                
                # Show tool call for this turn if applicable
                turn_tool_calls = [tc for tc in tool_calls if tc.get('turn') == turn_idx + 1]
                if turn_tool_calls:
                    for tc in turn_tool_calls:
                        f.write(f"\n  >> TOOL EXECUTED: {tc.get('name', 'unknown')}\n")
                        f.write(f"     Arguments: {tc.get('arguments', {})}\n")
                        f.write(f"     Result: {tc.get('result', 'N/A')}\n")
                        f.write(f"     Valid: {tc.get('valid', 'N/A')}\n")
                        f.write(f"     [Image was cropped and passed to next turn]\n")
            
            f.write("\n")
            
            # Tool calls with full details
            tool_calls = result.get('tool_calls', [])
            if tool_calls:
                f.write(f"TOOL EXECUTION ({len(tool_calls)} call{'s' if len(tool_calls) > 1 else ''}):\n")
                f.write("-" * 40 + "\n")
                for tc in tool_calls:
                    f.write(f"\n  Turn {tc.get('turn', '?')}: {tc.get('name', 'unknown')}\n")
                    f.write(f"    Arguments:\n")
                    args = tc.get('arguments', {})
                    for k, v in args.items():
                        f.write(f"      {k}: {v}\n")
                    f.write(f"    Result: {tc.get('result', 'N/A')}\n")
                    f.write(f"    Valid: {tc.get('valid', 'N/A')}\n")
                f.write("\n")
            
            # Zoom transform info (for grounding)
            if result.get('zoom_transform'):
                zt = result['zoom_transform']
                f.write(f"ZOOM TRANSFORM (applied to predictions):\n")
                f.write(f"  Offset: ({zt[0]:.4f}, {zt[1]:.4f})\n")
                f.write(f"  Scale: ({zt[2]:.4f}, {zt[3]:.4f})\n")
                if result.get('pred_pairs_raw'):
                    f.write(f"\n  Raw predictions (in zoomed view):\n")
                    for j, raw in enumerate(result.get('pred_pairs_raw', [])):
                        f.write(f"    Raw {j+1}: Person {raw.get('person_box', 'N/A')}, Object {raw.get('object_box', 'N/A')}\n")
                f.write("\n")
            
            # Predictions/Results
            if task_type == 'referring':
                f.write(f"RESULT:\n")
                f.write(f"  Predicted Action: {result.get('predicted', 'N/A')}\n")
                f.write(f"  Raw Answer: {result.get('raw_answer', 'N/A')}\n")
                gt = result.get('gt_action', '')
                pred = result.get('predicted', '')
                if gt and pred:
                    is_exact = pred.lower().strip() == gt.lower().strip()
                    f.write(f"  Exact Match: {'YES' if is_exact else 'NO'}\n")
            else:  # grounding
                f.write(f"PREDICTIONS ({len(result.get('predicted_pairs', []))}):\n")
                for j, pred in enumerate(result.get('predicted_pairs', [])):
                    f.write(f"  Pred {j+1}: Person {pred.get('person_box', 'N/A')}, Object {pred.get('object_box', 'N/A')}\n")
            
            # Parsed components
            f.write(f"\nPARSED COMPONENTS:\n")
            f.write(f"  Has <think>: {'YES' if result.get('thinking') else 'NO'}\n")
            f.write(f"  Has <tool_call>: {'YES' if result.get('has_tool_call') else 'NO'}\n")
            f.write(f"  Has <answer>: {'YES' if result.get('answer_tag') else 'NO'}\n")
            if result.get('thinking'):
                f.write(f"\n  Thinking content:\n")
                thinking = result.get('thinking', '')
                for line in thinking[:500].split('\n'):
                    f.write(f"    {line}\n")
                if len(thinking) > 500:
                    f.write(f"    ... (truncated)\n")
            
            f.write("\n\n")
        
        # Statistics summary at the end
        f.write("=" * 80 + "\n")
        f.write("STATISTICS SUMMARY:\n")
        f.write("=" * 80 + "\n")
        
        total = len(detailed_results)
        has_think = sum(1 for r in detailed_results if r.get('thinking'))
        has_tool = sum(1 for r in detailed_results if r.get('has_tool_call'))
        has_answer = sum(1 for r in detailed_results if r.get('answer_tag'))
        
        f.write(f"  Total samples: {total}\n")
        f.write(f"  Has <think>: {has_think} ({100*has_think/total:.1f}%)\n")
        f.write(f"  Has <tool_call>: {has_tool} ({100*has_tool/total:.1f}%)\n")
        f.write(f"  Has <answer>: {has_answer} ({100*has_answer/total:.1f}%)\n")
        
        if has_tool > 0:
            tool_counts = {}
            for r in detailed_results:
                for tc in r.get('tool_calls', []):
                    name = tc.get('name', 'unknown')
                    tool_counts[name] = tool_counts.get(name, 0) + 1
            f.write(f"\n  Tool usage breakdown:\n")
            for name, count in tool_counts.items():
                f.write(f"    {name}: {count}\n")
    
    logger.info(f"Log saved to {log_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Spatial Linking HOI Model")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "spatial_linking", "base"],
        help="Model type: 'auto' (detect), 'spatial_linking' (with spatial module), 'base' (standard Qwen3-VL with LoRA)"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to test data JSON file"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        choices=["referring", "grounding"],
        required=True,
        help="Task type to evaluate"
    )
    parser.add_argument(
        "--image_base_dir",
        type=str,
        default="/workspace/dataset",
        help="Base directory for images"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--visualize_attention",
        action="store_true",
        help="Generate attention visualizations"
    )
    parser.add_argument(
        "--viz_output_dir",
        type=str,
        default="./attention_viz",
        help="Output directory for attention visualizations"
    )
    parser.add_argument(
        "--save_detailed",
        action="store_true",
        help="Save detailed per-sample results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed per-sample output during evaluation"
    )
    
    # vLLM arguments
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vLLM server for inference instead of HuggingFace"
    )
    parser.add_argument(
        "--vllm_endpoint",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM server endpoint (default: http://localhost:8000/v1)"
    )
    parser.add_argument(
        "--vllm_model",
        type=str,
        default=None,
        help="Model name in vLLM server (defaults to model_path basename)"
    )
    
    args = parser.parse_args()
    
    # Determine vLLM model name
    if args.use_vllm and args.vllm_model is None:
        args.vllm_model = Path(args.model_path).name
    
    # Print configuration
    print("\n" + "=" * 70)
    print(" Spatial Linking HOI Model Evaluation")
    print("=" * 70)
    print(f"  Model:       {args.model_path}")
    print(f"  Model type:  {args.model_type}")
    print(f"  Test file:   {args.test_file}")
    print(f"  Task:        {args.task_type}")
    print(f"  Max samples: {args.max_samples or 'all'}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Verbose:     {args.verbose}")
    if args.use_vllm:
        print(f"  Mode:        vLLM Server")
        print(f"  Endpoint:    {args.vllm_endpoint}")
        print(f"  vLLM Model:  {args.vllm_model}")
    else:
        print(f"  Mode:        HuggingFace Local")
    print("=" * 70 + "\n")
    
    # Load test data
    test_data = load_test_data(args.test_file)
    
    # Evaluate based on mode
    if args.use_vllm:
        # vLLM mode - use vLLM server with real tool calling
        logger.info("Using vLLM server for evaluation...")
        
        if args.task_type == "referring":
            metrics, detailed_results = evaluate_referring_task_vllm(
                test_data=test_data,
                image_base_dir=args.image_base_dir,
                vllm_endpoint=args.vllm_endpoint,
                model_name=args.vllm_model,
                max_samples=args.max_samples,
                verbose=args.verbose,
            )
        else:
            metrics, detailed_results = evaluate_grounding_task_vllm(
                test_data=test_data,
                image_base_dir=args.image_base_dir,
                vllm_endpoint=args.vllm_endpoint,
                model_name=args.vllm_model,
                max_samples=args.max_samples,
                verbose=args.verbose,
            )
    else:
        # HuggingFace mode - load model locally
        logger.info("Using HuggingFace local model for evaluation...")
        model, processor = load_model(args.model_path, model_type=args.model_type)
        
        if args.task_type == "referring":
            metrics, detailed_results = evaluate_referring_task(
                model, processor, test_data,
                args.image_base_dir, args.max_samples,
                args.visualize_attention, args.viz_output_dir,
                args.verbose
            )
        else:
            metrics, detailed_results = evaluate_grounding_task(
                model, processor, test_data,
                args.image_base_dir, args.max_samples,
                args.visualize_attention, args.viz_output_dir,
                args.verbose
            )
    
    # Print results
    print_metrics(metrics, title=f"{args.task_type.upper()} Evaluation Results")
    
    # Print response pattern statistics
    print_response_statistics(detailed_results)
    
    # Save results
    save_results(
        metrics, detailed_results, 
        args.output_dir, args.task_type,
        args.save_detailed
    )
    
    # Log attention viz location
    if args.visualize_attention:
        logger.info(f"Attention visualizations saved to {args.viz_output_dir}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
