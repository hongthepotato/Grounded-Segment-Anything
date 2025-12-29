#!/usr/bin/env python3
"""
Export Grounding DINO to ONNX format.

Adapted from NVIDIA TAO Toolkit's export approach for TensorRT compatibility.
Reference: tao_pytorch_backend/nvidia_tao_pytorch/cv/grounding_dino/scripts/export.py

Usage:
    python scripts/export_grounding_dino_onnx.py \
        --checkpoint data/models/pretrained/groundingdino_swint_ogc.pth \
        --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --output data/models/groundingdino_swint.onnx \
        --input-width 800 \
        --input-height 800
"""

import os
import sys
import argparse
import logging

import torch
import torch.nn as nn
import numpy as np

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "GroundingDINO"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(config_path: str, checkpoint_path: str, device: str = "cuda"):
    """
    Load Grounding DINO model.
    
    Args:
        config_path: Path to config file
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.models import build_model
    from groundingdino.util.utils import clean_state_dict
    
    logger.info(f"Loading model from {checkpoint_path}")
    
    # Load config
    args = SLConfig.fromfile(config_path)
    args.device = device
    
    # Build model
    model = build_model(args)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    
    model.eval()
    model.to(device)
    
    return model, args


class GroundingDINOWrapper(nn.Module):
    """
    Wrapper for Grounding DINO that handles the forward pass for ONNX export.
    
    The original model's forward() is complex with optional arguments.
    This wrapper creates a clean interface for ONNX export.
    """
    
    def __init__(self, model, max_text_len: int = 256):
        super().__init__()
        self.model = model
        self.max_text_len = max_text_len
        
    def forward(
        self,
        samples: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        text_token_mask: torch.Tensor,
    ):
        """
        Forward pass for ONNX export.
        
        Args:
            samples: Image tensor [B, 3, H, W]
            input_ids: Token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            position_ids: Position IDs [B, seq_len]
            token_type_ids: Token type IDs [B, seq_len]
            text_token_mask: Text self-attention mask [B, seq_len, seq_len]
            
        Returns:
            pred_logits: [B, num_queries, max_text_len]
            pred_boxes: [B, num_queries, 4]
        """
        from groundingdino.util.misc import NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid
        
        # Encode text with BERT
        # Build tokenized dict for BERT
        tokenized_for_encoder = {
            "input_ids": input_ids,
            "attention_mask": text_token_mask,  # Use the full attention mask
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
        }
        
        bert_output = self.model.bert(**tokenized_for_encoder)
        
        # Get encoded text features
        encoded_text = self.model.feat_map(bert_output["last_hidden_state"])
        
        # Text token mask (1D mask for each token)
        text_token_mask_1d = attention_mask.bool()
        
        # Get text dict for transformer
        text_dict = {
            "encoded_text": encoded_text,
            "text_token_mask": text_token_mask_1d,
            "position_ids": position_ids,
            "text_self_attention_masks": text_token_mask,
        }
        
        # Convert samples to NestedTensor
        # Use the utility function from the original codebase
        nested_samples = nested_tensor_from_tensor_list(samples)
        
        # Image forward through backbone
        features, poss = self.model.backbone(nested_samples)
        
        # Multi-scale features
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.model.input_proj[l](src))
            masks.append(mask)
        
        # Additional feature levels if needed
        if self.model.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.model.num_feature_levels):
                if l == _len_srcs:
                    src = self.model.input_proj[l](features[-1].tensors)
                else:
                    src = self.model.input_proj[l](srcs[-1])
                m = nested_samples.mask
                mask = torch.nn.functional.interpolate(
                    m[None].float(), size=src.shape[-2:]
                ).to(torch.bool)[0]
                pos_l = self.model.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)
        
        # Transformer forward (no DN training for export)
        input_query_bbox = input_query_label = attn_mask = None
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.model.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict
        )
        
        # Deformable-DETR-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.model.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        
        # Class predictions
        outputs_class = torch.stack([
            layer_cls_embed(layer_hs, text_dict)
            for layer_cls_embed, layer_hs in zip(self.model.class_embed, hs)
        ])
        
        # Take last layer output
        pred_logits = outputs_class[-1]
        pred_boxes = outputs_coord_list[-1]
        
        return pred_logits, pred_boxes


def create_dummy_inputs(
    batch_size: int,
    input_height: int,
    input_width: int,
    text_prompt: str,
    model,
    device: str = "cuda",
):
    """
    Create dummy inputs for ONNX export.
    
    Args:
        batch_size: Batch size
        input_height: Input image height
        input_width: Input image width
        text_prompt: Text prompt for tokenization
        model: Model (to get tokenizer)
        device: Device
        
    Returns:
        Tuple of dummy inputs
    """
    from groundingdino.util.utils import get_phrases_from_posmap
    
    # Create dummy image
    dummy_image = torch.randn(batch_size, 3, input_height, input_width, device=device)
    
    # Tokenize text
    tokenizer = model.tokenizer
    tokenized = tokenizer(
        [text_prompt] * batch_size,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    )
    
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    token_type_ids = tokenized.get("token_type_ids", torch.zeros_like(input_ids)).to(device)
    
    # Position IDs
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Text self-attention mask (causal or full)
    text_token_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1).bool()
    
    return (
        dummy_image,
        input_ids,
        attention_mask,
        position_ids,
        token_type_ids,
        text_token_mask,
    )


def export_onnx(
    model,
    dummy_inputs,
    output_path: str,
    batch_size: int = 1,
    opset_version: int = 17,
    dynamic_batch: bool = True,
):
    """
    Export model to ONNX.
    
    Args:
        model: Model to export
        dummy_inputs: Tuple of dummy input tensors
        output_path: Output ONNX file path
        batch_size: Batch size
        opset_version: ONNX opset version
        dynamic_batch: Whether to use dynamic batch size
    """
    import onnx
    
    input_names = [
        "inputs",
        "input_ids", 
        "attention_mask",
        "position_ids",
        "token_type_ids",
        "text_token_mask",
    ]
    output_names = ["pred_logits", "pred_boxes"]
    
    if dynamic_batch:
        dynamic_axes = {
            "inputs": {0: "batch_size"},
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "position_ids": {0: "batch_size"},
            "token_type_ids": {0: "batch_size"},
            "text_token_mask": {0: "batch_size"},
            "pred_logits": {0: "batch_size"},
            "pred_boxes": {0: "batch_size"},
        }
    else:
        dynamic_axes = None
    
    logger.info(f"Exporting to ONNX: {output_path}")
    logger.info(f"  Opset version: {opset_version}")
    logger.info(f"  Dynamic batch: {dynamic_batch}")
    
    torch.onnx.export(
        model,
        dummy_inputs,
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
        verbose=False,
    )
    
    # Verify ONNX model
    logger.info("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verification passed!")
    
    return output_path


def optimize_onnx(onnx_path: str):
    """
    Optimize ONNX model using onnx_graphsurgeon.
    
    Args:
        onnx_path: Path to ONNX model
    """
    try:
        import onnx
        import onnx_graphsurgeon as gs
        
        logger.info("Optimizing ONNX graph...")
        
        graph = gs.import_onnx(onnx.load(onnx_path))
        
        # Constant folding
        graph.fold_constants(size_threshold=1024 * 1024 * 1024)  # 1GB threshold
        graph.cleanup().toposort()
        
        onnx.save(gs.export_onnx(graph), onnx_path)
        logger.info("ONNX optimization complete!")
        
    except ImportError:
        logger.warning("onnx_graphsurgeon not installed. Skipping optimization.")
        logger.warning("Install with: pip install onnx_graphsurgeon")


def main():
    parser = argparse.ArgumentParser(description="Export Grounding DINO to ONNX")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/models/pretrained/groundingdino_swint_ogc.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help="Path to model config",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/models/groundingdino_swint.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=800,
        help="Input image width",
    )
    parser.add_argument(
        "--input-height",
        type=int,
        default=800,
        help="Input image height",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (use -1 for dynamic)",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        default="the running dog .",
        help="Sample text prompt for export",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for export (cpu recommended to avoid CUDA kernel issues)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimize ONNX graph with graphsurgeon",
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not os.path.exists(args.config):
        logger.error(f"Config not found: {args.config}")
        sys.exit(1)
    
    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load model
    model, config = load_model(args.config, args.checkpoint, args.device)
    
    # Create wrapper for export
    logger.info("Creating export wrapper...")
    wrapped_model = GroundingDINOWrapper(model)
    wrapped_model.eval()
    
    # Create dummy inputs
    batch_size = 1 if args.batch_size == -1 else args.batch_size
    dummy_inputs = create_dummy_inputs(
        batch_size=batch_size,
        input_height=args.input_height,
        input_width=args.input_width,
        text_prompt=args.text_prompt,
        model=model,
        device=args.device,
    )
    
    # Export to ONNX
    dynamic_batch = args.batch_size == -1
    export_onnx(
        model=wrapped_model,
        dummy_inputs=dummy_inputs,
        output_path=args.output,
        batch_size=batch_size,
        opset_version=args.opset_version,
        dynamic_batch=dynamic_batch,
    )
    
    # Optimize if requested
    if args.optimize:
        optimize_onnx(args.output)
    
    logger.info(f"Export complete! ONNX model saved to: {args.output}")
    logger.info("")
    logger.info("To convert to TensorRT, run:")
    logger.info(f"  trtexec --onnx={args.output} --saveEngine={args.output.replace('.onnx', '.trt')} --fp16")


if __name__ == "__main__":
    main()

