"""
Low-Rank Adaptation (LoRA) for efficient fine-tuning.

LoRA reduces the number of trainable parameters by up to 100x while maintaining
model performance. This allows:
- 60% reduction in VRAM usage
- Faster training
- Larger batch sizes
- Ability to fine-tune larger models

Based on: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import math


class LoRALinear(nn.Module):
    """
    LoRA-adapted Linear layer.
    
    Replaces W with W + BA where:
    - W: Original weight matrix (frozen)
    - B: Low-rank matrix (trainable)
    - A: Low-rank matrix (trainable)
    - rank: Rank of B and A (typically 1-16)
    """
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 rank: int = 8,
                 alpha: float = 16.0,
                 dropout: float = 0.0,
                 merge_weights: bool = False):
        """
        Initialize LoRA Linear layer.
        
        Args:
            in_features: Input feature size
            out_features: Output feature size
            rank: LoRA rank (default: 8)
            alpha: LoRA alpha scaling factor (default: 16.0)
            dropout: Dropout rate for LoRA (default: 0.0)
            merge_weights: Whether to merge LoRA weights into base weights
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Base weight (will be set from original layer)
        self.register_buffer('weight', None)
        self.register_buffer('bias', None)
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.merged = False
        self.merge_weights = merge_weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.
        
        Args:
            x: Input tensor (batch, ..., in_features)
            
        Returns:
            Output tensor (batch, ..., out_features)
        """
        if self.weight is None:
            raise ValueError("Base weight not set. Call set_base_weight() first.")
        
        # Base forward pass
        base_output = F.linear(x, self.weight, self.bias)
        
        # LoRA forward pass
        x_dropout = self.lora_dropout(x)
        lora_output = F.linear(
            F.linear(x_dropout, self.lora_A.t()),
            self.lora_B.t()
        ) * self.scaling
        
        return base_output + lora_output
    
    def set_base_weight(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """
        Set base weight from original layer.
        
        Args:
            weight: Original weight matrix
            bias: Original bias vector (optional)
        """
        self.weight = weight.detach().clone()
        if bias is not None:
            self.bias = bias.detach().clone()
        else:
            self.bias = None
    
    def merge_weights(self):
        """Merge LoRA weights into base weights."""
        if self.merged:
            return
        
        if self.weight is None:
            raise ValueError("Base weight not set.")
        
        # Merge: W_new = W_old + (B @ A) * scaling
        lora_weight = (self.lora_B @ self.lora_A) * self.scaling
        self.weight.data += lora_weight
        self.merged = True
    
    def unmerge_weights(self):
        """Unmerge LoRA weights from base weights."""
        if not self.merged:
            return
        
        if self.weight is None:
            raise ValueError("Base weight not set.")
        
        # Unmerge: W_old = W_new - (B @ A) * scaling
        lora_weight = (self.lora_B @ self.lora_A) * self.scaling
        self.weight.data -= lora_weight
        self.merged = False
    
    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return self.lora_A.numel() + self.lora_B.numel()
    
    def get_total_params(self) -> int:
        """Get total number of parameters (base + LoRA)."""
        base_params = self.weight.numel() if self.weight is not None else 0
        if self.bias is not None:
            base_params += self.bias.numel()
        return base_params + self.get_trainable_params()


def apply_lora_to_linear(module: nn.Module,
                         target_modules: List[str],
                         rank: int = 8,
                         alpha: float = 16.0,
                         dropout: float = 0.0) -> Dict[str, LoRALinear]:
    """
    Apply LoRA to specified linear layers in a module.
    
    Args:
        module: PyTorch module to apply LoRA to
        target_modules: List of module names to apply LoRA to (e.g., ['W_q', 'W_k'])
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        dropout: LoRA dropout rate
        
    Returns:
        Dictionary mapping module names to LoRALinear instances
    """
    lora_modules = {}
    
    for name, child in module.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(child, nn.Linear):
                # Create LoRA layer
                lora_layer = LoRALinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout
                )
                
                # Set base weights
                lora_layer.set_base_weight(child.weight, child.bias)
                
                # Replace original layer with LoRA layer
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = module.get_submodule(parent_name)
                else:
                    parent = module
                
                setattr(parent, child_name, lora_layer)
                lora_modules[name] = lora_layer
    
    return lora_modules


def freeze_base_model(model: nn.Module, lora_modules: Dict[str, LoRALinear]):
    """
    Freeze base model weights, keep only LoRA weights trainable.
    
    Args:
        model: Model to freeze
        lora_modules: Dictionary of LoRA modules
    """
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze LoRA parameters
    for lora_module in lora_modules.values():
        lora_module.lora_A.requires_grad = True
        lora_module.lora_B.requires_grad = True


def get_lora_trainable_params(model: nn.Module, lora_modules: Dict[str, LoRALinear]) -> int:
    """
    Get total number of trainable parameters (LoRA only).
    
    Args:
        model: Model
        lora_modules: Dictionary of LoRA modules
        
    Returns:
        Total number of trainable parameters
    """
    total = 0
    for lora_module in lora_modules.values():
        total += lora_module.get_trainable_params()
    return total


def get_model_total_params(model: nn.Module) -> int:
    """
    Get total number of parameters in model.
    
    Args:
        model: Model
        
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def print_lora_summary(model: nn.Module, lora_modules: Dict[str, LoRALinear]):
    """
    Print summary of LoRA adaptation.
    
    Args:
        model: Model
        lora_modules: Dictionary of LoRA modules
    """
    total_params = get_model_total_params(model)
    trainable_params = get_lora_trainable_params(model, lora_modules)
    reduction_ratio = (1 - trainable_params / total_params) * 100
    
    print("=" * 60)
    print("LoRA ADAPTATION SUMMARY")
    print("=" * 60)
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable parameters (LoRA): {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print(f"Parameter reduction: {reduction_ratio:.2f}%")
    print(f"LoRA modules: {len(lora_modules)}")
    print("=" * 60)

