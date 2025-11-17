"""
Configuration for training GPT with Tropp's Randomized SVD compression on V matrices

This configuration demonstrates how to enable and tune randomized SVD compression
following the algorithms from Halko, Martinsson, and Tropp (2011).
"""

from src.mla_gpt.model.model import GPTConfig
import torch

# =============================================================================
# Base Configuration
# =============================================================================

# Model architecture (GPT-2 Small equivalent)
config = GPTConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=12,
    n_head=12, 
    n_embd=768,
    dropout=0.1,
    bias=True
)

# =============================================================================
# Tropp's Randomized SVD Configuration
# =============================================================================

# Enable SVD compression on V matrices
config.use_svd = True
config.use_randomized_svd = True

# Target rank for compression (key parameter for compression ratio)
# Rule of thumb: start with head_size // 2 to head_size // 4
head_size = config.n_embd // config.n_head  # 64 for GPT-2 small
config.svd_rank = 32  # 50% compression

# Tropp's algorithm parameters (following paper recommendations)
config.svd_oversampling = 10    # p parameter: 5-10 for good balance
config.svd_power_iter = 1       # q parameter: 1-2 for most cases

# =============================================================================
# Training Configuration
# =============================================================================

# Training hyperparameters
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# Evaluation
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False

# Checkpointing
ckpt_path = 'ckpt_randomized_svd.pt'
always_save_checkpoint = True

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# =============================================================================
# Advanced Randomized SVD Configurations
# =============================================================================

# Configuration for different scenarios:

def get_high_compression_config():
    """
    Configuration for maximum compression (lower quality, higher speed)
    """
    high_compression_config = config
    high_compression_config.svd_rank = 16          # 25% of original
    high_compression_config.svd_oversampling = 5   # Lower oversampling for speed
    high_compression_config.svd_power_iter = 0     # No power iterations for speed
    return high_compression_config

def get_high_quality_config():
    """
    Configuration for maximum quality (moderate compression, best accuracy)
    """
    high_quality_config = config
    high_quality_config.svd_rank = 48              # 75% of original
    high_quality_config.svd_oversampling = 15      # Higher oversampling for accuracy
    high_quality_config.svd_power_iter = 2         # More power iterations for accuracy
    return high_quality_config

def get_balanced_config():
    """
    Configuration for balanced compression and quality (recommended default)
    """
    balanced_config = config
    balanced_config.svd_rank = 32                  # 50% of original
    balanced_config.svd_oversampling = 10          # Standard oversampling
    balanced_config.svd_power_iter = 1             # One power iteration
    return balanced_config

def get_adaptive_rank_config(sequence_length):
    """
    Adaptive rank based on sequence length (for variable length sequences)
    
    Args:
        sequence_length: Current sequence length
        
    Returns:
        Config with adapted rank
    """
    adaptive_config = config
    
    # Adaptive rank: more compression for longer sequences
    if sequence_length <= 128:
        adaptive_config.svd_rank = 48       # Light compression
    elif sequence_length <= 512:
        adaptive_config.svd_rank = 32       # Moderate compression  
    else:
        adaptive_config.svd_rank = 24       # Heavy compression
    
    # Adjust oversampling based on rank
    adaptive_config.svd_oversampling = min(10, adaptive_config.svd_rank // 2)
    
    return adaptive_config

# =============================================================================
# Performance Estimation
# =============================================================================

def estimate_compression_ratio():
    """
    Estimate memory and computational savings from randomized SVD
    """
    head_size = config.n_embd // config.n_head
    rank = config.svd_rank
    
    # Memory compression ratio for V matrices
    original_params = head_size * head_size
    compressed_params = head_size * rank + rank + rank * head_size
    memory_ratio = original_params / compressed_params
    
    # Computational speedup (approximate)
    # Standard SVD: O(m * n * min(m,n))
    # Randomized SVD: O(m * n * (k+p)) + O(m * k^2) + O(k^2 * n)
    
    oversampling = config.svd_oversampling
    m, n = head_size, head_size  # Square matrices for attention
    k = rank
    p = oversampling
    
    standard_ops = m * n * min(m, n)
    randomized_ops = m * n * (k + p) + m * k**2 + k**2 * n
    computational_speedup = standard_ops / randomized_ops
    
    print(f"Configuration Summary:")
    print(f"  Head size: {head_size}")
    print(f"  Target rank: {rank}")
    print(f"  Oversampling: {oversampling}")
    print(f"  Power iterations: {config.svd_power_iter}")
    print(f"")
    print(f"Estimated Performance:")
    print(f"  Memory compression ratio: {memory_ratio:.2f}x")
    print(f"  Computational speedup: {computational_speedup:.2f}x")
    print(f"  Rank ratio: {rank / head_size:.2%}")

# =============================================================================
# Configuration Validation
# =============================================================================

def validate_config(cfg):
    """
    Validate that the randomized SVD configuration is reasonable
    """
    head_size = cfg.n_embd // cfg.n_head
    
    # Check rank is reasonable
    if cfg.svd_rank is not None:
        assert cfg.svd_rank > 0, "SVD rank must be positive"
        assert cfg.svd_rank <= head_size, f"SVD rank ({cfg.svd_rank}) cannot exceed head size ({head_size})"
        
        if cfg.svd_rank > head_size * 0.75:
            print(f"Warning: High rank ({cfg.svd_rank}/{head_size}) - limited compression benefit")
        
        if cfg.svd_rank < head_size * 0.2:
            print(f"Warning: Very low rank ({cfg.svd_rank}/{head_size}) - may hurt quality significantly")
    
    # Check oversampling
    if cfg.svd_oversampling < 5:
        print(f"Warning: Low oversampling ({cfg.svd_oversampling}) - may reduce accuracy")
    elif cfg.svd_oversampling > 20:
        print(f"Warning: High oversampling ({cfg.svd_oversampling}) - diminishing returns")
    
    # Check power iterations
    if cfg.svd_power_iter > 3:
        print(f"Warning: Many power iterations ({cfg.svd_power_iter}) - diminishing returns")
    
    print("✅ Configuration validation passed")

if __name__ == "__main__":
    print("Tropp's Randomized SVD Configuration")
    print("=" * 50)
    
    # Show default configuration
    print("\nDefault Configuration:")
    print(f"  use_svd: {config.use_svd}")
    print(f"  use_randomized_svd: {config.use_randomized_svd}")
    print(f"  svd_rank: {config.svd_rank}")
    print(f"  svd_oversampling: {config.svd_oversampling}")
    print(f"  svd_power_iter: {config.svd_power_iter}")
    
    # Validate and show estimates
    print("\n")
    validate_config(config)
    print("\n")
    estimate_compression_ratio()