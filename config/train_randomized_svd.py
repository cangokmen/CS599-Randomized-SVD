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

# Enable SVD compression with individual matrix and algorithm controls
config.use_svd = True

# Individual matrix compression controls
config.svd_apply_to_q = False    # Disable Q compression by default
config.svd_apply_to_k = False    # Disable K compression by default  
config.svd_apply_to_v = True     # Enable V compression (most beneficial)

# Individual randomized SVD controls (per matrix type)
config.use_randomized_svd_q = False    # Standard SVD for Q
config.use_randomized_svd_k = False    # Standard SVD for K
config.use_randomized_svd_v = True     # Randomized SVD for V

# Global fallback (used when individual settings not specified)
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

def get_mixed_compression_config():
    """
    Configuration with mixed standard and randomized SVD
    Q: Standard SVD, K: Standard SVD, V: Randomized SVD
    """
    mixed_config = config
    mixed_config.svd_apply_to_q = True
    mixed_config.svd_apply_to_k = True
    mixed_config.svd_apply_to_v = True
    mixed_config.use_randomized_svd_q = False   # Standard SVD
    mixed_config.use_randomized_svd_k = False   # Standard SVD
    mixed_config.use_randomized_svd_v = True    # Randomized SVD
    mixed_config.svd_rank = 32
    return mixed_config

def get_all_randomized_config():
    """
    Configuration with randomized SVD for all matrices
    """
    all_rsvd_config = config
    all_rsvd_config.svd_apply_to_q = True
    all_rsvd_config.svd_apply_to_k = True
    all_rsvd_config.svd_apply_to_v = True
    all_rsvd_config.use_randomized_svd_q = True
    all_rsvd_config.use_randomized_svd_k = True
    all_rsvd_config.use_randomized_svd_v = True
    all_rsvd_config.svd_rank = 32
    all_rsvd_config.svd_oversampling = 10
    all_rsvd_config.svd_power_iter = 1
    return all_rsvd_config

def get_all_standard_config():
    """
    Configuration with standard SVD for all matrices
    """
    all_std_config = config
    all_std_config.svd_apply_to_q = True
    all_std_config.svd_apply_to_k = True
    all_std_config.svd_apply_to_v = True
    all_std_config.use_randomized_svd_q = False
    all_std_config.use_randomized_svd_k = False
    all_std_config.use_randomized_svd_v = False
    all_std_config.svd_rank = 32
    return all_std_config

def get_research_comparison_config():
    """
    Configuration for comparing standard vs randomized SVD
    Q: Standard, K: Randomized, V: Randomized (for comparison)
    """
    research_config = config
    research_config.svd_apply_to_q = True
    research_config.svd_apply_to_k = True
    research_config.svd_apply_to_v = True
    research_config.use_randomized_svd_q = False   # Standard for comparison
    research_config.use_randomized_svd_k = True    # Randomized
    research_config.use_randomized_svd_v = True    # Randomized
    research_config.svd_rank = 32
    research_config.svd_oversampling = 10
    research_config.svd_power_iter = 1
    return research_config

def get_qkv_balanced_config():
    """
    Configuration for balanced Q, K, V compression
    """
    qkv_config = config
    qkv_config.svd_apply_to_q = True
    qkv_config.svd_apply_to_k = True
    qkv_config.svd_apply_to_v = True
    qkv_config.svd_rank = 32
    qkv_config.svd_oversampling = 10
    qkv_config.svd_power_iter = 1
    return qkv_config

def get_kv_only_config():
    """
    Configuration for K, V compression (preserves query precision)
    """
    kv_config = config
    kv_config.svd_apply_to_q = False  # Preserve query precision
    kv_config.svd_apply_to_k = True
    kv_config.svd_apply_to_v = True
    kv_config.svd_rank = 32
    return kv_config

def get_aggressive_compression_config():
    """
    Configuration for maximum compression across all matrices
    """
    aggressive_config = config
    aggressive_config.svd_apply_to_q = True
    aggressive_config.svd_apply_to_k = True
    aggressive_config.svd_apply_to_v = True
    aggressive_config.svd_rank = 16          # Lower rank for max compression
    aggressive_config.svd_oversampling = 5   # Lower oversampling for speed
    aggressive_config.svd_power_iter = 0     # No power iterations for speed
    return aggressive_config

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
    Estimate memory and computational savings from mixed SVD/randomized SVD
    """
    head_size = config.n_embd // config.n_head
    rank = config.svd_rank
    
    # Count matrices and their compression types
    compression_summary = {
        'Q': 'none',
        'K': 'none', 
        'V': 'none'
    }
    
    if config.svd_apply_to_q:
        compression_summary['Q'] = 'randomized_svd' if config.use_randomized_svd_q else 'standard_svd'
    if config.svd_apply_to_k:
        compression_summary['K'] = 'randomized_svd' if config.use_randomized_svd_k else 'standard_svd'
    if config.svd_apply_to_v:
        compression_summary['V'] = 'randomized_svd' if config.use_randomized_svd_v else 'standard_svd'
    
    num_compressed_matrices = sum([config.svd_apply_to_q, config.svd_apply_to_k, config.svd_apply_to_v])
    num_randomized_matrices = sum([
        config.svd_apply_to_q and config.use_randomized_svd_q,
        config.svd_apply_to_k and config.use_randomized_svd_k,
        config.svd_apply_to_v and config.use_randomized_svd_v
    ])
    
    if num_compressed_matrices == 0:
        print("No matrices selected for compression!")
        return
    
    # Memory compression ratio per matrix
    original_params = head_size * head_size
    compressed_params = head_size * rank + rank + rank * head_size
    memory_ratio_per_matrix = original_params / compressed_params
    
    # Overall memory impact
    total_original = 3 * original_params  # Q, K, V matrices
    total_compressed = (3 - num_compressed_matrices) * original_params + num_compressed_matrices * compressed_params
    overall_memory_ratio = total_original / total_compressed
    
    # Computational analysis
    oversampling = config.svd_oversampling
    m, n = head_size, head_size
    k = rank
    p = oversampling
    
    standard_ops_per_matrix = m * n * min(m, n)
    randomized_ops_per_matrix = m * n * (k + p) + m * k**2 + k**2 * n
    
    print(f"Configuration Summary:")
    print(f"  Head size: {head_size}")
    print(f"  Target rank: {rank}")
    print(f"  Oversampling: {oversampling}")
    print(f"  Power iterations: {config.svd_power_iter}")
    print(f"")
    print(f"Compression Selection:")
    for matrix, comp_type in compression_summary.items():
        if comp_type == 'none':
            print(f"  {matrix}: ✗ (no compression)")
        elif comp_type == 'standard_svd':
            print(f"  {matrix}: ✓ Standard SVD")
        elif comp_type == 'randomized_svd':
            print(f"  {matrix}: ✓ Randomized SVD")
    
    print(f"  Total compressed: {num_compressed_matrices}/3")
    print(f"  Randomized: {num_randomized_matrices}/{num_compressed_matrices if num_compressed_matrices > 0 else 1}")
    print(f"")
    print(f"Estimated Performance:")
    print(f"  Per-matrix memory compression: {memory_ratio_per_matrix:.2f}x")
    print(f"  Overall memory compression: {overall_memory_ratio:.2f}x")
    
    if num_randomized_matrices > 0:
        randomized_speedup = standard_ops_per_matrix / randomized_ops_per_matrix
        print(f"  Standard SVD ops per matrix: {standard_ops_per_matrix:,}")
        print(f"  Randomized SVD ops per matrix: {randomized_ops_per_matrix:,}")
        print(f"  Randomized speedup per matrix: {randomized_speedup:.2f}x")
    
    print(f"  Rank ratio: {rank / head_size:.2%}")

# =============================================================================
# Configuration Validation
# =============================================================================

def validate_config(cfg):
    """
    Validate configuration for mixed SVD/randomized SVD setup
    """
    head_size = cfg.n_embd // cfg.n_head
    
    # Check if any compression is enabled
    compression_enabled = cfg.use_svd and any([
        cfg.svd_apply_to_q,
        cfg.svd_apply_to_k,
        cfg.svd_apply_to_v
    ])
    
    if cfg.use_svd and not compression_enabled:
        print("Warning: SVD enabled but no matrices selected for compression")
    
    # Check rank is reasonable
    if cfg.svd_rank is not None:
        assert cfg.svd_rank > 0, "SVD rank must be positive"
        assert cfg.svd_rank <= head_size, f"SVD rank ({cfg.svd_rank}) cannot exceed head size ({head_size})"
        
        if cfg.svd_rank > head_size * 0.75:
            print(f"Warning: High rank ({cfg.svd_rank}/{head_size}) - limited compression benefit")
        
        if cfg.svd_rank < head_size * 0.2:
            print(f"Warning: Very low rank ({cfg.svd_rank}/{head_size}) - may hurt quality significantly")
    
    # Check oversampling for randomized SVD matrices
    randomized_matrices = [
        (cfg.svd_apply_to_q and cfg.use_randomized_svd_q, 'Q'),
        (cfg.svd_apply_to_k and cfg.use_randomized_svd_k, 'K'), 
        (cfg.svd_apply_to_v and cfg.use_randomized_svd_v, 'V')
    ]
    
    num_randomized = sum(enabled for enabled, _ in randomized_matrices)
    
    if num_randomized > 0:
        if cfg.svd_oversampling < 5:
            print(f"Warning: Low oversampling ({cfg.svd_oversampling}) - may reduce accuracy for randomized SVD")
        elif cfg.svd_oversampling > 20:
            print(f"Warning: High oversampling ({cfg.svd_oversampling}) - diminishing returns")
        
        if cfg.svd_power_iter > 3:
            print(f"Warning: Many power iterations ({cfg.svd_power_iter}) - diminishing returns")
    
    # Matrix-specific recommendations
    if compression_enabled:
        print(f"\nCompression Configuration Analysis:")
        
        matrix_configs = [
            ('Q', cfg.svd_apply_to_q, cfg.use_randomized_svd_q),
            ('K', cfg.svd_apply_to_k, cfg.use_randomized_svd_k),
            ('V', cfg.svd_apply_to_v, cfg.use_randomized_svd_v)
        ]
        
        for matrix, enabled, randomized in matrix_configs:
            if enabled:
                comp_type = 'Randomized SVD' if randomized else 'Standard SVD'
                print(f"  {matrix} matrix: {comp_type}")
            else:
                print(f"  {matrix} matrix: No compression")
        
        # Configuration recommendations
        if cfg.svd_apply_to_q and cfg.svd_apply_to_k and cfg.svd_apply_to_v:
            print("Note: Compressing all Q, K, V - monitor attention quality carefully")
        
        if num_randomized == 0 and compression_enabled:
            print("Note: Using standard SVD only - consider randomized SVD for larger matrices")
        elif num_randomized == sum([cfg.svd_apply_to_q, cfg.svd_apply_to_k, cfg.svd_apply_to_v]):
            print("Note: Using randomized SVD for all compressed matrices")
        else:
            print("Note: Mixed standard/randomized SVD configuration")
    
    print("✅ Configuration validation passed")

if __name__ == "__main__":
    print("Mixed Standard/Randomized SVD Configuration for Q, K, V Matrices")
    print("=" * 70)
    
    # Show default configuration
    print("\nDefault Configuration:")
    print(f"  use_svd: {config.use_svd}")
    print(f"  svd_rank: {config.svd_rank}")
    print(f"  svd_oversampling: {config.svd_oversampling}")
    print(f"  svd_power_iter: {config.svd_power_iter}")
    print(f"")
    print(f"Matrix Compression:")
    print(f"  Q compression: {config.svd_apply_to_q} ({'Randomized' if config.use_randomized_svd_q else 'Standard'} SVD)")
    print(f"  K compression: {config.svd_apply_to_k} ({'Randomized' if config.use_randomized_svd_k else 'Standard'} SVD)")
    print(f"  V compression: {config.svd_apply_to_v} ({'Randomized' if config.use_randomized_svd_v else 'Standard'} SVD)")
    
    # Validate and show estimates
    print("\n")
    validate_config(config)
    print("\n")
    estimate_compression_ratio()