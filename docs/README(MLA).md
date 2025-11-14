# Multi-Head Latent Attention (MLA) Implementation

This document describes the Multi-Head Latent Attention (MLA) implementation in the CS599 Randomized SVD project.

## Overview

Multi-Head Latent Attention (MLA) is an efficient alternative to standard multi-head attention that compresses key-value representations into a lower-dimensional latent space. This reduces both memory usage and computational requirements while maintaining good performance.

## Key Features

### 1. Compressed Key-Value Representations
- Keys and values are projected to a lower-dimensional latent space
- Latent dimension can be configured (default: `head_size // 2`)
- Significant memory savings during inference (especially for KV caching)

### 2. Learnable Compression/Decompression
- Learnable projection to latent space: `kv_latent_proj`
- Separate expansion matrices for keys and values: `k_expand`, `v_expand`
- Maintains expressiveness through learned transformations

### 3. SVD Compatibility
- Can be combined with existing SVD compression
- Applies SVD to the expanded value matrices
- Provides additional compression when enabled

### 4. Drop-in Replacement
- Compatible with existing GPT model architecture
- Same input/output interface as `CausalSelfAttention`
- Configurable via `GPTConfig`

## Architecture

```
Input (B, T, n_embd)
    |
    ├── Query Projection → (B, nh, T, head_size)
    |
    └── KV Latent Projection → (B, T, 2*nh*latent_dim)
            |
            ├── K Latent → K Expand → (B, nh, T, head_size)  
            └── V Latent → V Expand → (B, nh, T, head_size)
                                |
                                └── [Optional SVD] → (B, nh, T, head_size)
                                        |
                                        ↓
                                    Attention(Q, K, V)
                                        |
                                        ↓
                                Output Projection → (B, T, n_embd)
```

## Configuration

Add MLA configuration to your training config file:

```python
# Enable MLA
use_mla = True              # Use MLA instead of standard attention
mla_latent_dim = 32         # Latent dimension (None for default: head_size // 2)

# Optional: Combine with SVD
use_svd = True              # Enable SVD on value matrices
svd_rank = 16               # SVD compression rank
```

## Usage Examples

### 1. Basic MLA Model

```python
from mla_gpt.model import GPT, GPTConfig

config = GPTConfig(
    n_embd=768,
    n_head=12, 
    n_layer=12,
    use_mla=True,           # Enable MLA
    mla_latent_dim=128      # Latent dimension
)

model = GPT(config)
```

### 2. MLA with SVD Compression

```python
config = GPTConfig(
    n_embd=768,
    n_head=12,
    n_layer=12,
    use_mla=True,           # Enable MLA
    mla_latent_dim=128,     # Latent dimension
    use_svd=True,           # Enable SVD
    svd_rank=64             # SVD rank
)

model = GPT(config)
```

### 3. Training with MLA

```bash
# Use the provided config file
python -m mla_gpt.cli.train config/train_mla_test.py

# Or override parameters
python -m mla_gpt.cli.train config/train_shakespeare_char.py \
    --use_mla=True \
    --mla_latent_dim=32 \
    --use_svd=True \
    --svd_rank=16
```

## Performance Analysis

### Memory Efficiency

The MLA implementation provides several methods to analyze efficiency:

```python
from mla_gpt.model.attention import MultiHeadLatentAttention

# Create MLA layer
mla = MultiHeadLatentAttention(config)

# Analyze parameter compression
compression_ratio = mla.get_compression_ratio()
print(f"Parameter compression: {compression_ratio:.2f}x")

# Analyze memory savings for KV caching
memory_savings = mla.get_memory_savings(sequence_length=1024)
print(f"KV cache memory savings: {memory_savings:.2f}x")
```

### Expected Benefits

1. **Parameter Reduction**: 20-50% reduction in attention parameters
2. **Memory Savings**: 2-4x reduction in KV cache memory during inference
3. **Computational Efficiency**: Reduced FLOPs for attention computation
4. **Scalability**: Benefits increase with longer sequences and larger models

## Implementation Details

### Key Components

1. **`q_proj`**: Standard query projection (full dimensionality)
2. **`kv_latent_proj`**: Joint key-value projection to latent space
3. **`k_expand`**, **`v_expand`**: Separate expansion matrices
4. **`c_proj`**: Output projection (standard)

### Latent Dimension Selection

- **Default**: `head_size // 2` (provides good balance)
- **Aggressive**: `head_size // 4` (maximum compression)
- **Conservative**: `head_size // 1.5` (minimal compression)
- **Custom**: Any dimension < `head_size`

### SVD Integration

When both MLA and SVD are enabled:
1. Input → MLA latent compression
2. Latent → expanded key/value matrices  
3. SVD compression applied to expanded values
4. Standard attention computation

## Testing

Run the test suite to verify the implementation:

```bash
cd tests
python test_mla_implementation.py
```

Test coverage includes:
- Initialization and parameter validation
- Forward pass correctness
- Shape compatibility with standard attention
- Compression ratio calculations
- SVD integration
- Various latent dimensions

## Examples and Demos

### 1. Compare Attention Mechanisms

```bash
cd examples
python example_mla_usage.py
```

This script compares:
- Standard attention
- MLA with different latent dimensions
- MLA + SVD combinations
- Parameter counts and memory usage

### 2. Training Comparison

```bash
# Train standard model
python -m mla_gpt.cli.train config/train_shakespeare_char.py \
    --out_dir=out-standard

# Train MLA model  
python -m mla_gpt.cli.train config/train_mla_test.py \
    --out_dir=out-mla

# Compare results
python -m mla_gpt.cli.sample --out_dir=out-standard
python -m mla_gpt.cli.sample --out_dir=out-mla
```

## Configuration Reference

### GPTConfig Parameters

```python
@dataclass
class GPTConfig:
    # ... standard parameters ...
    
    # MLA Configuration
    use_mla: bool = False           # Enable MLA
    mla_latent_dim: int = None      # Latent dimension (default: head_size // 2)
    
    # SVD Configuration (compatible with MLA)
    use_svd: bool = False           # Enable SVD compression
    svd_rank: int = None            # SVD rank (default: full rank)
```

### Training Config Example

```python
# config/train_mla_custom.py

# Model architecture
n_layer = 12
n_head = 12  
n_embd = 768

# MLA settings
use_mla = True
mla_latent_dim = 192    # 768 // 12 // 2 = 32, but we use 192 for less aggressive compression

# Optional SVD
use_svd = True
svd_rank = 32

# ... other training parameters ...
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the `__init__.py` files are present and correctly configured
2. **Shape Mismatches**: Verify `mla_latent_dim` is appropriate for your model size
3. **Memory Issues**: Try reducing `mla_latent_dim` or disabling SVD
4. **Performance Regression**: Increase `mla_latent_dim` or adjust training hyperparameters

### Debug Mode

Enable debug prints by modifying the MLA implementation:

```python
# In multihead_latent_attention.py
def forward(self, x):
    print(f"MLA Input shape: {x.shape}")
    print(f"Latent dim: {self.latent_dim}")
    # ... rest of forward pass
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [GPT-2](https://openai.com/research/better-language-models) - Base model architecture
- [Efficient Attention Mechanisms](https://arxiv.org/abs/2009.14794) - Survey of efficient attention variants

## Contributing

To extend the MLA implementation:

1. Modify `multihead_latent_attention.py` for core changes
2. Update `test_mla_implementation.py` with new tests
3. Add configuration options to `GPTConfig`
4. Update documentation and examples