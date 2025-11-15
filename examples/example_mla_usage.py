"""
Example demonstrating Multi-Head Latent Attention (MLA) usage

This example shows how to:
1. Create a model with MLA instead of standard attention
2. Compare memory usage and parameters between MLA and standard attention
3. Test different latent dimensions for MLA
4. Combine MLA with SVD compression for maximum efficiency
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import torch
    import torch.nn as nn
    from mla_gpt.model import GPT, GPTConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure PyTorch is installed and you're in the correct directory")
    sys.exit(1)


def create_model_with_mla(use_mla=True, mla_latent_dim=None, use_svd=False, svd_rank=None):
    """
    Create a GPT model with MLA configuration
    
    Args:
        use_mla: Whether to use MLA instead of standard attention
        mla_latent_dim: Latent dimension for MLA (None for default)
        use_svd: Whether to enable SVD compression on value matrices
        svd_rank: Rank for SVD approximation (None for full rank)
    """
    config = GPTConfig(
        block_size=256,
        vocab_size=50304,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        bias=False,
        use_mla=use_mla,
        mla_latent_dim=mla_latent_dim,
        use_svd=use_svd,
        svd_rank=svd_rank
    )
    
    model = GPT(config)
    return model, config


def analyze_model_efficiency(model, config, model_name):
    """Analyze the efficiency characteristics of a model"""
    total_params = model.get_num_params()
    
    print(f"\n{model_name} Analysis:")
    print("=" * 50)
    print(f"Total Parameters: {total_params:,}")
    
    # If using MLA, show compression statistics
    if hasattr(config, 'use_mla') and config.use_mla:
        # Get the first attention layer to analyze
        first_attn = model.transformer.h[0].attn
        if hasattr(first_attn, 'get_compression_ratio'):
            compression_ratio = first_attn.get_compression_ratio()
            print(f"MLA Compression Ratio: {compression_ratio:.2f}x")
            
            # Calculate memory savings for KV cache during inference
            seq_len = 1024  # Example sequence length
            memory_savings = first_attn.get_memory_savings(seq_len)
            print(f"KV Cache Memory Savings: {memory_savings:.2f}x")
            print(f"MLA Latent Dimension: {first_attn.latent_dim}")
            print(f"Head Size: {first_attn.head_size}")
    
    if hasattr(config, 'use_svd') and config.use_svd:
        print(f"SVD Enabled: True (rank={config.svd_rank})")
    
    return total_params


def test_model_forward_pass(model, model_name):
    """Test a forward pass through the model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    seq_length = 32
    x = torch.randint(0, 50304, (batch_size, seq_length), device=device)
    
    print(f"\n{model_name} Forward Pass Test:")
    print("-" * 30)
    
    with torch.no_grad():
        try:
            logits, loss = model(x)
            print(f"✓ Forward pass successful")
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {logits.shape}")
            print(f"  Device: {device}")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            return False
    
    return True


def compare_attention_mechanisms():
    """Compare different attention mechanism configurations"""
    print("Multi-Head Latent Attention (MLA) Comparison")
    print("=" * 60)
    
    # Test configurations
    configs = [
        ("Standard Attention", {"use_mla": False, "use_svd": False}),
        ("MLA (latent_dim=16)", {"use_mla": True, "mla_latent_dim": 16, "use_svd": False}),
        ("MLA (latent_dim=32)", {"use_mla": True, "mla_latent_dim": 32, "use_svd": False}),
        ("MLA + SVD (rank=8)", {"use_mla": True, "mla_latent_dim": 16, "use_svd": True, "svd_rank": 8}),
        ("Standard + SVD (rank=8)", {"use_mla": False, "use_svd": True, "svd_rank": 8})
    ]
    
    results = []
    
    for name, config_args in configs:
        print(f"\nTesting: {name}")
        print("-" * 40)
        
        try:
            model, config = create_model_with_mla(**config_args)
            
            # Analyze efficiency
            param_count = analyze_model_efficiency(model, config, name)
            
            # Test forward pass
            success = test_model_forward_pass(model, name)
            
            results.append({
                'name': name,
                'params': param_count,
                'success': success,
                'config': config_args
            })
            
        except Exception as e:
            print(f"✗ Failed to create {name}: {e}")
            results.append({
                'name': name,
                'params': 0,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    successful_results = [r for r in results if r['success']]
    if successful_results:
        baseline = successful_results[0]['params'] if successful_results[0]['name'] == 'Standard Attention' else None
        
        for result in successful_results:
            params = result['params']
            reduction = ""
            if baseline and params < baseline:
                reduction = f" ({(baseline - params) / baseline * 100:.1f}% reduction)"
            
            print(f"{result['name']:25}: {params:,} parameters{reduction}")
    
    print("\nMLA Benefits:")
    print("• Reduced memory usage for key-value caching during inference")
    print("• Lower computational complexity for attention computation")
    print("• Configurable latent dimension for trading efficiency vs. quality")
    print("• Compatible with SVD compression for additional memory savings")
    print("• Maintains similar model expressiveness with fewer parameters")


def usage_example():
    """Show basic usage example"""
    print("\nBasic MLA Usage Example:")
    print("=" * 30)
    
    print("""
# Create a model with MLA
from mla_gpt.model import GPT, GPTConfig

config = GPTConfig(
    n_embd=768,
    n_head=12,
    n_layer=12,
    use_mla=True,           # Enable MLA
    mla_latent_dim=128,     # Latent dimension (default: head_size // 2)
    use_svd=True,           # Optional: combine with SVD
    svd_rank=64             # SVD compression rank
)

model = GPT(config)

# The model will automatically use MLA instead of standard attention
# in all transformer blocks
""")


if __name__ == "__main__":
    try:
        compare_attention_mechanisms()
        usage_example()
        
        print(f"\nTo train with MLA, use the configuration file:")
        print(f"python -m mla_gpt.cli.train config/train_mla_test.py")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()