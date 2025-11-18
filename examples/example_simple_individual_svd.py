#!/usr/bin/env python3
"""
Simple Examples for Individual SVD/Randomized SVD Selection

This script demonstrates the new functionality where each matrix (Q, K, V) 
can independently choose between:
- No compression
- Standard SVD compression
- Randomized SVD compression
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.mla_gpt.model.model import GPT, GPTConfig

def example_mixed_compression():
    """
    Example: Mixed compression strategy
    Q: Standard SVD, K: No compression, V: Randomized SVD
    """
    print("🔄 Example: Mixed Compression Strategy")
    print("=" * 50)
    
    config = GPTConfig(
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=128,
        
        # Enable SVD with mixed strategy
        use_svd=True,
        svd_rank=24,
        svd_oversampling=8,
        svd_power_iter=1,
        
        # Individual matrix and algorithm selection
        svd_apply_to_q=True,     # Compress Q
        svd_apply_to_k=False,    # Keep K uncompressed
        svd_apply_to_v=True,     # Compress V
        
        use_randomized_svd_q=False,  # Standard SVD for Q
        use_randomized_svd_k=False,  # N/A (K not compressed)
        use_randomized_svd_v=True,   # Randomized SVD for V
    )
    
    # Create model and analyze
    model = GPT(config)
    attn = model.transformer.h[0].attn
    
    # Get compression info
    head_size = config.n_embd // config.n_head
    matrix_shape = (head_size, head_size)
    info = attn.get_compression_info(matrix_shape)
    
    print(f"Configuration:")
    print(f"  Head size: {head_size}, Target rank: {config.svd_rank}")
    print(f"  Q: {'✓ Standard SVD' if info['q_compression'] else '✗ No compression'}")
    print(f"  K: {'✓' if info['k_compression'] else '✗ No compression'}")
    print(f"  V: {'✓ Randomized SVD' if info['v_compression'] and info['v_compression_type'] == 'randomized_svd' else '✗'}")
    
    # Test forward pass
    x = torch.randint(0, config.vocab_size, (2, 64))
    with torch.no_grad():
        logits, loss = model(x)
        print(f"\nForward pass: {x.shape} -> {logits.shape} ✅")
    
    # Calculate compression ratio
    original = head_size * head_size
    compressed = head_size * config.svd_rank + config.svd_rank + config.svd_rank * head_size
    ratio = original / compressed
    print(f"Memory impact: {ratio:.2f}x compression per matrix")
    
    return config, model

def example_all_randomized():
    """
    Example: All matrices with randomized SVD
    """
    print("\n\n🔬 Example: All Randomized SVD")
    print("=" * 50)
    
    config = GPTConfig(
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=128,
        
        use_svd=True,
        svd_rank=16,
        svd_oversampling=10,
        svd_power_iter=2,
        
        # All matrices with randomized SVD
        svd_apply_to_q=True,  use_randomized_svd_q=True,
        svd_apply_to_k=True,  use_randomized_svd_k=True,
        svd_apply_to_v=True,  use_randomized_svd_v=True,
    )
    
    model = GPT(config)
    attn = model.transformer.h[0].attn
    
    head_size = config.n_embd // config.n_head
    matrix_shape = (head_size, head_size)
    info = attn.get_compression_info(matrix_shape)
    
    print(f"All matrices: Randomized SVD")
    print(f"Rank: {config.svd_rank}/{head_size} ({config.svd_rank/head_size:.1%})")
    
    # Performance analysis
    if 'complexity' in info:
        complexity = info['complexity']
        print(f"\nComputational Analysis:")
        print(f"  Standard SVD ops: {complexity['standard_svd']:,}")
        print(f"  Randomized SVD ops: {complexity['randomized_svd']:,}")
        print(f"  Speedup: {complexity['speedup_ratio']:.2f}x")
    
    return config, model

def example_v_only():
    """
    Example: V-only compression (conservative approach)
    """
    print("\n\n🚀 Example: V-only Compression")
    print("=" * 50)
    
    config = GPTConfig(
        vocab_size=1000,
        n_layer=4,
        n_head=8,
        n_embd=256,
        
        use_svd=True,
        svd_rank=24,  # Conservative rank
        
        # V-only compression
        svd_apply_to_q=False,
        svd_apply_to_k=False,
        svd_apply_to_v=True,   # V-only
        use_randomized_svd_v=True,
    )
    
    model = GPT(config)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Conservative V-only strategy:")
    print(f"  Parameters: {total_params:,}")
    print(f"  Only V matrices compressed")
    print(f"  Preserves Q and K precision")
    
    # Test forward pass
    x = torch.randint(0, config.vocab_size, (1, 32))
    with torch.no_grad():
        logits, loss = model(x)
        print(f"  Forward pass successful ✅")
    
    return config, model

def compare_configurations():
    """
    Compare different configuration strategies
    """
    print("\n\n📊 Configuration Comparison")
    print("=" * 50)
    
    configs = [
        ("No Compression", GPTConfig(
            vocab_size=1000, n_layer=2, n_head=4, n_embd=128,
            use_svd=False)),
        
        ("V-only Standard", GPTConfig(
            vocab_size=1000, n_layer=2, n_head=4, n_embd=128,
            use_svd=True, svd_rank=16,
            svd_apply_to_v=True, use_randomized_svd_v=False)),
        
        ("V-only Randomized", GPTConfig(
            vocab_size=1000, n_layer=2, n_head=4, n_embd=128,
            use_svd=True, svd_rank=16,
            svd_apply_to_v=True, use_randomized_svd_v=True)),
        
        ("All Randomized", GPTConfig(
            vocab_size=1000, n_layer=2, n_head=4, n_embd=128,
            use_svd=True, svd_rank=16,
            svd_apply_to_q=True, svd_apply_to_k=True, svd_apply_to_v=True,
            use_randomized_svd_q=True, use_randomized_svd_k=True, use_randomized_svd_v=True)),
    ]
    
    print(f"{'Configuration':<20} {'Parameters':<12} {'Algorithm'}")
    print("-" * 50)
    
    for name, config in configs:
        model = GPT(config)
        total_params = sum(p.numel() for p in model.parameters())
        
        # Determine algorithm
        if not config.use_svd:
            algorithm = "None"
        elif config.svd_apply_to_v and not (config.svd_apply_to_q or config.svd_apply_to_k):
            alg = "R" if config.use_randomized_svd_v else "S"
            algorithm = f"V:{alg}"
        elif config.svd_apply_to_q and config.svd_apply_to_k and config.svd_apply_to_v:
            algorithm = "Q:R,K:R,V:R"
        else:
            algorithm = "Mixed"
        
        print(f"{name:<20} {total_params:<12,} {algorithm}")

def main():
    """
    Run all examples demonstrating individual SVD/RSVD selection
    """
    print("🎯 Individual SVD/Randomized SVD Selection Examples")
    print("=" * 70)
    print("Each matrix (Q, K, V) can independently choose:")
    print("  • No compression")
    print("  • Standard SVD") 
    print("  • Randomized SVD (Tropp's algorithm)")
    print()
    
    # Run examples
    example_mixed_compression()
    example_all_randomized()
    example_v_only()
    compare_configurations()
    
    print(f"\n\n✅ All examples completed!")
    print(f"\n🔧 Usage Tips:")
    print(f"  • Start with V-only compression")
    print(f"  • Use randomized SVD for speed")
    print(f"  • Standard SVD for higher accuracy")
    print(f"  • Monitor quality with Q/K compression")

if __name__ == "__main__":
    main()