#!/usr/bin/env python3
"""
Comprehensive Examples for Individual SVD/Randomized SVD Selection

This script demonstrates the new functionality where each matrix (Q, K, V) 
can independently choose between:
- No compression
- Standard SVD compression
- Randomized SVD compression

Usage:
    python examples/example_individual_svd_rsvd.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.mla_gpt.model.model import GPT, GPTConfig

def example_mixed_compression():
    """
    Example 1: Mixed compression strategy
    - Q matrix: Standard SVD (precise, moderate cost)
    - K matrix: No compression (preserve key matching)  
    - V matrix: Randomized SVD (fast approximation)
    """
    print("🔄 Example 1: Mixed Compression Strategy")
    print("=" * 50)
    
    config = GPTConfig(
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=128,
        block_size=256,
        
        # Enable SVD with mixed strategy
        use_svd=True,
        svd_rank=24,  # 75% of head size (32)
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
    print(f"  K: {'✓ ' + info['k_compression_type'] if info['k_compression'] else '✗ No compression'}")
    print(f"  V: {'✓ Randomized SVD' if info['v_compression'] and info['v_compression_type'] == 'randomized_svd' else '✗'}")
    
    # Test forward pass
    x = torch.randint(0, config.vocab_size, (2, 64))
    with torch.no_grad():
        logits, loss = model(x)
        print(f"\nForward pass: {x.shape} -> {logits.shape} ✅")
    
    print(f"Memory impact: {(head_size**2) / (head_size*config.svd_rank + config.svd_rank + config.svd_rank*head_size):.2f}x compression per matrix")
    
    return config, model

def example_research_comparison():
    \"\"\"
    Example 2: Research comparison setup
    - All matrices compressed for fair comparison
    - Q: Standard SVD (baseline)
    - K: Randomized SVD (test candidate)
    - V: Randomized SVD (test candidate)
    \"\"\"
    print(f\"\\n\\n🔬 Example 2: Research Comparison Setup\")
    print(\"=\" * 50)
    
    config = GPTConfig(
        vocab_size=5000,
        n_layer=4,
        n_head=8,
        n_embd=256,
        block_size=512,
        
        # Research configuration
        use_svd=True,
        svd_rank=32,  # 50% of head size (64)
        svd_oversampling=12,
        svd_power_iter=2,
        
        # All matrices compressed for comparison
        svd_apply_to_q=True,  use_randomized_svd_q=False,  # Baseline
        svd_apply_to_k=True,  use_randomized_svd_k=True,   # Test
        svd_apply_to_v=True,  use_randomized_svd_v=True,   # Test
    )
    
    model = GPT(config)
    attn = model.transformer.h[0].attn
    
    head_size = config.n_embd // config.n_head
    matrix_shape = (head_size, head_size)
    info = attn.get_compression_info(matrix_shape)
    
    print(f\"Research Configuration:\")
    print(f\"  Model size: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} embedding\")
    print(f\"  Head size: {head_size}, Compression rank: {config.svd_rank} ({config.svd_rank/head_size:.1%})\\n\")
    
    print(f\"Matrix Algorithms:\")
    print(f\"  Q: Standard SVD (baseline for comparison)\")
    print(f\"  K: Randomized SVD (accuracy test)\")
    print(f\"  V: Randomized SVD (efficiency test)\\n\")
    
    # Performance analysis
    if 'complexity' in info:
        complexity = info['complexity']
        print(f\"Computational Analysis:\")
        print(f\"  Standard SVD ops: {complexity['standard_svd']:,}\")
        print(f\"  Randomized SVD ops: {complexity['randomized_svd']:,}\")
        print(f\"  Speedup ratio: {complexity['speedup_ratio']:.2f}x\")
    
    # Memory analysis
    if 'memory' in info:
        memory = info['memory']
        print(f\"\\nMemory Analysis:\")
        print(f\"  Original matrix: {memory['original_matrix']} parameters\")
        print(f\"  Compressed + overhead: {memory['total_memory']} parameters\")
        print(f\"  Memory overhead ratio: {memory['memory_overhead_ratio']:.2f}x\")
    
    return config, model

def example_production_deployment():
    \"\"\"
    Example 3: Production deployment strategy
    - V-only compression (proven effective)
    - Randomized SVD (fast inference)
    - Conservative rank (quality preservation)
    \"\"\"
    print(f\"\\n\\n🚀 Example 3: Production Deployment Strategy\")
    print(\"=\" * 50)
    
    config = GPTConfig(
        vocab_size=50000,  # Large vocabulary
        n_layer=12,        # Medium-size model
        n_head=12,
        n_embd=768,
        block_size=1024,
        
        # Production-ready configuration
        use_svd=True,
        svd_rank=48,       # Conservative 75% of head size (64)
        svd_oversampling=10,
        svd_power_iter=1,  # Minimal iterations for speed
        
        # Conservative V-only compression
        svd_apply_to_q=False,  use_randomized_svd_q=False,
        svd_apply_to_k=False,  use_randomized_svd_k=False,
        svd_apply_to_v=True,   use_randomized_svd_v=True,
    )
    
    model = GPT(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f\"Model Configuration:\")
    print(f\"  Parameters: {total_params:,} ({total_params/1e6:.1f}M)\")
    print(f\"  Vocabulary: {config.vocab_size:,}\")
    print(f\"  Context length: {config.block_size}\")
    print(f\"  Architecture: {config.n_layer}L-{config.n_head}H-{config.n_embd}D\\n\")
    
    print(f\"Compression Strategy:\")
    print(f\"  Target: Value matrices only (proven effective)\")
    print(f\"  Algorithm: Randomized SVD (fast inference)\")
    print(f\"  Rank: {config.svd_rank}/64 ({config.svd_rank/64:.1%}) - Conservative\")
    
    # Estimate savings
    head_size = config.n_embd // config.n_head
    original_v_params = config.n_layer * head_size * head_size
    compressed_v_params = config.n_layer * (head_size * config.svd_rank + config.svd_rank + config.svd_rank * head_size)
    v_savings = original_v_params - compressed_v_params
    
    print(f\"\\nEstimated Savings:\")
    print(f\"  V matrix parameters: {original_v_params:,} -> {compressed_v_params:,}\")
    print(f\"  Parameters saved: {v_savings:,} ({v_savings/total_params:.2%} of total)\")
    print(f\"  Memory reduction: {original_v_params/compressed_v_params:.2f}x for V matrices\")
    
    return config, model

def example_aggressive_compression():
    \"\"\"
    Example 4: Aggressive compression for resource-constrained deployment
    - All matrices compressed
    - All randomized SVD (maximum speed)
    - Low rank (maximum compression)
    \"\"\"
    print(f\"\\n\\n⚡ Example 4: Aggressive Compression Strategy\")
    print(\"=\" * 50)
    
    config = GPTConfig(
        vocab_size=10000,
        n_layer=6,
        n_head=8,
        n_embd=512,
        block_size=512,
        
        # Aggressive compression
        use_svd=True,
        svd_rank=16,       # Low rank (25% of head size)
        svd_oversampling=8,
        svd_power_iter=1,
        
        # Compress everything with randomized SVD
        svd_apply_to_q=True,  use_randomized_svd_q=True,
        svd_apply_to_k=True,  use_randomized_svd_k=True,
        svd_apply_to_v=True,  use_randomized_svd_v=True,
    )
    
    model = GPT(config)
    
    # Calculate compression ratio
    head_size = config.n_embd // config.n_head
    original_qkv_params = 3 * config.n_layer * head_size * head_size
    compressed_qkv_params = 3 * config.n_layer * (head_size * config.svd_rank + config.svd_rank + config.svd_rank * head_size)
    compression_ratio = original_qkv_params / compressed_qkv_params
    
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f\"Aggressive Configuration:\")
    print(f\"  All QKV matrices: Randomized SVD\")
    print(f\"  Rank: {config.svd_rank}/64 ({config.svd_rank/64:.1%}) - Aggressive\")
    print(f\"  Target: Maximum speed and memory savings\\n\")
    
    print(f\"Compression Results:\")
    print(f\"  QKV parameters: {original_qkv_params:,} -> {compressed_qkv_params:,}\")
    print(f\"  QKV compression: {compression_ratio:.2f}x\")
    print(f\"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)\")
    
    # Test inference speed characteristics
    print(f\"\\nInference Characteristics:\")
    print(f\"  All matrices use fast randomized SVD\")
    print(f\"  Low rank reduces computation significantly\")
    print(f\"  Trade-off: Quality vs Speed/Memory\")
    
    return config, model

def benchmark_configurations():
    \"\"\"
    Compare different configuration strategies
    \"\"\"
    print(f\"\\n\\n📊 Configuration Benchmark Comparison\")
    print(\"=\" * 60)
    
    configs = [
        (\"No Compression\", GPTConfig(
            vocab_size=1000, n_layer=2, n_head=4, n_embd=128,
            use_svd=False)),
        
        (\"V-only Standard\", GPTConfig(
            vocab_size=1000, n_layer=2, n_head=4, n_embd=128,
            use_svd=True, svd_rank=16,
            svd_apply_to_v=True, use_randomized_svd_v=False)),
        
        (\"V-only Randomized\", GPTConfig(
            vocab_size=1000, n_layer=2, n_head=4, n_embd=128,
            use_svd=True, svd_rank=16,
            svd_apply_to_v=True, use_randomized_svd_v=True)),
        
        (\"All Standard\", GPTConfig(
            vocab_size=1000, n_layer=2, n_head=4, n_embd=128,
            use_svd=True, svd_rank=16,
            svd_apply_to_q=True, svd_apply_to_k=True, svd_apply_to_v=True,
            use_randomized_svd_q=False, use_randomized_svd_k=False, use_randomized_svd_v=False)),
        
        (\"All Randomized\", GPTConfig(
            vocab_size=1000, n_layer=2, n_head=4, n_embd=128,
            use_svd=True, svd_rank=16,
            svd_apply_to_q=True, svd_apply_to_k=True, svd_apply_to_v=True,
            use_randomized_svd_q=True, use_randomized_svd_k=True, use_randomized_svd_v=True)),
    ]
    
    print(f\"{'Configuration':<20} {'Parameters':<12} {'QKV Compression':<16} {'Algorithm'}\")
    print(\"-\" * 60)
    
    for name, config in configs:
        model = GPT(config)
        total_params = sum(p.numel() for p in model.parameters())
        
        # Calculate QKV compression
        if config.use_svd and any([config.svd_apply_to_q, config.svd_apply_to_k, config.svd_apply_to_v]):
            head_size = config.n_embd // config.n_head
            num_compressed = sum([config.svd_apply_to_q, config.svd_apply_to_k, config.svd_apply_to_v])
            original_params = head_size * head_size
            compressed_params = head_size * config.svd_rank + config.svd_rank + config.svd_rank * head_size
            compression_ratio = original_params / compressed_params
            qkv_compression = f\"{compression_ratio:.1f}x ({num_compressed}/3)\"
            
            # Determine algorithm mix
            algorithms = []
            if config.svd_apply_to_q:
                algorithms.append(f\"Q:{'R' if config.use_randomized_svd_q else 'S'}\")
            if config.svd_apply_to_k:
                algorithms.append(f\"K:{'R' if config.use_randomized_svd_k else 'S'}\")
            if config.svd_apply_to_v:
                algorithms.append(f\"V:{'R' if config.use_randomized_svd_v else 'S'}\")
            algorithm_str = \",\".join(algorithms)
        else:
            qkv_compression = \"None\"
            algorithm_str = \"None\"
        
        print(f\"{name:<20} {total_params:<12,} {qkv_compression:<16} {algorithm_str}\")

def main():
    \"\"\"
    Run all examples demonstrating individual SVD/RSVD selection
    \"\"\"
    print(\"🎯 Individual SVD/Randomized SVD Selection Examples\")
    print(\"=\" * 70)
    print(\"Demonstrating flexible compression strategies where each matrix\")
    print(\"(Q, K, V) can independently choose between:\")
    print(\"  • No compression\")
    print(\"  • Standard SVD\") 
    print(\"  • Randomized SVD (Tropp's algorithm)\")
    print()
    
    # Run examples
    example_mixed_compression()
    example_research_comparison()
    example_production_deployment()
    example_aggressive_compression()
    benchmark_configurations()
    
    print(f\"\\n\\n✅ All examples completed successfully!\")
    print(f\"\\n🔧 Usage Tips:\")
    print(f\"  • Start with V-only compression (proven effective)\")
    print(f\"  • Use randomized SVD for faster inference\")
    print(f\"  • Mix algorithms based on your requirements:\")
    print(f\"    - Standard SVD: Higher accuracy, slower\")
    print(f\"    - Randomized SVD: Lower accuracy, faster\")
    print(f\"  • Monitor model quality when compressing Q or K\")
    print(f\"  • Conservative ranks (>50%) preserve quality better\")

if __name__ == \"__main__\":
    main()