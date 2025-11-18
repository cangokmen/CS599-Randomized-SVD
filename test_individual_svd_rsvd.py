"""
Test script for individual SVD/Randomized SVD selection per matrix (Q, K, V)

Tests the new functionality where each matrix can independently choose between:
- No compression
- Standard SVD compression  
- Randomized SVD compression
"""

import torch
import sys
import os
sys.path.append(os.path.abspath('.'))

from src.mla_gpt.model.model import GPT, GPTConfig
from src.mla_gpt.model.compression.svd_compression import SVDCompression  
from src.mla_gpt.model.compression.randomized_svd_compression import RandomizedSVDCompression

def test_individual_svd_rsvd_selection():
    """
    Test that each matrix (Q, K, V) can independently select standard vs randomized SVD
    """
    print("🧪 Testing Individual SVD/Randomized SVD Selection")
    print("=" * 60)
    
    # Test configuration with mixed compression types
    config = GPTConfig(
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=128,
        
        # Enable SVD
        use_svd=True,
        svd_rank=16,
        svd_oversampling=8,
        svd_power_iter=1,
        
        # Mixed compression configuration
        svd_apply_to_q=True,    # Apply compression to Q
        svd_apply_to_k=True,    # Apply compression to K  
        svd_apply_to_v=True,    # Apply compression to V
        
        # Different algorithms for each matrix
        use_randomized_svd_q=False,  # Standard SVD for Q
        use_randomized_svd_k=True,   # Randomized SVD for K
        use_randomized_svd_v=True,   # Randomized SVD for V
    )
    
    print(f"Configuration:")
    print(f"  Head size: {config.n_embd // config.n_head}")
    print(f"  Target rank: {config.svd_rank}")
    print(f"")
    print(f"Matrix configurations:")
    print(f"  Q: Compression={'✓' if config.svd_apply_to_q else '✗'}, "
          f"Algorithm={'Randomized SVD' if config.use_randomized_svd_q else 'Standard SVD'}")
    print(f"  K: Compression={'✓' if config.svd_apply_to_k else '✗'}, "
          f"Algorithm={'Randomized SVD' if config.use_randomized_svd_k else 'Standard SVD'}")
    print(f"  V: Compression={'✓' if config.svd_apply_to_v else '✗'}, "
          f"Algorithm={'Randomized SVD' if config.use_randomized_svd_v else 'Standard SVD'}")
    print()
    
    # Create model 
    model = GPT(config)
    
    # Get the attention layer to examine compressors
    attn_layer = model.transformer.h[0].attn
    
    print(f"Attention layer compressor analysis:")
    
    # Check Q compressor
    if hasattr(attn_layer, 'q_compressor') and attn_layer.q_compressor is not None:
        q_type = type(attn_layer.q_compressor).__name__
        print(f"  Q compressor: {q_type}")
        assert config.svd_apply_to_q, "Q compressor exists but svd_apply_to_q is False!"
        
        if config.use_randomized_svd_q:
            assert isinstance(attn_layer.q_compressor, RandomizedSVDCompression), \
                f"Expected RandomizedSVDCompression for Q, got {q_type}"
        else:
            assert isinstance(attn_layer.q_compressor, SVDCompression), \
                f"Expected SVDCompression for Q, got {q_type}"
    else:
        print(f"  Q compressor: None")
        assert not config.svd_apply_to_q, "Q compressor is None but svd_apply_to_q is True!"
    
    # Check K compressor
    if hasattr(attn_layer, 'k_compressor') and attn_layer.k_compressor is not None:
        k_type = type(attn_layer.k_compressor).__name__
        print(f"  K compressor: {k_type}")
        assert config.svd_apply_to_k, "K compressor exists but svd_apply_to_k is False!"
        
        if config.use_randomized_svd_k:
            assert isinstance(attn_layer.k_compressor, RandomizedSVDCompression), \
                f"Expected RandomizedSVDCompression for K, got {k_type}"
        else:
            assert isinstance(attn_layer.k_compressor, SVDCompression), \
                f"Expected SVDCompression for K, got {k_type}"
    else:
        print(f"  K compressor: None")
        assert not config.svd_apply_to_k, "K compressor is None but svd_apply_to_k is True!"
    
    # Check V compressor
    if hasattr(attn_layer, 'v_compressor') and attn_layer.v_compressor is not None:
        v_type = type(attn_layer.v_compressor).__name__
        print(f"  V compressor: {v_type}")
        assert config.svd_apply_to_v, "V compressor exists but svd_apply_to_v is False!"
        
        if config.use_randomized_svd_v:
            assert isinstance(attn_layer.v_compressor, RandomizedSVDCompression), \
                f"Expected RandomizedSVDCompression for V, got {v_type}"
        else:
            assert isinstance(attn_layer.v_compressor, SVDCompression), \
                f"Expected SVDCompression for V, got {v_type}"
    else:
        print(f"  V compressor: None")
        assert not config.svd_apply_to_v, "V compressor is None but svd_apply_to_v is True!"
    
    print(f"✅ Individual compressor instantiation correct!")
    print()
    
    # Test forward pass with mixed compression
    batch_size, seq_len = 2, 64
    # Create input token indices instead of embeddings
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"Testing forward pass with mixed compression...")
    
    with torch.no_grad():
        # Forward pass through model
        logits, loss = model(x)
        
        # Check output shape (inference mode returns last position only)
        expected_shape = (batch_size, 1, config.vocab_size)
        assert logits.shape == expected_shape, \
            f"Expected output shape {expected_shape}, got {logits.shape}"
        
        # Check that output is finite
        assert torch.isfinite(logits).all(), "Output contains NaN or Inf values!"
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Output range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    
    print(f"✅ Forward pass successful with mixed SVD/RSVD compression!")
    print()
    
    return True

def test_all_configurations():
    """
    Test various combinations of SVD/RSVD settings
    """
    print("🧪 Testing All Configuration Combinations")
    print("=" * 60)
    
    # Define test configurations
    test_configs = [
        # All standard SVD
        {
            'svd_apply_to_q': True, 'use_randomized_svd_q': False,
            'svd_apply_to_k': True, 'use_randomized_svd_k': False,
            'svd_apply_to_v': True, 'use_randomized_svd_v': False,
        },
        
        # All randomized SVD
        {
            'svd_apply_to_q': True, 'use_randomized_svd_q': True,
            'svd_apply_to_k': True, 'use_randomized_svd_k': True,
            'svd_apply_to_v': True, 'use_randomized_svd_v': True,
        },
        
        # V-only randomized (original configuration)
        {
            'svd_apply_to_q': False, 'use_randomized_svd_q': False,
            'svd_apply_to_k': False, 'use_randomized_svd_k': False,
            'svd_apply_to_v': True, 'use_randomized_svd_v': True,
        },
        
        # Mixed: Q standard, K and V randomized
        {
            'svd_apply_to_q': True, 'use_randomized_svd_q': False,
            'svd_apply_to_k': True, 'use_randomized_svd_k': True,
            'svd_apply_to_v': True, 'use_randomized_svd_v': True,
        },
        
        # No compression
        {
            'svd_apply_to_q': False, 'use_randomized_svd_q': False,
            'svd_apply_to_k': False, 'use_randomized_svd_k': False,
            'svd_apply_to_v': False, 'use_randomized_svd_v': False,
        }
    ]
    
    config_names = [
        'All Standard SVD',
        'All Randomized SVD', 
        'V-only Randomized SVD',
        'Mixed: Q=Standard, K&V=Randomized',
        'No Compression'
    ]
    
    base_config = {
        'vocab_size': 1000,
        'n_layer': 2,
        'n_head': 4,
        'n_embd': 128,
        'use_svd': True,
        'svd_rank': 16,
        'svd_oversampling': 8,
        'svd_power_iter': 1,
    }
    
    for i, test_config in enumerate(test_configs):
        config_name = config_names[i]
        print(f"\nTesting: {config_name}")
        print(f"  Configuration: Q={test_config['svd_apply_to_q']}/" +
              f"{'R' if test_config['use_randomized_svd_q'] else 'S'}, " +
              f"K={test_config['svd_apply_to_k']}/" +
              f"{'R' if test_config['use_randomized_svd_k'] else 'S'}, " +
              f"V={test_config['svd_apply_to_v']}/" +
              f"{'R' if test_config['use_randomized_svd_v'] else 'S'}")
        
        # Create configuration
        config = GPTConfig(**{**base_config, **test_config})
        
        # Handle no compression case
        if not any([config.svd_apply_to_q, config.svd_apply_to_k, config.svd_apply_to_v]):
            config.use_svd = False
        
        # Create model and test
        try:
            model = GPT(config)
            
            # Quick forward pass test
            x = torch.randint(0, config.vocab_size, (1, 32))
            with torch.no_grad():
                logits, loss = model(x)
                assert torch.isfinite(logits).all()
            
            print(f"  ✅ Passed")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            return False
    
    print(f"\n✅ All configuration combinations passed!")
    return True

def test_compression_info():
    """
    Test the get_compression_info method with mixed configurations
    """
    print("🧪 Testing Compression Info Reporting")
    print("=" * 60)
    
    config = GPTConfig(
        vocab_size=1000,
        n_layer=1,
        n_head=4,
        n_embd=128,
        
        use_svd=True,
        svd_rank=16,
        
        # Mixed configuration for testing
        svd_apply_to_q=True,  use_randomized_svd_q=False,   # Standard SVD
        svd_apply_to_k=False, use_randomized_svd_k=True,    # No compression
        svd_apply_to_v=True,  use_randomized_svd_v=True,    # Randomized SVD
    )
    
    model = GPT(config)
    attn_layer = model.transformer.h[0].attn
    
    # Test get_compression_info method
    head_size = config.n_embd // config.n_head
    matrix_shape = (head_size, head_size)  # Q, K, V matrix shape
    info = attn_layer.get_compression_info(matrix_shape)
    
    print(f"Compression info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Validate info content
    assert info['q_compression'] == True, f"Expected True for Q compression, got {info['q_compression']}"
    assert info['k_compression'] == False, f"Expected False for K compression, got {info['k_compression']}"
    assert info['v_compression'] == True, f"Expected True for V compression, got {info['v_compression']}"
    assert info['q_compression_type'] == 'standard_svd', f"Expected 'standard_svd' for Q, got {info['q_compression_type']}"
    assert info['k_compression_type'] == 'none', f"Expected 'none' for K, got {info['k_compression_type']}"
    assert info['v_compression_type'] == 'randomized_svd', f"Expected 'randomized_svd' for V, got {info['v_compression_type']}"
    
    # Count compressed matrices
    num_compressed = sum([info['q_compression'], info['k_compression'], info['v_compression']])
    assert num_compressed == 2, f"Expected 2 compressed matrices, got {num_compressed}"
    
    print(f"✅ Compression info reporting correct!")
    return True

def run_all_tests():
    """
    Run all tests for individual SVD/RSVD functionality
    """
    print("🚀 Running All Tests for Individual SVD/RSVD Selection")
    print("=" * 80)
    
    tests = [
        test_individual_svd_rsvd_selection,
        test_all_configurations,
        test_compression_info,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            print(f"\n" + "="*60)
            result = test_func()
            if result:
                passed += 1
                print(f"✅ {test_func.__name__} PASSED")
            else:
                failed += 1
                print(f"❌ {test_func.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_func.__name__} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n" + "="*80)
    print(f"🏁 Test Summary")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total:  {passed + failed}")
    
    if failed == 0:
        print(f"🎉 ALL TESTS PASSED! Individual SVD/RSVD functionality is working correctly.")
        return True
    else:
        print(f"💥 Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)