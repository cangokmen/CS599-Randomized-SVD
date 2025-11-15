"""
Test suite for Multi-Head Latent Attention (MLA) implementation
"""

import unittest
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import torch
    import torch.nn as nn
    from mla_gpt.model import GPT, GPTConfig
    from mla_gpt.model.attention import MultiHeadLatentAttention, CausalSelfAttention
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure PyTorch is installed and you're in the correct directory")
    sys.exit(1)


class TestMLA(unittest.TestCase):
    
    def setUp(self):
        """Set up test configuration"""
        self.config = GPTConfig(
            block_size=64,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_embd=128,
            dropout=0.0,
            bias=False
        )
        
        self.batch_size = 2
        self.seq_len = 32
        
    def test_mla_initialization(self):
        """Test MLA layer initialization"""
        mla_config = GPTConfig(
            block_size=64,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_embd=128,
            dropout=0.0,
            bias=False,
            use_mla=True,
            mla_latent_dim=32
        )
        
        mla = MultiHeadLatentAttention(mla_config)
        
        # Check attributes
        self.assertEqual(mla.n_head, 4)
        self.assertEqual(mla.n_embd, 128)
        self.assertEqual(mla.head_size, 32)
        self.assertEqual(mla.latent_dim, 32)
        
        # Check layer dimensions
        self.assertEqual(mla.q_proj.in_features, 128)
        self.assertEqual(mla.q_proj.out_features, 128)
        self.assertEqual(mla.kv_latent_proj.out_features, 2 * 4 * 32)  # 2 * n_head * latent_dim
        
    def test_mla_forward_pass(self):
        """Test MLA forward pass"""
        mla_config = GPTConfig(
            block_size=64,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_embd=128,
            dropout=0.0,
            bias=False,
            use_mla=True,
            mla_latent_dim=16
        )
        
        mla = MultiHeadLatentAttention(mla_config)
        
        # Create dummy input
        x = torch.randn(self.batch_size, self.seq_len, self.config.n_embd)
        
        # Forward pass
        output = mla(x)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)
        
    def test_mla_vs_standard_attention_shapes(self):
        """Test that MLA and standard attention produce same output shapes"""
        # Standard attention
        standard_attn = CausalSelfAttention(self.config)
        
        # MLA attention
        mla_config = GPTConfig(
            block_size=64,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_embd=128,
            dropout=0.0,
            bias=False,
            use_mla=True,
            mla_latent_dim=16
        )
        mla_attn = MultiHeadLatentAttention(mla_config)
        
        # Create dummy input
        x = torch.randn(self.batch_size, self.seq_len, self.config.n_embd)
        
        # Forward passes
        standard_output = standard_attn(x)
        mla_output = mla_attn(x)
        
        # Check shapes match
        self.assertEqual(standard_output.shape, mla_output.shape)
        
    def test_model_with_mla(self):
        """Test full model with MLA"""
        mla_config = GPTConfig(
            block_size=64,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_embd=128,
            dropout=0.0,
            bias=False,
            use_mla=True,
            mla_latent_dim=16
        )
        
        model = GPT(mla_config)
        
        # Create dummy input and targets for full sequence output
        x = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        targets = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        
        # Forward pass with targets to get full sequence
        logits, loss = model(x, targets)
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, 1000)
        self.assertEqual(logits.shape, expected_shape)
        
        # Also test inference mode (no targets) 
        logits_inference, loss_inference = model(x)
        expected_inference_shape = (self.batch_size, 1, 1000)  # Only last position
        self.assertEqual(logits_inference.shape, expected_inference_shape)
        self.assertIsNone(loss_inference)
        
    def test_mla_compression_ratio(self):
        """Test MLA compression ratio calculation"""
        mla_config = GPTConfig(
            block_size=64,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_embd=128,
            dropout=0.0,
            bias=False,
            use_mla=True,
            mla_latent_dim=16
        )
        
        mla = MultiHeadLatentAttention(mla_config)
        compression_ratio = mla.get_compression_ratio()
        
        # Should have compression since latent_dim < head_size
        self.assertGreater(compression_ratio, 1.0)
        
    def test_mla_memory_savings(self):
        """Test MLA memory savings calculation"""
        mla_config = GPTConfig(
            block_size=64,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_embd=128,
            dropout=0.0,
            bias=False,
            use_mla=True,
            mla_latent_dim=16
        )
        
        mla = MultiHeadLatentAttention(mla_config)
        memory_savings = mla.get_memory_savings(sequence_length=1024)
        
        # Should have memory savings since latent_dim < head_size
        self.assertGreater(memory_savings, 1.0)
        
    def test_mla_with_svd(self):
        """Test MLA combined with SVD compression"""
        mla_config = GPTConfig(
            block_size=64,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_embd=128,
            dropout=0.0,
            bias=False,
            use_mla=True,
            mla_latent_dim=16,
            use_svd=True,
            svd_rank=8
        )
        
        mla = MultiHeadLatentAttention(mla_config)
        
        # Create dummy input
        x = torch.randn(self.batch_size, self.seq_len, self.config.n_embd)
        
        # Forward pass should work with both MLA and SVD
        output = mla(x)
        self.assertEqual(output.shape, x.shape)
        
    def test_different_latent_dimensions(self):
        """Test MLA with different latent dimensions"""
        latent_dims = [8, 16, 32, 64]
        
        for latent_dim in latent_dims:
            with self.subTest(latent_dim=latent_dim):
                mla_config = GPTConfig(
                    block_size=64,
                    vocab_size=1000,
                    n_layer=2,
                    n_head=4,
                    n_embd=128,
                    dropout=0.0,
                    bias=False,
                    use_mla=True,
                    mla_latent_dim=latent_dim
                )
                
                mla = MultiHeadLatentAttention(mla_config)
                
                # Create dummy input
                x = torch.randn(self.batch_size, self.seq_len, self.config.n_embd)
                
                # Forward pass
                output = mla(x)
                self.assertEqual(output.shape, x.shape)
                
                # Check latent dimension is set correctly
                self.assertEqual(mla.latent_dim, latent_dim)
    
    def test_default_latent_dimension(self):
        """Test default latent dimension calculation"""
        mla_config = GPTConfig(
            block_size=64,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_embd=128,
            dropout=0.0,
            bias=False,
            use_mla=True
            # No mla_latent_dim specified - should use default
        )
        
        mla = MultiHeadLatentAttention(mla_config)
        
        # Default should be head_size // 2 = 32 // 2 = 16
        expected_latent_dim = self.config.n_embd // self.config.n_head // 2
        self.assertEqual(mla.latent_dim, expected_latent_dim)
        

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)