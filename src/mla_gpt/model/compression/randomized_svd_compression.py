"""
Tropp's Randomized SVD Compression for V matrices

Implementation of randomized SVD following Halko, Martinsson, and Tropp (2011)
"Finding structure with randomness: Probabilistic algorithms for constructing 
approximate matrix decompositions"

Specifically implements:
- Algorithm 4.1: Basic Randomized SVD
- Algorithm 4.4: Randomized SVD with Power Iterations
"""

import torch
import torch.nn as nn
from .base_compression import BaseCompression


class RandomizedSVDCompression(BaseCompression):
    """
    Randomized SVD compression using Tropp's algorithms
    
    Implements Algorithm 4.1 (basic) and Algorithm 4.4 (with power iterations)
    from Halko, Martinsson, and Tropp (2011).
    
    Specifically optimized for V matrices in attention mechanisms.
    """
    
    def __init__(self, rank=None, oversampling=10, power_iterations=1, **kwargs):
        """
        Initialize Randomized SVD compression
        
        Args:
            rank: Target rank for compression (if None, uses full rank)
            oversampling: Oversampling parameter p (typically 5-10)
            power_iterations: Number of power iterations q (0-2, typically 1)
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(rank=rank, **kwargs)
        self.oversampling = oversampling
        self.power_iterations = power_iterations
        
        # Validate parameters according to Tropp's recommendations
        if oversampling < 0:
            raise ValueError("Oversampling must be non-negative")
        if power_iterations < 0:
            raise ValueError("Power iterations must be non-negative")
    
    def compress(self, matrix):
        """
        Apply Tropp's randomized SVD compression to the input matrix
        
        Args:
            matrix: Input tensor, typically V matrices of shape (B, nh, T, hs)
                   where B=batch, nh=num_heads, T=seq_len, hs=head_size
            
        Returns:
            Compressed tensor of same shape using randomized SVD
        """
        return self._randomized_svd_tropp(matrix)
    
    def _randomized_svd_tropp(self, v):
        """
        Tropp's Randomized SVD following Algorithm 4.1 with optional power iterations (4.4)
        
        Algorithm 4.1: Randomized SVD
        Given A ∈ ℝ^(m×n), target rank k, oversampling p:
        
        Stage A: Compute approximate basis for range of A
        1. Generate Ω ∈ ℝ^(n×(k+p))  
        2. Y = AΩ (with optional power iterations)
        3. Q, R = qr(Y)
        
        Stage B: Compute SVD of projected matrix
        4. B = Q^T A
        5. B = Ũ Σ Ṽ^T  (SVD)
        6. U = Q Ũ
        
        Return: A ≈ U Σ Ṽ^T
        
        Args:
            v: Input tensor of shape (B, nh, T, hs) - V matrices from attention
            
        Returns:
            Reconstructed tensor of same shape after randomized SVD
        """
        original_shape = v.shape
        
        # Handle different input shapes
        if len(original_shape) == 2:
            # Single matrix case
            return self._compress_single_matrix_randomized(v)
        elif len(original_shape) == 4:
            # Common attention case: (B, nh, T, hs)
            B, nh, T, hs = original_shape
            return self._compress_attention_matrices(v, B, nh, T, hs)
        else:
            # General case: flatten to 3D and process
            return self._compress_general_shape(v)
    
    def _compress_attention_matrices(self, v, B, nh, T, hs):
        """
        Compress V matrices from attention mechanism using randomized SVD
        
        Optimized for the common (B, nh, T, hs) shape in attention
        
        Args:
            v: V matrices of shape (B, nh, T, hs)
            B, nh, T, hs: Batch size, num heads, sequence length, head size
            
        Returns:
            Compressed V matrices of same shape
        """
        # Reshape to (B*nh, T, hs) for batch processing
        v_reshaped = v.reshape(B * nh, T, hs)
        
        # Initialize output tensor
        v_reconstructed = torch.zeros_like(v_reshaped)
        
        # Process each attention head matrix independently
        for i in range(B * nh):
            v_reconstructed[i] = self._compress_single_matrix_randomized(v_reshaped[i])
        
        # Reshape back to original dimensions
        return v_reconstructed.reshape(B, nh, T, hs)
    
    def _compress_single_matrix_randomized(self, A):
        """
        Apply Tropp's randomized SVD to a single 2D matrix
        
        Implements Algorithm 4.1 with optional power iterations (Algorithm 4.4)
        
        Args:
            A: 2D tensor of shape (m, n) - single matrix to compress
            
        Returns:
            Compressed 2D tensor of same shape
        """
        m, n = A.shape
        
        # Determine target rank
        if self.rank is not None:
            target_rank = min(self.rank, min(m, n))
        else:
            target_rank = min(m, n)  # Full rank
        
        # Skip compression if matrix is too small or rank is full
        if target_rank >= min(m, n) or min(m, n) <= 2:
            return A
        
        # Calculate sketch size: k + p (target rank + oversampling)
        sketch_size = min(target_rank + self.oversampling, min(m, n))
        
        # =============================================================
        # Algorithm 4.1: Randomized SVD
        # =============================================================
        
        # Stage A: Compute approximate basis for the range of A
        
        # Step 1: Generate random matrix Ω ∈ ℝ^(n×ℓ) where ℓ = k + p
        Omega = torch.randn(n, sketch_size, device=A.device, dtype=A.dtype)
        
        # Step 2: Form Y = AΩ  
        Y = A @ Omega  # Shape: (m, sketch_size)
        
        # Optional: Power iterations for improved accuracy (Algorithm 4.4)
        # For matrices with slowly decaying singular values
        if self.power_iterations > 0:
            for _ in range(self.power_iterations):
                # Y = A(A^T Y) - power iteration step
                Y = A @ (A.T @ Y)
        
        # Step 3: Orthogonalize Y to get Q using QR decomposition
        Q, _ = torch.linalg.qr(Y)  # Q ∈ ℝ^(m×ℓ)
        
        # Stage B: Compute SVD of projected matrix
        
        # Step 4: Form B = Q^T A  (project A onto range of Q)
        B = Q.T @ A  # Shape: (sketch_size, n)
        
        # Step 5: Compute SVD of the smaller matrix B = Ũ Σ Ṽ^T
        U_tilde, S, Vh = torch.linalg.svd(B, full_matrices=False)
        
        # Step 6: Form U = Q Ũ  (lift back to original space)
        U = Q @ U_tilde
        
        # =============================================================
        # Reconstruction with target rank
        # =============================================================
        
        # Truncate to target rank
        rank = min(target_rank, S.shape[0])
        U_k = U[:, :rank]
        S_k = S[:rank]  
        Vh_k = Vh[:rank, :]
        
        # Reconstruct: A_k = U_k Σ_k Ṽ_k^T
        S_diag = torch.diag(S_k)
        A_reconstructed = U_k @ S_diag @ Vh_k
        
        return A_reconstructed
    
    def _compress_general_shape(self, matrix):
        """
        Handle arbitrary tensor shapes by flattening and processing
        
        Args:
            matrix: Input tensor of arbitrary shape (..., M, N)
            
        Returns:
            Compressed tensor of same shape
        """
        original_shape = matrix.shape
        *batch_dims, M, N = original_shape
        
        # Calculate total batch size
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim
        
        # Reshape to (batch_size, M, N)
        matrix_reshaped = matrix.reshape(batch_size, M, N)
        compressed_reshaped = torch.zeros_like(matrix_reshaped)
        
        # Process each matrix in the batch
        for i in range(batch_size):
            compressed_reshaped[i] = self._compress_single_matrix_randomized(matrix_reshaped[i])
        
        # Reshape back to original shape
        return compressed_reshaped.reshape(original_shape)
    
    def get_computational_complexity(self, matrix_shape):
        """
        Estimate computational complexity of randomized SVD vs standard SVD
        
        Args:
            matrix_shape: Shape tuple (..., m, n) for the matrices
            
        Returns:
            Dictionary with complexity estimates
        """
        *_, m, n = matrix_shape
        k = self.rank if self.rank is not None else min(m, n)
        p = self.oversampling
        q = self.power_iterations
        
        # Standard SVD complexity: O(mn * min(m,n))
        standard_complexity = m * n * min(m, n)
        
        # Randomized SVD complexity following Tropp's analysis:
        # - Matrix-vector products: O(mn(k+p)) 
        # - Power iterations: O(mn(k+p) * q)
        # - QR decomposition: O(m(k+p)^2)
        # - Small SVD: O((k+p)^2 * n)
        # - Reconstruction: O(m*n*k)
        
        sketch_size = k + p
        randomized_complexity = (
            m * n * sketch_size * (1 + q) +  # Matrix products + power iterations
            m * sketch_size**2 +             # QR decomposition
            sketch_size**2 * n +             # Small SVD 
            m * n * k                        # Reconstruction
        )
        
        speedup_ratio = standard_complexity / randomized_complexity
        
        return {
            'standard_svd': standard_complexity,
            'randomized_svd': randomized_complexity,
            'speedup_ratio': speedup_ratio,
            'parameters': {
                'matrix_size': (m, n),
                'target_rank': k,
                'oversampling': p,
                'power_iterations': q,
                'sketch_size': sketch_size
            }
        }
    
    def get_memory_usage(self, matrix_shape):
        """
        Estimate memory usage for randomized SVD
        
        Args:
            matrix_shape: Shape tuple (..., m, n)
            
        Returns:
            Dictionary with memory estimates
        """
        *_, m, n = matrix_shape
        k = self.rank if self.rank is not None else min(m, n)
        sketch_size = k + self.oversampling
        
        # Original matrix memory
        original_memory = m * n
        
        # Additional memory needed during computation:
        # - Omega: n * sketch_size
        # - Y: m * sketch_size  
        # - Q: m * sketch_size
        # - B: sketch_size * n
        # - U, S, Vh from small SVD: sketch_size * (sketch_size + 1 + n)
        
        additional_memory = (
            n * sketch_size +           # Omega
            m * sketch_size +           # Y
            m * sketch_size +           # Q  
            sketch_size * n +           # B
            sketch_size * (sketch_size + 1 + n)  # Small SVD factors
        )
        
        return {
            'original_matrix': original_memory,
            'additional_memory': additional_memory,
            'total_memory': original_memory + additional_memory,
            'memory_overhead_ratio': additional_memory / original_memory
        }