"""
Compression algorithms for matrix factorization and dimensionality reduction
"""

from .base_compression import BaseCompression
from .svd_compression import SVDCompression
from .randomized_svd_compression import RandomizedSVDCompression

__all__ = [
    'BaseCompression',
    'SVDCompression', 
    'RandomizedSVDCompression'
]