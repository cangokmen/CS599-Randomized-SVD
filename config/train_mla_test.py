# Configuration for GPT-2 training with Multi-Head Latent Attention (MLA)
# This demonstrates the MLA implementation with various settings

wandb_log = False  # Disable wandb for quick testing
wandb_project = 'mla-experiments'
wandb_run_name = 'gpt2-mla-test'

# Smaller scale for testing MLA functionality
batch_size = 4
block_size = 256
gradient_accumulation_steps = 1

# Shorter training for testing
max_iters = 1000
lr_decay_iters = 1000

# More frequent evaluation for monitoring MLA performance
eval_interval = 100
eval_iters = 50
log_interval = 10

# Standard hyperparameters
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

# Model configuration
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False

# MLA Configuration - ENABLED for testing
use_mla = True              # Enable Multi-Head Latent Attention
mla_latent_dim = 32         # Latent dimension (default would be 128//4//2 = 16, but we set higher for testing)

# SVD Configuration - Can be combined with MLA
use_svd = True              # Enable SVD compression on value matrices
svd_rank = 16               # Rank for SVD approximation

# Data configuration
dataset = 'shakespeare_char'