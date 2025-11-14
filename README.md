### Abstract

Multi Head Latent Attention Uses SVD for matrix compression. Our goal it to implement randomized-SVD instead of SVD, and make the process more efficient.

### Standard SVD Implementation

[Standard SVD README](https://github.com/cangokmen/CS599-Randomized-SVD/blob/master/docs/README(SVD).md)

### How to run?
There are multiple versions of the GPT implementation in this repository. 

In any case you are required to install the dependencies:

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

and for testing purposes you can prepare a small dataset as follows:



**1. In order to run regular attention version:**

Train it 
```sh
python -m mla_gpt.cli.train config/train_shakespeare_char.py \
    --device=cpu \
    --compile=False \
    --eval_iters=20 \
    --log_interval=1 \
    --block_size=64 \
    --batch_size=12 \
    --n_layer=4 \
    --n_head=4 \
    --n_embd=128 \
    --max_iters=2000 \
    --lr_decay_iters=2000 \
    --dropout=0.0 \
    --dtype=float32
```
and run to see a sample output:

```sh
python sample.py --out_dir=out-shakespeare-char
```


**2. In order to run Multi-Head Latent Attention with Regular SVD version:**

Train the model with MLA enabled:
```sh
python -m mla_gpt.cli.train config/train_mla_test.py \
    --device=cpu \
    --compile=False \
    --eval_iters=20 \
    --log_interval=1 \
    --block_size=256 \
    --batch_size=4 \
    --n_layer=4 \
    --n_head=4 \
    --n_embd=128 \
    --max_iters=1000 \
    --lr_decay_iters=1000 \
    --dropout=0.0 \
    --dtype=float32
```

Or manually enable MLA with custom settings:
```sh
python -m mla_gpt.cli.train config/train_shakespeare_char.py \
    --use_mla=True \
    --mla_latent_dim=32 \
    --use_svd=True \
    --svd_rank=16 \
    --device=cpu \
    --compile=False
```

**3. In order to run Multi-Head Latent Attention with Randomized SVD version:**

*Note: Randomized SVD implementation is planned for future development.*

### Multi-Head Latent Attention (MLA)

This repository now includes a complete implementation of Multi-Head Latent Attention, which provides:

- **Memory Efficiency**: 2-4x reduction in key-value cache memory during inference
- **Parameter Reduction**: 20-50% fewer attention parameters
- **Configurable Compression**: Adjustable latent dimensions for efficiency/quality trade-off
- **SVD Compatibility**: Works with existing SVD compression features

For detailed documentation, see: [MLA Documentation](docs/MLA_README.md)

#### Quick MLA Example:
```python
from mla_gpt.model import GPT, GPTConfig

# Create model with MLA
config = GPTConfig(
    n_embd=768, n_head=12, n_layer=12,
    use_mla=True,           # Enable MLA
    mla_latent_dim=128,     # Latent dimension
    use_svd=True,           # Optional SVD compression
    svd_rank=64
)
model = GPT(config)
```

#### Compare MLA vs Standard Attention:
```sh
cd examples
python example_mla_usage.py
```






### acknowledgements
MLA implementation is based on nanoGPT implementation of Karpathy.
https://github.com/karpathy/nanoGPT
