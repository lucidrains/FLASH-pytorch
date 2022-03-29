<img src="./flash.png" width="500px"></img>

## FLASH - Pytorch

Implementation of the Transformer variant proposed in the paper <a href="https://arxiv.org/abs/2202.10447">Transformer Quality in Linear Time</a>

## Install

```bash
$ pip install FLASH-pytorch
```

## Usage

The main novel circuit in this paper is the "Gated Attention Unit", which they claim can replace multi-headed attention while reducing it to just one head.

It uses a relu squared activation in place of the softmax, the activation of which was first seen in the <a href="https://arxiv.org/abs/2109.08668">Primer paper</a>, and the use of ReLU in <a href="https://arxiv.org/abs/2104.07012">ReLA Transformer</a>. The gating style seems mostly inspired by <a href="https://arxiv.org/abs/2105.08050">gMLPs</a>.

```python
import torch
from flash_pytorch import GAU

gau = GAU(
    dim = 512,
    query_key_dim = 128,     # query / key dimension
    causal = True,           # autoregressive or not
    expansion_factor = 2,    # hidden dimension = dim * expansion_factor
)

x = torch.randn(1, 1024, 512)
out = gau(x) # (1, 1024, 512)
```

The authors then combine `GAU` with Katharopoulos linear attention, using grouping of the sequences to overcome a known issue with autoregressive linear attention.

This combination of the quadratic gated attention unit with grouped linear attention they named FLASH

You can also use this quite easily

```python
import torch
from flash_pytorch import FLASH

flash = FLASH(
    dim = 512,
    group_size = 256,             # group size
    causal = True,                # autoregressive or not
    query_key_dim = 128,          # query / key dimension
    expansion_factor = 2.         # hidden dimension = dim * expansion_factor
)

x = torch.randn(1, 1111, 512)     # sequence will be auto-padded to nearest group size
out = flash(x) # (1, 1111, 512)
```

Finally, you can use the full FLASH transformer as mentioned in the paper. This contains all the positional embeddings mentioned in the paper. Absolute positional embedding uses scaled sinusoidal. GAU quadratic attention will get one-headed T5 relative positional bias. On top of all this, both GAU attention as well as the linear attention will be rotary embedded (RoPE).

```python
import torch
from flash_pytorch import FLASHTransformer

model = FLASHTransformer(
    num_tokens = 20000,          # number of tokens
    dim = 512,                   # model dimension
    depth = 12,                  # depth
    causal = True,               # autoregressive or not
    group_size = 256,            # size of the groups
    query_key_dim = 128,         # dimension of queries / keys
    expansion_factor = 2.,       # hidden dimension = dim * expansion_factor
    norm_type = 'scalenorm',     # in the paper, they claimed scalenorm led to faster training at no performance hit. the other option is 'layernorm' (also default)
    shift_tokens = True          # discovered by an independent researcher in Shenzhen @BlinkDL, this simply shifts half of the feature space forward one step along the sequence dimension - greatly improved convergence even more in my local experiments
)

x = torch.randint(0, 20000, (1, 1024))
logits = model(x) # (1, 1024, 20000)
```

## Test on Autoregressive Enwik8

```bash
$ python train.py
```

## Citations

```bibtex
@article{Hua2022TransformerQI,
    title   = {Transformer Quality in Linear Time},
    author  = {Weizhe Hua and Zihang Dai and Hanxiao Liu and Quoc V. Le},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2202.10447}
}
```

```bibtex
@software{peng_bo_2021_5196578,
    author    = {PENG Bo},
    title     = {BlinkDL/RWKV-LM: 0.01},
    month     = {aug},
    year      = {2021},
    publisher = {Zenodo},
    version   = {0.01},
    doi       = {10.5281/zenodo.5196578},
    url       = {https://doi.org/10.5281/zenodo.5196578}
}
```
