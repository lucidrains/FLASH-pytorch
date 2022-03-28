<img src="./flash.png" width="500px"></img>

## FLASH - Pytorch (wip)

Implementation of the Transformer variant proposed in the paper <a href="https://arxiv.org/abs/2202.10447">Transformer Quality in Linear Time</a>

## Install

```bash
$ pip install FLASH-pytorch
```

## Usage

The main novel circuit in this paper is the "Gated Attention Unit", which they claim can replace multi-headed attention while reducing it to just one head. It uses a relu squared activation in place of the softmax, which was first seen in the <a href="https://arxiv.org/abs/2109.08668">Primer paper</a>

```python
from flash_pytorch import GAU
import torch

model = GAU(
    dim = 512,
    query_key_dim = 128,
    hidden_dim = 1024
)

x = torch.randn(1, 1024, 512)
out = model(x) # (1, 1024, 512)
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
