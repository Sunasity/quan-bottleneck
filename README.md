# Pytorch implementation of Google Quantization

### Experimental results:

| MOdel | Weight Bit | Act Bit | top-1 | 
|---|---|---|---|
| Res-50 | 8 | 8 | 75.46% |


### The pruning schemes follow these papers:
```
@article{jacob2017quantization,
  title={Quantization and training of neural networks for efficient integer-arithmetic-only inference},
  author={Jacob, Benoit and Kligys, Skirmantas and Chen, Bo and Zhu, Menglong and Tang, Matthew and Howard, Andrew and Adam, Hartwig and Kalenichenko, Dmitry},
  journal={arXiv preprint arXiv:1712.05877},
  year={2017}
}
```
