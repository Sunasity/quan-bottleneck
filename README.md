# Pytorch implementation of Google Quantization

### Experimental result on Res-50:

| Weight Bit | Act Bit | top-1 | 
|---|---|---|
| 8 | 8 | 75.46% |


### The quantization scheme follows this paper:
```
@article{jacob2017quantization,
  title={Quantization and training of neural networks for efficient integer-arithmetic-only inference},
  author={Jacob, Benoit and Kligys, Skirmantas and Chen, Bo and Zhu, Menglong and Tang, Matthew and Howard, Andrew and Adam, Hartwig and Kalenichenko, Dmitry},
  journal={arXiv preprint arXiv:1712.05877},
  year={2017}
}
```
