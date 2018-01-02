# nl-styletransfer
Codes for paper Photographic Style Transfer Based on Normalization Loss

## Disclaimer 
**This software is published for academic and non-commercial use only.**

## Setup
This code is based on torch. It has been tested on Ubuntu 16.04 LTS.

Dependencies:
* [Torch](https://github.com/torch/torch7) (with [matio-ffi](https://github.com/soumith/matio-ffi.torch) and [loadcaffe](https://github.com/szagoruyko/loadcaffe))
* [Matlab](https://www.mathworks.com/) or [Octave](https://www.gnu.org/software/octave/)

CUDA backend:
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [cudnn](https://developer.nvidia.com/cudnn)

Download VGG-19:
```
sh models/download_models.sh
```

Compile ``cuda_utils.cu`` (Adjust ``PREFIX`` and ``NVCC_PREFIX`` in ``makefile`` for your machine):
```
make clean && make
