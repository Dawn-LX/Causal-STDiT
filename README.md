our code is build upon open-sora (https://github.com/hpcaitech/Open-Sora), with the following features
 - autoregressive video generation, i.e., generating subsequent clips conditioned on last frames of
previous clip
 - calsual generaion (by causal temporal attention)
 - cache sharing, the kv-cache is shared across all the denoising steps. This is differnet to the kv-cache implementation in [live2diff](https://github.com/open-mmlab/Live2Diff)
 - kv-cache queue, i.e., autoregressive generation without the redundant computation of overlapped conditional frames. the old kv-cache will be deququed
 - cyclic temporal positional embeddings (TPEs). i.e., we use cyclic shift to support the kv-cache queue

 - the key difference of our implementation compared to [live2diff](https://github.com/open-mmlab/Live2Diff)
    - our kv-cache is shared across all the denoising steps. They store the kv-cache for all the denoising steps
    - we use a cache queue structure to support the autoregressive generation, facilitated by the cyclic-TPEs

The code is preparing