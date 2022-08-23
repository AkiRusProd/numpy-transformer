# numpy-transformer `(under development)`
This is a numpy implementation of the Transformer model in "Attention is All You Need", that runs at CPU

<p align="center">
<img src="images/The-Transformer-model-architecture.png" width=50% height=50%>
</p>

Some methods were borrowed from my [numpy-nn-model](https://github.com/AkiRusProd/numpy-nn-model) repository

Since the model is implemented at numpy, it runs on the CPU. Therefore, I have to set more gentle conditions for training the model. Otherwise CPU training will take **hundreds or thousands of hours**

### References:
 - https://arxiv.org/abs/1706.03762 - article "Attention is All You Need"

### TODO:
1) add pretrained model (very soon)
2) clean up and refactor code

