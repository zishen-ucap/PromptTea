# PromptTea: Let Prompts Tell TeaCache the Optimal Threshold

[![arXiv](https://img.shields.io/badge/Arxiv-2507.06739-b31b1b.svg?logo=arXiv)](http://arxiv.org/abs/2507.06739) 

![cover img](./asset/visual.jpg)

## 🫖 Introduction 
This repository contains the official implementation of our paper: PromptTea: Let Prompts Tell TeaCache the Optimal Threshold

## 🔥 Latest News 
- If you like our project, please give us a star ⭐ on GitHub for the latest update.
- **[2025/10/27]** 🔥 Support multi-gpu parallel inference for the [Wan2.2-I2V](https://github.com/Wan-Video/Wan2.2) and [Wan2.2-S2V](https://github.com/Wan-Video/Wan2.2).
- **[2025/10/09]** 🎉 The dataset of [Complex-Simple prompt](./our_data.py) in the paper, as well as the [dataset of the fitting model](./model_fit_prompt), have been made public. The dataset of the fitting model is from [T2V-CompBench](https://github.com/KaiyueSun98/T2V-CompBench).
- **[2025/08/06]** 🔥 Support [Wan2.2](https://github.com/Wan-Video/Wan2.2).
- **[2025/07/30]** 🔥 Support multi-gpu parallel inference for the [Wan2.1](https://github.com/Wan-Video/Wan2.1).  
- **[2025/07/10]** 🎉 Submitted to [arXiv](http://arxiv.org/abs/2507.06739) for review.  
- **[2025/07/10]** 🔥 Released open-source code of PromptTea. Support [CogVideoX1.5](https://github.com/THUDM/CogVideo), [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) and [Wan2.1](https://github.com/Wan-Video/Wan2.1).  

## 🎉 Supported Models 
**Text to Video**
- [PromptTea4CogVideoX1.5](./PromptTea4CogVideoX1.5/README.md)
- [PromptTea4HunyuanVideo](./PromptTea4HunyuanVideo/README.md)
- [PromptTea4Wan2.1](./PromptTea4Wan2.1/README.md)
- [PromptTea4Wan2.2](./PromptTea4Wan2.2/README.md)

**Image to Video**
- [PromptTea4Wan2.2-I2V](./PromptTea4Wan2.2-I2V/README.md)

**Audio-Driven**
- [PromptTea4Wan2.2-S2V](./PromptTea4Wan2.2-S2V/README.md)




## 💐 Acknowledgement 

This repository is built based on [TeaCache](https://github.com/ali-vilab/TeaCache/tree/main), [TeaCache-xDit](https://github.com/MingXiangL/Teacache-xDiT), [FasterCache](https://github.com/Vchitect/FasterCache), [Diffusers](https://github.com/huggingface/diffusers), [CogVideoX](https://github.com/THUDM/CogVideo), [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), [Wan2.1](https://github.com/Wan-Video/Wan2.1) and [Wan2.2](https://github.com/Wan-Video/Wan2.2). Thanks for their contributions!

## 📚 Citation

```bibtex
@article{huang2025prompttea,
  title={PromptTea: Let Prompts Tell TeaCache the Optimal Threshold},
  author={Huang, Zishen and Yang, Chunyu and Ren, Mengyuan},
  journal={arXiv preprint arXiv:2507.06739},
  year={2025}
}
```
## 📜 License
This project is licensed under the Apache 2.0 License.
