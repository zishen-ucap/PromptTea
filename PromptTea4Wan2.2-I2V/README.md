<!-- ## **PromptTea4Wan2.1** -->
# PromptTea4Wan2.2-I2V

https://github.com/user-attachments/assets/74e0ddb3-114e-4f59-9ad5-9b0084bcdaae

## 📈 Inference Latency Comparisons on 2*H100

| Method | Speedup ↑ | Latency (s) ↓ |
| --- | --- | --- |
| **Wan2.2 (480p)** |  1x | 225 |
| Ours (PromptTea) | 1.90x | 118 |


## Usage

Follow [Wan2.2](https://github.com/Wan-Video/Wan2.2) to clone the repo and finish the installation. Note: After setting up the environment, make sure to install the additional dependencies `joblib` and `pandas` using `pip install joblib pandas`. Then copy 'prompttea_i2v.py', 'codebook_wan2.2_i2v.csv', 'fit_model_high_i2v.pkl', 'fit_model_low_i2v.pkl', in this repo to the Wan2.2 repo.


To use BF16 precision with multi-GPU, add this code at Line 163 of wan/image2video.py:

```python
        if dit_fsdp:
            model = shard_fn(model) # Line 163
            # Fix BF16 mismatch in multi-GPU
            if convert_model_dtype:
                model.to(self.param_dtype)

```

For Single-GPU I2V inference, you can use the following command:

```bash
python generate.py --task i2v-A14B --size 832*480 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." --base_seed 42
```

For Multi-GPU inference using FSDP + DeepSpeed Ulysses, you can use the following command:

```bash
torchrun --nproc_per_node=2 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 2 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." --convert_model_dtype --base_seed 42
```


## Acknowledgements

We would like to thank the contributors to the [Wan2.2](https://github.com/Wan-Video/Wan2.2).