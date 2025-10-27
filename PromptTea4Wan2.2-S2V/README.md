<!-- ## **PromptTea4Wan2.1** -->
# PromptTea4Wan2.2-S2V

https://github.com/user-attachments/assets/bc6b1943-0258-4fea-82a7-3ba0514a4063

## 📈 Inference Latency Comparisons on 2*H100

| Method | Speedup ↑ | Latency (s) ↓ |
| --- | --- | --- |
| **Wan2.2 (480p)** |  1x | 555 |
| Ours (PromptTea) | 2.22x | 250 |


## Usage

Follow [Wan2.2](https://github.com/Wan-Video/Wan2.2) to clone the repo and finish the installation. Note: After setting up the environment, make sure to install the additional dependencies `joblib` and `pandas` using `pip install joblib pandas`. Then copy 'prompttea_s2v.py', 'codebook_wan2.2_s2v.csv', 'fit_model_s2v.pkl', in this repo to the Wan2.2 repo.

For Single-GPU Speech-to-Video inference, you can use the following command:

```bash
python prmpttea_s2v.py  --task s2v-14B --size 832*480 --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype --prompt "A woman giving a speech"  --image "examples/pose.png" --audio "examples/talk.wav" --base_seed 42
```

For Multi-GPU inference using FSDP + DeepSpeed Ulysses, you can use the following command:

```bash
torchrun --nproc_per_node=2 generate.py --task s2v-14B --size 480*832 --ckpt_dir ./Wan2.2-S2V-14B/ --dit_fsdp --t5_fsdp --ulysses_size 2 --prompt "A woman giving a speech" --image "examples/pose.png" --audio "examples/talk.wav" --base_seed 42
```


## Acknowledgements

We would like to thank the contributors to the [Wan2.2](https://github.com/Wan-Video/Wan2.2).