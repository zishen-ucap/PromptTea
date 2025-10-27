<!-- ## **PromptTea4Wan2.1** -->
# PromptTea4Wan2.2

https://github.com/user-attachments/assets/0c665733-a8a6-401d-9791-8dba6af54bdf

## 📈 Inference Latency Comparisons on a Single H100

| Method | Speedup ↑ | Latency (min) ↓ |
| --- | --- | --- |
| **Wan2.2 (81frames, 1280×720)** |  1x | 28 |
| Ours (PromptTea) | 1.65x~2.07× | 13.5~17 |


## Usage

Follow [Wan2.2](https://github.com/Wan-Video/Wan2.2) to clone the repo and finish the installation. Note: After setting up the environment, make sure to install the additional dependencies `joblib` and `pandas` using `pip install joblib pandas`. Then copy 'prompttea_sample.py', 'codebook_wan2.2.csv', 'fit_model_high_wan2.2.pkl', 'fit_model_low_wan2.2.pkl', 'reference_embeddings_wan2.2.pt' in this repo to the Wan2.2 repo.

For T2V with 14B model, you can use the following command:

```bash
python prompttea_sample.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --t5_cpu --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." --use_pca
```

## Acknowledgements

We would like to thank the contributors to the [Wan2.2](https://github.com/Wan-Video/Wan2.2).