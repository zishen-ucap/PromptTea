# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import merge_video_audio, save_video, str2bool

import matplotlib.pyplot as plt
import math
from contextlib import contextmanager
from wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from tqdm import tqdm
import gc
import pandas as pd
from wan.modules.model import sinusoidal_embedding_1d
from copy import deepcopy
from torchvision import transforms
import numpy as np
from wan.modules.s2v.s2v_utils import rope_precompute
import torch.cuda.amp as amp
from wan.distributed.sequence_parallel import (
    distributed_attention,
    gather_forward,
    get_rank,
    get_world_size,
)
import joblib

def prepare_codebook(csv_path):
    """
    读取指定路径的 CSV 文件，并将 'num_step' 和 'num_block' 设置为索引，作为 codebook 返回。

    Args:
        csv_path (str): CSV 文件路径

    Returns:
        pd.DataFrame: 设置好 MultiIndex 的 codebook DataFrame
    """
    df = pd.read_csv(csv_path)
    df.set_index(["num_step"], inplace=True)
    return df

def create_poly_features(x, t):
    X = np.column_stack((
        np.ones(len(x)),     # 常数项
        x,                   # x 的一次项
        t,                   # t 的一次项
        x**2,                # x 的平方项
        t**2,                # t 的平方项
        x*t,                 # x 和 t 的交叉项
        x**3,                # x 的立方项
        t**3,                # t 的立方项
        x**2 * t,            # x^2 * t
        x * t**2,            # x * t^2
        x**4,                # x 的四次项
        t**4                 # t 的四次项
    ))
    return X


def s2v_generate(
    self,
    input_prompt,
    ref_image_path,
    audio_path,
    num_repeat=1,
    pose_video=None,
    max_area=720 * 1280,
    infer_frames=80,
    shift=5.0,
    sample_solver='unipc',
    sampling_steps=40,
    guide_scale=5.0,
    n_prompt="",
    seed=-1,
    offload_model=True,
    init_first_frame=False,
    cache_threshould=0.1,
    cfg_threshould=0.02
):
    r"""
    Generates video frames from input image and text prompt using diffusion process.

    Args:
        input_prompt (`str`):
            Text prompt for content generation.
        ref_image_path ('str'):
            Input image path
        audio_path ('str'):
            Audio for video driven
        num_repeat ('int'):
            Number of clips to generate; will be automatically adjusted based on the audio length
        pose_video ('str'):
            If provided, uses a sequence of poses to drive the generated video
        max_area (`int`, *optional*, defaults to 720*1280):
            Maximum pixel area for latent space calculation. Controls video resolution scaling
        infer_frames (`int`, *optional*, defaults to 80):
            How many frames to generate per clips. The number should be 4n
        shift (`float`, *optional*, defaults to 5.0):
            Noise schedule shift parameter. Affects temporal dynamics
            [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
        sample_solver (`str`, *optional*, defaults to 'unipc'):
            Solver used to sample the video.
        sampling_steps (`int`, *optional*, defaults to 40):
            Number of diffusion sampling steps. Higher values improve quality but slow generation
        guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
            Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            If tuple, the first guide_scale will be used for low noise model and
            the second guide_scale will be used for high noise model.
        n_prompt (`str`, *optional*, defaults to ""):
            Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
        seed (`int`, *optional*, defaults to -1):
            Random seed for noise generation. If -1, use random seed
        offload_model (`bool`, *optional*, defaults to True):
            If True, offloads models to CPU during generation to save VRAM
        init_first_frame (`bool`, *optional*, defaults to False):
            Whether to use the reference image as the first frame (i.e., standard image-to-video generation)

    Returns:
        torch.Tensor:
            Generated video frames tensor. Dimensions: (C, N H, W) where:
            - C: Color channels (3 for RGB)
            - N: Number of frames (81)
            - H: Frame height (from max_area)
            - W: Frame width from max_area)
    """
    # preprocess
    size = self.get_gen_size(
        size=None,
        max_area=max_area,
        ref_image_path=ref_image_path,
        pre_video_path=None)
    HEIGHT, WIDTH = size
    channel = 3

    resize_opreat = transforms.Resize(min(HEIGHT, WIDTH))
    crop_opreat = transforms.CenterCrop((HEIGHT, WIDTH))
    tensor_trans = transforms.ToTensor()

    ref_image = None
    motion_latents = None

    if ref_image is None:
        ref_image = np.array(Image.open(ref_image_path).convert('RGB'))
    if motion_latents is None:
        motion_latents = torch.zeros(
            [1, channel, self.motion_frames, HEIGHT, WIDTH],
            dtype=self.param_dtype,
            device=self.device)

    # extract audio emb
    audio_emb, nr = self.encode_audio(audio_path, infer_frames=infer_frames)
    if num_repeat is None or num_repeat > nr:
        num_repeat = nr

    lat_motion_frames = (self.motion_frames + 3) // 4
    model_pic = crop_opreat(resize_opreat(Image.fromarray(ref_image)))

    ref_pixel_values = tensor_trans(model_pic)
    ref_pixel_values = ref_pixel_values.unsqueeze(1).unsqueeze(
        0) * 2 - 1.0  # b c 1 h w
    ref_pixel_values = ref_pixel_values.to(
        dtype=self.vae.dtype, device=self.vae.device)
    ref_latents = torch.stack(self.vae.encode(ref_pixel_values))

    # encode the motion latents
    videos_last_frames = motion_latents.detach()
    drop_first_motion = self.drop_first_motion
    if init_first_frame:
        drop_first_motion = False
        motion_latents[:, :, -6:] = ref_pixel_values
    motion_latents = torch.stack(self.vae.encode(motion_latents))

    # get pose cond input if need
    COND = self.load_pose_cond(
        pose_video=pose_video,
        num_repeat=num_repeat,
        infer_frames=infer_frames,
        size=size)

    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)

    if n_prompt == "":
        n_prompt = self.sample_neg_prompt

    # preprocess
    if not self.t5_cpu:
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
    else:
        context = self.text_encoder([input_prompt], torch.device('cpu'))
        context_null = self.text_encoder([n_prompt], torch.device('cpu'))
        context = [t.to(self.device) for t in context]
        context_null = [t.to(self.device) for t in context_null]

    out = []
    # evaluation mode
    with (
            torch.amp.autocast('cuda', dtype=self.param_dtype),
            torch.no_grad(),
    ):
        for r in range(num_repeat):
            seed_g = torch.Generator(device=self.device)
            seed_g.manual_seed(seed + r)

            lat_target_frames = (infer_frames + 3 + self.motion_frames
                                ) // 4 - lat_motion_frames
            target_shape = [lat_target_frames, HEIGHT // 8, WIDTH // 8]
            noise = [
                torch.randn(
                    16,
                    target_shape[0],
                    target_shape[1],
                    target_shape[2],
                    dtype=self.param_dtype,
                    device=self.device,
                    generator=seed_g)
            ]
            max_seq_len = np.prod(target_shape) // 4

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            latents = deepcopy(noise)
            with torch.no_grad():
                left_idx = r * infer_frames
                right_idx = r * infer_frames + infer_frames
                cond_latents = COND[r] if pose_video else COND[0] * 0
                cond_latents = cond_latents.to(
                    dtype=self.param_dtype, device=self.device)
                audio_input = audio_emb[..., left_idx:right_idx]
            input_motion_latents = motion_latents.clone()

            arg_c = {
                'context': context[0:1],
                'seq_len': max_seq_len,
                'cond_states': cond_latents,
                "motion_latents": input_motion_latents,
                'ref_latents': ref_latents,
                "audio_input": audio_input,
                "motion_frames": [self.motion_frames, lat_motion_frames],
                "drop_motion_frames": drop_first_motion and r == 0,
                'is_cond':True,
                "cache_threshould":cache_threshould,
                "cfg_threshould":cfg_threshould
            }
            if guide_scale > 1:
                arg_null = {
                    'context': context_null[0:1],
                    'seq_len': max_seq_len,
                    'cond_states': cond_latents,
                    "motion_latents": input_motion_latents,
                    'ref_latents': ref_latents,
                    "audio_input": 0.0 * audio_input,
                    "motion_frames": [
                        self.motion_frames, lat_motion_frames
                    ],
                    "drop_motion_frames": drop_first_motion and r == 0,
                    'is_cond':False,
                    "cache_threshould":cache_threshould,
                    "cfg_threshould":cfg_threshould
                }
            if offload_model or self.init_on_cpu:
                self.noise_model.to(self.device)
                torch.cuda.empty_cache()

            for i, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents[0:1]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                noise_pred_cond = self.noise_model(
                    latent_model_input, t=timestep, denoise_time=i, **arg_c)

                if guide_scale > 1:
                    noise_pred_uncond = self.noise_model(
                        latent_model_input, t=timestep, denoise_time=i, **arg_null)
                    noise_pred = [
                        u + guide_scale * (c - u)
                        for c, u in zip(noise_pred_cond, noise_pred_uncond)
                    ]
                else:
                    noise_pred = noise_pred_cond

                temp_x0 = sample_scheduler.step(
                    noise_pred[0].unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents[0] = temp_x0.squeeze(0)

            if offload_model:
                self.noise_model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            latents = torch.stack(latents)
            if not (drop_first_motion and r == 0):
                decode_latents = torch.cat([motion_latents, latents], dim=2)
            else:
                decode_latents = torch.cat([ref_latents, latents], dim=2)
            image = torch.stack(self.vae.decode(decode_latents))
            image = image[:, :, -(infer_frames):]
            if (drop_first_motion and r == 0):
                image = image[:, :, 3:]

            overlap_frames_num = min(self.motion_frames, image.shape[2])
            videos_last_frames = torch.cat([
                videos_last_frames[:, :, overlap_frames_num:],
                image[:, :, -overlap_frames_num:]
            ],
                                            dim=2)
            videos_last_frames = videos_last_frames.to(
                dtype=motion_latents.dtype, device=motion_latents.device)
            motion_latents = torch.stack(
                self.vae.encode(videos_last_frames))
            out.append(image.cpu())

    videos = torch.cat(out, dim=2)
    del noise, latents
    del sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    return videos[0] if self.rank == 0 else None


def prompttea_forward(
        self,
        x,
        t,
        context,
        seq_len,
        ref_latents,
        motion_latents,
        cond_states,
        denoise_time,
        audio_input=None,
        motion_frames=[17, 5],
        add_last_motion=2,
        drop_motion_frames=False,
        cache_threshould=0.1,
        cfg_threshould=0.02,
        is_cond=True,
        *extra_args,
        **extra_kwargs):
    """
    x:                  A list of videos each with shape [C, T, H, W].
    t:                  [B].
    context:            A list of text embeddings each with shape [L, C].
    seq_len:            A list of video token lens, no need for this model.
    ref_latents         A list of reference image for each video with shape [C, 1, H, W].
    motion_latents      A list of  motion frames for each video with shape [C, T_m, H, W].
    cond_states         A list of condition frames (i.e. pose) each with shape [C, T, H, W].
    audio_input         The input audio embedding [B, num_wav2vec_layer, C_a, T_a].
    motion_frames       The number of motion frames and motion latents frames encoded by vae, i.e. [17, 5]
    add_last_motion     For the motioner, if add_last_motion > 0, it means that the most recent frame (i.e., the last frame) will be added.
                        For frame packing, the behavior depends on the value of add_last_motion:
                        add_last_motion = 0: Only the farthest part of the latent (i.e., clean_latents_4x) is included.
                        add_last_motion = 1: Both clean_latents_2x and clean_latents_4x are included.
                        add_last_motion = 2: All motion-related latents are used.
    drop_motion_frames  Bool, whether drop the motion frames info
    """
    add_last_motion = self.add_last_motion * add_last_motion
    audio_input = torch.cat([
        audio_input[..., 0:1].repeat(1, 1, 1, motion_frames[0]), audio_input
    ],
                            dim=-1)
    audio_emb_res = self.casual_audio_encoder(audio_input)
    if self.enbale_adain:
        audio_emb_global, audio_emb = audio_emb_res
        self.audio_emb_global = audio_emb_global[:,
                                                    motion_frames[1]:].clone()
    else:
        audio_emb = audio_emb_res
    self.merged_audio_emb = audio_emb[:, motion_frames[1]:, :]

    device = self.patch_embedding.weight.device

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    # cond states
    cond = [self.cond_encoder(c.unsqueeze(0)) for c in cond_states]
    x = [x_ + pose for x_, pose in zip(x, cond)]

    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)

    original_grid_sizes = deepcopy(grid_sizes)
    grid_sizes = [[torch.zeros_like(grid_sizes), grid_sizes, grid_sizes]]

    # ref and motion
    self.lat_motion_frames = motion_latents[0].shape[1]

    ref = [self.patch_embedding(r.unsqueeze(0)) for r in ref_latents]
    batch_size = len(ref)
    height, width = ref[0].shape[3], ref[0].shape[4]
    ref_grid_sizes = [[
        torch.tensor([30, 0, 0]).unsqueeze(0).repeat(batch_size,
                                                        1),  # the start index
        torch.tensor([31, height,
                        width]).unsqueeze(0).repeat(batch_size,
                                                    1),  # the end index
        torch.tensor([1, height, width]).unsqueeze(0).repeat(batch_size, 1),
    ]  # the range
                        ]

    ref = [r.flatten(2).transpose(1, 2) for r in ref]  # r: 1 c f h w
    self.original_seq_len = seq_lens[0]

    seq_lens = seq_lens + torch.tensor([r.size(1) for r in ref],
                                        dtype=torch.long)

    grid_sizes = grid_sizes + ref_grid_sizes

    x = [torch.cat([u, r], dim=1) for u, r in zip(x, ref)]

    # Initialize masks to indicate noisy latent, ref latent, and motion latent.
    # However, at this point, only the first two (noisy and ref latents) are marked;
    # the marking of motion latent will be implemented inside `inject_motion`.
    mask_input = [
        torch.zeros([1, u.shape[1]], dtype=torch.long, device=x[0].device)
        for u in x
    ]
    for i in range(len(mask_input)):
        mask_input[i][:, self.original_seq_len:] = 1

    # compute the rope embeddings for the input
    x = torch.cat(x)
    b, s, n, d = x.size(0), x.size(
        1), self.num_heads, self.dim // self.num_heads
    self.pre_compute_freqs = rope_precompute(
        x.detach().view(b, s, n, d), grid_sizes, self.freqs, start=None)

    x = [u.unsqueeze(0) for u in x]
    self.pre_compute_freqs = [
        u.unsqueeze(0) for u in self.pre_compute_freqs
    ]

    x, seq_lens, self.pre_compute_freqs, mask_input = self.inject_motion(
        x,
        seq_lens,
        self.pre_compute_freqs,
        mask_input,
        motion_latents,
        drop_motion_frames=drop_motion_frames,
        add_last_motion=add_last_motion)

    x = torch.cat(x, dim=0)
    self.pre_compute_freqs = torch.cat(self.pre_compute_freqs, dim=0)
    mask_input = torch.cat(mask_input, dim=0)

    x = x + self.trainable_cond_mask(mask_input).to(x.dtype)

    # time embeddings
    if self.zero_timestep:
        t = torch.cat([t, torch.zeros([1], dtype=t.dtype, device=t.device)])
    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    if self.zero_timestep:
        e = e[:-1]
        zero_e0 = e0[-1:]
        e0 = e0[:-1]
        token_len = x.shape[1]
        e0 = torch.cat([
            e0.unsqueeze(2),
            zero_e0.unsqueeze(2).repeat(e0.size(0), 1, 1, 1)
        ],
                        dim=2)
        e0 = [e0, self.original_seq_len]
    else:
        e0 = e0.unsqueeze(2).repeat(1, 1, 2, 1)
        e0 = [e0, 0]

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    # grad ckpt args
    def create_custom_forward(module, return_dict=None):

        def custom_forward(*inputs, **kwargs):
            if return_dict is not None:
                return module(*inputs, **kwargs, return_dict=return_dict)
            else:
                return module(*inputs, **kwargs)

        return custom_forward

    if self.use_context_parallel:
        # sharded tensors for long context attn
        sp_rank = get_rank()
        x = torch.chunk(x, get_world_size(), dim=1)
        sq_size = [u.shape[1] for u in x]
        sq_start_size = sum(sq_size[:sp_rank])
        x = x[sp_rank]
        # Confirm the application range of the time embedding in e0[0] for each sequence:
        # - For tokens before seg_id: apply e0[0][:, :, 0]
        # - For tokens after seg_id: apply e0[0][:, :, 1]
        sp_size = x.shape[1]
        seg_idx = e0[1] - sq_start_size
        e0[1] = seg_idx

        self.pre_compute_freqs = torch.chunk(
            self.pre_compute_freqs, get_world_size(), dim=1)
        self.pre_compute_freqs = self.pre_compute_freqs[sp_rank]

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.pre_compute_freqs,
        context=context,
        context_lens=context_lens)
    # 判断环节
    if self.enable_teacache:
        modulated_inp = e0[0][0]
        if is_cond:
            if denoise_time==0:
                should_calc_cond = True
                self.accumulated_rel_l1_distance = 0
            else:
                delta_e0 = ((modulated_inp-self.previous_e0).abs().mean() / self.previous_e0.abs().mean()).cpu().item()
                x_and_t = create_poly_features(np.array([delta_e0]), np.array([denoise_time]))
                fit_model = self.fit_model
                l1 = fit_model.predict(x_and_t)
                self.accumulated_rel_l1_distance += l1
                if self.accumulated_rel_l1_distance < cache_threshould:
                    should_calc_cond = False
                else:
                    should_calc_cond = True
                    self.accumulated_rel_l1_distance = 0
            self.should_calc = should_calc_cond
            self.previous_e0 = modulated_inp
        
        # uncond
        else:
            should_calc_uncond = self.should_calc
            cfg_l1 = self.cfg_codebook.get(denoise_time, 0)
            self.accumulated_cfg_l1_distance += cfg_l1
            if self.accumulated_cfg_l1_distance < cfg_threshould:
                should_calc_cfg = False
            else:
                should_calc_cfg = True
                self.accumulated_cfg_l1_distance = 0

    if self.enable_teacache:
        if is_cond:
            if not should_calc_cond:
                x += self.previous_residual_cond
                # print(f'is_cond={is_cond}, denoise_time={denoise_time}成功跳过')
            else:
                ori_x = x.clone()
                for idx, block in enumerate(self.blocks):
                    x = block(x, **kwargs)
                    x = self.after_transformer_block(idx, x)
                self.previous_residual_cond = x - ori_x
        # uncond
        else:
            if not should_calc_uncond:
                x += self.previous_residual_uncond
                # print(f'is_cond={is_cond}, denoise_time={denoise_time}成功跳过')
            else:
                ori_x = x.clone()
                if should_calc_cfg:
                    for idx, block in enumerate(self.blocks):
                        x = block(x, **kwargs)
                        x = self.after_transformer_block(idx, x)
                else:
                    x = x + self.previous_residual_cond
                    # print(f"第{denoise_time}步使用cfgcache---------------------------")
                self.previous_residual_uncond = x - ori_x

    else:
        for idx, block in enumerate(self.blocks):
            x = block(x, **kwargs)
            x = self.after_transformer_block(idx, x)

    # for idx, block in enumerate(self.blocks):
    #     x = block(x, **kwargs)
    #     x = self.after_transformer_block(idx, x)

    # Context Parallel
    if self.use_context_parallel:
        x = gather_forward(x.contiguous(), dim=1)
    # unpatchify
    x = x[:, :self.original_seq_len]
    # head
    x = self.head(x, e)
    x = self.unpatchify(x, original_grid_sizes)
    return [u.float() for u in x]




EXAMPLE_PROMPT = {
    "t2v-A14B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "i2v-A14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
    "ti2v-5B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "s2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
        "audio":
            "examples/talk.wav",
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]
    if args.audio is None and "audio" in EXAMPLE_PROMPT[args.task]:
        args.audio = EXAMPLE_PROMPT[args.task]["audio"]

    if args.task == "i2v-A14B":
        assert args.image is not None, "Please specify the image path for i2v."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    if not 's2v' in args.task:
        assert args.size in SUPPORTED_SIZES[
            args.
            task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")

    # following args only works for s2v
    parser.add_argument(
        "--num_clip",
        type=int,
        default=None,
        help="Number of video clips to generate, the whole video will not exceed the length of audio."
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to the audio file, e.g. wav, mp3")
    parser.add_argument(
        "--pose_video",
        type=str,
        default=None,
        help="Provide Dw-pose sequence to do Pose Driven")
    parser.add_argument(
        "--start_from_ref",
        action="store_true",
        default=False,
        help="whether set the reference image as the starting point for generation"
    )
    parser.add_argument(
        "--infer_frames",
        type=int,
        default=80,
        help="Number of frames per clip, 48 or 80 or others (must be multiple of 4) for 14B s2v"
    )
    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=0.15,
        help="Higher speedup will cause to worse quality")
    parser.add_argument(
        "--codebook_path",
        type=str,
        default="codebook_wan2.2_s2v.csv",
        help="The path to the codebook.")  
    parser.add_argument(
        "--fit_model_path",
        type=str,
        default="fit_model_s2v.pkl",
        help="The path to the codebook.")  
    parser.add_argument(
        "--cfgcache_thresh",
        type=float,
        default=0.02,
        help="Higher speedup will cause to worse quality")

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), f"sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Input prompt: {args.prompt}")
    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        logging.info(f"Input image: {args.image}")

    # prompt extend
    if args.use_prompt_extend:
        logging.info("Extending prompt ...")
        if rank == 0:
            prompt_output = prompt_expander(
                args.prompt,
                image=img,
                tar_lang=args.prompt_extend_target_lang,
                seed=args.base_seed)
            if prompt_output.status == False:
                logging.info(
                    f"Extending prompt failed: {prompt_output.message}")
                logging.info("Falling back to original prompt.")
                input_prompt = args.prompt
            else:
                input_prompt = prompt_output.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")

    if "t2v" in args.task:
        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        logging.info(f"Generating video ...")
        video = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
    elif "ti2v" in args.task:
        logging.info("Creating WanTI2V pipeline.")
        wan_ti2v = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        logging.info(f"Generating video ...")
        video = wan_ti2v.generate(
            args.prompt,
            img=img,
            size=SIZE_CONFIGS[args.size],
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
    elif "s2v" in args.task:
        logging.info("Creating WanS2V pipeline.")
        wan_s2v = wan.WanS2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        if args.dit_fsdp:
            # print(f'交换前{wan_i2v.high_noise_model.module.forward}')
            wan_noise_model = wan_s2v.noise_model.module
           
        else:
            wan_noise_model = wan_s2v.noise_model

        # cfgcache
        codebook = prepare_codebook(args.codebook_path)
        codebook_dict = {(i): value for i, value in zip(codebook.index.get_level_values(0),
                                                        codebook['output_value'])}
        # wan_noise_model.__class__.cfgcache_thresh = args.cfgcache_thresh
        wan_noise_model.__class__.accumulated_cfg_l1_distance = 0
        wan_noise_model.__class__.cfg_codebook = codebook_dict


        # TeaCache
        wan_s2v.__class__.generate = s2v_generate
        # wan_s2v.__class__.low_step = -1
        wan_noise_model.__class__.enable_teacache = True
        wan_noise_model.__class__.forward = prompttea_forward
        # wan_noise_model.__class__.teacache_thresh_high = args.teacache_thresh_high
        # wan_noise_model.__class__.teacache_thresh_low = args.teacache_thresh_low
        wan_noise_model.__class__.accumulated_rel_l1_distance = 0
        wan_noise_model.__class__.previous_e0 = None
        wan_noise_model.__class__.previous_residual_even = None
        wan_noise_model.__class__.previous_residual_odd = None
        wan_noise_model.__class__.should_calc = True
        wan_noise_model.__class__.fit_model = joblib.load(args.fit_model_path)


        logging.info(f"Generating video ...")
        video = wan_s2v.generate(
            input_prompt=args.prompt,
            ref_image_path=args.image,
            audio_path=args.audio,
            num_repeat=args.num_clip,
            pose_video=args.pose_video,
            max_area=MAX_AREA_CONFIGS[args.size],
            infer_frames=args.infer_frames,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            init_first_frame=args.start_from_ref,
            cache_threshould=args.teacache_thresh,
            cfg_threshould=args.cfgcache_thresh
        )

    else:
        logging.info("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        logging.info("Generating video ...")
        video = wan_i2v.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.mp4'
            args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{formatted_prompt}_{formatted_time}" + suffix

        logging.info(f"Saving generated video to {args.save_file}")
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        if "s2v" in args.task:
            merge_video_audio(video_path=args.save_file, audio_path=args.audio)
    del video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
