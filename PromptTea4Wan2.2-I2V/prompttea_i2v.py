# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
import torchvision.transforms.functional as TF

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import save_video, str2bool
from wan.modules.model import sinusoidal_embedding_1d
import pandas as pd
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
import numpy as np
import joblib

from wan.modules.model import sinusoidal_embedding_1d
from wan.distributed.util import gather_forward, get_rank, get_world_size
from wan.distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
import types


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
}


def prompttea_configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                        convert_model_dtype):
    """
    Configures a model object. This includes setting evaluation modes,
    applying distributed parallel strategy, and handling device placement.

    Args:
        model (torch.nn.Module):
            The model instance to configure.
        use_sp (`bool`):
            Enable distribution strategy of sequence parallel.
        dit_fsdp (`bool`):
            Enable FSDP sharding for DiT model.
        shard_fn (callable):
            The function to apply FSDP sharding.
        convert_model_dtype (`bool`):
            Convert DiT model parameters dtype to 'config.param_dtype'.
            Only works without FSDP.

    Returns:
        torch.nn.Module:
            The configured model.
    """
    print(f"换过函数了换过函数了！！！！！！！！！！！！！！！")
    model.eval().requires_grad_(False)

    if use_sp:
        for block in model.blocks:
            block.self_attn.forward = types.MethodType(
                sp_attn_forward, block.self_attn)
        model.forward = types.MethodType(sp_dit_forward, model)

    if dist.is_initialized():
        dist.barrier()

    if dit_fsdp:
        model = shard_fn(model)
        if convert_model_dtype:
            model.to(self.param_dtype)
    else:
        if convert_model_dtype:
            model.to(self.param_dtype)
        if not self.init_on_cpu:
            model.to(self.device)

    return model

def calculate_PCA_threshold(reference_embeddings, context, high_min=0.3, high_max=0.5, low_min=0.2, low_max=0.4):
    """
    计算提示词复杂度感知后的TeaCache阈值

    Args:
        reference_embeddings (dict): 相对提示词嵌入
        context (list): 提示词嵌入列表
        t1 (float) : 阈值下界
        t2 (float) : 阈值上界

    Returns:
        threshold(float): 设置好 MultiIndex 的 codebook DataFrame
    """
    context = torch.mean(context[0],dim=0).unsqueeze(0)
    context = [t.to(reference_embeddings["complex"].device) for t in context]
    simple_sim = torch.mean(torch.cosine_similarity(
        context[0], 
        reference_embeddings["simple"], 
        dim=1
    ))
    complex_sim = torch.mean(torch.cosine_similarity(
        context[0],
        reference_embeddings["complex"],
        dim=1
    ))
    k = 50  # 幂指数，可根据需要调整
    ratio = complex_sim / (simple_sim + complex_sim + 1e-6)
    w = (1 / (1 + torch.exp(-k * (ratio - 0.5)))).item()
    threshould_high = w*high_min + (1-w)*high_max
    threshould_low = w*low_min + (1-w)*low_max
    return threshould_high, threshould_low

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

def t2v_generate(self,
                input_prompt,
                size=(1280, 720),
                frame_num=81,
                shift=5.0,
                sample_solver='unipc',
                sampling_steps=50,
                guide_scale=5.0,
                n_prompt="",
                seed=-1,
                offload_model=True):
    r"""
    Generates video frames from text prompt using diffusion process.

    Args:
        input_prompt (`str`):
            Text prompt for content generation
        size (`tuple[int]`, *optional*, defaults to (1280,720)):
            Controls video resolution, (width,height).
        frame_num (`int`, *optional*, defaults to 81):
            How many frames to sample from a video. The number should be 4n+1
        shift (`float`, *optional*, defaults to 5.0):
            Noise schedule shift parameter. Affects temporal dynamics
        sample_solver (`str`, *optional*, defaults to 'unipc'):
            Solver used to sample the video.
        sampling_steps (`int`, *optional*, defaults to 50):
            Number of diffusion sampling steps. Higher values improve quality but slow generation
        guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
            Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            If tuple, the first guide_scale will be used for low noise model and
            the second guide_scale will be used for high noise model.
        n_prompt (`str`, *optional*, defaults to ""):
            Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
        seed (`int`, *optional*, defaults to -1):
            Random seed for noise generation. If -1, use random seed.
        offload_model (`bool`, *optional*, defaults to True):
            If True, offloads models to CPU during generation to save VRAM

    Returns:
        torch.Tensor:
            Generated video frames tensor. Dimensions: (C, N H, W) where:
            - C: Color channels (3 for RGB)
            - N: Number of frames (81)
            - H: Frame height (from size)
            - W: Frame width from size)
    """
    # preprocess
    guide_scale = (guide_scale, guide_scale) if isinstance(
        guide_scale, float) else guide_scale
    F = frame_num
    target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                    size[1] // self.vae_stride[1],
                    size[0] // self.vae_stride[2])

    seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                        (self.patch_size[1] * self.patch_size[2]) *
                        target_shape[1] / self.sp_size) * self.sp_size

    if n_prompt == "":
        n_prompt = self.sample_neg_prompt
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)

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

    noise = [
        torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=self.device,
            generator=seed_g)
    ]

    @contextmanager
    def noop_no_sync():
        yield

    no_sync_low_noise = getattr(self.low_noise_model, 'no_sync',
                                noop_no_sync)
    no_sync_high_noise = getattr(self.high_noise_model, 'no_sync',
                                    noop_no_sync)

    # evaluation mode
    with (
            torch.amp.autocast('cuda', dtype=self.param_dtype),
            torch.no_grad(),
            no_sync_low_noise(),
            no_sync_high_noise(),
    ):
        boundary = self.boundary * self.num_train_timesteps

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

        # sample videos
        latents = noise
        if self.reference_embeddings != None:
            threshould_high, threshould_low = calculate_PCA_threshold(self.reference_embeddings, context)
            self.high_noise_model.teacache_thresh_high = threshould_high
            self.high_noise_model.teacache_thresh_low = threshould_low
            self.low_noise_model.teacache_thresh_high = threshould_high
            self.low_noise_model.teacache_thresh_low = threshould_low

        arg_c = {'context': context, 'seq_len': seq_len, 'is_cond':True}
        arg_null = {'context': context_null, 'seq_len': seq_len, 'is_cond':False}
        # init
        # reset_model_attributes(self.high_noise_model)
        # reset_model_attributes(self.low_noise_model)

        for i, t in enumerate(tqdm(timesteps)):
            latent_model_input = latents
            timestep = [t]

            timestep = torch.stack(timestep)

            model = self._prepare_model_for_timestep(
                t, boundary, offload_model)
            sample_guide_scale = guide_scale[1] if t.item(
            ) >= boundary else guide_scale[0]
            
            if t.item() < boundary:
                self.low_step = self.low_step+1
                self.high_noise_model.previous_residual_cond=None
                self.high_noise_model.previous_residual_uncond=None
                self.high_noise_model.previous_e0 = None
            noise_pred_cond = model(
                latent_model_input, t=timestep, low_step=self.low_step, denoise_time=i, **arg_c)[0]
            noise_pred_uncond = model(
                latent_model_input, t=timestep, low_step=self.low_step, denoise_time=i, **arg_null)[0]

            noise_pred = noise_pred_uncond + sample_guide_scale * (
                noise_pred_cond - noise_pred_uncond)

            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=seed_g)[0]
            latents = [temp_x0.squeeze(0)]
        self.low_step = -1
        

        x0 = latents
        if offload_model:
            self.low_noise_model.cpu()
            self.high_noise_model.cpu()
            torch.cuda.empty_cache()
        if self.rank == 0:
            videos = self.vae.decode(x0)

    del noise, latents
    # 清理高低噪声模型
    self.low_noise_model.previous_residual_cond=None
    self.low_noise_model.previous_residual_uncond=None
    self.low_noise_model.previous_e0 = None
    del sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    return videos[0] if self.rank == 0 else None

def i2v_generate(self,
                input_prompt,
                img,
                max_area=720 * 1280,
                frame_num=81,
                shift=5.0,
                sample_solver='unipc',
                sampling_steps=40,
                guide_scale=5.0,
                n_prompt="",
                seed=-1,
                offload_model=True):
    r"""
    Generates video frames from input image and text prompt using diffusion process.

    Args:
        input_prompt (`str`):
            Text prompt for content generation.
        img (PIL.Image.Image):
            Input image tensor. Shape: [3, H, W]
        max_area (`int`, *optional*, defaults to 720*1280):
            Maximum pixel area for latent space calculation. Controls video resolution scaling
        frame_num (`int`, *optional*, defaults to 81):
            How many frames to sample from a video. The number should be 4n+1
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

    Returns:
        torch.Tensor:
            Generated video frames tensor. Dimensions: (C, N H, W) where:
            - C: Color channels (3 for RGB)
            - N: Number of frames (81)
            - H: Frame height (from max_area)
            - W: Frame width from max_area)
    """
    # preprocess
    guide_scale = (guide_scale, guide_scale) if isinstance(
        guide_scale, float) else guide_scale
    img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

    F = frame_num
    h, w = img.shape[1:]
    aspect_ratio = h / w
    lat_h = round(
        np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
        self.patch_size[1] * self.patch_size[1])
    lat_w = round(
        np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
        self.patch_size[2] * self.patch_size[2])
    h = lat_h * self.vae_stride[1]
    w = lat_w * self.vae_stride[2]

    max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
        self.patch_size[1] * self.patch_size[2])
    max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)
    noise = torch.randn(
        16,
        21,
        lat_h,
        lat_w,
        dtype=torch.float32,
        generator=seed_g,
        device=self.device)

    msk = torch.ones(1, 81, lat_h, lat_w, device=self.device)
    msk[:, 1:] = 0
    msk = torch.concat([
        torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
    ],
                        dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]

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

    y = self.vae.encode([
        torch.concat([
            torch.nn.functional.interpolate(
                img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                    0, 1),
            torch.zeros(3, 80, h, w)
        ],
                        dim=1).to(self.device)
    ])[0]
    y = torch.concat([msk, y])

    @contextmanager
    def noop_no_sync():
        yield

    no_sync_low_noise = getattr(self.low_noise_model, 'no_sync',
                                noop_no_sync)
    no_sync_high_noise = getattr(self.high_noise_model, 'no_sync',
                                    noop_no_sync)

    # evaluation mode
    with (
            torch.amp.autocast('cuda', dtype=self.param_dtype),
            torch.no_grad(),
            no_sync_low_noise(),
            no_sync_high_noise(),
    ):
        boundary = self.boundary * self.num_train_timesteps

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

        # sample videos
        latent = noise

        arg_c = {
            'context': [context[0]],
            'seq_len': max_seq_len,
            'y': [y],
            'is_cond': True,
        }

        arg_null = {
            'context': context_null,
            'seq_len': max_seq_len,
            'y': [y],
            'is_cond': False,
        }

        if offload_model:
            torch.cuda.empty_cache()

        if dist.is_initialized():
            low_noise_model = self.low_noise_model.module
            high_noise_model = self.high_noise_model.module
        else:
            low_noise_model = self.low_noise_model
            high_noise_model = self.high_noise_model

        for i, t in enumerate(tqdm(timesteps)):
            latent_model_input = [latent.to(self.device)]
            timestep = [t]

            timestep = torch.stack(timestep).to(self.device)

            # 判断处于什么阶段，15为i2v的临界值
            low_step = i - 15
            if low_step < 0:
                low_noise_model.previous_residual_cond=None
                low_noise_model.previous_residual_uncond=None
                low_noise_model.previous_e0 = None

                del low_noise_model.previous_residual_cond
                del low_noise_model.previous_residual_uncond
                del low_noise_model.previous_e0

                low_noise_model.previous_residual_cond=None
                low_noise_model.previous_residual_uncond=None
                low_noise_model.previous_e0 = None
            else:
                high_noise_model.previous_residual_cond=None
                high_noise_model.previous_residual_uncond=None
                high_noise_model.previous_e0 = None

                del high_noise_model.previous_residual_cond
                del high_noise_model.previous_residual_uncond
                del high_noise_model.previous_e0

                high_noise_model.previous_residual_cond=None
                high_noise_model.previous_residual_uncond=None
                high_noise_model.previous_e0 = None

            model = self._prepare_model_for_timestep(
                t, boundary, offload_model)
            sample_guide_scale = guide_scale[1] if t.item(
            ) >= boundary else guide_scale[0]

            
            


            noise_pred_cond = model(
                latent_model_input, t=timestep, low_step=low_step, denoise_time=i, **arg_c)[0]
            if offload_model:
                torch.cuda.empty_cache()
            noise_pred_uncond = model(
                latent_model_input, t=timestep, low_step=low_step, denoise_time=i, **arg_null)[0]
            if offload_model:
                torch.cuda.empty_cache()
            noise_pred = noise_pred_uncond + sample_guide_scale * (
                noise_pred_cond - noise_pred_uncond)

            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latent.unsqueeze(0),
                return_dict=False,
                generator=seed_g)[0]
            latent = temp_x0.squeeze(0)

            x0 = [latent]
            del latent_model_input, timestep

        if offload_model:
            self.low_noise_model.cpu()
            self.high_noise_model.cpu()
            torch.cuda.empty_cache()

        if self.rank == 0:
            videos = self.vae.decode(x0)

    del noise, latent, x0

    self.low_noise_model.previous_residual_cond=None
    self.low_noise_model.previous_residual_uncond=None
    self.low_noise_model.previous_e0 = None
    # 清理高低噪声模型
    del self.low_noise_model.previous_residual_cond
    del self.low_noise_model.previous_residual_uncond
    del self.low_noise_model.previous_e0

    self.low_noise_model.previous_residual_cond=None
    self.low_noise_model.previous_residual_uncond=None
    self.low_noise_model.previous_e0 = None
    del sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    return videos[0] if self.rank == 0 else None

def sp_prompttea_forward(
    self,
    x,
    t,
    low_step,
    denoise_time,
    context,
    seq_len,
    is_cond,
    y=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == 'i2v':
        assert y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast('cuda', dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim,
                                    t).unflatten(0, (bt, seq_len)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    # Context Parallel
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]
    e = torch.chunk(e, get_world_size(), dim=1)[get_rank()]
    e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)
    
    '''多卡复用cache'''
    teacache_thresh = self.teacache_thresh_high if low_step<0 else self.teacache_thresh_low
    sp_world_size = get_world_size()
    sp_rank = get_rank()
    # sp_group = get_sp_group()
    if self.enable_teacache:
        # 同步cnt和累积距离值
        if dist.is_initialized():
            tensor_cnt = torch.tensor(denoise_time, device=device)
            tensor_accum = torch.tensor(self.accumulated_rel_l1_distance, device=device)
            # tensor_accum_odd = torch.tensor(self.accumulated_rel_l1_distance_uncond, device=device)
            dist.broadcast(tensor_cnt, src=0)
            dist.broadcast(tensor_accum, src=0)
            # dist.broadcast(tensor_accum_odd, src=0)
            denoise_time = tensor_cnt.item()
            self.accumulated_rel_l1_distance = tensor_accum.item()
            # self.accumulated_rel_l1_distance_odd = tensor_accum_odd.item()
        
        modulated_inp = e0
        

        if is_cond:
            if denoise_time==0 or low_step==0:
                should_calc_cond = True
                self.accumulated_rel_l1_distance = 0
            else:
                delta_e0 = ((modulated_inp-self.previous_e0).abs().mean() / self.previous_e0.abs().mean()).cpu().item()
                x_and_t = create_poly_features(np.array([delta_e0]), np.array([denoise_time]))
                fit_model = self.fit_model_high if low_step<0 else self.fit_model_low
                l1 = fit_model.predict(x_and_t)
                self.accumulated_rel_l1_distance += l1
                if self.accumulated_rel_l1_distance < teacache_thresh:
                    should_calc_cond = False
                else:
                    should_calc_cond = True
                    self.accumulated_rel_l1_distance = 0
            if dist.is_initialized():
                tensor_should_calc = torch.tensor(should_calc_cond, device=device)
                dist.broadcast(tensor_should_calc, src=0)
                should_calc_cond = tensor_should_calc.item()
            self.should_calc = should_calc_cond
            self.previous_e0 = modulated_inp
        
        # uncond
        else:
            should_calc_uncond = self.should_calc
            cfg_l1 = self.cfg_codebook.get(denoise_time, 0)
            self.accumulated_cfg_l1_distance += cfg_l1
            if self.accumulated_cfg_l1_distance < self.cfgcache_thresh:
                should_calc_cfg = False
            else:
                should_calc_cfg = True
                self.accumulated_cfg_l1_distance = 0
            if dist.is_initialized():
                    tensor_should_compute = torch.tensor(should_calc_uncond, device=device)
                    dist.broadcast(tensor_should_compute, src=0)
                    should_calc_uncond = tensor_should_compute.item()
                    tensor_should_calc_cfg = torch.tensor(should_calc_cfg, device=device)
                    dist.broadcast(tensor_should_calc_cfg, src=0)
                    should_calc_cfg_odd = tensor_should_calc_cfg.item()

    # calc
    if self.enable_teacache:
        if is_cond:
            if not should_calc_cond:
                chunk_residual = torch.chunk(self.previous_residual_cond, sp_world_size, dim=1)[sp_rank]
                x = x + chunk_residual
                # print(f'is_cond={is_cond}, denoise_time={denoise_time}成功跳过')
            else:
                ori_x = x.clone()
                for block in self.blocks:
                    x = block(x, **kwargs)
                local_residual = x - ori_x
                if dist.is_initialized():
                   full_residual = gather_forward(local_residual, dim=1)
                else:
                    full_residual = local_residual
                self.previous_residual_cond = full_residual
        # uncond
        else:
            if not should_calc_uncond:
                chunk_residual = torch.chunk(self.previous_residual_uncond, sp_world_size, dim=1)[sp_rank]
                x = x + chunk_residual
                # print(f'is_cond={is_cond}, denoise_time={denoise_time}成功跳过')
            else:
                ori_x = x.clone()
                if should_calc_cfg:
                    for block in self.blocks:
                        x = block(x, **kwargs)
                else:
                    x = x + torch.chunk(self.previous_residual_cond, sp_world_size, dim=1)[sp_rank]
                    # print(f"第{denoise_time}步使用cfgcache---------------------------")
                local_residual = x - ori_x
                # 收集所有GPU上的残差，构建完整残差
                if dist.is_initialized():
                    full_residual = gather_forward(local_residual, dim=1)
                else:
                    full_residual = local_residual
                self.previous_residual_uncond = full_residual    
    else:
        for block in self.blocks:
            x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = gather_forward(x , dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]




def prompttea_forward(
        self,
        x,
        t,
        low_step,
        denoise_time,
        context,
        seq_len,
        is_cond,
        y=None,
    ):
    r"""
    Forward pass through the diffusion model

    Args:
        x (List[Tensor]):
            List of input video tensors, each with shape [C_in, F, H, W]
        t (Tensor):
            Diffusion timesteps tensor of shape [B]
        context (List[Tensor]):
            List of text embeddings each with shape [L, C]
        seq_len (`int`):
            Maximum sequence length for positional encoding
        y (List[Tensor], *optional*):
            Conditional video inputs for image-to-video mode, same shape as x

    Returns:
        List[Tensor]:
            List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
    """
    if self.model_type == 'i2v':
        assert y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                    dim=1) for u in x
    ])

    # time embeddings
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast('cuda', dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim,
                                    t).unflatten(0, (bt, seq_len)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)
    
    teacache_thresh = self.teacache_thresh_high if low_step<0 else self.teacache_thresh_low
    # print(f'self.teacache_thresh_high={self.teacache_thresh_high}, self.teacache_thresh_low={self.teacache_thresh_low}')
    # prompttea
    if self.enable_teacache:
        modulated_inp = e0
        if is_cond:
            if denoise_time==0 or low_step==0:
                should_calc_cond = True
                self.accumulated_rel_l1_distance = 0
            else:
                delta_e0 = ((modulated_inp-self.previous_e0).abs().mean() / self.previous_e0.abs().mean()).cpu().item()
                x_and_t = create_poly_features(np.array([delta_e0]), np.array([denoise_time]))
                fit_model = self.fit_model_high if low_step<0 else self.fit_model_low
                l1 = fit_model.predict(x_and_t)
                self.accumulated_rel_l1_distance += l1
                if self.accumulated_rel_l1_distance < teacache_thresh:
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
            if self.accumulated_cfg_l1_distance < self.cfgcache_thresh:
                should_calc_cfg = False
            else:
                should_calc_cfg = True
                self.accumulated_cfg_l1_distance = 0

        # if denoise_time!=0 and low_step!=0:
        #     if is_cond:
        #         print(f'is_cond={is_cond}, denoise_time={denoise_time}, l1={l1}, self.should_calc={self.should_calc}------------------')
        #     else:
        #         print(f'is_cond={is_cond}, denoise_time={denoise_time}, self.should_calc={self.should_calc}------------------')
        #         print(f'cfg_l1={cfg_l1}, should_calc_cfg={should_calc_cfg}-----------------------------------------------------------')

    # calc
    if self.enable_teacache:
        if is_cond:
            if not should_calc_cond:
                x += self.previous_residual_cond
                # print(f'is_cond={is_cond}, denoise_time={denoise_time}成功跳过')
            else:
                ori_x = x.clone()
                for block in self.blocks:
                    x = block(x, **kwargs)
                self.previous_residual_cond = x - ori_x
        # uncond
        else:
            if not should_calc_uncond:
                x += self.previous_residual_uncond
                # print(f'is_cond={is_cond}, denoise_time={denoise_time}成功跳过')
            else:
                ori_x = x.clone()
                if should_calc_cfg:
                    for block in self.blocks:
                        x = block(x, **kwargs)
                else:
                    x = x + self.previous_residual_cond
                    # print(f"第{denoise_time}步使用cfgcache---------------------------")
                self.previous_residual_uncond = x - ori_x

    else:
        for block in self.blocks:
            x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]

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
    parser.add_argument(
        "--teacache_thresh_high",
        type=float,
        default=0.4,
        help="Higher speedup will cause to worse quality")
    parser.add_argument(
        "--teacache_thresh_low",
        type=float,
        default=0.25,
        help="Higher speedup will cause to worse quality")
    parser.add_argument(
        "--codebook_path",
        type=str,
        default="codebook_wan2.2_i2v.csv",
        help="The path to the codebook.")  
    parser.add_argument(
        "--fit_model_high_path",
        type=str,
        default="fit_model_i2v_high.pkl",
        help="The path to the codebook.")  
    parser.add_argument(
        "--fit_model_low_path",
        type=str,
        default="fit_model_i2v_low.pkl",
        help="The path to the codebook.")
    parser.add_argument(
        "--cfgcache_thresh",
        type=float,
        default=0.02,
        help="Higher speedup will cause to worse quality")
    parser.add_argument(
        "--use_pca",
        action="store_true",
        default=False,
        help="Using Prompt Complexity Aware will automatically select the threshold for TeaCache based on prompt words.")
    parser.add_argument(
        "--ref_emb_path",
        type=str,
        default="reference_embeddings_wan2.2.pt",
        help="The path to the reference embeddings.")
        

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

        # PCA
        wan_t2v.reference_embeddings = None
        if args.use_pca:
            reference_embeddings = torch.load(args.ref_emb_path)
            for emb in reference_embeddings:
                reference_embeddings[emb] = reference_embeddings[emb].to(device)
            wan_t2v.reference_embeddings = reference_embeddings

        # cfgcache
        codebook = prepare_codebook(args.codebook_path)
        codebook_dict = {(i): value for i, value in zip(codebook.index.get_level_values(0),
                                                        codebook['output_value'])}
        wan_t2v.high_noise_model.__class__.cfgcache_thresh = args.cfgcache_thresh
        wan_t2v.high_noise_model.__class__.accumulated_cfg_l1_distance = 0
        wan_t2v.high_noise_model.__class__.cfg_codebook = codebook_dict
        wan_t2v.low_noise_model.__class__.cfgcache_thresh = args.cfgcache_thresh
        wan_t2v.low_noise_model.__class__.accumulated_cfg_l1_distance = 0
        wan_t2v.low_noise_model.__class__.cfg_codebook = codebook_dict

        # TeaCache
        wan_t2v.__class__.generate = t2v_generate
        wan_t2v.__class__.low_step = -1
        wan_t2v.high_noise_model.__class__.enable_teacache = True
        wan_t2v.high_noise_model.__class__.forward = prompttea_forward
        wan_t2v.high_noise_model.__class__.teacache_thresh_high = args.teacache_thresh_high
        wan_t2v.high_noise_model.__class__.teacache_thresh_low = args.teacache_thresh_low
        wan_t2v.high_noise_model.__class__.accumulated_rel_l1_distance = 0
        wan_t2v.high_noise_model.__class__.previous_e0 = None
        wan_t2v.high_noise_model.__class__.previous_residual_even = None
        wan_t2v.high_noise_model.__class__.previous_residual_odd = None
        wan_t2v.high_noise_model.__class__.should_calc = True
        wan_t2v.high_noise_model.__class__.fit_model_high = joblib.load(args.fit_model_high_path)
        wan_t2v.high_noise_model.__class__.fit_model_low = joblib.load(args.fit_model_low_path)

        wan_t2v.low_noise_model.__class__.enable_teacache = True
        wan_t2v.low_noise_model.__class__.forward = prompttea_forward
        wan_t2v.low_noise_model.__class__.teacache_thresh_high = args.teacache_thresh_high
        wan_t2v.low_noise_model.__class__.teacache_thresh_low = args.teacache_thresh_low
        wan_t2v.low_noise_model.__class__.accumulated_rel_l1_distance = 0
        wan_t2v.low_noise_model.__class__.previous_e0 = None
        wan_t2v.low_noise_model.__class__.previous_residual_even = None
        wan_t2v.low_noise_model.__class__.previous_residual_odd = None
        wan_t2v.low_noise_model.__class__.should_calc = True
        wan_t2v.low_noise_model.__class__.fit_model_high = joblib.load(args.fit_model_high_path)
        wan_t2v.low_noise_model.__class__.fit_model_low = joblib.load(args.fit_model_low_path)


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
        n_p = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，"
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
            offload_model=args.offload_model,
            n_prompt = n_p
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

        if args.ulysses_size > 1:
            new_forward_func = sp_prompttea_forward
            logging.info(f'Use sp_prompttea_forward')
        else:
            new_forward_func  = prompttea_forward
            logging.info(f'Use prompttea_forward')

        if args.dit_fsdp:
            # print(f'交换前{wan_i2v.high_noise_model.module.forward}')
            wan_high_noise_model = wan_i2v.high_noise_model.module
            wan_low_noise_model = wan_i2v.low_noise_model.module
        else:
            wan_high_noise_model = wan_i2v.high_noise_model
            wan_low_noise_model = wan_i2v.low_noise_model

        codebook = prepare_codebook(args.codebook_path)
        codebook_dict = {(i): value for i, value in zip(codebook.index.get_level_values(0),
                                                        codebook['output_value'])}
        wan_high_noise_model.__class__.cfgcache_thresh = args.cfgcache_thresh
        wan_high_noise_model.__class__.accumulated_cfg_l1_distance = 0
        wan_high_noise_model.__class__.cfg_codebook = codebook_dict
        wan_low_noise_model.__class__.cfgcache_thresh = args.cfgcache_thresh
        wan_low_noise_model.__class__.accumulated_cfg_l1_distance = 0
        wan_low_noise_model.__class__.cfg_codebook = codebook_dict

        # TeaCache
        wan_i2v.__class__.generate = i2v_generate
        # wan_i2v._configure_model = prompttea_configure_model.__get__(wan_i2v)
        wan_i2v.__class__.low_step = -1
        wan_high_noise_model.__class__.enable_teacache = True
        # wan_high_noise_model.__class__.forward = new_forward_func
        wan_high_noise_model.forward = new_forward_func.__get__(wan_high_noise_model)
        # print(f'交换后{wan_i2v.high_noise_model.module.forward}')
        wan_high_noise_model.__class__.teacache_thresh_high = args.teacache_thresh_high
        wan_high_noise_model.__class__.teacache_thresh_low = args.teacache_thresh_low
        wan_high_noise_model.__class__.accumulated_rel_l1_distance = 0
        wan_high_noise_model.__class__.previous_e0 = None
        wan_high_noise_model.__class__.previous_residual_cond = None
        wan_high_noise_model.__class__.previous_residual_uncond = None
        wan_high_noise_model.__class__.should_calc = True
        wan_high_noise_model.__class__.fit_model_high = joblib.load(args.fit_model_high_path)
        wan_high_noise_model.__class__.fit_model_low = joblib.load(args.fit_model_low_path)

        wan_low_noise_model.__class__.enable_teacache = True
        # wan_low_noise_model.__class__.forward = new_forward_func
        wan_low_noise_model.forward = new_forward_func.__get__(wan_low_noise_model)
        wan_low_noise_model.__class__.teacache_thresh_high = args.teacache_thresh_high
        wan_low_noise_model.__class__.teacache_thresh_low = args.teacache_thresh_low
        wan_low_noise_model.__class__.accumulated_rel_l1_distance = 0
        wan_low_noise_model.__class__.previous_e0 = None
        wan_low_noise_model.__class__.previous_residual_cond = None
        wan_low_noise_model.__class__.previous_residual_uncond = None
        wan_low_noise_model.__class__.should_calc = True
        wan_low_noise_model.__class__.fit_model_high = joblib.load(args.fit_model_high_path)
        wan_low_noise_model.__class__.fit_model_low = joblib.load(args.fit_model_low_path)


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
    del video
    gc.collect()
    torch.cuda.empty_cache()

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)