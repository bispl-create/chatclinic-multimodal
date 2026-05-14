import copy
import glob
import math
import os
import random
from collections import namedtuple
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from PIL import Image
from torch import einsum, nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import utils
from tqdm.auto import tqdm
from PIL import Image
ModelResPrediction = namedtuple(
    'ModelResPrediction', ['pred_res', 'pred_noise', 'pred_x_start'])

# helpers functions

def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# normalization functions

def normalize_to_neg_one_to_one(img):
    if isinstance(img, list):
        return [img[k] * 2 - 1 for k in range(len(img))]
    else:
        return img * 2 - 1

def unnormalize_to_zero_to_one(img):
    if isinstance(img, list):
        return [(img[k] + 1) * 0.5 for k in range(len(img))]
    else:
        return (img + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )

def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)

class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1',
                     partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(
            half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        v = v / (h * w)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q * self.scale
        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        condition=False,
        input_condition=False
    ):
        super().__init__()
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels + channels * \
            (1 if self_condition else 0) + channels * \
            (1 if condition else 0) + channels * (1 if input_condition else 0)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                    dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                    dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)

class UnetRes(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        share_encoder=1,
        condition=False,
        input_condition=False
    ):
        super().__init__()
        self.condition = condition
        self.input_condition = input_condition
        self.share_encoder = share_encoder
        self.channels = channels
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        self.self_condition = self_condition

        if self.share_encoder == 1:
            input_channels = channels + channels * \
                (1 if self_condition else 0) + \
                channels * (1 if condition else 0) + channels * \
                (1 if input_condition else 0)
            init_dim = default(init_dim, dim)
            self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)
            dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
            in_out = list(zip(dims[:-1], dims[1:]))
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)
            time_dim = dim * 4
            if self.random_or_learned_sinusoidal_cond:
                sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                    learned_sinusoidal_dim, random_fourier_features)
                fourier_dim = learned_sinusoidal_dim + 1
            else:
                sinu_pos_emb = SinusoidalPosEmb(dim)
                fourier_dim = dim
            self.time_mlp = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(fourier_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )
            self.downs = nn.ModuleList([])
            self.ups = nn.ModuleList([])
            self.ups_no_skip = nn.ModuleList([])
            num_resolutions = len(in_out)
            for ind, (dim_in, dim_out) in enumerate(in_out):
                is_last = ind >= (num_resolutions - 1)
                self.downs.append(nn.ModuleList([
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                        dim_in, dim_out, 3, padding=1)
                ]))
            mid_dim = dims[-1]
            self.mid_block1 = block_klass(
                mid_dim, mid_dim, time_emb_dim=time_dim)
            self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
            self.mid_block2 = block_klass(
                mid_dim, mid_dim, time_emb_dim=time_dim)
            for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
                is_last = ind == (len(in_out) - 1)
                self.ups.append(nn.ModuleList([
                    block_klass(dim_out + dim_in, dim_out,
                                time_emb_dim=time_dim),
                    block_klass(dim_out + dim_in, dim_out,
                                time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                        dim_out, dim_in, 3, padding=1)
                ]))
                self.ups_no_skip.append(nn.ModuleList([
                    block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                    block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                        dim_out, dim_in, 3, padding=1)
                ]))
            self.final_res_block_1 = block_klass(
                dim, dim, time_emb_dim=time_dim)
            self.final_conv_1 = nn.Conv2d(dim, self.out_dim, 1)
            self.final_res_block_2 = block_klass(
                dim * 2, dim, time_emb_dim=time_dim)
            self.final_conv_2 = nn.Conv2d(dim, self.out_dim, 1)
        elif self.share_encoder == 0:
            self.unet0 = Unet(dim, init_dim=init_dim, out_dim=out_dim, dim_mults=dim_mults,
                              channels=channels, self_condition=self_condition, resnet_block_groups=resnet_block_groups,
                              learned_variance=learned_variance, learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features, learned_sinusoidal_dim=learned_sinusoidal_dim,
                              condition=condition, input_condition=input_condition)
            self.unet1 = Unet(dim, init_dim=init_dim, out_dim=out_dim, dim_mults=dim_mults,
                              channels=channels, self_condition=self_condition, resnet_block_groups=resnet_block_groups,
                              learned_variance=learned_variance, learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features, learned_sinusoidal_dim=learned_sinusoidal_dim,
                              condition=condition, input_condition=input_condition)
        elif self.share_encoder == -1:
            self.unet0 = Unet(dim, init_dim=init_dim, out_dim=out_dim, dim_mults=dim_mults,
                              channels=channels, self_condition=self_condition, resnet_block_groups=resnet_block_groups,
                              learned_variance=learned_variance, learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features, learned_sinusoidal_dim=learned_sinusoidal_dim,
                              condition=condition, input_condition=input_condition)

    def forward(self, x, time, x_self_cond=None):
        if self.share_encoder == 1:
            if self.self_condition:
                x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
                x = torch.cat((x_self_cond, x), dim=1)
            x = self.init_conv(x)
            r = x.clone()
            t = self.time_mlp(time)
            h = []
            for block1, block2, attn, downsample in self.downs:
                x = block1(x, t)
                h.append(x)
                x = block2(x, t)
                x = attn(x)
                h.append(x)
                x = downsample(x)
            x = self.mid_block1(x, t)
            x = self.mid_attn(x)
            x = self.mid_block2(x, t)
            out_res = x
            for block1, block2, attn, upsample in self.ups_no_skip:
                out_res = block1(out_res, t)
                out_res = block2(out_res, t)
                out_res = attn(out_res)
                out_res = upsample(out_res)
            out_res = self.final_res_block_1(out_res, t)
            out_res = self.final_conv_1(out_res)
            for block1, block2, attn, upsample in self.ups:
                x = torch.cat((x, h.pop()), dim=1)
                x = block1(x, t)
                x = torch.cat((x, h.pop()), dim=1)
                x = block2(x, t)
                x = attn(x)
                x = upsample(x)
            x = torch.cat((x, r), dim=1)
            x = self.final_res_block_2(x, t)
            out_res_add_noise = self.final_conv_2(x)
            return out_res, out_res_add_noise
        elif self.share_encoder == 0:
            return self.unet0(x, time, x_self_cond=x_self_cond), self.unet1(x, time, x_self_cond=x_self_cond)
        elif self.share_encoder == -1:
            return [self.unet0(x, time, x_self_cond=x_self_cond)]

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def gen_coefficients(timesteps, schedule="increased", sum_scale=1):
    if schedule == "increased":
        x = torch.linspace(1, timesteps, timesteps, dtype=torch.float64)
        scale = 0.5*timesteps*(timesteps+1)
        alphas = x/scale
    elif schedule == "decreased":
        x = torch.linspace(1, timesteps, timesteps, dtype=torch.float64)
        x = torch.flip(x, dims=[0])
        scale = 0.5*timesteps*(timesteps+1)
        alphas = x/scale
    elif schedule == "average":
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float64)
    else:
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float64)
    assert alphas.sum()-torch.tensor(1) < torch.tensor(1e-10)
    return alphas*sum_scale

class ResidualDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        loss_type='l1',
        objective='pred_res_noise',
        ddim_sampling_eta=0.,
        condition=False,
        sum_scale=None,
        input_condition=False,
        input_condition_mask=False
    ):
        super().__init__()
        assert not (
            type(self) == ResidualDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.image_size = image_size
        self.objective = objective
        self.condition = condition
        self.input_condition = input_condition
        self.input_condition_mask = input_condition_mask

        if self.condition:
            self.sum_scale = sum_scale if sum_scale else 0.01
            ddim_sampling_eta = 0.
        else:
            self.sum_scale = sum_scale if sum_scale else 1.

        alphas = gen_coefficients(timesteps, schedule="decreased")
        alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
        alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
        betas2 = gen_coefficients(
            timesteps, schedule="increased", sum_scale=self.sum_scale)
        betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)
        betas_cumsum = torch.sqrt(betas2_cumsum)
        betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
        posterior_variance = betas2*betas2_cumsum_prev/betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('alphas', alphas)
        register_buffer('alphas_cumsum', alphas_cumsum)
        register_buffer('one_minus_alphas_cumsum', 1-alphas_cumsum)
        register_buffer('betas2', betas2)
        register_buffer('betas', torch.sqrt(betas2))
        register_buffer('betas2_cumsum', betas2_cumsum)
        register_buffer('betas_cumsum', betas_cumsum)
        register_buffer('posterior_mean_coef1',
                        betas2_cumsum_prev/betas2_cumsum)
        register_buffer('posterior_mean_coef2', (betas2 *
                        alphas_cumsum_prev-betas2_cumsum_prev*alphas)/betas2_cumsum)
        register_buffer('posterior_mean_coef3', betas2/betas2_cumsum)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

    def predict_noise_from_res(self, x_t, t, x_input, pred_res):
        return (
            (x_t-x_input-(extract(self.alphas_cumsum, t, x_t.shape)-1)
             * pred_res)/extract(self.betas_cumsum, t, x_t.shape)
        )

    def predict_start_from_xinput_noise(self, x_t, t, x_input, noise):
        return (
            (x_t-extract(self.alphas_cumsum, t, x_t.shape)*x_input -
             extract(self.betas_cumsum, t, x_t.shape) * noise)/extract(self.one_minus_alphas_cumsum, t, x_t.shape)
        )

    def predict_start_from_res_noise(self, x_t, t, x_res, noise):
        return (
            x_t-extract(self.alphas_cumsum, t, x_t.shape) * x_res -
            extract(self.betas_cumsum, t, x_t.shape) * noise
        )

    def q_posterior_from_res_noise(self, x_res, noise, x_t, t):
        return (x_t-extract(self.alphas, t, x_t.shape) * x_res -
                (extract(self.betas2, t, x_t.shape)/extract(self.betas_cumsum, t, x_t.shape)) * noise)

    def q_posterior(self, pred_res, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_t +
            extract(self.posterior_mean_coef2, t, x_t.shape) * pred_res +
            extract(self.posterior_mean_coef3, t, x_t.shape) * x_start
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_input, x, t, x_input_condition=0, x_self_cond=None, clip_denoised=True):
        if not self.condition:
            x_in = x
        else:
            if self.input_condition:
                x_in = torch.cat((x, x_input, x_input_condition), dim=1)
            else:
                x_in = torch.cat((x, x_input), dim=1)
        model_output = self.model(x_in, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_denoised else identity

        if self.objective == 'pred_res_noise':
            pred_res, pred_noise = model_output[0], model_output[1]
            pred_res = maybe_clip(pred_res)
            x_start = self.predict_start_from_res_noise(x, t, pred_res, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == 'pred_res_add_noise':
            pred_res, pred_noise = model_output[0], model_output[1] - model_output[0]
            pred_res = maybe_clip(pred_res)
            x_start = self.predict_start_from_res_noise(x, t, pred_res, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == 'pred_x0_noise':
            pred_res, pred_noise = x_input-model_output[0], model_output[1]
            pred_res = maybe_clip(pred_res)
            x_start = maybe_clip(model_output[0])
        elif self.objective == 'pred_x0_add_noise':
            x_start, pred_noise = model_output[0], model_output[1] - model_output[0]
            pred_res = x_input-x_start
            pred_res = maybe_clip(pred_res)
            x_start = maybe_clip(model_output[0])
        elif self.objective == "pred_noise":
            pred_noise = model_output[0]
            x_start = self.predict_start_from_xinput_noise(x, t, x_input, pred_noise)
            x_start = maybe_clip(x_start)
            pred_res = maybe_clip(x_input - x_start)
        elif self.objective == "pred_res":
            pred_res = maybe_clip(model_output[0])
            pred_noise = self.predict_noise_from_res(x, t, x_input, pred_res)
            x_start = maybe_clip(x_input - pred_res)
        return ModelResPrediction(pred_res, pred_noise, x_start)

    def p_mean_variance(self, x_input, x, t, x_input_condition=0, x_self_cond=None):
        preds = self.model_predictions(
            x_input, x, t, x_input_condition, x_self_cond)
        pred_res, x_start = preds.pred_res, preds.pred_x_start
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            pred_res=pred_res, x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x_input, x, t: int, x_input_condition=0, x_self_cond=None):
        batched_times = torch.full(
            (x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x_input, x=x, t=batched_times, x_input_condition=x_input_condition, x_self_cond=x_self_cond)
        noise = torch.randn_like(x) if t > 0 else 0.
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, x_input, shape, last=True):
        if self.input_condition:
            x_input_condition = x_input[1]
        else:
            x_input_condition = 0
        x_input = x_input[0]
        batch, device = shape[0], self.betas.device
        if self.condition:
            img = x_input+math.sqrt(self.sum_scale) * torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)
        x_start = None
        if not last:
            img_list = []
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps, leave=False):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(
                x_input, img, t, x_input_condition, self_cond)
            if not last:
                img_list.append(img)
        if self.condition:
            img_list = [input_add_noise] + (img_list if not last else [img])
            return unnormalize_to_zero_to_one(img_list)
        else:
            img_list = img_list if not last else [img]
            return unnormalize_to_zero_to_one(img_list)

    @torch.no_grad()
    def ddim_sample(self, x_input, shape, last=True):
        if self.input_condition:
            x_input_condition = x_input[1]
        else:
            x_input_condition = 0
        x_input = x_input[0]
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        if self.condition:
            img = x_input+math.sqrt(self.sum_scale) * torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)
        x_start, type = None, "use_pred_noise"
        if not last: img_list = []
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step', leave=False):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            preds = self.model_predictions(x_input, img, time_cond, x_input_condition, self_cond)
            pred_res, pred_noise, x_start = preds.pred_res, preds.pred_noise, preds.pred_x_start
            if time_next < 0:
                img = x_start
                if not last: img_list.append(img)
                continue
            alpha_cumsum, alpha_cumsum_next = self.alphas_cumsum[time], self.alphas_cumsum[time_next]
            alpha = alpha_cumsum - alpha_cumsum_next
            betas2_cumsum, betas2_cumsum_next = self.betas2_cumsum[time], self.betas2_cumsum[time_next]
            betas2 = betas2_cumsum - betas2_cumsum_next
            betas, betas_cumsum, betas_cumsum_next = betas2.sqrt(), self.betas_cumsum[time], self.betas_cumsum[time_next]
            sigma2 = eta * (betas2 * betas2_cumsum_next / betas2_cumsum)
            sqrt_betas2_cumsum_next_m_sigma2_div_betas_cumsum = (betas2_cumsum_next - sigma2).sqrt() / betas_cumsum
            noise = torch.randn_like(img) if eta != 0 else 0
            if type == "use_pred_noise":
                img = img - alpha * pred_res - (betas_cumsum - (betas2_cumsum_next - sigma2).sqrt()) * pred_noise + sigma2.sqrt() * noise
            if not last: img_list.append(img)
        img_list = ([input_add_noise] if self.condition else []) + (img_list if not last else [img])
        return unnormalize_to_zero_to_one(img_list)

    @torch.no_grad()
    def sample(self, x_input=0, batch_size=16, last=True):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        if self.condition:
            if self.input_condition and self.input_condition_mask:
                x_input[0] = normalize_to_neg_one_to_one(x_input[0])
                x_input[1] = normalize_to_neg_one_to_one(x_input[1])
            else:
                x_input = normalize_to_neg_one_to_one(x_input)
            batch_size, channels, h, w = x_input[0].shape
            size = (batch_size, channels, h, w)
        else:
            size = (batch_size, channels, image_size, image_size)
        return sample_fn(x_input, size, last=last)

    def q_sample(self, x_start, x_res, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (x_start + extract(self.alphas_cumsum, t, x_start.shape) * x_res + extract(self.betas_cumsum, t, x_start.shape) * noise)

    @property
    def loss_fn(self):
        if self.loss_type == 'l1': return F.l1_loss
        elif self.loss_type == 'l2': return F.mse_loss
        else: raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, imgs, t, noise=None):
        if isinstance(imgs, list):
            x_input_condition = imgs[2] if self.input_condition else 0
            x_input, x_start = imgs[1], imgs[0]
        else:
            x_input, x_start = 0, imgs
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_res = x_input - x_start
        x = self.q_sample(x_start, x_res, t, noise=noise)
        x_self_cond = None
        if self.self_condition and random.random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x_input, x, t, x_input_condition if self.input_condition else 0).pred_x_start
                x_self_cond.detach_()
        x_in = torch.cat((x, x_input, x_input_condition), dim=1) if (self.condition and self.input_condition) else (torch.cat((x, x_input), dim=1) if self.condition else x)
        model_out = self.model(x_in, t, x_self_cond)
        target = []
        if self.objective == 'pred_res_noise':
            target.extend([x_res, noise])
        elif self.objective == 'pred_res_add_noise':
            target.extend([x_res, x_res + noise])
        elif self.objective == 'pred_x0_noise':
            target.extend([x_start, noise])
        elif self.objective == 'pred_x0_add_noise':
            target.extend([x_start, x_start + noise])
        elif self.objective == "pred_noise":
            target.append(noise)
        elif self.objective == "pred_res":
            target.append(x_res)
        loss = 0
        for i in range(len(model_out)):
            loss += self.loss_fn(model_out[i], target[i], reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        if isinstance(img, list): b, device = img[0].shape[0], img[0].device
        else: b, device = img.shape[0], img.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        if self.input_condition and self.input_condition_mask:
            img[0], img[1] = normalize_to_neg_one_to_one(img[0]), normalize_to_neg_one_to_one(img[1])
        else:
            img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)


# ==========================================
# INFERENCE SCRIPT
# ==========================================

import os
import sys
import argparse
import scipy.io as sio
import numpy as np
import torch
from tqdm.auto import tqdm
from PIL import Image
import h5py
from pathlib import Path
from datetime import datetime

# Add project root to sys.path to ensure 'src' can be found.

class FixedResidualDiffusion(ResidualDiffusion):
    def forward(self, img, *args, **kwargs):
        """
        Custom forward pass that normalizes input and condition images 
        to the range [-1, 1] before computing the loss.
        """
        if isinstance(img, list):
            img = [normalize_to_neg_one_to_one(i) for i in img]
        else:
            img = normalize_to_neg_one_to_one(img)
        b, c, h, w = img[0].shape
        t = torch.randint(0, self.num_timesteps, (b,), device=img[0].device).long()
        return self.p_losses(img, t, *args, **kwargs)

def load_mat_data(file_path):
    """Loads a .mat file, handles h5py / loadmat differences, similar to GAN codebase."""
    try:
        mat = sio.loadmat(file_path)
        keys = [k for k in mat.keys() if not k.startswith('__')]
        data = mat[keys[0]]
    except Exception:
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            data = f[keys[0]][:]
    return data.astype(np.float32)

def preprocess_pad(x):
    """
    Normalize the input image using percentiles and pad it to the target size 64x64.
    Returns:
        padded: Padded image array.
        params: Padding parameters and original dimensions to restore the size.
    """
    mi, ma = np.percentile(x, 0.1), np.percentile(x, 99.9)
    norm = np.clip((x - mi) / (ma - mi + 1e-8), 0, 1) if ma > mi else np.zeros_like(x)
    
    h, w = norm.shape
    pad_h, pad_w = (64 - h) // 2, (64 - w) // 2
    
    # Pad to 64x64 using constant mode (0 filling)
    padded = np.pad(norm, ((pad_h, 64 - h - pad_h), (pad_w, 64 - w - pad_w)), mode='constant')
    return padded, (pad_h, pad_w, h, w)

def main():
    parser = argparse.ArgumentParser(description="Inference script for the RDDM model.")
    parser.add_argument('-i', '--input', type=str, default="./input", 
                        help="Relative path to the input data directory or file (.mat)")
    parser.add_argument('-p', '--pth', type=str, default="./best_model.pth", 
                        help="Relative path to the model weights (.pt or .pth)")
    # Generate dynamic timestamped output path
    default_out_dir = f"../outputs/rddm/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    parser.add_argument('-o', '--output', type=str, default=default_out_dir, 
                        help="Directory to save the inference results")
    parser.add_argument('--project-path', type=str, default=".", 
                        help="Project path to append to sys.path if 'src' is not found")
    parser.add_argument('--seed', type=int, default=None, 
                        help="Optional random seed for reproducible inference")
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)
        print(f">>> Set random seed to {args.seed}")

    # Append project path for the 'src' module if specified
    project_path_abs = os.path.abspath(args.project_path)
    if project_path_abs not in sys.path:
        sys.path.append(project_path_abs)

    # Initialize the model structure with specific hyperparameters
    print("Initializing the model...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = UnetRes(dim=64, dim_mults=(1, 2, 4, 8), share_encoder=-1, 
                    condition=True, input_condition=True, channels=1)
    diffusion = FixedResidualDiffusion(model, image_size=64, timesteps=1000, 
                                       sampling_timesteps=1000, objective='pred_res', 
                                       loss_type='l1', condition=True, sum_scale=0.1, 
                                       input_condition=True, input_condition_mask=False)
    diffusion.to(device)

    # Load the best model path
    best_model_path = Path(args.pth)
    if not best_model_path.exists():
        print(f"Error: Model weights not found at {best_model_path}")
        sys.exit(1)

    print(f"Loading weights from {best_model_path}...")
    try:
        checkpoint = torch.load(best_model_path, map_location=device)
    except Exception as e:
        print(f"Error loading the checkpoint: {e}")
        sys.exit(1)
    
    # Process weights (remove 'ema_model.' prefix if present)
    model_state_dict = checkpoint.get('ema', checkpoint.get('model', checkpoint))
    cleaned_state_dict = {}
    for k, v in model_state_dict.items():
        if k.startswith('ema_model.'):
            cleaned_state_dict[k[10:]] = v  # 10 is the length of 'ema_model.'
        else:
            cleaned_state_dict[k] = v

    # Load cleaned weights to the diffusion model
    diffusion.load_state_dict(cleaned_state_dict, strict=False)
    diffusion.eval()

    # Prepare output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"Results will be saved to: {args.output}")

    # Gather input files
    input_path = Path(args.input)
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = list(input_path.glob("*.png"))
        if len(files) == 0:
            print(f"No .mat files found in {input_path}")
            sys.exit(0)
    else:
        print(f"Error: Input path {input_path} does not exist")
        sys.exit(1)

    results = []
    all_preds = {}

    print("\nStarting final inference (restoring original dimensions)...")
    with torch.no_grad():
        for fpath in tqdm(files, desc="Processing files"):
            try:
                # Read .mat file using the aligned function
                img_pil = Image.open(str(fpath)).convert('L')
                raw = np.array(img_pil).astype(np.float32) / 255.0
                
                # Preprocess and prepare tensor
                img_64, params = preprocess_pad(raw)
                inp = torch.from_numpy(img_64).unsqueeze(0).unsqueeze(0).float().to(device)
                
                # Perform inference (pass the input twice as a condition)
                recovered = diffusion.sample([inp, inp], batch_size=1, last=True)
                if isinstance(recovered, list): 
                    recovered = recovered[-1]
                
                out_64 = np.clip(recovered.cpu().numpy().squeeze(), 0, 1)
                ph, pw, h, w = params
                
                # Crop back to the original size
                out_original = out_64[ph:ph+h, pw:pw+w]
                
                # Save as .npy
                save_name = fpath.name.replace('.png', '.npy')
                save_fpath = os.path.join(args.output, save_name)
                np.save(save_fpath, out_original)
                all_preds[fpath.name] = out_original
                
                
                results.append((out_original, fpath.name))
            except Exception as e:
                print(f"Error processing {fpath.name}: {e}")
                continue

    # (Visualization section has been removed as requested)
    
    # Create a 2-row summary grid image
    # Create a dynamic layout summary grid image for any N inputs
    import math
    import PIL.Image
    
    file_keys = sorted(list(all_preds.keys()))
    num_files = len(file_keys)
    
    if num_files > 0:
        cell_h, cell_w = 49, 49
        pad = 2
        
        # Calculate dynamic grid dimensions (roughly square)
        cols = math.ceil(math.sqrt(num_files))
        rows = math.ceil(num_files / cols)
        
        grid_h = rows * cell_h + pad * (rows + 1)
        grid_w = cols * cell_w + pad * (cols + 1)
        grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
        
        for idx, k in enumerate(file_keys):
            r = idx // cols
            c = idx % cols
            y = pad + r * (cell_h + pad)
            x = pad + c * (cell_w + pad)
            
            img_np = all_preds[k]
            if img_np.shape != (cell_h, cell_w):
                h_c, w_c = img_np.shape
                y_s = max(0, (h_c - cell_h) // 2)
                x_s = max(0, (w_c - cell_w) // 2)
                img_np = img_np[y_s : y_s + cell_h, x_s : x_s + cell_w]
                
            grid[y : y + cell_h, x : x + cell_w] = (img_np * 255).clip(0, 255).astype(np.uint8)
        
        summ_path = os.path.join(args.output, "summary_grid.png")
        PIL.Image.fromarray(grid).save(summ_path)


    print(f"\n>>> Final Inference Complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()
