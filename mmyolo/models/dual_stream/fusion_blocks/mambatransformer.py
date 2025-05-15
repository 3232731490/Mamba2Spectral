import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import numpy as np
from typing import Tuple, List, Optional, Callable, Any
from einops import rearrange, repeat
from functools import partial
from collections import OrderedDict
# from timm.models.layers import DropPath, trunc_normal_, to_2tuple # Not strictly needed for the core logic here

# --- Placeholder for OpenMMLab's BaseModule and MODELS ---
# If using OpenMMLab, replace these with actual imports:
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from mmyolo.models.utils import make_divisible

# --- Selective Scan Implementation (from mamba.py, simplified for CPU reference) ---
def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1] # A is (dim, dstate)
    
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    
    x = A.new_zeros((batch, dim, dstate)) # x is the state, (B, D, N)
    ys = []
    
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A)) # (B, D, L, N)
    
    if not is_variable_B: # B is (D, N)
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u) # (B, D, L, N)
    else: # B is (B, N, L) or (B, D, N, L) or (B, G, N, L)
        if B.dim() == 3: # B is (B, N, L)
             # *** Fix: Removed B.unsqueeze(1) ***
             deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        elif B.dim() == 4: # B is (B, D, N, L) or (B, G, N, L)
            if B.shape[1] == dim : # (B, D, N, L)
                 deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
            else: # B is (B, G, N, L), needs repeat
                B_ = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B_, u)
        else:
            # Should not happen with current SSM structure where B_for_scan is (B,N,L)
            raise ValueError(f"Unexpected B dimensions: {B.dim()}")


    if is_variable_C and C.dim() == 4: # (B, D, N, L) or (B, G, N, L)
        if C.shape[1] != dim and C.shape[1] != 1: # Grouped C
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])

    last_state = None
    for i in range(u.shape[2]): # L
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i] # (B, D, N)
        if not is_variable_C: # C is (D, N)
            y = torch.einsum('bdn,dn->bd', x, C) # (B, D)
        else: # C is (B, N, L) or (B, D, N, L)
            if C.dim() == 3: # (B, N, L)
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i]) # (B, D)
            else: # (B, D, N, L) or (B, G, N, L) -> (B, D, N, L) after repeat
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i]) # (B, D)
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2 
        ys.append(y)
    
    y = torch.stack(ys, dim=2) # (batch, dim, L)
    
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)

# Attempt to import optimized scan if available, otherwise use reference
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as mamba_selective_scan_fn
    selective_scan_fn = mamba_selective_scan_fn
    print("INFO: Using mamba_ssm's optimized selective_scan_fn.")
except ImportError:
    selective_scan_fn = selective_scan_ref
    print("INFO: mamba_ssm's optimized selective_scan_fn not found. Using Python reference (selective_scan_ref).")


# --- Components from mamba.py ---
class SSM(BaseModule):
    def __init__(
            self,
            d_model=96,
            d_state="auto", 
            ssm_ratio=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None} 
        super().__init__()
        self.d_model = d_model # This is d_inner for the SSM block itself
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else int(d_state)
        self.expand = ssm_ratio # ssm_ratio is for the parent block (SS2D), actual expansion for SSM is 1 as d_model is already d_inner
        self.d_inner = int(self.expand * self.d_model) # For SSM, d_inner should be d_model if ssm_ratio=1
        
        # If ssm_ratio is passed to SSM and d_model is already d_inner,
        # then self.d_inner here will be expand * d_inner.
        # The x_proj should be from self.d_model (which is d_inner for SSM)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.x_proj = nn.Linear(self.d_model, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.dt_proj = self.dt_init(self.dt_rank, self.d_model, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                    **factory_kwargs)
        self.A_log = self.A_log_init(self.d_state, self.d_model) # A is (D_inner, N)
        self.D = self.D_init(self.d_model) # D is (D_inner)
        self.out_norm = nn.LayerNorm(self.d_model) # Norm output of d_model (d_inner for SSM)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) 
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward(self, x: torch.Tensor): # x: (B, L, D_inner_ssm)
        B, L, d_inner_ssm = x.shape 
        assert d_inner_ssm == self.d_model, f"Input dim {d_inner_ssm} to SSM forward should be self.d_model (d_inner for SSM) {self.d_model}"

        x_permuted = x.permute(0, 2, 1) # (B, D_inner_ssm, L)
        
        x_rearranged = rearrange(x_permuted, "b d l -> (b l) d") 
        x_dbl = self.x_proj(x_rearranged) 
        
        dt_val, B_val, C_val = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = self.dt_proj.weight @ dt_val.t() 
        dt = rearrange(dt, "d (b l) -> b d l", l=L) 
        
        A = -torch.exp(self.A_log.float()) 
        
        B_reshaped = rearrange(B_val, "(b l) n -> b l n", l=L) 
        C_reshaped = rearrange(C_val, "(b l) n -> b l n", l=L) 

        B_for_scan = B_reshaped.permute(0, 2, 1).contiguous() # (B, d_state, L)
        C_for_scan = C_reshaped.permute(0, 2, 1).contiguous() # (B, d_state, L)

        y = selective_scan_fn(
            x_permuted, dt, 
            A, B_for_scan, C_for_scan, self.D.float(), 
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        ) 
        
        y = rearrange(y, "b d l -> b l d") 
        y = self.out_norm(y) 
        return y


class SS2D_intra(BaseModule):
    def __init__(
            self, d_model=96, size=(80,80), bi=False, d_state="auto", ssm_ratio=2, dt_rank="auto",
            d_conv=3, conv_bias=True, dropout=0., bias=False, cfg=None, **kwargs ):
        super().__init__()
        self.size = tuple(size)
        self.d_model = d_model # d_model of the parent (e.g. MM_SS2D's in_channels)
        self.d_state_cfg = d_state # Store config for SSM
        self.dt_rank_cfg = dt_rank # Store config for SSM
        
        self.d_conv = d_conv
        self.expand = ssm_ratio # Expansion factor for this SS2D block
        self.d_inner = int(self.expand * self.d_model) # d_inner for this SS2D block
        self.bi = bi

        self.in_proj_r = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.in_proj_t = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        
        self.conv2d_r = nn.Sequential(nn.Conv2d(
            in_channels=self.d_inner, out_channels=self.d_inner, groups=self.d_inner,
            bias=conv_bias, kernel_size=d_conv, padding=(d_conv - 1) // 2),
            nn.BatchNorm2d(self.d_inner), nn.SiLU())
        self.conv2d_t = nn.Sequential(nn.Conv2d(
            in_channels=self.d_inner, out_channels=self.d_inner, groups=self.d_inner,
            bias=conv_bias, kernel_size=d_conv, padding=(d_conv - 1) // 2),
            nn.BatchNorm2d(self.d_inner), nn.SiLU())

        # SSM's d_model is the channel dim it operates on, which is d_inner of this SS2D block
        # SSM's ssm_ratio should be 1.0 as its input is already d_inner
        _ssm_d_state = math.ceil(self.d_inner / 6) if self.d_state_cfg == "auto" else int(self.d_state_cfg)
        _ssm_dt_rank = math.ceil(self.d_inner / 16) if self.dt_rank_cfg == "auto" else int(self.dt_rank_cfg)

        self.SSM_r_f = SSM(d_model=self.d_inner, d_state=_ssm_d_state, ssm_ratio=1.0, dt_rank=_ssm_dt_rank)
        self.SSM_t_f = SSM(d_model=self.d_inner, d_state=_ssm_d_state, ssm_ratio=1.0, dt_rank=_ssm_dt_rank)
        if self.bi:
            self.SSM_r_b = SSM(d_model=self.d_inner, d_state=_ssm_d_state, ssm_ratio=1.0, dt_rank=_ssm_dt_rank)
            self.SSM_t_b = SSM(d_model=self.d_inner, d_state=_ssm_d_state, ssm_ratio=1.0, dt_rank=_ssm_dt_rank)
        
        # *** Fix: out_proj takes d_inner and outputs d_model, consistent with mamba.py logic ***
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias) 
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()


    def ssm_stage_1_corrected(self, r_conv_out, t_conv_out): # Inputs are (B, N, D_inner)
        y_r = self.SSM_r_f(r_conv_out) 
        y_t = self.SSM_t_f(t_conv_out) 
        if self.bi:
            y_r_b = torch.flip(self.SSM_r_b(torch.flip(r_conv_out, dims=[1])), dims=[1])
            y_t_b = torch.flip(self.SSM_t_b(torch.flip(t_conv_out, dims=[1])), dims=[1])
            y_r = y_r + y_r_b
            y_t = y_t + y_t_b
        # Concatenate along sequence dim (B, 2N, D_inner) as per mamba.py's SS2D_intra
        return torch.cat([y_r, y_t], dim=1) 

    def conv_sep(self, r, t): # r, t are (B, N, D_model)
        B, N, D_model_dim = r.shape
        assert D_model_dim == self.d_model

        xz_r = self.in_proj_r(r) 
        xz_t = self.in_proj_t(t) 

        x_r, z_r = xz_r.chunk(2, dim=-1) 
        x_t, z_t = xz_t.chunk(2, dim=-1) 

        x_r_conv_in = x_r.reshape(B, self.size[0], self.size[1], self.d_inner).permute(0, 3, 1, 2).contiguous()
        x_t_conv_in = x_t.reshape(B, self.size[0], self.size[1], self.d_inner).permute(0, 3, 1, 2).contiguous()

        x_r_conv_out = self.conv2d_r(x_r_conv_in) 
        x_t_conv_out = self.conv2d_t(x_t_conv_in) 

        x_r_ssm_in = x_r_conv_out.permute(0, 2, 3, 1).reshape(B, N, self.d_inner)
        x_t_ssm_in = x_t_conv_out.permute(0, 2, 3, 1).reshape(B, N, self.d_inner)
        
        y_combined_seq = self.ssm_stage_1_corrected(x_r_ssm_in, x_t_ssm_in) # (B, 2N, D_inner)
        z_combined_seq = torch.cat([z_r, z_t], dim=1) # (B, 2N, D_inner) for SiLU activation
        
        y_activated_seq = y_combined_seq * F.silu(z_combined_seq) # (B, 2N, D_inner)
        # out_proj takes (..., d_inner) and outputs (..., d_model)
        out_projected_seq = self.dropout(self.out_proj(y_activated_seq)) # (B, 2N, D_model)

        new_r = out_projected_seq[:, :N]
        new_t = out_projected_seq[:, N:]
        return new_r, new_t

    def forward(self, r, t, **kwargs): 
        new_r, new_t = self.conv_sep(r, t)
        return r + new_r, t + new_t


class SS2D_inter(BaseModule): 
    def __init__(
            self, d_model=96, size=(80,80), bi=False, d_state="auto", ssm_ratio=2, dt_rank="auto",
            d_conv=3, conv_bias=True, dropout=0., bias=False, cfg=None, **kwargs ):
        super().__init__()
        self.size = tuple(size)
        self.d_model = d_model
        self.d_state_cfg = d_state
        self.dt_rank_cfg = dt_rank
        
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.bi = bi

        self.in_proj_r = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.in_proj_t = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        
        self.conv2d_r = nn.Sequential(nn.Conv2d(
            in_channels=self.d_inner, out_channels=self.d_inner, groups=self.d_inner,
            bias=conv_bias, kernel_size=d_conv, padding=(d_conv - 1) // 2),
            nn.BatchNorm2d(self.d_inner), nn.SiLU())
        self.conv2d_t = nn.Sequential(nn.Conv2d(
            in_channels=self.d_inner, out_channels=self.d_inner, groups=self.d_inner,
            bias=conv_bias, kernel_size=d_conv, padding=(d_conv - 1) // 2),
            nn.BatchNorm2d(self.d_inner), nn.SiLU())
        
        _ssm_d_state = math.ceil(self.d_inner / 6) if self.d_state_cfg == "auto" else int(self.d_state_cfg)
        _ssm_dt_rank = math.ceil(self.d_inner / 16) if self.dt_rank_cfg == "auto" else int(self.dt_rank_cfg)

        self.SSM_inter_f = SSM(d_model=self.d_inner, d_state=_ssm_d_state, ssm_ratio=1.0, dt_rank=_ssm_dt_rank)
        if self.bi:
            self.SSM_inter_b = SSM(d_model=self.d_inner, d_state=_ssm_d_state, ssm_ratio=1.0, dt_rank=_ssm_dt_rank)
        
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias) 
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def ssm_stage_inter(self, r_conv_out, t_conv_out): 
        x_combined_seq = torch.cat([r_conv_out, t_conv_out], dim=1) 
        y_fused_seq = self.SSM_inter_f(x_combined_seq) 
        if self.bi:
            y_fused_seq_b = torch.flip(self.SSM_inter_b(torch.flip(x_combined_seq, dims=[1])), dims=[1])
            y_fused_seq = y_fused_seq + y_fused_seq_b
        return y_fused_seq 

    def conv_sep(self, inputs1, inputs2, **kwargs): 
        B, N, D_model_dim = inputs1.shape
        assert D_model_dim == self.d_model

        xz_r = self.in_proj_r(inputs1) 
        xz_t = self.in_proj_t(inputs2) 

        x_r, z_r = xz_r.chunk(2, dim=-1) 
        x_t, z_t = xz_t.chunk(2, dim=-1) 

        x_r_conv_in = x_r.reshape(B, self.size[0], self.size[1], self.d_inner).permute(0, 3, 1, 2).contiguous()
        x_t_conv_in = x_t.reshape(B, self.size[0], self.size[1], self.d_inner).permute(0, 3, 1, 2).contiguous()

        x_r_conv_out = self.conv2d_r(x_r_conv_in) 
        x_t_conv_out = self.conv2d_t(x_t_conv_in) 

        x_r_ssm_in = x_r_conv_out.permute(0, 2, 3, 1).reshape(B, N, self.d_inner)
        x_t_ssm_in = x_t_conv_out.permute(0, 2, 3, 1).reshape(B, N, self.d_inner)
        
        z_combined_seq = torch.cat([z_r, z_t], dim=1) 

        y_fused_seq = self.ssm_stage_inter(x_r_ssm_in, x_t_ssm_in) 
        y_activated_seq = y_fused_seq * F.silu(z_combined_seq) 
        
        out_projected_seq = self.dropout(self.out_proj(y_activated_seq)) 
        
        new_r = out_projected_seq[:, :N] 
        new_t = out_projected_seq[:, N:] 
        return new_r, new_t

    def forward(self, inputs1, inputs2, **kwargs):
        new_r, new_t = self.conv_sep(inputs1, inputs2)
        return inputs1 + new_r, inputs2 + new_t


class PatchEmbed(BaseModule):
    def __init__(self, img_size=(224,224), patch_size=(16,16), stride=(16,16), in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        self.img_size = tuple(img_size)
        self.patch_size = tuple(patch_size)
        self.stride = tuple(stride)
        self.grid_size = ((self.img_size[0] - self.patch_size[0]) // self.stride[0] + 1, 
                          (self.img_size[1] - self.patch_size[1]) // self.stride[1] + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # Allow for dynamic input sizes if img_size is not strictly enforced,
        # but grid_size calculation depends on it. For now, keep assert.
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  
        x = self.norm(x)
        return x

class Reconstruct(BaseModule):
    def __init__(self, size, patch_grid_size):
        super().__init__()
        self.patch_grid_size = tuple(patch_grid_size)
        self.size = tuple(size)

    def forward(self, x): 
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, self.patch_grid_size[0], self.patch_grid_size[1])
        x = F.interpolate(x, size=(self.size[0], self.size[1]), mode='bilinear', align_corners=False)
        return x


# @MODELS.register_module()
class MM_SS2D(BaseModule):
    def __init__(self, in_channels, size, mamba_operations=1, cfg=None, dt_rank="auto", d_state="auto", **kwargs):
        super().__init__()
        
        if isinstance(size, int):
            _size = (size, size)
        else:
            _size = tuple(size)

        self.patch_embed = PatchEmbed(img_size=_size, 
                                      patch_size=(_size[0] // 8, _size[1] // 8), 
                                      stride=(_size[0] // 8, _size[1] // 8), 
                                      in_chans=in_channels, embed_dim=in_channels)
        patch_grid_size = self.patch_embed.grid_size
        
        # Pass d_state and dt_rank configurations to SS2D blocks
        self.SSM_intra = SS2D_intra(d_model=in_channels, size=patch_grid_size, 
                                    d_state=d_state, cfg=cfg, dt_rank=dt_rank, 
                                    **kwargs) # Example ssm_ratio, bi
        self.SSM_inter = SS2D_inter(d_model=in_channels, size=patch_grid_size, 
                                    d_state=d_state, cfg=cfg, dt_rank=dt_rank, 
                                    **kwargs) # Example ssm_ratio, bi
        self.reconstruct = Reconstruct(_size, patch_grid_size)
        self.mamba_operations = mamba_operations
        
        self.pos_embed_r = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, in_channels))
        self.pos_embed_t = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, in_channels))
        nn.init.trunc_normal_(self.pos_embed_r, std=.02)
        nn.init.trunc_normal_(self.pos_embed_t, std=.02)


    def forward(self, inputs1: torch.Tensor, inputs2: torch.Tensor, **kwargs):
        r = self.patch_embed(inputs1) + self.pos_embed_r
        t = self.patch_embed(inputs2) + self.pos_embed_t

        for _ in range(self.mamba_operations):
            r, t = self.SSM_intra(r, t)
            r, t = self.SSM_inter(r, t)
            
        out_inputs1 = self.reconstruct(r)
        out_inputs2 = self.reconstruct(t)
        return out_inputs1, out_inputs2

# --- Components from transformer.py ---
class ModalityWeightingLayer(BaseModule):
    def __init__(self, in_channels, reduction=16):
        super(ModalityWeightingLayer, self).__init__()
        self.fc1 = nn.Conv2d(in_channels * 2, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels * 2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs1: torch.Tensor, inputs2: torch.Tensor):
        combined = torch.cat([inputs1, inputs2], dim=1)
        se_weight = F.adaptive_avg_pool2d(combined, (1, 1))
        se_weight = F.relu(self.fc1(se_weight))
        se_weight = self.sigmoid(self.fc2(se_weight))
        alpha, beta = torch.split(se_weight, inputs1.size(1), dim=1)
        return (inputs1 * alpha, inputs2 * beta)

class Concat(BaseModule):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension
    def forward(self, x: List[torch.Tensor]): 
        return torch.cat(x, self.d)

def autopad(k, p=None, d=1):  
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(BaseModule):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ConcatFusion(BaseModule):
    def __init__(self, in_channels):
        super(ConcatFusion, self).__init__()
        self.concat = Concat(dimension=1)
        self.conv1x1_out = Conv(c1=in_channels * 2, c2=in_channels, k=1, s=1, p=0, g=1, act=True)
    def forward(self, inputs1: torch.Tensor, inputs2: torch.Tensor):
        new_fea = self.concat([inputs1, inputs2])
        new_fea = self.conv1x1_out(new_fea)
        return [new_fea] 

class AdaptivePool2d(BaseModule): 
    def __init__(self, output_h, output_w, pool_type='avg'):
        super(AdaptivePool2d, self).__init__()
        self.output_h = output_h
        self.output_w = output_w
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, input_h, input_w = x.shape
        if (input_h > self.output_h) or (input_w > self.output_w):
            stride_h = max(1, input_h // self.output_h) # Ensure stride is at least 1
            stride_w = max(1, input_w // self.output_w)
            
            kernel_h = input_h - (self.output_h - 1) * stride_h
            kernel_w = input_w - (self.output_w - 1) * stride_w

            # Ensure kernel size is positive
            kernel_h = max(1, kernel_h)
            kernel_w = max(1, kernel_w)
            
            if self.pool_type == 'avg':
                y = F.avg_pool2d(x, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w), padding=0)
            else: 
                y = F.max_pool2d(x, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w), padding=0)
        else: 
            y = x
        # Ensure output size matches target, might need F.interpolate if pooling logic isn't exact
        if y.shape[2] != self.output_h or y.shape[3] != self.output_w:
             y = F.interpolate(y, size=(self.output_h, self.output_w), mode='bilinear', align_corners=False)
        return y

class LearnableCoefficient(BaseModule):
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.tensor([1.0])) 
    def forward(self, x):
        return x * self.bias

class LearnableWeights(BaseModule):
    def __init__(self):
        super(LearnableWeights, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([0.5]))
        self.w2 = nn.Parameter(torch.tensor([0.5]))
    def forward(self, x1, x2):
        return x1 * self.w1 + x2 * self.w2

class CrossAttention(BaseModule):
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        super(CrossAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.out_proj_vis = nn.Linear(h * self.d_v, d_model)
        self.out_proj_ir = nn.Linear(h * self.d_v, d_model)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.init_weights_custom() 

    def init_weights_custom(self): 
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, q_vis, k_vis, v_vis, q_ir, k_ir, v_ir, b_s, nq, attention_mask=None, attention_weights=None):
        att_vis = torch.matmul(q_ir, k_vis) / np.sqrt(self.d_k) 
        att_ir = torch.matmul(q_vis, k_ir) / np.sqrt(self.d_k)  

        att_vis = torch.softmax(att_vis, -1)
        att_vis = self.attn_drop(att_vis)
        att_ir = torch.softmax(att_ir, -1)
        att_ir = self.attn_drop(att_ir)
        
        out_vis = torch.matmul(att_vis, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out_vis = self.resid_drop(self.out_proj_vis(out_vis))
        out_ir = torch.matmul(att_ir, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out_ir = self.resid_drop(self.out_proj_ir(out_ir))
        return [out_vis, out_ir]

class CrossTransformerBlock(BaseModule):
    def __init__(self, d_model, h, block_exp, attn_pdrop, resid_pdrop, loops_num=1): 
        super(CrossTransformerBlock, self).__init__()
        self.loops = loops_num
        self.d_model = d_model
        assert d_model % h == 0, "d_model must be divisible by h (num_heads)"
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        self.ln1_vis = nn.LayerNorm(d_model) 
        self.ln1_ir = nn.LayerNorm(d_model)  
        
        self.que_proj_vis = nn.Linear(d_model, h * self.d_k)
        self.key_proj_vis = nn.Linear(d_model, h * self.d_k)
        self.val_proj_vis = nn.Linear(d_model, h * self.d_v)
        self.que_proj_ir = nn.Linear(d_model, h * self.d_k)
        self.key_proj_ir = nn.Linear(d_model, h * self.d_k)
        self.val_proj_ir = nn.Linear(d_model, h * self.d_v)

        self.crossatt = CrossAttention(d_model, self.d_k, self.d_v, self.h, attn_pdrop, resid_pdrop)
        
        self.ln2_vis = nn.LayerNorm(d_model) 
        self.ln2_ir = nn.LayerNorm(d_model)

        self.mlp_vis = nn.Sequential(nn.Linear(d_model, block_exp * d_model), nn.GELU(),
                                     nn.Linear(block_exp * d_model, d_model), nn.Dropout(resid_pdrop))
        self.mlp_ir = nn.Sequential(nn.Linear(d_model, block_exp * d_model), nn.GELU(),
                                    nn.Linear(block_exp * d_model, d_model), nn.Dropout(resid_pdrop))
        
        self.coef_att_vis = LearnableCoefficient()
        self.coef_mlp_vis = LearnableCoefficient()
        self.coef_att_ir = LearnableCoefficient()
        self.coef_mlp_ir = LearnableCoefficient()


    def forward(self, x: List[torch.Tensor]): 
        vis_fea_in, ir_fea_in = x[0], x[1] 
        bs, nx, c = vis_fea_in.size()

        for _ in range(self.loops):
            vis_norm = self.ln1_vis(vis_fea_in)
            ir_norm = self.ln1_ir(ir_fea_in)

            q_vis = self.que_proj_vis(vis_norm).view(bs, nx, self.h, self.d_k).permute(0, 2, 1, 3)
            k_vis = self.key_proj_vis(vis_norm).view(bs, nx, self.h, self.d_k).permute(0, 2, 3, 1) 
            v_vis = self.val_proj_vis(vis_norm).view(bs, nx, self.h, self.d_v).permute(0, 2, 1, 3)
            
            q_ir = self.que_proj_ir(ir_norm).view(bs, nx, self.h, self.d_k).permute(0, 2, 1, 3)
            k_ir = self.key_proj_ir(ir_norm).view(bs, nx, self.h, self.d_k).permute(0, 2, 3, 1) 
            v_ir = self.val_proj_ir(ir_norm).view(bs, nx, self.h, self.d_v).permute(0, 2, 1, 3)

            att_out_vis, att_out_ir = self.crossatt(q_vis, k_vis, v_vis, q_ir, k_ir, v_ir, bs, nx)
            
            vis_fea_in = vis_fea_in + self.coef_att_vis(att_out_vis) 
            ir_fea_in = ir_fea_in + self.coef_att_ir(att_out_ir)

            vis_mlp_in = self.ln2_vis(vis_fea_in)
            ir_mlp_in = self.ln2_ir(ir_fea_in)
            
            vis_fea_in = vis_fea_in + self.coef_mlp_vis(self.mlp_vis(vis_mlp_in))
            ir_fea_in = ir_fea_in + self.coef_mlp_ir(self.mlp_ir(ir_mlp_in))
            
        return [vis_fea_in, ir_fea_in]


class TransformerFusionBlock(BaseModule):
    def __init__(self, d_model, vert_anchors=16, horz_anchors=16, fusion=False, 
                 h=8, block_exp=4, n_layer=1, attn_pdrop=0.1, resid_pdrop=0.1, loops_num=1):
        super(TransformerFusionBlock, self).__init__()
        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        
        self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
        self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
        nn.init.trunc_normal_(self.pos_emb_vis, std=.02)
        nn.init.trunc_normal_(self.pos_emb_ir, std=.02)

        self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')
        self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')
        self.vis_pool_weights = LearnableWeights() 
        self.ir_pool_weights = LearnableWeights()  

        self.crosstransformer = nn.Sequential(
            *[CrossTransformerBlock(d_model, h, block_exp, attn_pdrop, resid_pdrop, loops_num) 
              for _ in range(n_layer)]
        )
        self.fusion_enabled = fusion 
        self.out_fusion_module = None
        if self.fusion_enabled:
            self.out_fusion_module = ConcatFusion(in_channels=d_model)
        
        self.apply(self._init_custom_weights) 

    def _init_custom_weights(self, module): 
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None: nn.init.ones_(module.weight)
            if module.bias is not None: nn.init.zeros_(module.bias)


    def forward(self, ir_fea_orig, rgb_fea_orig): 
        # bs, c, h_orig, w_orig = rgb_fea_orig.shape

        # pooled_rgb_fea = self.vis_pool_weights(self.avgpool(rgb_fea_orig), self.maxpool(rgb_fea_orig))
        # pooled_ir_fea = self.ir_pool_weights(self.avgpool(ir_fea_orig), self.maxpool(ir_fea_orig))
        
        # rgb_fea_flat = pooled_rgb_fea.flatten(2).permute(0, 2, 1) + self.pos_emb_vis
        # ir_fea_flat = pooled_ir_fea.flatten(2).permute(0, 2, 1) + self.pos_emb_ir
        ir , rgb = ir_fea_orig, rgb_fea_orig
        fused_rgb_flat, fused_ir_flat = self.crosstransformer([rgb, ir])
        
        # _, _, h_pooled, w_pooled = pooled_rgb_fea.shape
        # rgb_fea_CFE = fused_rgb_flat.permute(0, 2, 1).view(bs, c, h_pooled, w_pooled)
        # ir_fea_CFE = fused_ir_flat.permute(0, 2, 1).view(bs, c, h_pooled, w_pooled)

        # interp_mode = 'bilinear' if not self.training else 'nearest' 
        # align_corners_val = False if interp_mode == 'bilinear' else None
        
        # rgb_fea_CFE_upsampled = F.interpolate(rgb_fea_CFE, size=(h_orig, w_orig), mode=interp_mode, align_corners=align_corners_val)
        # ir_fea_CFE_upsampled = F.interpolate(ir_fea_CFE, size=(h_orig, w_orig), mode=interp_mode, align_corners=align_corners_val)
        
        out_rgb_fea = fused_rgb_flat + rgb_fea_orig
        out_ir_fea = fused_ir_flat + ir_fea_orig

        return [out_ir_fea, out_rgb_fea]


# @MODELS.register_module()
class DMFF(BaseModule):
    def __init__(
        self, in_channels: int = 256, vert_anchor: int = 16, horz_anchor: int = 16,
        loops_num: int = 1, fusion: bool = True, mod_weight: bool = False,
        tf_num_heads: int = 8, tf_block_exp: int = 4, tf_n_layers: int = 1,
        tf_attn_pdrop: float = 0.1, tf_resid_pdrop: float = 0.1):
        super().__init__()
        self.block = TransformerFusionBlock(
            d_model=in_channels, vert_anchors=vert_anchor, horz_anchors=horz_anchor,
            fusion=fusion, h=tf_num_heads, block_exp=tf_block_exp, n_layer=tf_n_layers,
            attn_pdrop=tf_attn_pdrop, resid_pdrop=tf_resid_pdrop, loops_num=loops_num
        )
        self.mw = None
        if mod_weight:
            self.mw = ModalityWeightingLayer(in_channels) 
        
    def forward(self, inputs1: torch.Tensor, inputs2: torch.Tensor):
        # if self.mw is not None:
        #     inputs1_w, inputs2_w = self.mw(inputs1, inputs2)
        # else:
        #     inputs1_w, inputs2_w = inputs1, inputs2
        
        out = self.block(inputs1, inputs2) 
        return out


# --- New Combined MambaTransformerDMFF Model ---
@MODELS.register_module()
class MambaTransformerBlock(BaseModule):
    def __init__(
        self,
        in_channels: int = 256,
        mamba_size: Tuple[int, int] = (80, 80), 
        mamba_dt_rank: Any = "auto", 
        mamba_d_state: Any = "auto", 
        mamba_ssm_ratio: float = 2.0, # Added for SS2D blocks
        mamba_bi: bool = False,       # Added for SS2D blocks
        transformer_vert_anchor: int = 16,
        transformer_horz_anchor: int = 16,
        transformer_loops_num: int = 1,
        transformer_fusion: bool = True,
        transformer_mod_weight: bool = False,
        transformer_num_heads: int = 8,
        transformer_block_exp: int = 4,
        transformer_n_layers: int = 1,
        transformer_attn_pdrop: float = 0.1,
        transformer_resid_pdrop: float = 0.1,
        init_cfg: Optional[dict] = None 
    ):
        super().__init__(init_cfg=init_cfg)

        self.patch_embed = PatchEmbed(img_size=mamba_size, patch_size=(mamba_size[0] // 8, mamba_size[1] // 8), stride=(mamba_size[0] // 8, mamba_size[1] // 8), in_chans=in_channels, embed_dim=in_channels)
        patch_grid_size = self.patch_embed.grid_size  # (grid_h, grid_w)
        self.mamba_fusion_stage = SS2D_intra(
            d_model=in_channels,
            size=patch_grid_size,
            d_state=mamba_d_state,
            ssm_ratio=mamba_ssm_ratio,
            bi = mamba_bi,
            dt_rank=mamba_dt_rank,
        )

        self.transformer_fusion_stage = DMFF(
            in_channels=in_channels, 
            vert_anchor=transformer_vert_anchor,
            horz_anchor=transformer_horz_anchor,
            loops_num=transformer_loops_num,
            fusion=transformer_fusion,
            mod_weight=transformer_mod_weight,
            tf_num_heads=transformer_num_heads,
            tf_block_exp=transformer_block_exp,
            tf_n_layers=transformer_n_layers,
            tf_attn_pdrop=transformer_attn_pdrop,
            tf_resid_pdrop=transformer_resid_pdrop
        )

        self.reconstruct = Reconstruct(mamba_size , patch_grid_size)
        self.pos_embed_r = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, in_channels))
        self.pos_embed_t = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, in_channels))

        self.out_fusion_module = ConcatFusion(in_channels=in_channels)

        self.mw = None
        if transformer_mod_weight:
            self.mw = ModalityWeightingLayer(in_channels)

    def forward(self, inputs1: torch.Tensor, inputs2: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass for MambaTransformerDMFF.
        """
        # Stage 1: Mamba Fusion
        # patch_embed_mamba = self.mamba_fusion_stage.patch_embed
        # if inputs1.shape[2:] != patch_embed_mamba.img_size:
        #      # This warning can be helpful during debugging if input sizes are not as expected.
        #      # print(f"Warning: Input 1 H,W {inputs1.shape[2:]} does not match Mamba stage's expected size {patch_embed_mamba.img_size}. Resizing.")
        #      inputs1 = F.interpolate(inputs1, size=patch_embed_mamba.img_size, mode='bilinear', align_corners=False)
        # if inputs2.shape[2:] != patch_embed_mamba.img_size:
        #      # print(f"Warning: Input 2 H,W {inputs2.shape[2:]} does not match Mamba stage's expected size {patch_embed_mamba.img_size}. Resizing.")
        #      inputs2 = F.interpolate(inputs2, size=patch_embed_mamba.img_size, mode='bilinear', align_corners=False)
        if self.mw is not None:
            inputs1_w, inputs2_w = self.mw(inputs1, inputs2)
        r , t = self.patch_embed(inputs1_w) + self.pos_embed_r, self.patch_embed(inputs2_w) + self.pos_embed_t
        r, t = self.mamba_fusion_stage(r , t)
        # Stage 2: Transformer Fusion
        r , t = self.transformer_fusion_stage(r, t)
        r, t = self.reconstruct(r), self.reconstruct(t)
        return self.out_fusion_module(r + inputs1, t + inputs2)

if __name__ == '__main__':
    bs = 2
    in_channels = 64 
    feature_map_h, feature_map_w = 80, 80 

    mamba_size_param = (feature_map_h, feature_map_w)
    mamba_ops = 1
    mamba_ssm_r = 2.0
    mamba_bi_flag = False

    tf_vert_anchor = feature_map_h // 4 
    tf_horz_anchor = feature_map_w // 4
    tf_loops = 1
    tf_fusion = True 
    tf_mod_weight = False

    combined_model = MambaTransformerBlock(
        in_channels=in_channels,
        mamba_size=mamba_size_param,
        mamba_operations=mamba_ops,
        mamba_d_state="auto", 
        mamba_dt_rank="auto", 
        mamba_ssm_ratio=mamba_ssm_r,
        mamba_bi=mamba_bi_flag,
        transformer_vert_anchor=tf_vert_anchor,
        transformer_horz_anchor=tf_horz_anchor,
        transformer_loops_num=tf_loops,
        transformer_fusion=tf_fusion,
        transformer_mod_weight=tf_mod_weight,
        transformer_num_heads=4, 
        transformer_block_exp=2, 
        transformer_n_layers=1  
    ).eval() # Set to eval mode for testing

    print(f"MambaTransformerDMFF model instantiated.")

    dummy_inputs1 = torch.randn(bs, in_channels, feature_map_h, feature_map_w)
    dummy_inputs2 = torch.randn(bs, in_channels, feature_map_h, feature_map_w)
    
    # For CUDA testing if available and mamba_ssm installed with CUDA ops
    # if torch.cuda.is_available():
    #     print("CUDA available, moving model and inputs to GPU.")
    #     combined_model = combined_model.cuda()
    #     dummy_inputs1 = dummy_inputs1.cuda()
    #     dummy_inputs2 = dummy_inputs2.cuda()
    # else:
    #     print("CUDA not available, running on CPU.")

    with torch.no_grad(): # Disable gradient calculations for inference
        output_features = combined_model(dummy_inputs1, dummy_inputs2)

    print("\nOutput from MambaTransformerDMFF:")
    if isinstance(output_features, list):
        for i, tensor in enumerate(output_features):
            print(f"Output tensor {i} shape: {tensor.shape}")
    else: 
        print(f"Output tensor shape: {output_features.shape}")

    print("\n--- Example with transformer_fusion=False ---")
    combined_model_two_outputs = MambaTransformerBlock(
        in_channels=in_channels,
        mamba_size=mamba_size_param,
        mamba_ssm_ratio=mamba_ssm_r,
        mamba_bi=mamba_bi_flag,
        transformer_fusion=False, 
        transformer_vert_anchor=tf_vert_anchor,
        transformer_horz_anchor=tf_horz_anchor,
    ).eval()
    
    # if torch.cuda.is_available():
    #     combined_model_two_outputs = combined_model_two_outputs.cuda()

    with torch.no_grad():
        output_features_two = combined_model_two_outputs(dummy_inputs1, dummy_inputs2)
        
    print("Output from MambaTransformerDMFF (transformer_fusion=False):")
    if isinstance(output_features_two, list):
        for i, tensor in enumerate(output_features_two):
            print(f"Output tensor {i} shape: {tensor.shape}")
