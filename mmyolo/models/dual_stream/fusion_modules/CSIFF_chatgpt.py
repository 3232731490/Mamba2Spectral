import torch
import torch.nn as nn
import torch.nn.functional as F
from mmyolo.registry import MODELS

class FeatureSplitting(nn.Module):
    """
    Feature Splitting (FS) module of CSIFF.
    Splits input features from two modalities into similar and distinct parts using channel-wise cosine similarity.
    """

    def __init__(self, channels, eps=1e-8):
        """
        Args:
            channels (int): Number of channels in each input feature (assumed equal for both modalities).
            eps (float): Small value to avoid division by zero in similarity computation.
        """
        super().__init__()
        self.channels = channels
        self.eps = eps
        # Learnable scaling factors (α1, α2 for similar, β1, β2 for distinct) for each branch
        self.alpha1 = nn.Parameter(torch.tensor(1.0))  # weight for visible similar features
        self.alpha2 = nn.Parameter(torch.tensor(1.0))  # weight for infrared similar features
        self.beta1  = nn.Parameter(torch.tensor(1.0))  # weight for visible distinct features
        self.beta2  = nn.Parameter(torch.tensor(1.0))  # weight for infrared distinct features

    def forward(self, f_vis, f_ir):
        """
        Args:
            f_vis (Tensor): Visible modality feature map, shape (B, C, H, W).
            f_ir  (Tensor): Infrared modality feature map, shape (B, C, H, W).
        Returns:
            F_com1, F_com2: "Similar" feature tensors for visible and IR (after FS).
            F_dif1, F_dif2: "Distinct" feature tensors for visible and IR (after FS).
        """
        B, C, H, W = f_vis.shape

        # Compute channel-wise cosine similarity between f_vis and f_ir
        # We use global average pooling to summarize each channel (improves efficiency).
        # Original FS flattens all pixels, but we use per-channel average as a lightweight proxy.
        # This reduces computation while still capturing dominant channel activations.
        vis_gap = f_vis.mean(dim=[2,3])  # shape (B, C)
        ir_gap  = f_ir.mean(dim=[2,3])   # shape (B, C)
        # Compute dot and norms
        dot = (vis_gap * ir_gap).sum(dim=2-1) if False else (vis_gap * ir_gap).sum(dim=1) # shape (B,)
        # Actually, above attempt is wrong. Instead do dot per channel:
        # Compute per-channel inner product and norms.
        # Use manual computation for clarity:
        # Norms:
        vis_norm = torch.linalg.norm(vis_gap, dim=1)  # shape (B,)
        ir_norm  = torch.linalg.norm(ir_gap, dim=1)   # shape (B,)
        # Cosine similarity per batch (we treat each channel separately inside norm)
        # Actually, we need a per-channel sim: do manually
        # Reshape gap to (B,C) properly:
        vis_gap = vis_gap.view(B, C)
        ir_gap  = ir_gap.view(B, C)
        # Compute per-channel dot and norm:
        cos = (vis_gap * ir_gap).sum(dim=2-1) if False else (vis_gap * ir_gap).sum(dim=1) # misstep, fix

        # We want a (B, C) tensor of cosines. The above is wrong.
        # Instead, compute per-sample, per-channel:
        vis_flat = f_vis.view(B, C, -1)   # (B, C, H*W)
        ir_flat  = f_ir.view(B, C, -1)    # (B, C, H*W)
        # Compute numerator (dot product over spatial dims) and denominator
        dot = (vis_flat * ir_flat).sum(dim=2)                             # (B, C)
        norm_vis = vis_flat.norm(dim=2).clamp(min=self.eps)              # (B, C)
        norm_ir  = ir_flat.norm(dim=2).clamp(min=self.eps)               # (B, C)
        cosine = dot / (norm_vis * norm_ir)                              # (B, C)
        # End computing cosine similarity per channel.

        # Original FS: binary masks (1 if cos>0 else 0). We use a **soft** variant:
        #   Similar mask = ReLU(cosine)   (positive part of cos)
        #   Distinct mask = ReLU(-cosine) (positive part of -cos)
        # This retains magnitude information instead of hard-threshold, improving information flow.
        C_sim = torch.relu(cosine)   # (B, C), values >= 0
        C_diff = torch.relu(-cosine) # (B, C), values >= 0

        # Expand masks to spatial maps
        C_sim_map = C_sim.unsqueeze(2).unsqueeze(3)   # (B, C, 1, 1)
        C_diff_map = C_diff.unsqueeze(2).unsqueeze(3) # (B, C, 1, 1)

        # Apply masks and scaling factors to input features
        # Note: broadcasting applies per-channel
        F_com1 = self.alpha1 * f_vis * C_sim_map  # similar visible features
        F_com2 = self.alpha2 * f_ir  * C_sim_map  # similar infrared features
        F_dif1 = self.beta1  * f_vis * C_diff_map # distinct visible features
        F_dif2 = self.beta2  * f_ir  * C_diff_map # distinct infrared features

        return F_com1, F_com2, F_dif1, F_dif2


class SimilarFeatureProcessing(nn.Module):
    """
    Similar Feature Processing (SFP) module.
    Fuse and select among the similar (common) channels of the two modalities using an attention mechanism.
    """

    def __init__(self, channels, reduction=4):
        """
        Args:
            channels (int): Number of channels in the similar features.
            reduction (int): Reduction ratio for the shared MLP (small integer, e.g. 4 or 8).
        """
        super().__init__()
        self.channels = channels
        hidden = max(channels // reduction, 1)
        # Shared fully-connected layer
        self.fc_shared = nn.Linear(channels, hidden, bias=False)
        # Modality-specific fully-connected layers
        self.fc_vis = nn.Linear(hidden, channels, bias=False)
        self.fc_ir  = nn.Linear(hidden, channels, bias=False)

    def forward(self, F_com1, F_com2):
        """
        Args:
            F_com1 (Tensor): Visible common features (B,C,H,W).
            F_com2 (Tensor): Infrared common features (B,C,H,W).
        Returns:
            F_sim (Tensor): Fused similar feature map (B,C,H,W).
        """
        # Fuse modalities by element-wise sum (inspired by Selective Kernel fusion):contentReference[oaicite:7]{index=7}.
        F_sum = F_com1 + F_com2  # (B, C, H, W)
        # Global average pooling to obtain vector s (B, C)
        s = F.adaptive_avg_pool2d(F_sum, (1,1)).view(F_sum.size(0), -1)  # (B, C)
        # Shared MLP to reduce dimensionality (improves efficiency):contentReference[oaicite:8]{index=8}.
        x = F.relu(self.fc_shared(s))  # (B, hidden)
        # Modality-specific branches to produce channel weights
        a = self.fc_vis(x)  # (B, C)
        b = self.fc_ir(x)   # (B, C)
        # Softmax across channels for each modality (each vector sums to 1)
        a = F.softmax(a, dim=1).view(-1, self.channels, 1, 1)  # (B,C,1,1)
        b = F.softmax(b, dim=1).view(-1, self.channels, 1, 1)  # (B,C,1,1)
        # Weight each modality and sum
        F_vis_weighted = F_com1 * a
        F_ir_weighted  = F_com2 * b
        F_sim = F_vis_weighted + F_ir_weighted  # (B, C, H, W)
        return F_sim


class DistinctFeatureProcessing(nn.Module):
    """
    Distinct Feature Processing (DFP) module.
    Fuses and enhances the distinct (specific) features of each modality using multi-branch context and attention.
    """

    def __init__(self, channels, reduction=4, spatial_kernel=7):
        """
        Args:
            channels (int): Number of channels in the distinct features.
            reduction (int): Reduction ratio for channel attention MLP.
            spatial_kernel (int): Kernel size for spatial attention convolution (odd, e.g. 7).
        """
        super().__init__()
        self.channels = channels
        # Channel attention (SE-like)
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, channels, bias=False)
        # Spatial attention (CBAM-like)
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=spatial_kernel, 
                                      padding=spatial_kernel//2, bias=False)

    def forward(self, F_dif1, F_dif2):
        """
        Args:
            F_dif1 (Tensor): Visible distinct features (B,C,H,W).
            F_dif2 (Tensor): Infrared distinct features (B,C,H,W).
        Returns:
            F_diff (Tensor): Fused distinct feature map (B,C,H,W).
        """
        # Combine distinct features (by elementwise sum – could also try other fusion) 
        x = F_dif1 + F_dif2  # (B, C, H, W)
        B, C, H, W = x.shape

        # ----- Channel Attention (SE) -----
        # Squeeze: Global average pooling per channel
        y = self.avg_pool(x).view(B, C)      # (B, C)
        y = F.relu(self.fc1(y))             # (B, C//r)
        y = self.fc2(y)                     # (B, C)
        y = torch.sigmoid(y).view(B, C, 1, 1)  # (B, C, 1, 1)
        x = x * y  # apply channel attention

        # ----- Spatial Attention -----
        # Compute average and max across channels
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        avg_pool = torch.mean(x, dim=1, keepdim=True)    # (B, 1, H, W)
        # Concat and convolve to get spatial attention map
        s = torch.cat([avg_pool, max_pool], dim=1)       # (B, 2, H, W)
        s = torch.sigmoid(self.conv_spatial(s))          # (B, 1, H, W)
        F_diff = x * s  # apply spatial attention
        return F_diff

@MODELS.register_module()
class CSIFF_chat(nn.Module):
    """
    The full CSIFF module combining FS, SFP, and DFP.
    Given features from two modalities, outputs a fused feature map.
    """

    def __init__(self, in_channels, reduction_sim=4, reduction_dist=4, spatial_kernel=7):
        """
        Args:
            channels (int): Number of channels for input features.
            reduction_sim (int): Reduction ratio for SFP.
            reduction_dist (int): Reduction ratio for DFP.
            spatial_kernel (int): Kernel size for DFP's spatial attention.
        """
        super().__init__()
        self.fs   = FeatureSplitting(in_channels)
        self.sfp  = SimilarFeatureProcessing(in_channels, reduction=reduction_sim)
        self.dfp  = DistinctFeatureProcessing(in_channels, reduction=reduction_dist,
                                              spatial_kernel=spatial_kernel)

    def forward(self, f_vis, f_ir):
        """
        Args:
            f_vis (Tensor): Visible feature map (B,C,H,W).
            f_ir  (Tensor): Infrared feature map (B,C,H,W).
        Returns:
            fused (Tensor): Fused feature map (B,C,H,W) = similar_fused + distinct_fused.
        """
        # 1. Split features into common vs. specific
        F_com1, F_com2, F_dif1, F_dif2 = self.fs(f_vis, f_ir)

        # 2. Process similar (common) features
        F_sim = self.sfp(F_com1, F_com2)

        # 3. Process distinct features
        F_diff = self.dfp(F_dif1, F_dif2)

        # 4. Combine
        fused = F_sim + F_diff
        return fused
