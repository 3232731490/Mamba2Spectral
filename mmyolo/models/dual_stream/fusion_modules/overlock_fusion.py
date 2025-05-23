import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Helper for ConvFFN (Feed-Forward Network with Convolutions)
class ConvFFN(nn.Module):
    """
    Convolutional Feed-Forward Network block.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        self.gelu = nn.GELU() # Using GELU activation as commonly used in modern networks
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        return self.conv2(self.gelu(self.conv1(x)))

# Helper for SELayer (Squeeze-and-Excitation Layer) - Can be adapted for gating
class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Layer, used here as inspiration for a gating mechanism.
    It computes channel-wise weights based on global spatial information.
    """
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid() # Use Sigmoid to output gating weights between 0 and 1
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y # Output gating weights

# Simplified Contextual Dynamic Modulation (Inspired by ContMix)
class ContextualDynamicModulation(nn.Module):
    """
    A module inspired by OverLoCK's ContMix for dynamic spatial modulation
    guided by a global context prior. It calculates affinity between feature
    locations and context locations to generate spatially-varying modulation weights.
    """
    def __init__(self, in_channels, context_channels, output_channels, S=7): # Renamed out_channels to output_channels
        super().__init__()
        self.S = S # Spatial size for context pooling (determines number of context regions)
        self.output_channels = output_channels # Store output_channels

        # Projections for Q (from input features) and K (from context prior) for affinity calculation.
        # Project to a common dimension for batch matrix multiplication.
        # The affinity dimension can be related to input or context channels, or a fixed value.
        # Let's use a dimension related to the input channels for Q and K projections.
        affinity_dim = in_channels // 2 # Example affinity dimension
        if affinity_dim == 0: affinity_dim = 1 # Ensure affinity_dim is at least 1

        self.q_proj_affinity = nn.Conv2d(in_channels, affinity_dim, 1)
        # K comes from the context prior, so its projection should match context_channels
        self.k_proj_affinity = nn.Conv2d(context_channels, affinity_dim, 1)


        # Adaptive pooling to get context region features for K
        self.context_pool = nn.AdaptiveAvgPool2d(S)

        # Linear layer to generate modulation weights from the affinity map.
        # Affinity map shape: (N, H*W, S*S)
        # Linear layer maps S*S (number of context regions) to output_channels (number of output channels).
        self.modulation_layer = nn.Linear(S * S, output_channels)

        # Output projection for the input features before applying modulation.
        # Ensures the input features have the correct channel dimension for element-wise modulation.
        self.out_proj = nn.Conv2d(in_channels, output_channels, 1)

    def forward(self, x, context_prior):
        """
        Args:
            x (Tensor): Input features (e.g., concatenated multispectral features).
                        Shape (N, in_channels, H, W).
            context_prior (Tensor): Global context prior feature map.
                                    Shape (N, context_channels, H, W).

        Returns:
            Tensor: Modulated features. Shape (N, output_channels, H, W).
        """
        N, C_in, H, W = x.size()
        C_ctx = context_prior.size(1)
        C_out = self.output_channels # Use the stored output_channels

        # Compute Q from input features for affinity calculation
        # (N, in_channels, H, W) -> (N, affinity_dim, H, W) -> (N, affinity_dim, H*W) -> (N, H*W, affinity_dim)
        q_affinity = self.q_proj_affinity(x).view(N, self.q_proj_affinity.out_channels, H * W).transpose(1, 2)

        # Compute K from context prior with pooling for affinity calculation
        # (N, context_channels, H, W) -> (N, context_channels, S, S) -> (N, context_channels, S*S) -> (N, S*S, context_channels)
        k_pooled = self.context_pool(context_prior)
        k_affinity = self.k_proj_affinity(k_pooled).view(N, self.k_proj_affinity.out_channels, self.S * self.S) # (N, affinity_dim, S*S)

        # Calculate affinity matrix: Q_affinity @ K_affinity^T
        # (N, H*W, affinity_dim) @ (N, affinity_dim, S*S) -> (N, H*W, S*S)
        # This calculates the affinity between each feature location (token) and each context region.
        affinity = torch.bmm(q_affinity, k_affinity)

        # Generate modulation weights from the affinity matrix
        # (N, H*W, S*S) -> Linear layer -> (N, H*W, output_channels)
        modulation_weights = self.modulation_layer(affinity)
        # Reshape to spatial dimensions: (N, H*W, output_channels) -> (N, output_channels, H*W) -> (N, output_channels, H, W)
        modulation_weights = modulation_weights.transpose(1, 2).view(N, C_out, H, W)

        # Project input features to the desired output channel dimension
        x_proj_out = self.out_proj(x) # (N, output_channels, H, W)

        # Apply modulation by element-wise multiplication
        modulated_x = x_proj_out * modulation_weights

        return modulated_x

# Fusion Gated Dynamic Spatial Aggregator (Inspired by OverLoCK's GDSA)
class FusionGDSA(nn.Module):
    """
    A module inspired by OverLoCK's Gated Dynamic Spatial Aggregator (GDSA).
    It combines the dynamic modulation with a gating mechanism.
    """
    def __init__(self, in_channels, context_channels, output_channels, S=7): # Renamed out_channels to output_channels
        super().__init__()
        self.output_channels = output_channels # Store output_channels

        # Contextual Dynamic Modulation module
        self.dynamic_modulation = ContextualDynamicModulation(
            in_channels=in_channels,
            context_channels=context_channels,
            output_channels=output_channels, # Pass the target output_channels
            S=S
        )

        # Gating mechanism to refine the modulated features.
        # It takes concatenated features and context prior as input to determine gate weights.
        gate_channels_in = in_channels + context_channels
        self.gate_proj = nn.Sequential(
            nn.Conv2d(gate_channels_in, gate_channels_in // 2, 1), # Reduce channels
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_channels_in // 2, output_channels, 1), # Project to output_channels
            nn.Sigmoid() # Output gate weights between 0 and 1
        )

        # 1x1 convolution and SiLU activation applied to the concatenated
        # features and context prior before feeding to the gate projection,
        # similar to the structure in the paper's GDSA.
        self.gate_input_proj = nn.Sequential(
            nn.Conv2d(gate_channels_in, gate_channels_in, 1),
            nn.SiLU(inplace=True)
        )


    def forward(self, x, context_prior):
        """
        Args:
            x (Tensor): Input features (e.g., concatenated multispectral features).
                        Shape (N, in_channels, H, W).
            context_prior (Tensor): Global context prior feature map.
                                    Shape (N, context_channels, H, W).

        Returns:
            Tensor: Gated and modulated features. Shape (N, output_channels, H, W).
        """
        # Concatenate input features and context prior for the gating mechanism input
        x_concat = torch.cat([x, context_prior], dim=1)
        gate_input = self.gate_input_proj(x_concat)

        # Compute output from the dynamic modulation module
        dynamic_mod_out = self.dynamic_modulation(x, context_prior) # (N, output_channels, H, W)

        # Compute gate weights based on the concatenated features and context
        gate_weights = self.gate_proj(gate_input) # (N, output_channels, H, W)

        # Apply gating by element-wise multiplication
        gated_out = dynamic_mod_out * gate_weights

        return gated_out

# Main OverLoCK-inspired Multispectral Fusion Module
from mmyolo.registry import MODELS

@MODELS.register_module()
class OverLoCKFusionModule(nn.Module):
    """
    Multispectral Feature Fusion Module inspired by OverLoCK.
    It takes visible and infrared feature maps, projects them to a common
    channel space, concatenates them, generates a global context prior,
    and fuses them using a FusionGDSA block with residual connections and FFN.
    The output feature map has the same shape as the input feature maps,
    assuming input modalities have the same channel dimension.
    """
    def __init__(self, in_channels, S=7):
        super().__init__()

        output_channels = in_channels
        self.vis_channels = in_channels
        self.ir_channels = in_channels
        self.common_channels = in_channels
        self.context_channels = in_channels // 2
        self.output_channels = output_channels # Store the determined output channels
        self.S = S # Spatial size for context pooling

        # Initial feature projections for each modality to a common channel dimension.
        self.vis_proj = nn.Conv2d(self.vis_channels, self.common_channels, 1)
        self.ir_proj = nn.Conv2d(self.ir_channels, self.common_channels, 1)

        # Simplified context generator.
        # This network processes the concatenated features to produce a global context prior.
        # Input channels are 2 * common_channels.
        self.context_generator = nn.Sequential(
            nn.Conv2d(2 * self.common_channels, self.context_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.context_channels, self.context_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.context_channels, self.context_channels, 1) # Final 1x1 conv to refine channels
        )

        # The core Fusion Dynamic Block.
        # Input to FusionGDSA is the concatenated projected features (2 * common_channels).
        # Output channels of FusionGDSA should be the desired output_channels.
        gdsa_in_channels = 2 * self.common_channels
        gdsa_output_channels = output_channels

        self.fusion_gdsa = FusionGDSA(
            in_channels=gdsa_in_channels,
            context_channels=self.context_channels,
            output_channels=gdsa_output_channels, # Pass the target output_channels
            S=S
        )

        # ConvFFN after the GDSA.
        # Input and output channels of FFN should be the desired output_channels.
        ffn_in_channels = output_channels
        ffn_hidden_channels = ffn_in_channels * 4 # Example expansion ratio
        ffn_out_channels = output_channels
        self.conv_ffn = ConvFFN(ffn_in_channels, ffn_hidden_channels, ffn_out_channels)

        # Residual connection. Input is the concatenated projected features (2 * common_channels).
        # Output needs to match the output_channels.
        residual_in_channels = 2 * self.common_channels
        residual_out_channels = output_channels
        self.residual_conv = nn.Conv2d(residual_in_channels, residual_out_channels, 1) if residual_in_channels != residual_out_channels else None


    def forward(self, x_vis, x_ir):
        """
        Args:
            x_vis (Tensor): Visible modality feature map.
                            Shape (N, vis_channels, H, W).
            x_ir (Tensor): Infrared modality feature map.
                           Shape (N, ir_channels, H, W).

        Returns:
            Tensor: Fused feature map. Shape (N, output_channels, H, W).
        """
        N, C_vis, H, W = x_vis.size()
        C_ir = x_ir.size(1)

        # Initial feature projections to a common channel dimension
        x_vis_proj = self.vis_proj(x_vis) # (N, common_channels, H, W)
        x_ir_proj = self.ir_proj(x_ir)   # (N, common_channels, H, W)

        # Concatenate the projected features
        x_fused_in = torch.cat([x_vis_proj, x_ir_proj], dim=1) # (N, 2 * common_channels, H, W)

        # Generate the global context prior from the concatenated features
        context_prior_pooled = self.context_generator(x_fused_in) # (N, context_channels, H/4, W/4)

        # Upsample the context prior back to the original spatial dimensions
        context_prior_upsampled = F.interpolate(
            context_prior_pooled,
            size=(H, W),
            mode='bilinear', # Use bilinear interpolation for upsampling
            align_corners=False # Recommended for spatial transformations
        ) # (N, context_channels, H, W)

        # Pass the concatenated features and upsampled context prior through the Fusion GDSA block.
        # This performs the dynamic modulation and gating.
        gdsa_out = self.fusion_gdsa(x_fused_in, context_prior_upsampled) # (N, output_channels, H, W)

        # Apply the residual connection before the ConvFFN.
        # The residual input is the concatenated projected features (x_fused_in).
        residual = x_fused_in
        # Adjust residual channels if necessary to match the output channels of GDSA/FFN
        if self.residual_conv:
            residual = self.residual_conv(residual) # (N, output_channels, H, W)

        # Add the residual to the output of the GDSA. This is the input to the FFN.
        ffn_in = gdsa_out + residual # (N, output_channels, H, W)

        # Pass through the ConvFFN.
        ffn_out = self.conv_ffn(ffn_in) # (N, output_channels, H, W)

        # Apply the second residual connection (as in the paper's Dynamic Block).
        # Add the output of the FFN to its input (which already includes the first residual).
        fused_features = ffn_out + ffn_in # (N, output_channels, H, W)

        return fused_features

# Example Usage:
# Assuming input feature maps from visible and infrared modalities
# Batch size N = 4
# Visible channels = 64, Infrared channels = 64
# Spatial dimensions H = 128, W = 128
vis_features = torch.randn(4, 64, 128, 128)
ir_features = torch.randn(4, 64, 128, 128)

# Instantiate the fusion module
# Parameters: vis_channels, ir_channels, common_channels, context_channels, S
# The output_channels will be automatically determined.
# If vis_channels == ir_channels, output_channels = vis_channels.
# If vis_channels != ir_channels, output_channels = common_channels.
# Example: Input channels are 64, common_channels=64, context_channels=32, S=7.
# The output channels will be 64.
fusion_module = OverLoCKFusionModule(
    in_channels=64,
)

# Perform a forward pass
fused_output = fusion_module(vis_features, ir_features)

# Print shapes to verify
print("Input visible features shape:", vis_features.shape)
print("Input infrared features shape:", ir_features.shape)
print("Fused output shape:", fused_output.shape)

# Example with different input channels (output channels will be common_channels=128)
# vis_features_diff = torch.randn(4, 64, 128, 128)
# ir_features_diff = torch.randn(4, 32, 128, 128)
# fusion_module_diff = OverLoCKFusionModule(
#     vis_channels=64,
#     ir_channels=32,
#     common_channels=128, # Output channels will be 128
#     context_channels=32,
#     S=7
# )
# fused_output_diff = fusion_module_diff(vis_features_diff, ir_features_diff)
# print("Input visible features shape (diff):", vis_features_diff.shape)
# print("Input infrared features shape (diff):", ir_features_diff.shape)
# print("Fused output shape (diff):", fused_output_diff.shape)
