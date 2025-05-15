import torch
import torch.nn as nn
import torch.nn.functional as F


from mmyolo.registry import MODELS

@MODELS.register_module()
class SHFModule(nn.Module):
    def __init__(self, in_channels, sfe_reduction_factor=4, sic_intermediate_channels=None, hfo_bottleneck_factor=2, out_channels=None, use_residual=True):
        """
        Selective Harmony Fusion (SHF) Module.

        Args:
            in_channels (int): Number of input channels for both F_A and F_B.
                               Assumes F_A and F_B have the same number of channels.
            sfe_reduction_factor (int): Reduction factor for the SFE module's FC layers.
                                        Hidden channels = in_channels // sfe_reduction_factor.
            sic_intermediate_channels (int, optional): Number of intermediate channels for the SIC module's
                                                       PWConv. Defaults to in_channels // 2.
            hfo_bottleneck_factor (int): Factor to determine the bottleneck channels in HFO.
                                         Concatenated channels // hfo_bottleneck_factor.
            out_channels (int, optional): Number of output channels for the module.
                                          Defaults to in_channels.
            use_residual (bool): Whether to use a residual connection in the HFO stage.
        """
        super(SHFModule, self).__init__()

        if out_channels is None:
            out_channels = in_channels
        if sic_intermediate_channels is None:
            sic_intermediate_channels = in_channels // 2
            if sic_intermediate_channels == 0: # Ensure at least 1 channel
                sic_intermediate_channels = 1


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_residual = use_residual

        # --- 1. Selective Feature Enhancement (SFE) ---
        sfe_hidden_channels = max(1, in_channels // sfe_reduction_factor) # Ensure at least 1 channel

        # SFE for Modality A (M_A)
        self.sfe_A_fc1 = nn.Conv2d(in_channels * 2, sfe_hidden_channels, kernel_size=1, bias=False)
        self.sfe_A_relu = nn.ReLU(inplace=True)
        self.sfe_A_fc2 = nn.Conv2d(sfe_hidden_channels, in_channels, kernel_size=1, bias=False)
        self.sfe_A_sigmoid = nn.Sigmoid()

        # SFE for Modality B (M_B) - layers can be shared if designed symmetrically or kept separate
        self.sfe_B_fc1 = nn.Conv2d(in_channels * 2, sfe_hidden_channels, kernel_size=1, bias=False)
        self.sfe_B_relu = nn.ReLU(inplace=True)
        self.sfe_B_fc2 = nn.Conv2d(sfe_hidden_channels, in_channels, kernel_size=1, bias=False)
        self.sfe_B_sigmoid = nn.Sigmoid()

        # --- 2. Shared Information Condenser (SIC) ---
        self.sic_pwconv = nn.Conv2d(in_channels, sic_intermediate_channels, kernel_size=1, bias=False)
        self.sic_relu = nn.ReLU(inplace=True)
        self.sic_dwconv = nn.Conv2d(sic_intermediate_channels, sic_intermediate_channels, kernel_size=3, padding=1, groups=sic_intermediate_channels, bias=False)
        # Output of SIC has sic_intermediate_channels

        # --- 3. Harmonized Fusion Output (HFO) ---
        hfo_concat_channels = in_channels + in_channels + sic_intermediate_channels # F_A_unique, F_B_unique, F_shared_condensed
        hfo_mid_bottleneck_channels = max(1, hfo_concat_channels // hfo_bottleneck_factor)

        self.hfo_pwconv_reduce = nn.Conv2d(hfo_concat_channels, hfo_mid_bottleneck_channels, kernel_size=1, bias=False)
        self.hfo_relu1 = nn.ReLU(inplace=True)
        self.hfo_dwconv = nn.Conv2d(hfo_mid_bottleneck_channels, hfo_mid_bottleneck_channels, kernel_size=3, padding=1, groups=hfo_mid_bottleneck_channels, bias=False)
        self.hfo_relu2 = nn.ReLU(inplace=True)
        self.hfo_pwconv_expand = nn.Conv2d(hfo_mid_bottleneck_channels, self.out_channels, kernel_size=1, bias=False)

        if self.use_residual:
            self.residual_pwconv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, bias=False)
            # Batch norms can be added after each conv layer if desired, typically before ReLU
            # For brevity, they are omitted here but are common in practice.
            self.bn_res = nn.BatchNorm2d(self.out_channels)


        # Example of adding Batch Norms (optional, add as needed)
        self.bn_sfe_A_fc1 = nn.BatchNorm2d(sfe_hidden_channels)
        self.bn_sfe_A_fc2 = nn.BatchNorm2d(in_channels)
        self.bn_sfe_B_fc1 = nn.BatchNorm2d(sfe_hidden_channels)
        self.bn_sfe_B_fc2 = nn.BatchNorm2d(in_channels)

        self.bn_sic_pw = nn.BatchNorm2d(sic_intermediate_channels)
        self.bn_sic_dw = nn.BatchNorm2d(sic_intermediate_channels)

        self.bn_hfo_pw_reduce = nn.BatchNorm2d(hfo_mid_bottleneck_channels)
        self.bn_hfo_dw = nn.BatchNorm2d(hfo_mid_bottleneck_channels)
        self.bn_hfo_pw_expand = nn.BatchNorm2d(self.out_channels)


    def forward(self, F_A, F_B):
        # Input F_A, F_B: (batch_size, in_channels, height, width)
        # --- 1. Selective Feature Enhancement (SFE) ---
        P_A = F.adaptive_avg_pool2d(F_A, (1, 1))
        P_B = F.adaptive_avg_pool2d(F_B, (1, 1))


        # SFE for Modality A
        Diff_A_vs_B = P_A - P_B
        Concat_SFE_A = torch.cat((P_A, Diff_A_vs_B), dim=1) # (B, 2*C, 1, 1)

        M_A = self.sfe_A_fc1(Concat_SFE_A)
        M_A = self.bn_sfe_A_fc1(M_A)
        M_A = self.sfe_A_relu(M_A)
        M_A = self.sfe_A_fc2(M_A)
        M_A = self.bn_sfe_A_fc2(M_A)
        M_A = self.sfe_A_sigmoid(M_A) # Channel weights (B, C, 1, 1)
        F_A_unique = F_A * M_A

        # SFE for Modality B
        Diff_B_vs_A = P_B - P_A
        Concat_SFE_B = torch.cat((P_B, Diff_B_vs_A), dim=1)

        M_B = self.sfe_B_fc1(Concat_SFE_B)
        M_B = self.bn_sfe_B_fc1(M_B)
        M_B = self.sfe_B_relu(M_B)
        M_B = self.sfe_B_fc2(M_B)
        M_B = self.bn_sfe_B_fc2(M_B)
        M_B = self.sfe_B_sigmoid(M_B) # Channel weights (B, C, 1, 1)
        F_B_unique = F_B * M_B

        # --- 2. Shared Information Condenser (SIC) ---
        F_sum = F_A + F_B # Using original F_A and F_B for shared info

        shared_feat = self.sic_pwconv(F_sum)
        shared_feat = self.bn_sic_pw(shared_feat)
        shared_feat = self.sic_relu(shared_feat)

        shared_feat = self.sic_dwconv(shared_feat)
        shared_feat = self.bn_sic_dw(shared_feat)
        F_shared_condensed = self.sic_relu(shared_feat) # Added ReLU after DWConv

        # --- 3. Harmonized Fusion Output (HFO) ---
        F_combined = torch.cat((F_A_unique, F_B_unique, F_shared_condensed), dim=1)

        fused_intermediate = self.hfo_pwconv_reduce(F_combined)
        fused_intermediate = self.bn_hfo_pw_reduce(fused_intermediate)
        fused_intermediate = self.hfo_relu1(fused_intermediate)

        fused_intermediate = self.hfo_dwconv(fused_intermediate)
        fused_intermediate = self.bn_hfo_dw(fused_intermediate)
        fused_intermediate = self.hfo_relu2(fused_intermediate)

        fused_intermediate = self.hfo_pwconv_expand(fused_intermediate)
        fused_output = self.bn_hfo_pw_expand(fused_intermediate)
        # No final ReLU here, common practice if it's the output of a block before summation or loss

        if self.use_residual:
            F_residual_sum = (F_A + F_B) / 2.0 # Average of original inputs
            residual_path = self.residual_pwconv(F_residual_sum)
            residual_path = self.bn_res(residual_path)
            fused_output += residual_path

        return F.relu(fused_output) # Apply final ReLU if needed, or based on network design

if __name__ == '__main__':
    # Example Usage
    batch_size = 4
    in_channels_example = 64 # Example: features from ResNet block
    height, width = 32, 32
    out_channels_example = 64 # or 128, etc.

    # Create dummy input tensors
    F_A_dummy = torch.randn(batch_size, in_channels_example, height, width)
    F_B_dummy = torch.randn(batch_size, in_channels_example, height, width)

    # Instantiate the SHF module
    shf_block = SHFModule(
        in_channels=in_channels_example,
        sfe_reduction_factor=4,
        sic_intermediate_channels=in_channels_example // 2,
        hfo_bottleneck_factor=2,
        out_channels=out_channels_example,
        use_residual=True
    )

    # Pass inputs through the module
    fused_features = shf_block(F_A_dummy, F_B_dummy)

    # Print output shape
    print("Input F_A shape:", F_A_dummy.shape)
    print("Input F_B shape:", F_B_dummy.shape)
    print("Fused features shape:", fused_features.shape)

    # Check number of parameters
    num_params = sum(p.numel() for p in shf_block.parameters() if p.requires_grad)
    print(f"Number of parameters in SHFModule: {num_params:,}")

    # Test with different channel configurations
    print("\n--- Test with different channels ---")
    in_channels_test = 32
    out_channels_test = 64
    F_A_test = torch.randn(2, in_channels_test, 16, 16)
    F_B_test = torch.randn(2, in_channels_test, 16, 16)
    shf_block_test = SHFModule(
        in_channels=in_channels_test,
        out_channels=out_channels_test,
        sic_intermediate_channels=8 # Be explicit
    )
    fused_test = shf_block_test(F_A_test, F_B_test)
    print("Input test shape:", F_A_test.shape)
    print("Fused test shape:", fused_test.shape)
    num_params_test = sum(p.numel() for p in shf_block_test.parameters() if p.requires_grad)
    print(f"Number of parameters (test config): {num_params_test:,}")