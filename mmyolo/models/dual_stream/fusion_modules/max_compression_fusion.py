import torch
import torch.nn as nn
import torch.nn.functional as F

from mmyolo.registry import MODELS

@MODELS.register_module()
class CosineSortedMaxFusionModule(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
        Initializes the CosineSortedMaxFusion Module.

        This module:
        1. Creates an object-centric reference map from the two input feature maps.
        2. Calculates the cosine similarity of each channel in the input feature maps
           with this reference map.
        3. Sorts the channels of each input feature map based on these similarities.
        4. Fuses the channel-sorted feature maps using element-wise maximum.

        Args:
            epsilon (float): A small value to prevent division by zero in cosine similarity.
        """
        super(CosineSortedMaxFusionModule, self).__init__()
        self.epsilon = epsilon

    def _create_object_centric_reference(self, f_a: torch.Tensor, f_b: torch.Tensor) -> torch.Tensor:
        """
        Creates an object-centric reference map C_o' as described in the CSCR module[cite: 1].
        """
        # Channel Max Pooling (CMP) and Channel Average Pooling (CAP) for feature_map_a
        f_a_cmp, _ = torch.max(f_a, dim=1, keepdim=True)
        f_a_cap = torch.mean(f_a, dim=1, keepdim=True)
        # Combine CMP and CAP for f_a (e.g., by averaging, as C_RGB in paper is not explicitly defined from CMP/CAP)
        c_a = (f_a_cmp + f_a_cap) / 2.0

        # Channel Max Pooling (CMP) and Channel Average Pooling (CAP) for feature_map_b
        f_b_cmp, _ = torch.max(f_b, dim=1, keepdim=True)
        f_b_cap = torch.mean(f_b, dim=1, keepdim=True)
        # Combine CMP and CAP for f_b
        c_b = (f_b_cmp + f_b_cap) / 2.0

        # Concatenate along channel dimension to get C_F [cite: 1]
        c_f = torch.cat((c_a, c_b), dim=1) # Shape: (B, 2, H, W)

        # Max compression to get C_o [cite: 1]
        c_o, _ = torch.max(c_f, dim=1, keepdim=True) # Shape: (B, 1, H, W)

        # Sigmoid activation to get C_o' [cite: 1]
        c_o_prime = torch.sigmoid(c_o) # Shape: (B, 1, H, W)
        
        return c_o_prime

    def _sort_feature_map_by_similarity(self, feature_map: torch.Tensor, reference_map: torch.Tensor) -> torch.Tensor:
        """
        Sorts the channels of a feature map based on their cosine similarity to the reference map.
        """
        B, C, H, W = feature_map.shape
        
        sim_scores = F.cosine_similarity(feature_map.view(B, C, -1), reference_map.view(B, 1, -1), dim=2) # Shape: (B, C)
        
        # Get sorting indices (descending order)
        # Negative sign for ascending sort to behave like descending
        sorted_indices = torch.argsort(sim_scores, dim=1, descending=True) # Shape: (B, C)
        
        # Gather/Reorder the channels
        # sorted_indices needs to be expanded for gather: (B, C, H, W)
        sorted_indices_expanded = sorted_indices.view(B, C, 1, 1).expand(B, C, H, W)
        
        feature_map_sorted = torch.gather(feature_map, dim=1, index=sorted_indices_expanded)
        
        return feature_map_sorted

    def forward(self, feature_map_a: torch.Tensor, feature_map_b: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the CosineSortedMaxFusion Module.

        Args:
            feature_map_a (torch.Tensor): The feature map from the first modality.
                                          Shape: (B, C, H, W)
            feature_map_b (torch.Tensor): The feature map from the second modality.
                                          Shape: (B, C, H, W)

        Returns:
            torch.Tensor: The fused feature map.
                          Shape: (B, C, H, W)
        """
        if feature_map_a.shape != feature_map_b.shape:
            raise ValueError(f"Input feature maps must have the same shape. "
                             f"Got {feature_map_a.shape} and {feature_map_b.shape}")

        # 1. Create object-centric reference map C_o' [cite: 1]
        object_centric_ref = self._create_object_centric_reference(feature_map_a, feature_map_b)

        # 2. Sort channels of feature_map_a based on similarity to C_o' [cite: 1]
        f_a_sorted = self._sort_feature_map_by_similarity(feature_map_a, object_centric_ref)
        
        # 3. Sort channels of feature_map_b based on similarity to C_o' [cite: 1]
        f_b_sorted = self._sort_feature_map_by_similarity(feature_map_b, object_centric_ref)
        
        # 4. Fuse the sorted feature maps using element-wise maximum
        fused_feature_map = torch.maximum(f_a_sorted, f_b_sorted)
        
        return fused_feature_map

# Example Usage:
if __name__ == '__main__':
    # Example parameters
    batch_size = 2 # Reduced batch size for easier inspection if printing values
    num_channels = 3 # Reduced channels for easier inspection
    height = 4       # Reduced spatial dims
    width = 4

    # Create dummy input feature maps
    fm_a = torch.randn(batch_size, num_channels, height, width)
    fm_b = torch.rand(batch_size, num_channels, height, width) * 2 # Different distribution

    # Initialize the fusion module
    cosine_fusion_module = CosineSortedMaxFusionModule()

    # Perform fusion
    fused_features = cosine_fusion_module(fm_a, fm_b)

    # Print shapes to verify
    print(f"Shape of Feature Map A: {fm_a.shape}")
    print(f"Shape of Feature Map B: {fm_b.shape}")
    print(f"Shape of Fused Feature Map: {fused_features.shape}")

    # For detailed checking, you might want to inspect:
    # ref_map_example = cosine_fusion_module._create_object_centric_reference(fm_a, fm_b)
    # print(f"\nShape of Object-Centric Reference: {ref_map_example.shape}")
    # f_a_sorted_example = cosine_fusion_module._sort_feature_map_by_similarity(fm_a, ref_map_example)
    # print(f"Shape of Sorted Feature Map A: {f_a_sorted_example.shape}")

    # Verify values (optional, for a small example)
    # print("\nOriginal fm_a[0,:,0,0]:\n", fm_a[0,:,0,0])
    # sim_scores_a = F.cosine_similarity(fm_a[0].view(num_channels, -1), ref_map_example[0].view(1, -1), dim=1)
    # print("Sim scores for fm_a[0]:", sim_scores_a)
    # print("Sorted fm_a[0,:,0,0] (according to scores):\n", f_a_sorted_example[0,:,0,0])

    # print("\nFused_features[0,:,0,0]:\n", fused_features[0,:,0,0])