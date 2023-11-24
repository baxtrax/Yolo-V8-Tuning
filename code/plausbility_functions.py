import torch

def vanilla_grad_attributions(gradients):
    """
    Vanilla gradient attribution method, takes absolute.
    Works with batch size > 1
    """
    attribution_map = torch.sum(torch.abs(gradients), dim=1)  # Sum along the color channels (dimension 1)
    max_values, _ = torch.max(attribution_map.view(attribution_map.size(0), -1), dim=1, keepdim=True)  # Find the maximum value for each image
    attribution_map /= max_values.view(-1, 1, 1)  # Normalize by dividing by the maximum value for each image