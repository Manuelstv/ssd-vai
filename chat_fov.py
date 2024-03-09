import torch
from numpy import deg2rad

def xcyc_to_deg(xc, yc):
    """
    Convert bounding box center coordinates to angular values in degrees.

    Parameters:
    - xc (float or Tensor): The x center coordinates.
    - yc (float or Tensor): The y center coordinates.

    Returns:
    - tuple: theta (longitude angle in degrees), phi (latitude angle in degrees)
    """
    w, h = 1920, 960
    theta = (xc / w - 0.5) * 360  # Full range conversion [-180, 180] degrees
    phi = (yc / h - 0.5) * 180  # Full range conversion [-90, 90] degrees
    return theta, phi

def find_foviou(Bg, Bd):
    """
    Calculate the Field of View (FoV) Intersection over Union (IoU) for all combinations of bounding boxes.

    Parameters:
    - Bg (torch.Tensor): Ground truth bounding boxes in format [xmin, ymin, xmax, ymax].
    - Bd (torch.Tensor): Detected bounding boxes in the same format.

    Returns:
    - torch.Tensor: FoV-IoU matrix, each element (i, j) is the IoU between i-th Bg and j-th Bd box.
    """
    # Convert boundary to center coordinates and sizes
    xc_g, yc_g = (Bg[:, 0] + Bg[:, 2]) / 2, (Bg[:, 1] + Bg[:, 3]) / 2
    xc_d, yc_d = (Bd[:, 0] + Bd[:, 2]) / 2, (Bd[:, 1] + Bd[:, 3]) / 2
    alpha_g, beta_g = Bg[:, 2] - Bg[:, 0], Bg[:, 3] - Bg[:, 1]
    alpha_d, beta_d = Bd[:, 2] - Bd[:, 0], Bd[:, 3] - Bd[:, 1]

    # Convert center coordinates to angles in radians
    theta_g, phi_g, alpha_g, beta_g = torch.deg2rad(torch.tensor((xc_g, yc_g, alpha_g, beta_g)))
    theta_d, phi_d, alpha_d, beta_d = torch.deg2rad(torch.tensor((xc_d, yc_d, alpha_d, beta_d)))

    # Calculate FoV area for Bg and Bd
    A_Bg = alpha_g * beta_g
    A_Bd = alpha_d * beta_d

    # Calculate FoV distance and intersection
    delta_fov = (theta_d - theta_g) * torch.cos((phi_g + phi_d) / 2)
    theta_I_min = torch.max(-alpha_g / 2, delta_fov - alpha_d / 2)
    theta_I_max = torch.min(alpha_g / 2, delta_fov + alpha_d / 2)
    phi_I_min = torch.max(-beta_g / 2, -beta_d / 2)
    phi_I_max = torch.min(beta_g / 2, beta_d / 2)

    # Area of FoV Intersection and Union
    A_I = (theta_I_max - theta_I_min).clamp(min=0) * (phi_I_max - phi_I_min).clamp(min=0)
    A_U = A_Bg + A_Bd - A_I

    # Calculate FoV-IoU
    FoV_IoU = A_I / A_U.clamp(min=1e-6)
    return FoV_IoU

# Test with provided coordinates, converting to the expected format for this version
b1 = torch.tensor([[30,  85,  30,  40]])  # xmin, ymin, xmax, ymax
b2 = torch.tensor([[60,  78,  40,  30]])   # Same format

# Convert sizes to center coordinates and degrees to radians before calling find_foviou
print(find_foviou(b1, b2))
