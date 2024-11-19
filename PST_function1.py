import torch
import torch.fft
import math
import numpy as np
import mahotas as mh  # Needed for morphological operations; ensure it's installed

# Define functions
def cart2pol(x, y):
    theta = torch.atan2(y, x)
    rho = torch.hypot(x, y)
    return theta, rho

def PST(I, LPF, Phase_strength, Warp_strength, Threshold_min, Threshold_max, Morph_flag):
    L = 0.5
    device = I.device  # Ensure computations are on the same device as the input image
    
    x = torch.linspace(-L, L, I.shape[0], device=device)
    y = torch.linspace(-L, L, I.shape[1], device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    THETA, RHO = cart2pol(X, Y)

    # Apply localization kernel to reduce noise
    Image_orig_f = torch.fft.fft2(torch.from_numpy(I).float())  
    expo = torch.fft.fftshift(torch.exp(-((RHO / math.sqrt((LPF**2) / math.log(2))) ** 2)))
    Image_orig_filtered = torch.real(torch.fft.ifft2(Image_orig_f * expo))

    # Construct the PST Kernel
    PST_Kernel_1 = RHO * Warp_strength * torch.atan(RHO * Warp_strength) - 0.5 * torch.log(1 + (RHO * Warp_strength) ** 2)
    PST_Kernel = (PST_Kernel_1 / PST_Kernel_1.max()) * Phase_strength

    # Apply the PST Kernel
    temp = torch.fft.fftshift(torch.exp(-1j * PST_Kernel)) * torch.fft.fft2(Image_orig_filtered)
    Image_orig_filtered_PST = torch.fft.ifft2(temp)

    # Calculate phase of the transformed image
    PHI_features = torch.angle(Image_orig_filtered_PST)

    if Morph_flag == 0:
        out = PHI_features
    else:
        # Find image sharp transitions by thresholding the phase
        features = torch.zeros_like(PHI_features, dtype=torch.uint8)
        features[PHI_features > Threshold_max] = 1  # Bi-threshold decision
        features[PHI_features < Threshold_min] = 1  # Capture positive and negative transitions
        features[I < (I.max() / 20)] = 0  # Suppress edges in dark areas

        # Apply binary morphological operations to clean the transformed image
        out = features.cpu().numpy()  # Convert to NumPy for mahotas operations
        out = mh.thin(out, 1)
        out = mh.bwperim(out, 4)
        out = mh.thin(out, 1)
        out = mh.erode(out, np.ones((1, 1)))
        out = torch.tensor(out, device=device, dtype=torch.uint8)  # Convert back to Torch

    return out, PST_Kernel
