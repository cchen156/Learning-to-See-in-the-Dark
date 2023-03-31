from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
import numpy as np


# Load the two PNG files
img1 = Image.open("C:\\Users\\jeroe\\Downloads\\Learning-to-See-in-the-Dark-master\\Learning-to-See-in-the-Dark-master\\result_Sony\\10003_00_100_gt.png")
img2 = Image.open("C:\\Users\\jeroe\\Downloads\\Learning-to-See-in-the-Dark-master\\Learning-to-See-in-the-Dark-master\\result_Sony\\10003_00_100_out.png")

# Convert the PIL Image objects to numpy arrays
arr1 = np.array(img1)
arr2 = np.array(img2)

# Calculate the PSNR
psnr = peak_signal_noise_ratio(arr1, arr2)

# Print the result
print(f"PSNR: {psnr:.2f} dB")

img1 = img1.resize((256, 256))
img2 = img2.resize((256, 256))

# Convert the PIL Image objects to numpy arrays
arr1 = np.array(img1)
arr2 = np.array(img2)

# Calculate the SSIM
ssim_score, ssim_image = ssim(arr1, arr2, win_size=7, channel_axis=2, full=True)

# Print the SSIM score
print(f"SSIM: {ssim_score:.2f}")
