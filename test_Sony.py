from __future__ import division
import os
import numpy as np
import rawpy
import glob
from PIL import Image
import torch

from unet import UNet_original, UNet_single_batchnorm, UNet_double_batchnorm


# Pack the raw image into 4 channels using the bayer pattern
def pack_raw(raw):
    im = np.maximum(raw - 512, 0) / (16383 - 512)  # subtract the black level
    im = np.expand_dims(im, axis=2)  # Add a channel dimension
    img_shape = im.shape  # Get the shape of the image
    H = img_shape[0]  # Get the height of the image
    W = img_shape[1]  # Get the width of the image

    # Channel concatenation for the bayer pattern
    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

# loss function using absolute difference between the output and ground truth
def loss_function(out_image, gt_image):
    loss = torch.mean(torch.abs(out_image - gt_image))
    return loss


def test_sony(unet, result_folder, DEBUG=True, device='cuda:0'):
    """Function to train the model"""

    # Required paths to the datasets
    input_dir = './dataset/Sony/short/' # Path to the short exposure images
    gt_dir = './dataset/Sony/long/' # Path to the long exposure images
    checkpoint_dir = './result_Sony/' # Path to the checkpoint directory
    # result_dir = './result_Sony/final/' # Path to the result directory
    result_dir = './results/' + result_folder + '/' # Path to the result directory
    ckpt = checkpoint_dir + 'model.ckpt' # Path to the model

    # get test IDs
    test_fns = glob.glob(gt_dir + '/1*.ARW')
    test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

    # Debug mode that only uses 5 images from the dataset
    if DEBUG:
        test_ids = test_ids[0:5]

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # check if GPU is available
    device = torch.device(device)

    unet.load_state_dict(torch.load(ckpt,map_location={'cuda:1':'cuda:0'}))
    model = unet.to(device)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    for test_id in test_ids: # Loop through all test_ids
        in_files = glob.glob(input_dir + '%05d_00*.ARW' % test_id) # Get input image files (first image in each sequence) based on the test_id

        for k in range(len(in_files)): # Iterate through all input files
            in_path = in_files[k]
            _, in_fn = os.path.split(in_path)
            print(in_fn)
            gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)  # Get the ground truth files for the current test_id

            _, gt_fn = os.path.split(gt_files[0])
            in_exposure = float(in_fn[9:-5]) # Extract exposure values from input
            gt_exposure = float(gt_fn[9:-5]) # Extract exposure values from ground truth
            ratio = min(gt_exposure / in_exposure, 300) # Calculate the exposure ratio and limit it to 300

            raw = rawpy.imread(in_path) # Read the raw input image
            im = raw.raw_image_visible.astype(np.float32) # Convert it to a visible float32 image
            input_full = np.expand_dims(pack_raw(im), axis=0) * ratio # Multiply image with exposure ratio

            im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

            gt_raw = rawpy.imread(gt_files[0])  # Read the raw ground truth image and post-process it
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

            input_full = np.minimum(input_full, 1.0) # Clip the input image to the range [0, 1]

            in_img = torch.from_numpy(input_full).permute(0, 3, 1, 2).to(device) # Convert the input image to a PyTorch tensor
            out_img = unet(in_img) # Perform the image enhancement using the UNet model

            # Convert to numpy array and clip between 0 and 1
            output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
            output = np.minimum(np.maximum(output, 0), 1)

            # Remove the batch dimension from the images
            output = output[0, :, :, :]
            gt_full = gt_full[0, :, :, :]
            scale_full = scale_full[0, :, :, :]
            scale_full = scale_full * np.mean(gt_full) / np.mean(scale_full)

            # Image.fromarray((scale_full * 255).astype('uint8')).save(result_dir + '%5d_00_%d_ori.png' % (test_id, ratio))
            Image.fromarray((output * 255).astype('uint8')).save(result_dir + '%5d_00_%d_out.png' % (test_id, ratio))
            # Image.fromarray((scale_full * 255).astype('uint8')).save(result_dir + '%5d_00_%d_scale.png' % (test_id, ratio))
            Image.fromarray((gt_full * 255).astype('uint8')).save(result_dir + '%5d_00_%d_gt.png' % (test_id, ratio))

    print("\nFinished testing!\n")

if __name__ == '__main__':
    unet = UNet_original() # Create the model
    DEBUG = True # Set the debug flag
    device = "cuda:0" if torch.cuda.is_available() else "cpu" # Set the device to use

    test_sony(unet, DEBUG=DEBUG, device=device) # Train the model
