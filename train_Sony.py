from __future__ import division
import os, time
import numpy as np
import rawpy
import glob
from PIL import Image
import torch
import shutil

from unet import UNet_original, UNet_single_batchnorm, UNet_double_batchnorm


# Pack the raw image into 4 channels using the bayer pattern
def pack_raw(raw):
    im = raw.raw_image_visible.astype(np.float32) # Change data to float32
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2) # Add a channel dimension
    img_shape = im.shape # Get the shape of the image
    H = img_shape[0] # Get the height of the image
    W = img_shape[1] # Get the width of the image

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


def train_sony(unet, n_epochs=40, DEBUG=True, TRAIN_FROM_SCRATCH=False, device='cuda:0'):
    """Function to train the model"""

    # Required paths to the datasets
    input_dir = './dataset/Sony/short/' # Path to the short exposure images
    gt_dir = './dataset/Sony/long/' # Path to the long exposure images
    checkpoint_dir = './result_Sony/' # Path to the checkpoint directory
    result_dir = './result_Sony/' # Path to the result directory

    # get train IDs
    train_fns = glob.glob(gt_dir + '0*.ARW')
    train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

    ps = 512  # patch size for training
    save_freq = 500 

    # If debug is set to true, then set the save frequency to 2 and train on 5 images
    if DEBUG:
        save_freq = 2
        train_ids = train_ids[0:5]
    
    # Delete the result directory if boolean to rerun the training is set to true
    if TRAIN_FROM_SCRATCH:
        if os.path.isdir(result_dir):
            shutil.rmtree(result_dir)

    # Raw data takes long time to load. Keep them in memory after loaded.
    gt_images = [None] * 6000
    input_images = {}
    input_images['300'] = [None] * len(train_ids)
    input_images['250'] = [None] * len(train_ids)
    input_images['100'] = [None] * len(train_ids)

    # Array to store the loss values
    g_loss = np.zeros((5000, 1)) 

    allfolders = glob.glob(result_dir + '*0') # Get all the folders in the result directory
    lastepoch = 0 # Initialize the last epoch to 0

    # Get the last epoch number
    for folder in allfolders:
        lastepoch = np.maximum(lastepoch, int(folder[-4:]))
        
    device = torch.device(device) # check if GPU is available
    unet.to(device) # assign the model to the GPU or CPU
    unet.train() # Set the model to training mode

    learning_rate = 1e-4 # Set the learning rate

    G_opt = torch.optim.Adam(unet.parameters(), lr=learning_rate) # Set the optimizer

    # Training loop
    for epoch in range(lastepoch, n_epochs):
        # Check if the folder exists and skip the epoch if it does
        if os.path.isdir(result_dir + '%04d' % epoch):
            continue
        cnt = 0 # Initialize the counter to 0

        # Set the learning rate to 1e-5 after 2000 epochs
        if epoch > 2000:
            learning_rate = 1e-5

        # Loop through the training images
        for ind in np.random.permutation(len(train_ids)):
            train_id = train_ids[ind] # Get the training id
            in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id) # Get the input images
            in_path = in_files[np.random.randint(0, len(in_files))] # Get a random input image
            in_fn = os.path.basename(in_path) # Get the file name of the image

            gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id) # Get the ground truth images
            gt_path = gt_files[0] # Get the first ground truth image    
            gt_fn = os.path.basename(gt_path) # Get the file name of the ground truth image
            in_exposure = float(in_fn[9:-5]) # Get the exposure time for the input image
            gt_exposure = float(gt_fn[9:-5]) # Get the exposure time for the ground truth image
            ratio = min(gt_exposure / in_exposure, 300) # Get the ratio between the exposure times

            st = time.time() # Get the current time
            cnt += 1 # Increment the counter

            # Check if the image is loaded and load it if it is not
            if input_images[str(ratio)[0:3]][ind] is None:
                raw = rawpy.imread(in_path) # Read the raw image
                input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio # Pack the raw image and store it in the input images array depending on the ratio

                gt_raw = rawpy.imread(gt_path) # Read the ground truth image
                im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16) # Postprocess the ground truth image
                gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0) # Store the ground truth image in the ground thruth images array

            # Crop the image to the required size
            H = input_images[str(ratio)[0:3]][ind].shape[1] # Get the height of the image
            W = input_images[str(ratio)[0:3]][ind].shape[2] # Get the width of the image

            # Get a random start location on the image
            xx = np.random.randint(0, W - ps)
            yy = np.random.randint(0, H - ps)

            # Get the data used for training
            input_patch = input_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]
            gt_patch = gt_images[ind][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

            # Data augmentation
            if np.random.randint(2, size=1)[0] == 1:  # random flip around axis 1
                input_patch = np.flip(input_patch, axis=1)
                gt_patch = np.flip(gt_patch, axis=1)
            if np.random.randint(2, size=1)[0] == 1: # random flip around axis 2
                input_patch = np.flip(input_patch, axis=2)
                gt_patch = np.flip(gt_patch, axis=2)
            if np.random.randint(2, size=1)[0] == 1:  # random transpose
                input_patch = np.transpose(input_patch, (0, 2, 1, 3))
                gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

            input_patch = np.minimum(input_patch, 1.0) # Input patch should be between 0 and 1
            input_img = torch.from_numpy(input_patch) # Convert the input patch to a tensor
            input_img = input_img.permute(0, 3, 1, 2) # Permute the tensor
            input_img = input_img.to(device) # Assign the tensor to the GPU or CPU

            gt_patch = np.minimum(gt_patch, 1.0) # Ground truth patch should be between 0 and 1
            gt_img = torch.from_numpy(gt_patch) # Convert the ground truth patch to a tensor
            gt_img = gt_img.permute(0, 3, 1, 2) # Permute the tensor
            gt_img = gt_img.to(device) # Assign the tensor to the GPU or CPU

            # Run the model
            G_opt.zero_grad() # Set the gradient to 0
            out_image = unet(input_img) # Get the output image

            # Calculate the loss
            loss = loss_function(out_image, gt_img)
            loss.backward() # Calculate the gradients
            G_opt.step() # Update the weights
            g_loss[ind] = loss.item() # Store the loss

            print("%d %d Loss=%.3f Time=%.3f" % (epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st)) # Print the loss and time

            output = out_image.permute(0, 2, 3, 1).cpu().data.numpy()   # Get the output image and convert it to numpy
            output = np.minimum(np.maximum(output, 0), 1) # Output should be between 0 and 1

            # Save the results
            if epoch % save_freq == 0:
                # Create the directory if it does not exist
                if not os.path.isdir(result_dir + '%04d' % epoch):
                    os.makedirs(result_dir + '%04d' % epoch)

                # Save the image
                temp = np.concatenate((gt_patch[0, :, :, :], output[0, :, :, :]), axis=1) # Concatenate the ground truth and output images
                Image.fromarray((temp * 255).astype(np.uint8)).save(result_dir + '%04d/%05d_00_train_%d.jpg' % (epoch, train_id, ratio)) # Save the image

        # Save the model
        torch.save(unet.state_dict(), checkpoint_dir + 'model.ckpt')

    print("\nFinished training!\n")

if __name__ == '__main__':
    unet = UNet_double_batchnorm() # Create the model
    n_epochs = 40 # Set the number of epochs
    DEBUG = True # Set the debug flag
    TRAIN_FROM_SCRATCH = True # Set the flag to train from scratch
    device = "cuda:0" if torch.cuda.is_available() else "cpu" # Set the device to GPU or CPU

    train_sony(unet, n_epochs=n_epochs, DEBUG=DEBUG, TRAIN_FROM_SCRATCH=TRAIN_FROM_SCRATCH, device=device) # Train the model
