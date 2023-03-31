import matplotlib.pyplot as plt
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import csv
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim


from utils import calculate_psnr, calculate_ssim


def calculate_metrics(results_file, model_name):

    result_dir = './result_Sony/final/'

    total_psnr = 0
    total_ssim = 0
    count = 0

    for image in os.listdir(result_dir):
        if image.endswith('out.png'):
            # img_out = plt.imread(result_dir + image)
            # img_gt = plt.imread(result_dir + image.replace('out.png', 'gt.png'))
            # psnr_score = calculate_psnr(img_out, img_gt)
            # ssim_score = calculate_ssim(img_out, img_gt)

            img_out = np.array(Image.open(result_dir + image))
            img_gt = np.array(Image.open(result_dir + image.replace('out.png', 'gt.png')))
            psnr_score = psnr(img_gt, img_out)
            ssim_score = ssim(img_gt, img_out, win_size=7, channel_axis=2, full=True)[0]
            
            print(image, psnr_score, ssim_score)
            total_psnr += psnr_score
            total_ssim += ssim_score
            count += 1

    mean_psnr = total_psnr / count
    mean_ssim = total_ssim / count
    print('mean psnr: ', mean_psnr)
    print('mean ssim: ', mean_ssim)

    # Check if the csv file exists, if it does, append the results to the file, if not, create a new file and write the results to it
    if os.path.isfile(results_file):
        with open(results_file, 'a') as csvfile:
            fieldnames = ['Model', 'Mean psnr', 'Mean ssim']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'Model': model_name, 'Mean psnr': mean_psnr, 'Mean ssim': mean_ssim})
    else:
        with open(results_file, 'w') as csvfile:
            # Create a header for rhe csv file that tores the mean metrics for different models, in this case the model is using double batch normalization
            fieldnames = ['Model', 'Mean psnr', 'Mean ssim']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'Model': model_name, 'Mean psnr': mean_psnr, 'Mean ssim': mean_ssim})


if __name__ == '__main__':
    ###  CHANGE THESE PARAMETERS TO CHANGE THE NAME OF THE CSV FILE AND THE MODEL NAME IN THE CSV FILE  ###
    results_file = 'results_40.csv'
    model_name = 'Without Normalization'
    calculate_metrics(results_file=results_file, model_name=model_name)   