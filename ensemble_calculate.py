import cv2
import os
import shutil
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def ensemble(path1=None, path2=None, outputpath=None):

    SR_list_1 = os.listdir(path1)

    if os.path.exists(outputpath) and os.path.isdir(outputpath):
        shutil.rmtree(outputpath)
    try:
        os.mkdir(outputpath)
    except OSError:
        print("Creation of the output directory '%s' failed " % outputpath)

    for idx in range(0, len(SR_list_1), 2):
        SR_left_1 = cv2.imread(path1 + SR_list_1[idx]).astype(np.float64)
        SR_left_2 = cv2.imread(path2 + SR_list_1[idx]).astype(np.float64)


        SR_right_1 = cv2.imread(path1 + SR_list_1[idx + 1]).astype(np.float64)
        SR_right_2 = cv2.imread(path2 + SR_list_1[idx + 1]).astype(np.float64)


        SR_left = np.array((SR_left_1 + SR_left_2) / 2., dtype='float64')
        SR_right = np.array((SR_right_1 + SR_right_2) / 2., dtype='float64')

        cv2.imwrite(outputpath + SR_list_1[idx], SR_left)
        cv2.imwrite(outputpath + SR_list_1[idx + 1], SR_right)


def calculate(outputpath=None, gtpath=None):

    SR_list = os.listdir(outputpath)

    PSNR_stereo = 0
    SSIM_stereo = 0

    for idx in range(0, len(SR_list), 2):
        SR_left = cv2.imread(outputpath + SR_list[idx])
        SR_right = cv2.imread(outputpath + SR_list[idx + 1])

        label_left = cv2.imread(gtpath + SR_list[idx])
        label_right = cv2.imread(gtpath + SR_list[idx + 1])

        PSNR_left = peak_signal_noise_ratio(SR_left, label_left)
        PSNR_right = peak_signal_noise_ratio(SR_right, label_right)
        PSNR_stereo += (PSNR_left + PSNR_right) / 2.

        SSIM_left = structural_similarity(SR_left, label_left, multichannel=True)
        SSIM_right = structural_similarity(SR_right, label_right, multichannel=True)
        SSIM_stereo += (SSIM_left + SSIM_right) / 2.

    print(PSNR_stereo / (len(SR_list) / 2))
    print(SSIM_stereo / (len(SR_list) / 2))

if __name__ == '__main__':

    print("Start ensemble...")
    ensemble(path1=r'',
             path2=r'',
             outputpath=r'')
    print("Finish ensemble...")

    print("Start calculate metric...")
    # calculate(outputpath=r'./results_final/ensemble/',
    #           gtpath=r'./data/test/Validation/HR/')
    # print("Finish calculate metric...")
