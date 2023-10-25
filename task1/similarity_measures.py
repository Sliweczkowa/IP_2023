from PIL import Image
import numpy as np
import argparse
import sys


# Squared differences sum
def sqd_dif_sum(img1, img2):
    sqd_dif = np.square(img1 - img2)
    sum = np.sum(sqd_dif)
    return sum


# E1 | Mean square error
def mse(org_img, noise_img, fil_img):
    height = len(org_img[0])
    width = len(org_img)

    #original and filtered
    sum_org_fil = sqd_dif_sum(org_img, fil_img)
    err_org_fil = sum_org_fil / (width*height)

    #original and noise
    sum_org_noise = sqd_dif_sum(org_img, noise_img)
    err_org_noise = sum_org_noise / (width*height)

    return err_org_noise, err_org_fil


# E2 | Peak mean square error
def pmse(org_img, noise_img, fil_img):
    max_org_img = np.max(org_img)

    mse_res = mse(org_img, noise_img, fil_img)

    #original and filtered
    err_org_fil = mse_res[1] / np.square(max_org_img)

    #original anf noise
    err_org_noise = mse_res[0] / np.square(max_org_img)

    return err_org_fil, err_org_noise


# E3 | Signal to noise ratio [dB]
def snr(org_img, noise_img, fil_img):

    # squared sum
    sqd_sum = np.sum(np.square(org_img))

    # original and filtered
    fil_x = 10 * np.log10(abs(sqd_sum / sqd_dif_sum(org_img, fil_img)))

    # original and noise
    noise_x = 10 * np.log10(abs(sqd_sum / sqd_dif_sum(org_img, noise_img)))

    return fil_x, noise_x


# E4 | Peak signal to noise ratio [dB]
def psnr(org_img, noise_img, fil_img):

    # squared max sum
    sqd_max_sum = np.sum(np.square(np.max(org_img)))

    # original and filtered
    fil_x = 10 * np.log10(abs(sqd_max_sum / sqd_dif_sum(org_img, fil_img)))

    # original and noise
    noise_x = 10 * np.log10(abs(sqd_max_sum / sqd_dif_sum(org_img, noise_img)))

    return fil_x, noise_x


# E5 | Maximum difference
def md(org_img, noise_img, fil_img):

    #original and filtered
    err_org_fil = np.max(np.absolute(org_img - fil_img))

    #original and noise
    err_org_noise = np.max(np.absolute(org_img - noise_img))

    return err_org_fil, err_org_noise
