import fbp
from PIL import Image, ImageChops, ImageOps
import pydicom
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft, fftshift
import logging
import pickle
import json


if __name__ == '__main__':

    # generate output logging
    logging.basicConfig(filename='output.log', level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # --------------------------------------------------Load data----------------------------------------------------------
    dcm = pydicom.dcmread('C:\\Users\\Kathy\\Desktop\\DL\FBP\\1.2.826.0.1.3680043.5876\\1.dcm')
    img = Image.fromarray(dcm.pixel_array).convert('L')
    # img = np.asarray(img)
    # img = (img - np.min(img)) / np.ptp(img) * 255
    # img = img.astype(np.uint8)
    # img = Image.fromarray(img)

    logger.debug(f"Original_img: {img}")
    plt.imshow(img, cmap='gray')
    plt.title("original_image")
    plt.show()

    # -----------------------------------------------Forward Projection-----------------------------------------------------
    # Define rotation angle and the number of rotation
    numAngles = 720
    theta = np.linspace(0., 180., numAngles, endpoint=False)
    sinogram = fbp.getProj(img, theta)
    logger.debug(f"sinogram: {sinogram}")

    # -----------------------------------------------sinogram filteration---------------------------------------------------
    filter_sinogram=fbp.filterProj2(sinogram)
    logger.debug(f"filter_sinogram: {filter_sinogram}")

    # --------------------------------------------- Back Projection/ Recon -------------------------------------------------
    reconImg = fbp.backPorj(filter_sinogram, theta)
    logger.debug(f"reconImg: {reconImg}")

    # generate lineProfile
    fbp.lineProfile(200,img,reconImg)


    # Load logging
    with open('output.log', 'r') as f:
        content = f.readlines()
        print(content)






