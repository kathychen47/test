import pydicom
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft, fftshift

#if __name__ == '__main__':

dcm = pydicom.dcmread('C:\\Users\\Kathy\\Desktop\\DL\FBP\\1.2.826.0.1.3680043.5876\\1.dcm')
img = Image.fromarray(dcm.pixel_array)

plt.imshow(img, cmap='gray')
plt.title("original_image")
plt.show()

#define rotation angle and the number of rotation
numAngles = 180
theta = np.linspace(0., 180., numAngles, endpoint=False)

# ==================================forward projection==================================================================
# define empty sinogram matrix
print("generate_sinogram")
sinogram = np.zeros((img.size[0], numAngles))

# Iteratively calculate the integral value of each projection
for n in range(numAngles):
    # rotate the image
    rotImgObj = img.rotate(90-theta[n], resample=Image.BICUBIC)
    # calculate the integral value of porjection
    sinogram[:,n] = np.sum(rotImgObj, axis=0)


# =====================================filter the sinogram==============================================================
print("filter the sinogram")
projLen,numAngles=sinogram.shape
step = 2*np.pi/projLen
w = np.arange(-np.pi, np.pi, step)
ramp_filter = np.abs(w)
ramp_filter = fftshift(ramp_filter) # shifts the zero-frequency component of the discrete Fourier transform (DFT) to the center of the array

filter_sinogram = np.zeros_like(sinogram)
for i in range(numAngles):
    projfft = fft(sinogram[:, i]) #one dimentional Fourier transform
    filter_projfft = projfft*ramp_filter # *ramp_filter
    filter_proj = np.real(ifft(filter_projfft)) # inverse Fourier transform
    filter_sinogram[:, i] = filter_proj
plt.imshow(filter_sinogram, cmap='gray')
plt.title("filter_sinogram")
plt.show()

# =========================================== backprojection/recon=======================================================
print("backprojection/recon")

#define a empty reconstruction image
N = filter_sinogram.shape[0]
reconImg = np.zeros((N, N)) # the length of image is equal to the length of detector

# Create a grid coordinate system with the center point at (0,0) in which the origin of the image coordinate system,
# which is located at the upper left corner, is shifted to the center point.
X = np.arange(N)-N/2  # [-N/2,...0..+N/2]
Y = X.copy()
x, y = np.meshgrid(X, Y)  # generate grid coordinate

# Convert degrees to radians
theta = theta * np.pi/180

numAngles = len(theta)
for n in range(numAngles):

    t = x*np.sin(theta[n])+y*np.cos(theta[n]) #x-y -> t-s
    # s = -x*np.sin(theta[n])+y*np.cos(theta[n])
    tCor = np.round(t+N/2) #Shift the coordinate axis origin to match the image origin. round() may result in floating-point numbers, needs to be rounded to an integer.
    tCor = tCor.astype("int")
    tIndex, sIndex = np.where((tCor >= 0) & (tCor <= N-1)) #tIndex: row index; sIndex: column Index
    sino_angle_n = filter_sinogram[:,n]
    reconImg[tIndex,sIndex] += sino_angle_n[tCor[tIndex, sIndex]]
    # plt.imshow(reconImg, cmap='gray')
    # plt.title('Image {}'.format(n))
    # plt.show()
reconImg = Image.fromarray((reconImg - np.min(reconImg)) / np.ptp(reconImg) * 255) #normilization
fig=plt.imshow(reconImg)
plt.title('reconImg')
plt.show()

print("done!")






