import torch
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import data, img_as_float
from skimage.util import random_noise
from Split_Bregman import split_bregman
import numpy as np
import time


dog= cv.imread('/Users/zkang/Desktop/VSCODE/WechatIMG151.png')
dog_original = img_as_float(dog)

# add noise to image
sigma = 0.155
noisy = random_noise(dog_original, var=sigma**2)

# the denoised image is returned as torch tensor
# use .numpy() to convert into numpy array for plotting
Split_Bregman_denoised = split_bregman(torch.FloatTensor(noisy), weight=0.5)[0].numpy()
iteration=split_bregman(torch.FloatTensor(noisy), weight=0.5)[1]
#Covert float image to unit8 image
noisy1=noisy*255
noisy2=noisy1.astype(np.uint8)
dst = cv.fastNlMeansDenoisingColored(noisy2,None,10,10,7,21)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5),
                       sharex=True, sharey=True)

ax[0].imshow(noisy)
ax[0].axis('off')
ax[0].set_title('Noisy')
ax[1].imshow(Split_Bregman_denoised)
ax[1].axis('off')
ax[1].set_title('Split Bregman')
ax[2].imshow(dst)
ax[2].axis('off')
ax[2].set_title('Denoised2')

plt.show()
print('iteration=',iteration)
