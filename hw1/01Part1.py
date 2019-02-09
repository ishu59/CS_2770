import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.io import loadmat
from scipy.ndimage import convolve

filepath = {'cardinal1':'img/cardinal1.jpg','cardinal2':'img/cardinal2.jpg','leopard1':'img/leopard1.jpg',
            'leopard2':'img/leopard2.jpg','panda1':'img/panda1.jpg','panda2':'img/panda2.jpg'}

procImg = dict()
for k,v in filepath.items():
    img = None
    img = cv2.imread(v,0)
    img = cv2.resize(img,(200,200))
    procImg[k] = img

lmFilterBank =  loadmat('filter/filters.mat')
lmFilter = lmFilterBank["F"]
print(lmFilter.shape)

z = 1
for k, v in sorted(procImg.items()):
    plt.subplot(2, 3, z)
    plt.title(k)
    plt.imshow(v)
    z = z + 1

imgName = 'actualImg.png'
plt.savefig(imgName)

nFilter = lmFilter.shape[2]
for i in range(nFilter):
    z = 3
    plt.subplot(2, 4, 1)
    plt.imshow(lmFilter[:, :, i])
    for k, v in sorted(procImg.items()):
        img = None
        img = convolve(v, lmFilter[:, :, i])
        plt.subplot(2, 4, z)
        plt.title(k)
        plt.imshow(img)
        z = z + 1

    imgName = 'outputImg/resultWithFilter' + str(i) + '.png'
    plt.savefig(imgName)

