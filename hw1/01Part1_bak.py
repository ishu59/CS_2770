import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.io import loadmat
from scipy.ndimage import convolve

car1 = cv2.imread('img/cardinal1.jpg')
car2 = cv2.imread('img/cardinal2.jpg')
leo1 = cv2.imread('img/leopard1.jpg')
leo2 = cv2.imread('img/leopard2.jpg')
pan1 = cv2.imread('img/panda1.jpg')
pan2 = cv2.imread('img/panda2.jpg')
# cv2.imshow('img',pan1)
car1 = cv2.resize(car1, (200, 200))
car2 = cv2.resize(car2, (200, 200))
leo1 = cv2.resize(leo1, (200, 200))
leo2 = cv2.resize(leo2, (200, 200))
pan1 = cv2.resize(pan1, (200, 200))
pan2 = cv2.resize(pan2, (200, 200))

filepath = {'car1': 'img/cardinal1.jpg', 'car2': 'img/cardinal2.jpg', 'leo1': 'img/leopard1.jpg',
            'leo2': 'img/leopard2.jpg', 'pan1': 'img/panda1.jpg', 'pan2': 'img/panda2.jpg'}
# print(filepath)
procImg = dict()
for k, v in filepath.items():
    img = None
    img = cv2.imread(v, 0)
    img = cv2.resize(img, (200, 200))
    procImg[k] = img

for k, v in procImg.items():
    pass

car1_gray = cv2.cvtColor(car1, cv2.COLOR_BGR2GRAY)
pan1_gray = cv2.cvtColor(pan1, cv2.COLOR_BGR2GRAY)
leungMalikFilter = loadmat('filter/filters.mat')
dst = cv2.filter2D(pan1_gray, -1, leungMalikFilter["F"])

cv2.imshow('', dst)

# cv2.imshow('car gray',car1_gray)
# cv2.imshow("pan1",pan1)
# cv2.imshow('resized',cv2.resize(pan1,(100,100)))

print(car1.shape)
print(leo1.shape)
print(pan1.shape)

lmFilterBank =  loadmat('filter/filters.mat')
lmFilter = lmFilterBank["F"]
print(lmFilter.shape)
plt.rcParams['figure.figsize'] = [10, 5]
plt.subplot(2,4,1)
plt.imshow(lmFilter[:,:,1])
plt.subplot(2,4,3)
plt.imshow(convolve(procImg['cardinal1'],lmFilter[:,:,1]))
plt.subplot(2,4,4)
plt.imshow(convolve(procImg['cardinal2'],lmFilter[:,:,1]))
plt.subplot(2,4,5)
plt.imshow(convolve(procImg['leopard1'],lmFilter[:,:,1]))
plt.subplot(2,4,6)
plt.imshow(convolve(procImg['leopard2'],lmFilter[:,:,1]))
plt.subplot(2,4,7)
plt.imshow(convolve(procImg['panda1'],lmFilter[:,:,1]))
plt.subplot(2,4,8)
plt.imshow(convolve(procImg['panda2'],lmFilter[:,:,1]))
plt.savefig('img1.png')
