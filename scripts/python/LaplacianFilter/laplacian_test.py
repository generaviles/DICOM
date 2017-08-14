#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  laplacian_test.py
#  
#  Copyright 2017 Ricardo <ricardo@ricardo-Latitude-E6510>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  


from medpy.io import load
import cv2,csv
import numpy as np
import funciones as fn
#~ import timePython as tm
from matplotlib import pyplot as plt

#~ img = cv2.imread('spine.jpg')
#~ gray = fn.makeGray(img)

img1, image_header = load('IMG2')
#~ times = []

#~ tm.tic()
img = np.transpose(img1)
#~ times.append(tm.toc(save=True))
h,x = fn.histograma_16(img)
hac = fn.histogramaAc(h)
ecu = fn.equalizar(img,hac,x)

laplacian = cv2.Laplacian(ecu,cv2.CV_64F)
#~ print laplacian.shape
#~ gray = cv2.merge([gray,gray,gray])
#~ gray = cv2.merge([img,img,img])

#~ laplacian = img - laplacian

fig = plt.figure(1)
plt.subplot(1,3,1),plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(ecu,cmap = 'gray')
plt.title('Ecualizada'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(laplacian,cmap='gray')
plt.title('Bordes'), plt.xticks([]), plt.yticks([])
plt.show()
