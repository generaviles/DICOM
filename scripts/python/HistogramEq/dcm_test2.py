#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  dcm_test.py
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
import timePython as tm
from matplotlib import pyplot as plt

image_data, image_header = load('IMG2')
times = []

tm.tic()
image_data = np.transpose(image_data)
times.append(tm.toc(save=True))

#~ image_data = [[1,2,2],[4,5,4],[7,8,8]]


print "Histograma..."
tm.tic()
h,x = fn.histograma_16(image_data)
times.append(tm.toc(save=True))
#~ print image_data
print x, len(x)
print h

print "Histograma acumulado..."
tm.tic()
hac = fn.histogramaAc(h)
times.append(tm.toc(save=True))
print hac

print "Ecualizando..."
tm.tic()
equ = fn.equalizar(image_data,hac,x)
times.append(tm.toc(save=True))
#~ print equ
#~ Inorm = fn.normalizar(equ)
#~ Inorm =  fn.inverso(Inorm)

txtH = open("histograma.txt","w")
for x in range(len(h)):
	txtH.write(str(h[x])+" ")
txtH.close()
	
txtHac = open("histogramaAc.txt","w")
for x in range(len(hac)):
	txtHac.write(str(hac[x])+" ")
txtHac.close()

M,N = fn.tamano(equ)
txtEcu = open("ecualizada.txt","w")
for i in range(M):
	for j in range(N):
		txtEcu.write(str(equ[i][j])+" ")
	txtEcu.write("\n")
txtEcu.close()

M,N = fn.tamano(image_data)
txtOrig = open("original.txt","w")
for i in range(M):
	for j in range(N):
		txtOrig.write(str(image_data[i][j])+" ")
	txtOrig.write("\n")
txtOrig.close()	

	
#~ fig = plt.figure(1)
#~ plt.subplot(141),plt.imshow(image_data,cmap='gray')
#~ plt.title('Imagen Original:')
#~ plt.subplot(142),plt.bar(x,h, color='blue')
#~ plt.title('Histograma:')
#~ plt.subplot(143),plt.bar(x,hac, color='blue')
#~ plt.title('Histograma acumulado:')
#~ plt.subplot(144),plt.imshow(Inorm,cmap='gray')
#~ plt.title('Imagen Original:')
#~ ##########################################################
#~ plt.show()


arc = open("time.txt","w")
for i in range(len(times)):
	arc.write(str(times[i])+"\n")
arc.close()
