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

#~ image_data = [[0,1,2],[3,4,5],[6,7,8]]

tm.tic()
print "Diviendo imagen..."
T = np.mean(image_data)
I1,I2 = fn.sub_images(image_data,T)
times.append(tm.toc(save=True))
print "Valor medio",T
print "Histogramas..."
tm.tic()
h1,x1 = fn.histograma_16(I1)
h2,x2 = fn.histograma_16(I2)
times.append(tm.toc(save=True))
#~ print image_data
#~ print I1,I2
print x1,x2
print h1,h2

print "Histograma acumulado..."
tm.tic()
hac1 = fn.histogramaAc(h1)
hac2 = fn.histogramaAc(h2)
times.append(tm.toc(save=True))
print hac1,hac2

print "Ecualizando..."
tm.tic()
equ1 = fn.equalizar(I1,hac1,x1)
equ2 = fn.equalizar(I2,hac2,x2)
equ = equ1 + equ2
times.append(tm.toc(save=True))
#~ print equ
#~ Inorm = fn.normalizar(equ)
#~ Inorm =  fn.inverso(Inorm)

#~ txtH = open("histograma.txt","w")
#~ for x in range(len(h)):
	#~ txtH.write(str(h[x])+" ")
#~ txtH.close()
	
#~ txtHac = open("histogramaAc.txt","w")
#~ for x in range(len(hac)):
	#~ txtHac.write(str(hac[x])+" ")
#~ txtHac.close()

M,N = fn.tamano(equ1)
txtEcu1 = open("ecualizada1.txt","w")
for i in range(M):
	for j in range(N):
		txtEcu1.write(str(equ1[i][j])+" ")
	txtEcu1.write("\n")
txtEcu1.close()

M,N = fn.tamano(equ2)
txtEcu2 = open("ecualizada2.txt","w")
for i in range(M):
	for j in range(N):
		txtEcu2.write(str(equ2[i][j])+" ")
	txtEcu2.write("\n")
txtEcu2.close()

#~ M,N = fn.tamano(image_data)
#~ txtOrig = open("original.txt","w")
#~ for i in range(M):
	#~ for j in range(N):
		#~ txtOrig.write(str(image_data[i][j])+" ")
	#~ txtOrig.write("\n")
#~ txtOrig.close()	

	
#~ fig = plt.figure(1)
#~ plt.subplot(141),plt.imshow(image_data,cmap='gray')
#~ plt.title('Imagen Original:')
#~ plt.subplot(142),plt.imshow(equ1,cmap='gray')
#~ plt.title('Imagen Original:')
#~ plt.subplot(142),plt.bar(x,h, color='blue')
#~ plt.title('Histograma:')
#~ plt.subplot(143),plt.bar(x,hac, color='blue')
#~ plt.title('Histograma acumulado:')
#~ plt.subplot(143),plt.imshow(equ2,cmap='gray')
#~ plt.title('Imagen Original:')
#~ plt.subplot(144),plt.imshow(equ,cmap='gray')
#~ plt.title('Imagen Original:')
#~ ##########################################################
#~ plt.show()


arc = open("time.txt","w")
for i in range(len(times)):
	arc.write(str(times[i])+"\n")
arc.close()
