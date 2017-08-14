#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  funciones.py
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

import cv2
import numpy as np
from matplotlib import pyplot as plt


def makeGray(imagen):
	RED = imagen[:,:,0]
	GREEN = imagen[:,:,1]
	BLUE = imagen[:,:,2]
	gray =  np.uint8(0.29 * RED) + np.uint8(0.59 * GREEN) + np.uint8(0.11 * BLUE)
	return gray
#================================================================
def tamano(imagen):
	tam= np.shape(imagen)
	M = tam[0]
	N = tam[1]
	return M,N
#================================================================
def MuValues(hist,T):
	R1 = []#vector de valores en R1
	R2 = []#vector de valores en R2
	c1 = 0
	c2 = 0
	for l in range(256):
		if l > T:
			R1.append(hist[l]*l)
			c1=c1++hist[l]
		else:
			if l <= T:
				R2.append(hist[l]*l)
				c2=c2++hist[l]

	np.asarray(R1)
	np.asarray(R2)
	mu1 = np.sum(R1)/c1
	mu2 = np.sum(R2)/c2
	
	return mu1,mu2	
#=======================================================================
def histograma(imagen):
	vMin = np.min(imagen)
	vMax = np.max(imagen)
	#~ rango = vMax - vMin
	#~ print vMin,vMax
	histograma = np.zeros(vMax+1, dtype=int)
	for l in range(1,len(histograma)):
		histograma[l]=sum(sum(imagen == l))
		
	return histograma		
#~ #=======================================================================
#~ def histograma_16(imagen):
	#~ M,N = tamano(imagen)
	#~ vMin = np.min(imagen)
	#~ vMax = np.max(imagen)
	#~ print vMin,vMax 
	#~ x = np.arange(vMin,vMax+1)
	#~ histograma = np.zeros(len(x), dtype=int)
	#~ for i in range(M):
		#~ for j in range(N):
			#~ for l in range(len(histograma)):
				#~ a = np.argwhere(x==l+1)
				#~ print a
				#~ if imagen[i][j] == x[a]:
					#~ print np.where(x==l)
					#~ histograma[l] = histograma[l]+1
	#~ for l in range(len(histograma)):
		#~ histograma[l]=sum(sum(imagen == l))
	#~ return histograma,x		
#~ #=======================================================================
#=======================================================================
def histograma_16(imagen):
	M,N = tamano(imagen)
	vMin = np.min(imagen)
	vMax = np.max(imagen)
	print vMin,vMax 
	x = np.arange(vMin,vMax+1)
	print x
	histograma = np.zeros(len(x), dtype=int)
	for l in range(len(histograma)):
		#~ a = np.argwhere(x==l)
		#~ print a
		histograma[l]=sum(sum(imagen == x[l]))
	return histograma,x		
#=======================================================================
def suitableThreshold(h):
	T = 128
	mu1,mu2 =  MuValues(h,T)

	Tnew = round((mu1 + mu2)/2)
	Told = 0

	while(Tnew != Told):
		mu1,mu2 =  MuValues(h,Tnew)
		Told = Tnew
		Tnew = round((mu1 + mu2)/2)
		#print Told,Tnew
	T = int(Tnew)
	return T	
#=======================================================================
def edge_ShannonEn(bin1):
	M,N = tamano(bin1)
	m = 3
	n = 3
	a = (m-1)/2
	b = (n-1)/2
	s = (M,N)
	bordes = np.zeros((s), dtype=np.uint8)
	for y in range(b,(N-b)):
		for x in range(a,(M-a)):
			suma = 0
			for k in range(m):
				for j in range(n):
					if bin1[x][y] == bin1[x+k-a][y+j-b]:
						suma = suma + 1
			p = suma/float(9)
			H = -p * np.log(p)
			if H <	(-(1/float(9)) * np.log(1/float(9))):
				bordes[x][y] = 0
			else:
				bordes[x][y] = 255
	return bordes			
#========================================================================	
def histogramaAc(histograma):
	hac = np.zeros(len(histograma), dtype=int)
	suma = 0
	for i in range(len(histograma)):
		suma = suma + histograma[i]
		hac[i] = suma
	return hac
#=======================================================================
def equalizar(imagen,hac,x):
	n,m = tamano(imagen)
	P = n * m
	L = np.max(imagen)
	#~ print P,L
	s = (n,m)
	imgEqDec = np.zeros((s), dtype=float)
	#imgNorm = np.zeros((s), dtype=np.uint8)
	for i in range(n):
		for j in range(m):
			#~ indx = imagen[i][j]
			#~ for k  in range(len(x)):
				#~ if indx == x[k]:
					#~ indx = k
			#~ print hac[imagen[i][j]]
			#~ b = hac[imagen[i][j]] * (L / float(P))
			indx = np.argwhere(x==imagen[i][j])
			b = hac[indx] * (L / float(P))
			#~ print indx, b
			imgEqDec[i][j] = b	
	return imgEqDec
#========================================================================	
def vectorCreciente(valor):
	x = np.arange(valor)
	return x
#========================================================================
def sub_images(image,T):
	M,N = tamano(image)
	s = (M,N)
	IMG1 = np.zeros((s), dtype=np.int32)
	IMG2 = np.zeros((s), dtype=np.int32)
	for i in range(M):
		for j in range(N):
			if image[i][j] <= T:
				IMG1[i][j] = image[i][j]
			else:
				IMG2[i][j] = image[i][j]
	return IMG1,IMG2
				
#========================================================================
def normalizar(imagen):
	tam= np.shape(imagen)
	n = tam[0]
	m = tam[1]
	s = (n,m)
	mx = np.max(imagen)
	mn = np.min(imagen)
	norm = np.zeros((s), dtype=np.uint8)		
	for i in range(0,n):
		for j in range(0,m):
			valor = float(imagen[i][j])
			norm[i][j] = np.uint8(round(((valor-mn)/(mx-mn))*255))
			
	return norm
#========================================================================
def normalizar_01(imagen):
	tam= np.shape(imagen)
	n = tam[0]
	m = tam[1]
	s = (n,m)
	mx = np.max(imagen)
	#~ mn = np.min(imagen)
	norm = np.zeros((s), dtype=np.float)		
	for i in range(n):
		for j in range(m):
			norm[i][j] = float(imagen[i][j]) / float(mx)
			
	return norm		
#========================================================================	
def inverso(imagen):
	n,m = tamano(imagen)
	s = (n,m)
	inversa = np.zeros((s), dtype=np.uint8)
	for i in range(0,n):
		for j in range(0,m):
			inversa[i][j] = 255 - imagen[i][j]
	return inversa				
#========================================================================
def espejoDoble(imagen):
	n,m = tamano(imagen)
	s = (n,m)
	volteada = np.zeros((s), dtype=np.int32)
	volteada2 = np.zeros((s), dtype=np.int32)
	for i in range(n):
		for j in range(m):
			volteada[i][j] = imagen[i][m-j-1]
	for i in range(n):
		for j in range(m):
			volteada2[i][j] = volteada[n-i-1][j]			
	return volteada2
#===============================================================
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))
