# perform 2d convolution
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#Conv2D(in_channel, o_channel, kernel_size, stride, mode)


class Conv2D(object):

	def __init__(self, in_channel, o_channel, kernel_size, stride, mode):
		self.in_channel = in_channel
		self.o_channel = o_channel
		self.kernel_size = kernel_size
		self.stride = stride
		self.mode = mode

	def forward(self, input_image):

		def out_image(K, imgR, imgG, imgB, kernel_size, stride):

			[r, c] = imgR.size()
			d = int(kernel_size / 2)

			outR = torch.zeros([r-kernel_size+1, c-kernel_size+1], dtype = torch.float64)
			outG = torch.zeros([r-kernel_size+1, c-kernel_size+1], dtype = torch.float64)
			outB = torch.zeros([r-kernel_size+1, c-kernel_size+1], dtype = torch.float64)

			numCalulates = 0

			for i in range(d, r-d, stride):
				for j in range(d, c-d, stride):
					blkR = imgR[i-d: i+d+1, j-d: j+d+1]
					outR[i-d, j-d] = (blkR * K).sum()

					blkG = imgG[i-d: i+d+1, j-d: j+d+1]
					outG[i-d, j-d] = (blkG * K).sum()

					blkB = imgB[i-d: i+d+1, j-d: j+d+1]
					outB[i-d, j-d] = (blkB * K).sum()

					numCalulates = numCalulates + (K.size()[0] * K.size()[1]*2 -1)*3

			outImg = (outR + outG + outB) / 3.0

			return numCalulates, outImg

		print(input_image.size())
		imgR = input_image[0]
		imgG = input_image[1]
		imgB = input_image[2]

		[r, c] = imgR.size()

		# output image
		outImg = torch.zeros(self.o_channel, r-self.kernel_size+1, c-self.kernel_size+1)  # output grayscale image

		# Part B
		if(self.kernel_size == 3 and self.mode == 'rand' and self.stride == 1):
			numOperates = 0

			for i in range(0, self.o_channel):
				Kb = torch.rand(3,3) *2 -1  # in range [-1, 1]
				[numCalulates, outImg1] = out_image(Kb, imgR, imgG, imgB, self.kernel_size, self.stride)
				numOperates = numOperates + numCalulates
				outImg[i] = outImg1
			return numOperates, outImg

		
		# Part C
		#if False:
		elif(self.o_channel == 2 and self.mode == 'rand' and self.stride == 1):
			Kc1 = torch.rand(self.kernel_size, self.kernel_size) *2 -1 
			Kc2 = torch.rand(self.kernel_size, self.kernel_size) *2 -1 
			numOperates = 0

			[numCalulates, outImg1] = out_image(Kc1, imgR, imgG, imgB, self.kernel_size, self.stride)
			outImg[0] = outImg1
			numOperates = numOperates + numCalulates

			[numCalulates, outImg1] = out_image(Kc2, imgR, imgG, imgB, self.kernel_size, self.stride)
			outImg[1] = outImg1
			numOperates = numOperates + numCalulates

			return numOperates, outImg


		# task 1
		elif(self.o_channel == 1 and self.kernel_size == 3 and self.mode == 'known' and self.stride == 1):
			numOperates = 0
			K1 = torch.tensor([[-1., -1., -1.], [0., 0., 0.], [1., 1., 1.]])
			K1 = K1.transpose(0, 1)

			[numOperates, outImg1] = out_image(K1, imgR, imgG, imgB, self.kernel_size, self.stride)
			outImg = outImg1
			return numOperates, outImg

		# task 2
		#if False:
		elif(self.o_channel == 2 and self.kernel_size == 5 and self.mode == 'known' and self.stride == 1):
			K4 = torch.tensor([[-1., -1., -1., -1., -1.], [-1., -1., -1., -1., -1.], [0., 0., 0., 0., 0.], [-1., -1., -1., -1., -1.], [-1., -1., -1., -1., -1.]])
			K4 = K4.transpose(0, 1)

			K5 = torch.tensor([[-1., -1., 0., 1., 1.], [-1., -1., 0., 1., 1.], [-1., -1., 0., 1., 1.], [-1., -1., 0., 1., 1.], [-1., -1., 0., 1., 1.]])
			K5 = K5.transpose(0, 1)

			numOperates = 0

			[numCalulates, outImg1] = out_image(K4, imgR, imgG, imgB, self.kernel_size, self.stride)
			outImg[0] = outImg1
			numOperates = numOperates + numCalulates

			[numCalulates, outImg1] = out_image(K5, imgR, imgG, imgB, self.kernel_size, self.stride)
			outImg[1] = outImg1
			numOperates = numOperates + numCalulates

			return numOperates, outImg

		# task 3
		#if False:
		elif(self.o_channel == 3 and self.kernel_size == 3 and self.mode == 'known' and self.stride == 2):
			K1 = torch.tensor([[-1., -1., -1.], [0., 0., 0.], [1., 1., 1.]])
			K1 = K1.transpose(0, 1)

			K2 = torch.tensor([[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]])
			K2 = K1.transpose(0, 1)

			K3 = torch.tensor([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
			K3 = K1.transpose(0, 1)

			numOperates = 0

			[numCalulates, outImg1] = out_image(K1, imgR, imgG, imgB, self.kernel_size, self.stride)
			outImg[0] = outImg1
			numOperates = numOperates + numCalulates

			[numCalulates, outImg1] = out_image(K2, imgR, imgG, imgB, self.kernel_size, self.stride)
			outImg[1] = outImg1
			numOperates = numOperates + numCalulates

			[numCalulates, outImg1] = out_image(K3, imgR, imgG, imgB, self.kernel_size, self.stride)
			outImg[2] = outImg1
			numOperates = numOperates + numCalulates

			return numOperates, outImg


			

			
