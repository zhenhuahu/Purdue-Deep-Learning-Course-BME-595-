# perform 2d convolution
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import time


from conv import Conv2D

# convert PIL to tensor
pil2tensor = transforms.ToTensor()
tensor2pil = transforms.ToPILImage()

img_pil = Image.open('1280x720.jpg')
img = pil2tensor(img_pil) # convert JpegImageFile object to tensor

# task1
conv2d = Conv2D(in_channel=3, o_channel=1, kernel_size=3, stride=1, mode='known')
[numOperates, outImg] = conv2d.forward(img)
print(numOperates)
torchvision.utils.save_image(outImg, 'task1_1280x720.jpg', padding = 0, normalize = True)

if False:
	# task2
	conv2d = Conv2D(in_channel=3, o_channel=2, kernel_size=5, stride=1, mode='known')
	[numOperates, outImg] = conv2d.forward(img)
	print(numOperates)
	torchvision.utils.save_image(outImg[0], 'task2_1280x720_K4.jpg', padding = 0, normalize = True)
	torchvision.utils.save_image(outImg[1], 'task2_1280x720_K5.jpg', padding = 0, normalize = True)

if False:
	# task3
	conv2d = Conv2D(in_channel=3, o_channel=3, kernel_size=3, stride=2, mode='known')
	[numOperates, outImg] = conv2d.forward(img)
	print(numOperates)
	torchvision.utils.save_image(outImg[0], 'task3_1920x1080_K1.jpg', padding = 0, normalize = True)
	torchvision.utils.save_image(outImg[1], 'task3_1920x1080_K2.jpg', padding = 0, normalize = True)
	torchvision.utils.save_image(outImg[2], 'task3_1920x1080_K3.jpg', padding = 0, normalize = True)


# Part B
if False:
	timeB = torch.zeros([11, 1], dtype = torch.float64)
	for i in range(0,11):
		a = time.time()
		conv2d = Conv2D(in_channel=3, o_channel= 2**i, kernel_size=3, stride=1, mode='rand')
		[numOperates, outImg] = conv2d.forward(img)
		timeB[i] = time.time() - a
		print( str(i) + ': ' + str(timeB[i]) )

	plt.plot(timeB)
	plt.show()


# Part C
if False:
	numC = torch.zeros([5, 1], dtype = torch.int32)
	for i in range(0, 5):
		i
		a = time.time()
		conv2d = Conv2D(in_channel=3, o_channel= 2, kernel_size=(i+1)*2+1, stride=1, mode='rand')
		[numOperates, outImg] = conv2d.forward(img)
		print('time is ' + str(time.time() - a) )
		numC[i] = numOperates
		print(numOperates)

	plt.plot(numC)
	plt.show()






