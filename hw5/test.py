from img2num import img2num
from nn_img2num import NnImg2Num
from img2obj import img2obj
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt



import time

def main():
	pil2tensor = transforms.ToTensor()
	tensor2pil = transforms.ToPILImage()
	resizeTensor = transforms.Resize([32, 32]) # resize the image to desired size

	start_time = time.time()
	#myNN = img2num()
	#myNN = NnImg2Num()

	myNN = img2obj()
	#myNN.train()
	myNN.load_state_dict(torch.load('model_params'))

	''' view code 
	img_pil = Image.open('visual_truck.jpg')
	img32_pil = resizeTensor(img_pil)	

	# plt.figure
	# plt.imshow(img32_pil)
	# plt.show()

	img = pil2tensor(img32_pil) # convert JpegImageFile object to tensor
	#print('img32.size is {}'.format(img.size()))
	myNN.view(img)
	'''

	# cam() code
	myNN.cam()

	print('time elapsed is {} seconds'.format(time.time() - start_time))

main()
