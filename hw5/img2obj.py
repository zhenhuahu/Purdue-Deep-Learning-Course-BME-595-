# hw5: create the LeNet-5 convolutional neural network
# LeNet: (input: 1@ o32*32) convolution (6@28*28) ->subsample(6@14*14)->convolution(16A@10*10)
#        ->subsample(16@5*5)->full connected(120)->full connected(84)->gaussian connection (10 outputs)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from torch.autograd import Variable
import pickle  # save variables to a file
import numpy as numpy
import cv2
# show image and add caption to image
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

class img2obj(nn.Module):
	def __init__(self):
		super(img2obj, self).__init__()
		# super() lets you avoid referring to the base class explicitly

		# image dimension: 32*32, no need to pad. pixel-wise convolution
		# image is (R,G,B)
		self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5, stride = 1, padding = 0)
		self.layer1 = nn.Sequential(self.conv1, nn.BatchNorm2d(6))

		self.conv2 = nn.Conv2d(6, 16, 5)
		self.layer2 = nn.Sequential(self.conv2, nn.BatchNorm2d(16))

		# affine operation
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 100)	
		self.learning_rate = 0.0001

		self.max_epoch = 50 # number of total training/validating
		self.class_no = 100

		# Set loss function as Mean Square Error
		#self.loss_function = nn.MSELoss()
		self.loss_function = nn.CrossEntropyLoss()
		#self.optimizer = torch.optim.SGD(self.parameters(), lr = self.learning_rate)
		self.optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)

		# download data
		root = './data_cifar100'
		if not os.path.exists(root):
			os.mkdir(root)

		trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))])
		train_set = dset.CIFAR100(root = root, train = True, transform = trans, download = True)
		validation_set = dset.CIFAR100(root = root, train = False, transform = trans, download = True)

		self.train_batch_size = 50
		self.validation_batch_size = 1000

		self.train_loader = torch.utils.data.DataLoader(dataset=train_set, 
			batch_size = self.train_batch_size, shuffle = True)
		self.validation_loader = torch.utils.data.DataLoader(dataset = validation_set,
			batch_size = self.validation_batch_size, shuffle = True)

		self.train_batch_no = len(self.train_loader.dataset) / self.train_batch_size
		self.validation_batch_no = len(self.validation_loader.dataset) / self.validation_batch_size

		# class label
		self.classes = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 
        				'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 
        				'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 
        				'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 
        				'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 
        				'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 
        				'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 
        				'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 
        				'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 
        				'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        				)


	# forward function
	def forward(self, x):
		# change multi-dimensional features into 1d
		def num_flat_features(x):
			size = x.size()[1:] # all dimensions except the batch dimension
			num_features = 1
			for s in size:
				num_features *= s
			return num_features

		# Max-pooling over a 2*2 window
		x = F.max_pool2d(F.relu(self.layer1(x)), (2,2)) # relu(a) = max(0, a)
		# it has convolution and subsampling

		# if the size is a square, need only specify a single number
		x = F.max_pool2d(F.relu(self.layer2(x)), 2)
		x = x.view(-1, num_flat_features(x)) # change all features of 1 image to 1 dimension

		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


	def train(self):
		# values of all epochs
		train_loss_all = list()
		validation_loss_all = list()
		epoch_time = list()	
		accuracies = list()

		# loop through epochs
		for i in range(self.max_epoch):
			print('\nEpoch ' + str(i+1) + ' start: ')
			start_time = time.time()

			# train process
			train_loss = 0.

			# start
			for batch_idx, (data, target) in enumerate(self.train_loader):
				# train_target = torch.zeros(self.train_batch_size, self.class_no) # onehot label
				# #print('target size is {}'.format(target.size()))
				# for i in range(self.train_batch_size):
				# 	train_target[i][target[i]] = 1
				# target = train_target

				#data, target = Variable(data), Variable(target, requires_grad = False)
				data.requires_grad = True

				output = self.forward(data)

				train_batch_loss = self.loss_function(output, target)

				train_loss = train_loss + train_batch_loss.data

				# zero all gradients being used
				self.optimizer.zero_grad()

				train_batch_loss.backward()

				# uodate model parameters
				self.optimizer.step()

			average_train_loss = train_loss / self.train_batch_no
			print('train average loss is {:.5f}'.format(average_train_loss))
			train_loss_all.append(average_train_loss)

			# validation process
			validation_loss = 0.
			num_validation_correct = 0

			for data, target in self.validation_loader:
				# validation_target = torch.zeros(self.validation_batch_size, self.class_no)
				# for i in range(self.validation_batch_size):
				# 	validation_target[i][target[i]] = 1

				#data, target = Variable(data), Variable(target)
				data.requires_grad = False
				#data = data.view(-1, 3, 32, 32)
				output = self.forward(data)

				validation_batch_loss = self.loss_function(output, target)
				validation_loss = validation_loss + validation_batch_loss.data

				value, class_idx = torch.max(output.data, 1)

				for i in range(self.validation_batch_size):
					if class_idx[i] == target[i]:
						num_validation_correct = num_validation_correct + 1

				# pred = output.data.max(1, keepdim = True)
				# num_validation_correct += pred.eq(target.data.view_as(pred)).cpu().sum()

			average_validation_loss = validation_loss / self.validation_batch_no
			accuracy = num_validation_correct / len(self.validation_loader.dataset)
			print('validation average_loss is {}, accuracy is {}'.format(average_validation_loss, accuracy))
			validation_loss_all.append(average_validation_loss)
			accuracies.append(accuracy)
			
			# one epoch time
			current_epoch_time = time.time() - start_time
			epoch_time.append(current_epoch_time)
			print('compute time: {:.5f} seconds'.format(current_epoch_time))

		torch.save(self.state_dict(), 'model_params')

		# plot average_loss vs. epoch
		epoch_indices = range(1, self.max_epoch + 1)

		title = 'img2obj: CNN Loss Changes in Different Epochs'
		plt.figure(1)
		plt.plot(epoch_indices, train_loss_all, color = 'red', linestyle = 'solid', linewidth = '2.0',
			marker = '*', markerfacecolor = 'red', markersize = '5', label = 'training loss')
		plt.plot(epoch_indices, validation_loss_all, color = 'green', linestyle = 'solid', linewidth = '2.0',
			marker = 'o', markerfacecolor = 'green', markersize = '5', label = 'validation loss')
		plt.xlabel('epochs', fontsize = 10)
		plt.ylabel('loss', fontsize = 10)
		plt.title(title, fontsize = 10)
		plt.legend(fontsize = 10)
		plt.grid(True)
		#plt.show()
		plt.savefig('cnn_cifar100_loss.png')

		# plot time vs epoch
		title = 'img2obj: CNN Time Changes in Different Epochs'
		plt.figure(2)
		plt.plot(epoch_indices, epoch_time, color = 'blue', linestyle = 'solid', linewidth = '2.0',
			marker = '*', markerfacecolor = 'red', markersize = '5')
		plt.xlabel('epochs', fontsize = 10)
		plt.ylabel('epoch time', fontsize = 10)
		plt.title(title, fontsize = 10)
		plt.legend(fontsize = 10)
		plt.grid(True)
		#plt.show()
		plt.savefig('cnn_cifar100_epoch_time.png')

		# save variables to a file
		with open('CNN_CIFAR100', 'wb') as f:
			pickle.dump([accuracies, epoch_time, train_loss_all, validation_loss_all], f)



	#view image and predict
	def view(self, img):  # img type: 3*32*32 ByteTensor
		img.requires_grad = False  #don't do back-propagation
		img2 = img
		img = img.reshape(1, 3, 32, 32)
		output = self.forward(img.float())

		value, class_idx = torch.max(output.data, 1)
		print('predicted result is ' + self.classes[class_idx])

		# show image, and add caption
		tensor2pil = transforms.ToPILImage()
		img_PIL = tensor2pil(img2)
		draw = ImageDraw.Draw(img_PIL)
		#font_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fonts')
		#font = ImageFont.truetype(os.path.join(font_path,"sans-serif.ttf"), 6)
		font = ImageFont.truetype("arial.ttf", 10)
		
		draw.text((0,0), self.classes[class_idx], (0,0,0), font = font )
		#img.save('out.jpg')
		plt.imshow(img_PIL)
		plt.show()

	
	# fetch image from the camera and predict it
	def cam(self):
		pil2tensor = transforms.ToTensor()
		tensor2pil = transforms.ToPILImage()
		resizeTensor = transforms.Resize([32, 32])
		font = cv2.FONT_HERSHEY_SIMPLEX

		trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))])


		video = cv2.VideoCapture(0)
		cv2.namedWindow('cam')
		print('press q to exit')

		while True:  # for streaming
			check, frame = video.read()
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			#print(frame)
			#cv2.imshow("Capturing", frame)
			height, width, channels = frame.shape
			frame_resize = cv2.resize(frame, (32, 32))
			img = trans(frame_resize)
			img = img.reshape(1, 3, 32, 32)
			img = img.float()
			img.requires_grad = False

			output = self.forward(img)
			value, class_idx = torch.max(output.data, 1)
			#print('predicted result is ' + self.classes[class_idx])

			frame_caption = cv2.putText(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), self.classes[class_idx], (80,80), font, 1, (0,0,0), 2, cv2.LINE_AA)
			cv2.imshow('cam', frame_caption)

			# frameTensor = torch.from_numpy(numpy.array(frame))
			# frameTensor2 = frameTensor.view(3, frameTensor.size()[0], frameTensor.size()[1] )
			#print('frameTensor size is {}'.format(frameTensor.size()))

			# for press any key to out(milliseconds)
			key = cv2.waitKey(1)

			if key == ord('q'):
				break


		# release camera
		video.release()






















