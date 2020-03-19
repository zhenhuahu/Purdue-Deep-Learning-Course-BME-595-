# implement AlexNet, last layer: 4096-> 200
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.datasets.folder as folder
import matplotlib.pyplot as plt
import time
import pickle  # save variables to a file
import numpy as numpy
import cv2
# show image and add caption to image
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import argparse

# receive input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', default='tiny-imagenet-200')
parser.add_argument('-s', '--save', default='self_model')
args = parser.parse_args()


class AlexNet(nn.Module):
	def __init__(self):
		super(AlexNet, self).__init__()

		self.class_no = 200
		
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size=3, stride = 2),
			nn.Conv2d(64, 192, kernel_size=5, padding=2),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace = True),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace = True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			)
		self.classifier = nn.Sequential(
			nn.Dropout(p = 0.5),
			nn.Linear(256*6*6, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(p = 0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace = True),
			nn.Linear(4096, self.class_no),
			)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), 256*6*6)
		x = self.classifier(x)
		x = F.softmax(x) # apply softmax activation function to the output
		return x


class train_self_model:
	def __init__(self):

		train_dir = os.path.join(args.data, 'train')
		validation_dir = os.path.join(args.data, 'val')

		# split validation set
		val_folder = os.path.join(validation_dir,'images')
		filename = os.path.join(validation_dir, 'val_annotations.txt')
		fp = open(filename, 'r')
		data = fp.readlines()

		# create 
		val_dict = {}
		for line in data:
			words = line.split('\t')
			val_dict[words[0]] = words[1]
		fp.close()

		# move images to their corresponding class folders
		for img, folder in val_dict.items():
			sub_folder = os.path.join(val_folder, folder)
			if not os.path.exists(sub_folder):
				os.makedirs(sub_folder)
			if os.path.exists(os.path.join(val_folder, img)):
				os.rename(os.path.join(val_folder, img), os.path.join(sub_folder, img))


		# tiny imageNet has 200 classes 
		# each class has 500 train images, 50 validation images, 50 test images
		self.train_batch_size = 200
		self.validation_batch_size = 200
		
		# normalize input
		trans = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), 
					transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
		train_set = dset.ImageFolder(root = train_dir, transform = trans)
		self.train_loader = torch.utils.data.DataLoader(dataset = train_set, 
			batch_size = self.train_batch_size, shuffle = True)
		self.train_batch_no = len(self.train_loader.dataset) / self.train_batch_size

		validation_set = dset.ImageFolder(root = validation_dir, transform = trans)
		self.validation_loader = torch.utils.data.DataLoader(dataset = validation_set, 
			batch_size = self.validation_batch_size, shuffle = False)

		self.validation_batch_no = len(self.validation_loader.dataset) / self.validation_batch_size
		

		# create dict or label string for each of 200 classes
		self.class_names = train_set.classes

		# find tiny class labels from words.txt
		filename = os.path.join(args.data, 'words.txt')
		fp = open(filename, 'r')
		data = fp.readlines()

		# create a dict with numerical class names as key and corresponding string as values
		whole_class = {}
		for line in data:
			words = line.split("\t")
			super_label = words[1].split(",")
			whole_class[words[0]] = super_label[0].rstrip()

		fp.close()

		# create a small dict with only 200 classes in tiny image-net
		self.tiny_class = {}
		for tiny_label in self.class_names:
			for k, v in whole_class.items(): # sreach whole dict
				if tiny_label == k:
					self.tiny_class[k] = v
					continue

		self.model = AlexNet()
		pretrained_alexnet = torchvision.models.alexnet(pretrained = True)

		# copy wieghts from pretrained model except for last layer
		for i, j in zip(self.model.modules(), pretrained_alexnet.modules()):
			if not list(i.children()):
				if len(i.state_dict()) > 0:
					if i.weight.size() == j.weight.size():
						i.weight.data = j.weight.data
						i.bias.data = j.bias.data

		# freeze weights except for the last layer
		for param in self.model.parameters():
			param.requires_grad = False

		for param in self.model.classifier[6].parameters():
			param.requires_grad = True

		# hyper parameters
		self.max_epoch = 1
		self.learning_rate = 1e-3
		self.loss_function = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(self.model.classifier[6].parameters(), lr = self.learning_rate)


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

			for batch_idx, (data, target) in enumerate(self.train_loader):
				print(batch_idx)
	
				data.requires_grad = True

				self.optimizer.zero_grad()
				outputs = self.model.forward(data)
				train_batch_loss = self.loss_function(outputs, target)
				train_loss = train_loss + train_batch_loss
				
				train_batch_loss.backward()
				self.optimizer.step()

			average_train_loss = train_loss / self.train_batch_no
			print('train average loss is {:.5f}'.format(average_train_loss))
			train_loss_all.append(average_train_loss)

			torch.save(self.state_dict(), save_dir)

			# --------------- validation process
			validation_loss = 0.
			num_validation_correct = 0

			for data, target in self.validation_loader:
				data.requires_grad = False
				#data = data.view(-1, 3, 32, 32)
				output = self.model.forward(data)

				validation_batch_loss = self.loss_function(output, target)
				validation_loss = validation_loss + validation_batch_loss.data

				value, class_idx = torch.max(output.data, 1)

				for i in range(self.validation_batch_size):
					if class_idx[i] == target[i]:
						num_validation_correct = num_validation_correct + 1

				pred = output.data.max(1, keepdim = True)
				num_validation_correct += pred.eq(target.data.view_as(pred)).cpu().sum()

			average_validation_loss = validation_loss / self.validation_batch_no
			accuracy = num_validation_correct / len(self.validation_loader.dataset)
			print('validation average_loss is {}, accuracy is {}'.format(average_validation_loss, accuracy))
			validation_loss_all.append(average_validation_loss)
			accuracies.append(accuracy)

			save_dir = args.save

			torch.save(self.state_dict(), save_dir)
			
			# one epoch time
			current_epoch_time = time.time() - start_time
			epoch_time.append(current_epoch_time)
			print('compute time: {:.5f} seconds'.format(current_epoch_time))

		
		# plot average_loss vs. epoch
		epoch_indices = range(1, self.max_epoch + 1)

		title = 'hw6: CNN Loss Changes in Different Epochs'
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
		plt.savefig('alexnet_loss.png')


if __name__ == '__main__':
	a = train_self_model()
	a.train()



