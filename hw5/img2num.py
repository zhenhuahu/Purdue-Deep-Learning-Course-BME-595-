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
#from torch.autograd import Variable
import pickle  # save variables to a file

class img2num(nn.Module):

	def __init__(self):
		super(img2num, self).__init__()
		# super() lets you avoid referring to the base class explicitly

		# image dimension: 32*32 -> 28*28. pixel-wise convolution
		self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5, stride = 1, padding = 2)
		# use padding to make the input 28*28 image into 32*32

		self.conv2 = nn.Conv2d(6, 16, 5)

		# affine operation
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)	

		self.learning_rate = 3.

		# Set loss function as Mean Square Error
		self.loss_function = nn.MSELoss()

		self.optimizer = torch.optim.SGD(self.parameters(), lr = self.learning_rate)


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
		x = F.max_pool2d(F.relu(self.conv1(x)), (2,2)) # relu(a) = max(0, a)
		# it has convolution and subsampling

		# if the size is a square, need only specify a single number
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = x.view(-1, num_flat_features(x)) # change all features of 1 image to 1 dimension

		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


	def train(self):
		# download data
		root = './data'
		if not os.path.exists(root):
			os.mkdir(root)

		trans = transforms.Compose([transforms.ToTensor()])
		train_set = dset.MNIST(root = root, train = True, transform = trans, download = True)
		validation_set = dset.MNIST(root = root, train = False, transform = trans, download = True)

		self.train_batch_size = 60
		self.validation_batch_size = 1000

		self.train_loader = torch.utils.data.DataLoader(dataset=train_set, 
			batch_size = self.train_batch_size, shuffle = True)
		self.validation_loader = torch.utils.data.DataLoader(dataset = validation_set,
			batch_size = self.validation_batch_size, shuffle = True)

		self.train_batch_no = len(self.train_loader.dataset) / self.train_batch_size
		self.validation_batch_no = len(self.validation_loader.dataset) / self.validation_batch_size

		self.max_epoch = 50 # number of total training/validating
		self.input_size = 28 * 28 # change 2D data to 1D, also # of input neurons
		self.class_no = 10 
		

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
			# self.nn_network.train() # train mode

			# start
			for batch_idx, (data, target) in enumerate(self.train_loader):
				train_target = torch.zeros(self.train_batch_size, self.class_no) # onehot label
				#print('target size is {}'.format(target.size()))
				for i in range(self.train_batch_size):
					train_target[i][target[i]] = 1
				#target = train_label

				data.requires_grad = True
				#train_target.requires_grad = True

				#data, train_target = Variable(data), Variable(train_target)
				output = self.forward(data)
				#output.requires_grad = True

				train_batch_loss = self.loss_function(output, train_target)

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
				validation_target = torch.zeros(self.validation_batch_size, self.class_no)
				for i in range(self.validation_batch_size):
					validation_target[i][target[i]] = 1

				#data, validation_target = Variable(data), Variable(validation_target)
				data.requires_grad = False
				output = self.forward(data)

				validation_batch_loss = self.loss_function(output, validation_target)
				validation_loss = validation_loss + validation_batch_loss.data

				value, class_idx = torch.max(output.data, 1)

				for i in range(self.validation_batch_size):
					if class_idx[i] == target[i]:
						num_validation_correct = num_validation_correct + 1

			average_validation_loss = validation_loss / self.validation_batch_no
			accuracy = num_validation_correct / len(self.validation_loader.dataset)
			print('validation average_loss is {}, accuracy is {}'.format(average_validation_loss, accuracy))
			validation_loss_all.append(average_validation_loss)
			accuracies.append(accuracy)
			
			# one epoch time
			current_epoch_time = time.time() - start_time
			epoch_time.append(current_epoch_time)
			print('compute time: {:.5f} seconds'.format(current_epoch_time))

		# plot average_loss vs. epoch
		epoch_indices = range(1, self.max_epoch + 1)

		title = 'img2num: CNN Loss Changes in Different Epochs'
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
		plt.savefig('cnn_loss.png')

		# plot time vs epoch
		title = 'img2num: CNN Time Changes in Different Epochs'
		plt.figure(2)
		plt.plot(epoch_indices, epoch_time, color = 'blue', linestyle = 'solid', linewidth = '2.0',
			marker = '*', markerfacecolor = 'red', markersize = '5')
		plt.xlabel('epochs', fontsize = 10)
		plt.ylabel('epoch time', fontsize = 10)
		plt.title(title, fontsize = 10)
		plt.legend(fontsize = 10)
		plt.grid(True)
		#plt.show()
		plt.savefig('cnn_epoch_time.png')

		# save variables to a file
		with open('MNIST_LeNet-5', 'wb') as f:
			pickle.dump([accuracies, epoch_time, train_loss_all, validation_loss_all], f)
