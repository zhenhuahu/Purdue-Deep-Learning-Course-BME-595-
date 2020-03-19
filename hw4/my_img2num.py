from neural_network import NeuralNetwork
import os
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time


# MNIST dataset contains 7,000 handwritten images
# 6,000 for training, 1,000 for validation
# each batch has 100 images, therefore 60 batches
class MyImg2Num:
	def __init__(self):
		self.max_epoch = 50 # number of total training/validating
		self.input_size = 28 * 28 # change 2D data to 1D, also # of input neurons
		self.class_no = 10 # 10

		self.learning_rate = 0.000001

		# download data
		root = './data'
		if not os.path.exists(root):
			os.mkdir(root)

		trans = transforms.Compose([transforms.ToTensor()])
		train_set = dset.MNIST(root = root, train = True, transform = trans, download = True)
		validation_set = dset.MNIST(root = root, train = False, transform = trans, download = True)

		self.train_batch_size = 20
		self.validation_batch_size = 1000

		self.train_loader = torch.utils.data.DataLoader(dataset=train_set, 
			batch_size = self.train_batch_size, shuffle = True)
		self.validation_loader = torch.utils.data.DataLoader(dataset = validation_set,
			batch_size = self.validation_batch_size, shuffle = True)

		self.train_batch_no = len(self.train_loader.dataset) / self.train_batch_size
		self.validation_batch_no = len(self.validation_loader.dataset) / self.validation_batch_size

		# 4 layers of neural network. 
		self.my_network = NeuralNetwork([self.input_size, 300, 60, self.class_no])

		# print('train size is {}'.format(len(self.train_loader.dataset)))
		# print('validation size is {}'.format(len(self.validation_loader.dataset)))

	def train(self):
		# values of all epochs
		train_loss_all = list()
		validation_loss_all = list()
		epoch_time = list()	

		# loop through epochs
		for i in range(self.max_epoch):
			print('\nEpoch ' + str(i+1) + ' start: ')
			start_time = time.time()

			train_loss = 0.

			# train start
			for batch_idx, (data, target) in enumerate(self.train_loader):
				#print('batch_id is {}'.format(batch_id)
				# onehot labeling
				train_target = torch.zeros(self.class_no, self.train_batch_size)
				for i in range(self.train_batch_size):
					train_target[target[i]][i] = 1

				output = self.my_network.forward(data.view(self.input_size, self.train_batch_size))
				# print('train_target is {}'.format(train_target))
				# print('output is {}'.format(output))

				self.my_network.backward(train_target)

				#print(self.my_network.loss)

				train_loss = train_loss + self.my_network.loss

				self.my_network.updateParams(self.learning_rate)
				#break

			average_train_loss = train_loss / self.train_batch_no
			print('train average loss is {:.5f}'.format(average_train_loss))
			train_loss_all.append(average_train_loss)

			# validation process
			validation_loss = 0.
			num_validation_correct = 0

			for data, target in self.validation_loader:
				output = self.my_network.forward(data.view(self.input_size, self.validation_batch_size))

				validation_target = torch.zeros(self.class_no, self.validation_batch_size)
				for i in range(self.validation_batch_size):
					validation_target[target[i]][i] = 1

				validation_loss =validation_loss + (((validation_target - output)**2).sum() * 0.5)/self.validation_batch_size

				value, class_idx = torch.max(output.t(), 1) #index of max value along each row
				for i in range(self.validation_batch_size):
					if class_idx[i] == target[i]:
						num_validation_correct = num_validation_correct + 1

			average_validation_loss = validation_loss / self.validation_batch_no
			accuracy = num_validation_correct / len(self.validation_loader.dataset)
			print('validation average_loss is {}, accuracy is {}'.format(average_validation_loss, accuracy))
			validation_loss_all.append(average_validation_loss)
			
			# one epoch time
			current_epoch_time = time.time() - start_time
			epoch_time.append(current_epoch_time)
			print('compute time: {:.5f} seconds'.format(current_epoch_time))
		

		# plot average_loss vs. epoch
		epoch_indices = range(1, self.max_epoch + 1)

		title = 'MyImg2Num: Loss Changes in Different Epochs'
		plt.figure(1)
		plt.plot(epoch_indices, train_loss_all, color = 'red', linestyle = 'solid', linewidth = '2.0',
			marker = '*', markerfacecolor = 'red', markersize = '5', label = 'training loss')
		plt.plot(epoch_indices, validation_loss_all, color = 'green', linestyle = 'solid', linewidth = '2.0',
			marker = 'o', markerfacecolor = 'green', markersize = '5', label = 'validation loss')
		plt.xlabel('epochs', fontsize = 15)
		plt.ylabel('loss', fontsize = 15)
		plt.title(title, fontsize = 15)
		plt.legend(fontsize = 12)
		plt.grid(True)
		plt.show()

		# plot time vs epoch
		title = 'MyImg2Num: Time Changes in Different Epochs'
		plt.figure(2)
		plt.plot(epoch_indices, epoch_time, color = 'blue', linestyle = 'solid', linewidth = '2.0',
			marker = '*', markerfacecolor = 'red', markersize = '5')
		plt.xlabel('epochs', fontsize = 10)
		plt.ylabel('epoch time', fontsize = 10)
		plt.title(title, fontsize = 10)
		plt.legend(fontsize = 10)
		plt.grid(True)
		plt.show()


	def forward(self, img: torch.ByteTensor):
		# take 1 image as input and output its label
		predictRes = self.my_network.forward(img.view(self.input_size,1)) # reshape the matrix
		value, num_label = torch.max(predictRes.t(), 1)
		return num_label













