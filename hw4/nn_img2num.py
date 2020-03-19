import os
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from torch.autograd import Variable

class NnImg2Num:

	def __init__(self):
		self.max_epoch = 50 # number of total training/validating
		self.input_size = 28 * 28 # change 2D data to 1D, also # of input neurons
		self.class_no = 10 # 10

		self.learning_rate = 3.
				
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

		# create neural network. 
		self.nn_network = nn.Sequential(
            nn.Linear(self.input_size, 300), nn.Sigmoid(),
            nn.Linear(300, 60), nn.Sigmoid(),
            nn.Linear(60, self.class_no), nn.Sigmoid()
        )
        # Set loss function as Mean Square Error
		self.loss_function = nn.MSELoss()

		self.optimizer = torch.optim.SGD(self.nn_network.parameters(), lr = self.learning_rate)


	def train(self):
		# values of all epochs
		train_loss_all = list()
		validation_loss_all = list()
		epoch_time = list()	

		# loop through epochs
		for i in range(self.max_epoch):
			print('\nEpoch ' + str(i+1) + ' start: ')
			start_time = time.time()

			# train process
			train_loss = 0.
			self.nn_network.train() # train mode

			# start
			for batch_idx, (data, target) in enumerate(self.train_loader):
				train_target = torch.zeros(self.train_batch_size, self.class_no) # onehot label
				#print('target size is {}'.format(target.size()))
				for i in range(self.train_batch_size):
					train_target[i][target[i]] = 1
				#target = train_label

				data, train_target = Variable(data), Variable(train_target, requires_grad = False)
				output = self.nn_network(data.view(self.train_batch_size, self.input_size))

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
			self.nn_network.eval()
			validation_loss = 0.
			num_validation_correct = 0

			for data, target in self.validation_loader:
				validation_target = torch.zeros(self.validation_batch_size, self.class_no)
				for i in range(self.validation_batch_size):
					validation_target[i][target[i]] = 1

				data, validation_target = Variable(data), Variable(validation_target)
				output = self.nn_network(data.view(self.validation_batch_size, self.input_size))

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
			
			# one epoch time
			current_epoch_time = time.time() - start_time
			epoch_time.append(current_epoch_time)
			print('compute time: {:.5f} seconds'.format(current_epoch_time))

		# plot average_loss vs. epoch
		epoch_indices = range(1, self.max_epoch + 1)

		title = 'NnImg2Num: Loss Changes in Different Epochs'
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
		plt.show()

		# plot time vs epoch
		title = 'NnImg2Num: Time Changes in Different Epochs'
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
		img = Variable(img)
		self.nn_network.eval()
		output = self.nn_network(img.view(1, self.size1D))  # Forward pass using trained model
		value, pred_label = torch.max(output, 1)  # get index of max value among output class
		return pred_label







