# implementation of resNet to do orientation detection

import torchvision.models as models
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.datasets.folder as folder
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler

import time
import copy

# normalize input
trans = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), 
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

train_dir = os.path.join('50dpi', 'train')
val_dir = os.path.join('50dpi', 'val')

train_batch_size = 50
val_batch_size = 50

train_set = dset.ImageFolder(root = train_dir, transform = trans)
train_loader = torch.utils.data.DataLoader(dataset = train_set, 
	batch_size = train_batch_size, shuffle = True)
train_batch_no = len(train_loader.dataset) / train_batch_size


val_set = dset.ImageFolder(root = val_dir, transform = trans)
val_loader = torch.utils.data.DataLoader(dataset = val_set, 
	batch_size = val_batch_size, shuffle = True)
val_batch_no = len(val_loader.dataset) / val_batch_size

class_names = train_set.classes  #['0_degree', '180_degrees', '270_degrees', '90_degrees']
class_no = len(class_names)

# select device
if torch.cuda.is_available():
	device = torch.device('cuda:0')
	print('Using GPU')
else:
	device = torch.device('cpu')
	print('Using CPU')


model_conv = models.resnet18(pretrained=True)
for param in model_conv.parameters():
	param.requires_grad = False

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, class_no)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

#hyper parameters
optimizer_conv = torch.optim.SGD(model_conv.parameters(), lr = 0.001, momentum = 0.9)
# decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size = 7, gamma = 0.1)


def train_model(model, criterion, optimizer, scheduler, num_epochs):
	since = time.time()
	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	# store training information
	train_accuracy = list()
	val_accuracy = list()
	train_loss = list()
	val_loss = list()

	for epoch in range(25):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

        # each epoch has training and testing
		for phase in ['train', 'val']:
			if phase == 'train':
				scheduler.step()
				model.train()  # set model to training mode

				running_loss = 0.0
				running_corrects = 0

				# iterate over data
				for inputs, labels in train_loader:
					inputs = inputs.to(device)
					labels = labels.to(device)

					# zero the parameter gradients
					optimizer.zero_grad()

					# forward. track history iff in train
					with torch.set_grad_enabled(phase == 'train'):
						outputs = model(inputs)
						_, preds = torch.max(outputs, 1)
						loss = criterion(outputs, labels)

						# backward & optimizer only if in training phase
						loss.backward()
						optimizer.step()

					# statistics
					running_loss += loss.item() * inputs.size(0)
					running_corrects += torch.sum(preds == labels.data)

				epoch_loss = running_loss / len(train_loader.dataset)
				epoch_acc = running_corrects.double() / len(train_loader.dataset)

				train_loss.append(epoch_loss)
				train_accuracy.append(epoch_acc)

				print('training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

			else:			
				model.eval()   # set model to evaluation

				running_loss = 0.0
				running_corrects = 0

				# iterate over data
				for inputs, labels in val_loader:
					#val_start = time.time()

					inputs = inputs.to(device)
					labels = labels.to(device)

					# zero the parameter gradients
					optimizer.zero_grad()

					# forward. track history iff in train
					with torch.set_grad_enabled(phase == 'train'):
						outputs = model(inputs)
						_, preds = torch.max(outputs, 1)
						loss = criterion(outputs, labels)

					# statistics
					running_loss += loss.item() * inputs.size(0)
					running_corrects += torch.sum(preds == labels.data)

					#val_time = time.time() - val_start
					#print('one validation time is {:.0f}m {:.0f}s'.format(val_time // 60, val_time % 60))

				epoch_loss = running_loss / len(val_loader.dataset)
				epoch_acc = running_corrects.double() / len(val_loader.dataset)

				val_loss.append(epoch_loss)
				val_accuracy.append(epoch_acc)

				print('validation Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

				if(epoch_acc > best_acc):
					best_acc = epoch_acc
					best_model_wts = copy.deepcopy(model.state_dict())

		print()

	time_elapsed = time.time() - since

	print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
	model.load_state_dict(best_model_wts)

	torch.save(best_model_wts, 'resnet18_centerCrop')

	# plot average_loss vs. epoch
	epoch_indices = range(1, num_epochs + 1)

	title = 'resnet18_centerCrop_pretrained Accuracy in Different Epochs'
	plt.figure(1)
	plt.plot(epoch_indices, train_accuracy, color = 'red', linestyle = 'solid', linewidth = '2.0',
		marker = '*', markerfacecolor = 'red', markersize = '5', label = 'training accuracy')
	plt.plot(epoch_indices, val_accuracy, color = 'green', linestyle = 'solid', linewidth = '2.0',
		marker = 'o', markerfacecolor = 'green', markersize = '5', label = 'validation accuracy')
	plt.xlabel('epochs', fontsize = 10)
	plt.ylabel('accuracy', fontsize = 10)
	plt.title(title, fontsize = 10)
	plt.legend(fontsize = 10)
	plt.grid(True)
	#plt.show()
	plt.savefig('resnet18_centerCrop_pretrained.png')

	return model


model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs = 25)
			


        
