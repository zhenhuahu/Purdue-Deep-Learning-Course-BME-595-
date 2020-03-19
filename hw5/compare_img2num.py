# hw5:  draw accuracy and comparison of hw5 and hw4

import os
import torch
import matplotlib.pyplot as plt
import pickle  # save variables to a file

FCN_results = pickle.load(open('MNIST_nn', 'rb')) # read binary
CNN_results = pickle.load(open('MNIST_LeNet-5', 'rb'))

# plot average_loss vs. epoch
max_epoch = 50
epoch_indices = range(1, max_epoch + 1)

# plot accuracy vs. epoch
title = 'Validation Accuracies of Fully-connected and Convolution Neural Networks'
plt.figure(1)
plt.plot(epoch_indices, CNN_results[0], color = 'red', linestyle = 'solid', linewidth = '2.0',
	marker = '*', markerfacecolor = 'red', markersize = '5', label = 'CNN Accuracy')
plt.plot(epoch_indices, FCN_results[0], color = 'green', linestyle = 'solid', linewidth = '2.0',
	marker = 'o', markerfacecolor = 'green', markersize = '5', label = 'FCN Accuracy')
plt.xlabel('epochs', fontsize = 10)
plt.ylabel('loss', fontsize = 10)
plt.title(title, fontsize = 10)
plt.legend(fontsize = 10)
plt.grid(True)
#plt.show()
plt.savefig('compare_validation_accuracy.png')

# plot time vs. epoch
title = 'Epoch Time of Fully-connected and Convolution Neural Networks'
plt.figure(2)
plt.plot(epoch_indices, CNN_results[1], color = 'red', linestyle = 'solid', linewidth = '2.0',
	marker = '*', markerfacecolor = 'red', markersize = '5', label = 'CNN Epoch Time')
plt.plot(epoch_indices, FCN_results[1], color = 'green', linestyle = 'solid', linewidth = '2.0',
	marker = 'o', markerfacecolor = 'green', markersize = '5', label = 'FCN Epoch Time')
plt.xlabel('epochs', fontsize = 10)
plt.ylabel('loss', fontsize = 10)
plt.title(title, fontsize = 10)
plt.legend(fontsize = 10)
plt.grid(True)
#plt.show()
plt.savefig('compare_epoch_time.png')

# plot training loss vs. epoch
title = 'Training Loss of Fully-connected and Convolution Neural Networks'
plt.figure(3)
plt.plot(epoch_indices, CNN_results[2], color = 'red', linestyle = 'solid', linewidth = '2.0',
	marker = '*', markerfacecolor = 'red', markersize = '5', label = 'CNN Training Loss')
plt.plot(epoch_indices, FCN_results[2], color = 'green', linestyle = 'solid', linewidth = '2.0',
	marker = 'o', markerfacecolor = 'green', markersize = '5', label = 'FCN Training Loss')
plt.xlabel('epochs', fontsize = 10)
plt.ylabel('loss', fontsize = 10)
plt.title(title, fontsize = 10)
plt.legend(fontsize = 10)
plt.grid(True)
#plt.show()
plt.savefig('compare_train_loss.png')

# plot validation loss vs. epoch
title = 'Validation Loss of Fully-connected and Convolution Neural Networks'
plt.figure(4)
plt.plot(epoch_indices, CNN_results[3], color = 'red', linestyle = 'solid', linewidth = '2.0',
	marker = '*', markerfacecolor = 'red', markersize = '5', label = 'CNN Validation Loss')
plt.plot(epoch_indices, FCN_results[3], color = 'green', linestyle = 'solid', linewidth = '2.0',
	marker = 'o', markerfacecolor = 'green', markersize = '5', label = 'FCN Validation Loss')
plt.xlabel('epochs', fontsize = 10)
plt.ylabel('loss', fontsize = 10)
plt.title(title, fontsize = 10)
plt.legend(fontsize = 10)
plt.grid(True)
#plt.show()
plt.savefig('compare_validation_loss.png')