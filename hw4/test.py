from my_img2num import MyImg2Num
from nn_img2num import NnImg2Num
import time

def main():
	start_time = time.time()
	myNN = MyImg2Num()
	#myNN = NnImg2Num()
	myNN.train()
	print('time elapsed is {} seconds'.format(time.time() - start_time))

main()