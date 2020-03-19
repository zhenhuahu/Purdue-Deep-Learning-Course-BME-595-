import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import time
import pickle  # save variables to a file
import numpy as numpy
import cv2
# show image and add caption to image
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from train import AlexNet
import argparse


# receive input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default='self_model')
args = parser.parse_args()


class test:
	def __init__(self):
		super(test, self).__init__

		self.classes = {0: 'goldfish, Carassius auratus',
			 1: 'European fire salamander, Salamandra salamandra',
			 2: 'bullfrog, Rana catesbeiana',
			 3: 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',
			 4: 'American alligator, Alligator mississipiensis',
			 5: 'boa constrictor, Constrictor constrictor',
			 6: 'trilobite',
			 7: 'scorpion',
			 8: 'black widow, Latrodectus mactans',
			 9: 'tarantula',
			 10: 'centipede',
			 11: 'goose',
			 12: 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus',
			 13: 'jellyfish',
			 14: 'brain coral',
			 15: 'snail',
			 16: 'slug',
			 17: 'sea slug, nudibranch',
			 18: 'American lobster, Northern lobster, Maine lobster, Homarus americanus',
			 19: 'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish',
			 20: 'black stork, Ciconia nigra',
			 21: 'king penguin, Aptenodytes patagonica',
			 22: 'albatross, mollymawk',
			 23: 'dugong, Dugong dugon',
			 24: 'Chihuahua',
			 25: 'Yorkshire terrier',
			 26: 'golden retriever',
			 27: 'Labrador retriever',
			 28: 'German shepherd, German shepherd dog, German police dog, alsatian',
			 29: 'standard poodle',
			 30: 'tabby, tabby cat',
			 31: 'Persian cat',
			 32: 'Egyptian cat',
			 33: 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
			 34: 'lion, king of beasts, Panthera leo',
			 35: 'brown bear, bruin, Ursus arctos',
			 36: 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle',
			 37: 'fly',
			 38: 'bee',
			 39: 'grasshopper, hopper',
			 40: 'walking stick, walkingstick, stick insect',
			 41: 'cockroach, roach',
			 42: 'mantis, mantid',
			 43: "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
			 44: 'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus',
			 45: 'sulphur butterfly, sulfur butterfly',
			 46: 'sea cucumber, holothurian',
			 47: 'guinea pig, Cavia cobaya',
			 48: 'hog, pig, grunter, squealer, Sus scrofa',
			 49: 'ox',
			 50: 'bison',
			 51: 'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis',
			 52: 'gazelle',
			 53: 'Arabian camel, dromedary, Camelus dromedarius',
			 54: 'orangutan, orang, orangutang, Pongo pygmaeus',
			 55: 'chimpanzee, chimp, Pan troglodytes',
			 56: 'baboon',
			 57: 'African elephant, Loxodonta africana',
			 58: 'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens',
			 59: 'abacus',
			 60: "academic gown, academic robe, judge's robe",
			 61: 'altar',
			 62: 'apron',
			 63: 'backpack, back pack, knapsack, packsack, rucksack, haversack',
			 64: 'bannister, banister, balustrade, balusters, handrail',
			 65: 'barbershop',
			 66: 'barn',
			 67: 'barrel, cask',
			 68: 'basketball',
			 69: 'bathtub, bathing tub, bath, tub',
			 70: 'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon',
			 71: 'beacon, lighthouse, beacon light, pharos',
			 72: 'beaker',
			 73: 'beer bottle',
			 74: 'bikini, two-piece',
			 75: 'binoculars, field glasses, opera glasses',
			 76: 'birdhouse',
			 77: 'bow tie, bow-tie, bowtie',
			 78: 'brass, memorial tablet, plaque',
			 79: 'broom',
			 80: 'bucket, pail',
			 81: 'bullet train, bullet',
			 82: 'butcher shop, meat market',
			 83: 'candle, taper, wax light',
			 84: 'cannon',
			 85: 'cardigan',
			 86: 'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM',
			 87: 'CD player',
			 88: 'chain',
			 89: 'chest',
			 90: 'Christmas stocking',
			 91: 'cliff dwelling',
			 92: 'computer keyboard, keypad',
			 93: 'confectionery, confectionary, candy store',
			 94: 'convertible',
			 95: 'crane',
			 96: 'dam, dike, dyke',
			 97: 'desk',
			 98: 'dining table, board',
			 99: 'drumstick',
			 100: 'dumbbell',
			 101: 'flagpole, flagstaff',
			 102: 'fountain',
			 103: 'freight car',
			 104: 'frying pan, frypan, skillet',
			 105: 'fur coat',
			 106: 'gasmask, respirator, gas helmet',
			 107: 'go-kart',
			 108: 'gondola',
			 109: 'hourglass',
			 110: 'iPod',
			 111: 'jinrikisha, ricksha, rickshaw',
			 112: 'kimono',
			 113: 'lampshade, lamp shade',
			 114: 'lawn mower, mower',
			 115: 'lifeboat',
			 116: 'limousine, limo',
			 117: 'magnetic compass',
			 118: 'maypole',
			 119: 'military uniform',
			 120: 'miniskirt, mini',
			 121: 'moving van',
			 122: 'nail',
			 123: 'neck brace',
			 124: 'obelisk',
			 125: 'oboe, hautboy, hautbois',
			 126: 'organ, pipe organ',
			 127: 'parking meter',
			 128: 'pay-phone, pay-station',
			 129: 'picket fence, paling',
			 130: 'pill bottle',
			 131: "plunger, plumber's helper",
			 132: 'pole',
			 133: 'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria',
			 134: 'poncho',
			 135: 'pop bottle, soda bottle',
			 136: "potter's wheel",
			 137: 'projectile, missile',
			 138: 'punching bag, punch bag, punching ball, punchball',
			 139: 'reel',
			 140: 'refrigerator, icebox',
			 141: 'remote control, remote',
			 142: 'rocking chair, rocker',
			 143: 'rugby ball',
			 144: 'sandal',
			 145: 'school bus',
			 146: 'scoreboard',
			 147: 'sewing machine',
			 148: 'snorkel',
			 149: 'sock',
			 150: 'sombrero',
			 151: 'space heater',
			 152: "spider web, spider's web",
			 153: 'sports car, sport car',
			 154: 'steel arch bridge',
			 155: 'stopwatch, stop watch',
			 156: 'sunglasses, dark glasses, shades',
			 157: 'suspension bridge',
			 158: 'swimming trunks, bathing trunks',
			 159: 'syringe',
			 160: 'teapot',
			 161: 'teddy, teddy bear',
			 162: 'thatch, thatched roof',
			 163: 'torch',
			 164: 'tractor',
			 165: 'triumphal arch',
			 166: 'trolleybus, trolley coach, trackless trolley',
			 167: 'turnstile',
			 168: 'umbrella',
			 169: 'vestment',
			 170: 'viaduct',
			 171: 'volleyball',
			 172: 'water jug',
			 173: 'water tower',
			 174: 'wok',
			 175: 'wooden spoon',
			 176: 'comic book',
			 177: 'plate',
			 178: 'guacamole',
			 179: 'ice cream, icecream',
			 180: 'ice lolly, lolly, lollipop, popsicle',
			 181: 'pretzel',
			 182: 'mashed potato',
			 183: 'cauliflower',
			 184: 'bell pepper',
			 185: 'mushroom',
			 186: 'orange',
			 187: 'lemon',
			 188: 'banana',
			 189: 'pomegranate',
			 190: 'meat loaf, meatloaf',
			 191: 'pizza, pizza pie',
			 192: 'potpie',
			 193: 'espresso',
			 194: 'alp',
			 195: 'cliff, drop, drop-off',
			 196: 'coral reef',
			 197: 'lakeside, lakeshore',
			 198: 'seashore, coast, seacoast, sea-coast',
			 199: 'acorn'}

		save_dir = args.model
		self.myAlexNet = AlexNet()
		self.myAlexNet.load_state_dict(torch.load(args.model))
	
	# fetch image from the camera and predict it
	def cam(self):
		pil2tensor = transforms.ToTensor()
		tensor2pil = transforms.ToPILImage()
		trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

		font = cv2.FONT_HERSHEY_SIMPLEX

		video = cv2.VideoCapture(0)
		cv2.namedWindow('cam')
		print('press q to exit')

		while True:  # for streaming
			check, frame = video.read()
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			#print(frame)
			#cv2.imshow("Capturing", frame)
			height, width, channels = frame.shape
			frame_resize = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_CUBIC)
			img = trans(frame_resize)
			#img = img.reshape(1, 3, 224, 224)
			img = img.float()
			img = img.unsqueeze(0)
			img.requires_grad = False

			output = self.myAlexNet.forward(img)
			value, class_idx = torch.max(output.data, 1)
			#print('predicted result is ' + self.classes[class_idx])
			#print('class_idx is ' + str(int(class_idx.data[0])))

			frame_caption = cv2.putText(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), self.classes[int(class_idx.data[0])], (80,80), font, 1, (0,0,0), 2, cv2.LINE_AA)
			cv2.imshow('cam', frame_caption)

			# for press any key to out(milliseconds)
			key = cv2.waitKey(1)

			if key == ord('q'):
				break

		# release camera
		video.release()


if __name__ == '__main__':
	a = test()
	a.cam()




















