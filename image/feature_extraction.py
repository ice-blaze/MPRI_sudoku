from tools import list_images
import numpy as np
from PIL import Image
from skimage.transform import resize
from sklearn.feature_extraction.image import img_to_graph


def extract_features(im):
	""" Returns a feature vector for an image patch. """

	# TODO: find other features to use
	temp = im.flatten()

	#temp = img_to_graph(im)
	'''for i in range(0,len(temp)):
		if temp[i] > 64:
			temp[i] = 255
		else:
			temp[i] = 0'''

	return temp


def process_image(im, border_size=5, im_size=50):
	""" Remove borders and resize """

	im = im[border_size:-border_size, border_size:-border_size]

	
	'''for i in range(0,len(im)):
		for j in range(0,len(im[i])):
			im[i][j] = 255 if im[i][j] > 64 else 0'''
				
	im = resize(im, (im_size, im_size))

	return im


def load_data(path):
	""" Return labels and features for all jpg images in path. """

	# Create a list of all files ending in .jpg
	im_list = list_images(path, '.jpg')

	# Create labels
	labels = [int(im_name.split('/')[-1][0]) for im_name in im_list]
	features = []

	# Create features from the images
	# TOD.O: iterate over images paths
	for im_path in im_list:
		# TOD.O: load image as a gray level image
		im = np.array(Image.open(im_path).convert('L'))
		# TOD.O: process the image to remove borders and resize
		im = process_image(im)
		# TOD.O: append extracted features to the a list
		features.append(extract_features(im))

	# TOD.O: return features, and labels
	return features, labels
