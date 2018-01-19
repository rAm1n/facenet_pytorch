from __future__ import print_function

import torchvision.datasets as datasets
import os
import numpy as np
from tqdm import tqdm
from utils import default_loader

class TripletFaceDataset(datasets.ImageFolder):

	def __init__(self, dir, n_triplets, files=None,  transform=None, *arg, **kw):
		if files:
			f = open(files, 'r')
			self.classes = list()
			self.class_to_idx = dict()
			self.imgs = list()
			self.transform = transform
			self.loader = default_loader

			for line in f:
				line = line.strip().split(' ')
				img_path = os.path.join(dir, line[0])
				self.imgs.append((img_path , int(line[1])))
				if line[1] not in self.class_to_idx:
					self.classes.append(line[1])
					self.class_to_idx[line[1]] = int(line[1])
					
			self.classes.sort()

		else:
			super(TripletFaceDataset, self).__init__(dir,transform) 

		self.n_triplets = n_triplets

		print('Generating {} triplets'.format(self.n_triplets))
		self.training_triplets = self.generate_triplets(self.imgs, self.n_triplets,len(self.classes))

	@staticmethod
	def generate_triplets(imgs, num_triplets,n_classes):
		def create_indices(_imgs):
			inds = dict()
			for idx, (img_path,label) in enumerate(_imgs):
				if label not in inds:
					inds[label] = []
				inds[label].append(img_path)
			return inds

		triplets = []
		# Indices = array of labels and each label is an array of indices
		indices = create_indices(imgs)

		for x in tqdm(range(num_triplets)):
			c1 = np.random.randint(0, n_classes-1)
			c2 = np.random.randint(0, n_classes-1)
			while len(indices[c1]) < 2:
				c1 = np.random.randint(0, n_classes-1)

			while c1 == c2:
				c2 = np.random.randint(0, n_classes-1)
			if len(indices[c1]) == 2:  # hack to speed up process
				n1, n2 = 0, 1
			else:
				n1 = np.random.randint(0, len(indices[c1]) - 1)
				n2 = np.random.randint(0, len(indices[c1]) - 1)
				while n1 == n2:
					n2 = np.random.randint(0, len(indices[c1]) - 1)
			if len(indices[c2]) ==1:
				n3 = 0
			else:
				n3 = np.random.randint(0, len(indices[c2]) - 1)

			triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3],c1,c2])
		return triplets

	def __getitem__(self, index):
		'''

		Args:
			index: Index of the triplet or the matches - not of a single image

		Returns:

		'''
		def transform(img_path):
			"""Convert image into numpy array and apply transformation
			   Doing this so that it is consistent with all other datasets
			   to return a PIL Image.
			"""

			img = self.loader(img_path)
			return self.transform(img)

		# Get the index of each image in the triplet
		a, p, n,c1,c2 = self.training_triplets[index]

		# transform images if required
		img_a, img_p, img_n = transform(a), transform(p), transform(n)
		return img_a, img_p, img_n,c1,c2

	def __len__(self):
		return len(self.training_triplets)



