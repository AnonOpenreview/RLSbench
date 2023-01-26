import gzip
import os
import pickle
import urllib

from PIL import Image

import logging

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

logger = logging.getLogger("label_shift")


class Subset(Dataset):
	"""
	Subset of a dataset at specified indices.

	Arguments:
		dataset (Dataset): The whole Dataset
		indices (sequence): Indices in the whole set selected for subset
	"""
	def __init__(self, dataset, indices, transform=None):
		self.dataset = dataset
		self.indices = indices
		self.transform = transform

	def __getitem__(self, idx):
		
		# logger.debug(f"IDx recieved {idx}")
		# logger.debug(f"Indices type {type(self.indices[idx])} value {self.indices[idx]}") 
		x  = self.dataset[self.indices[idx]]

		if self.transform is not None:
			transformed_img = self.transform(x[0])

			return transformed_img, x[1], x[2:]

		else: 
			return x

	@property
	def y_array(self):
		return self.dataset.y_array[self.indices] 

	def __len__(self):
		return len(self.indices)


class DatasetwithPseudoLabels(Dataset):
	"""
	Dataset with pseudo labels
	"""
	def __init__(self, dataset, pseudo_labels):
		self.dataset = dataset
		self.pseudo_labels = pseudo_labels

	def __getitem__(self, idx):
		data  = self.dataset[idx]
		label = self.pseudo_labels[idx]

		return data[0], label, data[1:]

	@property
	def y_array(self):
		return self.dataset.y_array

	def __len__(self):
		return len(self.dataset)
	
def dataset_with_targets(cls):
    """
    Modifies the dataset class to return target
    """ 

    def y_array(self):
        return np.array(self.targets).astype(int)

    return type(cls.__name__, (cls,), {
        'y_array': property(y_array)
    })

# def dataset_with_get_item(cls):
# 	"""
# 	Modifies the dataset class to return target
# 	""" 

# 	def get_item(self, idx):
# 		x = self.get_input(idx)
# 		y = self.y_array[idx]
# 		metadata = self.metadata_array[idx]
# 		return x, y, metadata

# 	def y_array(self):
# 		return self.y_array

# 	return type(cls.__name__, (cls,), {
# 		'__getitem__': get_item, 
# 		'y_array': property(y_array)
# 	})

def split_idx(targets, num_classes, source_frac, seed):
	"""
	Returns the indices of the source and target sets
	Input:
		dataset_len: length of the dataset
		source_frac: fraction of the dataset to use as source
		seed: seed for the random number generator
	Output:
		source_idx: indices of the source set
		target_idx: indices of the target set
	"""

	np.random.seed(seed)
	idx_per_label = []
	for i in range(num_classes):
		idx_per_label.append(np.where(targets == i)[0])

	source_idx = []
	target_idx = []
	for i in range(num_classes):
		source_idx.extend(np.random.choice(idx_per_label[i], int(source_frac * len(idx_per_label[i])), replace=False))
		target_idx.extend(np.setdiff1d(idx_per_label[i], source_idx, assume_unique=True))
	
	return np.array(source_idx), np.array(target_idx)


class RandomSplit():
	def __init__(self, dataset, indices):
		self.dataset = dataset

		self.size = len(indices)
		self.indices = indices

	def __len__(self):  
		return self.size

	def __getitem__(self, index):
		out = self.dataset[self.indices[index]]

		return out

class CIFAR10v2(torchvision.datasets.CIFAR10):
    
	def __init__(self, root, train=True, transform=None, target_transform=None,
					download=False):
		self.transform = transform
		self.target_transform = target_transform

		if train: 
			data = np.load(root + "/" + 'cifar102_train.npz', allow_pickle=True)
		else: 
			data = np.load(root + "/" + 'cifar102_test.npz', allow_pickle=True)
			
		self.data = data["images"]
		self.targets = data["labels"]

	def __len__(self): 
		return len(self.targets)

	@property
	def y_array(self):
		return self.targets

	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]

		# doing this so that it is consistent with all other datasets
		# to return a PIL Image
		img = Image.fromarray(img)

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target

class CIFAR_C(torchvision.datasets.CIFAR10):
    
	def __init__(self, root, data_type=None, severity=1, transform=None, target_transform=None,
		download=False):
		self.transform = transform
		self.target_transform = target_transform

		data = np.load(root + "/" + data_type + '.npy')
		labels = np.load(root + "/" + 'labels.npy')
			
		self.data = data[(severity-1)*10000: (severity)*10000]
		self.targets = labels[(severity-1)*10000: (severity)*10000].astype(np.int_)

	def __len__(self): 
		return len(self.targets)

	@property
	def y_array(self):
		return self.targets

	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]

		# doing this so that it is consistent with all other datasets
		# to return a PIL Image
		img = Image.fromarray(img)

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target


class USPS(torch.utils.data.Dataset):
	"""USPS Dataset.
	Args:
		root (string): Root directory of dataset where dataset file exist.
		train (bool, optional): If True, resample from dataset randomly.
		download (bool, optional): If true, downloads the dataset
			from the internet and puts it in root directory.
			If dataset is already downloaded, it is not downloaded again.
		transform (callable, optional): A function/transform that takes in
			an PIL image and returns a transformed version.
			E.g, ``transforms.RandomCrop``
	"""

	url = "https://raw.githubusercontent.com/mingyuliutw/CoGAN/master/cogan_pytorch/data/uspssample/usps_28x28.pkl"

	def __init__(self, root, train=True, transform=None, download=False):
		"""Init USPS dataset."""
		# init params
		self.root = os.path.expanduser(root)
		self.filename = "usps_28x28.pkl"
		self.train = train
		# Num of Train = 7438, Num ot Test 1860
		self.transform = transform
		self.dataset_size = None

		# download dataset.
		if download:
			self.download()
		if not self._check_exists():
			raise RuntimeError("Dataset not found." +
								" You can use download=True to download it")

		self.train_data, self.train_labels = self.load_samples()
		if self.train:
			total_num_samples = self.train_labels.shape[0]
			indices = np.arange(total_num_samples)
			np.random.shuffle(indices)
			self.train_data = self.train_data[indices[0:self.dataset_size], ::]
			self.train_labels = self.train_labels[indices[0:self.dataset_size]]
		self.train_data *= 255.0
		self.train_data = self.train_data.transpose(
			(0, 2, 3, 1))  # convert to HWC

	def __getitem__(self, index):
		"""Get images and target for data loader.
		Args:
			index (int): Index
		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		img, label = self.train_data[index, ::], self.train_labels[index]
		# img = 
		# print(img.shape)
		img = Image.fromarray(img.squeeze().astype(np.int8), mode='L')
		if self.transform is not None:
			img = self.transform(img)
		label = int(label)
		# label = torch.FloatTensor([label.item()])
		return img, label

	def __len__(self):
		"""Return size of dataset."""
		return self.dataset_size

	def _check_exists(self):
		"""Check if dataset is download and in right place."""
		return os.path.exists(os.path.join(self.root, self.filename))

	def download(self):
		"""Download dataset."""
		filename = os.path.join(self.root, self.filename)
		dirname = os.path.dirname(filename)
		if not os.path.isdir(dirname):
			os.makedirs(dirname)
		if os.path.isfile(filename):
			return
		print("Download %s to %s" % (self.url, os.path.abspath(filename)))
		urllib.request.urlretrieve(self.url, filename)
		print("[DONE]")
		return

	def load_samples(self):
		"""Load sample images from dataset."""
		filename = os.path.join(self.root, self.filename)
		f = gzip.open(filename, "rb")
		data_set = pickle.load(f, encoding="bytes")
		f.close()
		if self.train:
			images = data_set[0][0]
			labels = data_set[0][1]
			self.dataset_size = labels.shape[0]
		else:
			images = data_set[1][0]
			labels = data_set[1][1]
			self.dataset_size = labels.shape[0]
		return images, labels