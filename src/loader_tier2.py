import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image
import cv2

############################################################## Loader Utilities #######################################################
def resizer(image_size):
	def _resize(img):
		return cv2.resize(img, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)

	return _resize


def mask_resizer(image_size):
	def _resize(img):
		return cv2.resize(img, dsize=(image_size, image_size), interpolation=cv2.INTER_NEAREST)

	return _resize


def normalizer(mean, stddev):
	def _normalize(img):
		img = ((img / 255.0).astype(np.float32) - mean) / stddev
		return img

	return _normalize


def tensorify(img):
	return img.transpose((2, 0, 1))


def mask_tensorify(img):
	img = np.expand_dims(img, 0)
	img[img > 0] = 1
	return img.astype(np.int32)


def compose(f1, *args):
	def _composed(img):
		i = f1(img)
		for arg in args:
			i = arg(i)
		return i

	return _composed


def read_rgb(path):
	with open(path, "rb") as f:
		return np.array(Image.open(f).convert('RGB'))


def read_mask(path):
	if path.find("aug/null") != -1:
		return np.zeros((512, 512)).astype(np.uint8)
	with open(path, "rb") as f:
		return np.array(Image.open(f).convert('L'))


def declass(filename):
	if filename.find("aug") != -1:
		return "_".join( filename.split("_")[2:] )
	return "_".join(filename.split("_")[1:])


def clsname(filename):
	return filename.split("_")[0]

class FSSPairIndex:
	def __init__(self, imagedir, maskdir, listfile):
		self.imagedir = imagedir
		self.maskdir = maskdir
		self.files = np.genfromtxt(listfile, dtype=str)

	def __getitem__(self, index):
		if self.files[index][0].startswith("aug"):
			classidx = self.files[index][0].strip().split('_')[1]
		else:
			classidx = self.files[index][0].strip().split('_')[0]
		queryimage = os.path.join(self.imagedir, declass(self.files[index][0]) + ".jpg")
		querymask = os.path.join(self.maskdir, self.files[index][0] + "_gt.png")
		supportimage = os.path.join(self.imagedir, declass(self.files[index][1]) + ".jpg")
		supportmask = os.path.join(self.maskdir, self.files[index][1] + "_gt.png")
		weight = float( self.files[index][2] )
		scoretype = int( self.files[index][3] )
		return queryimage, querymask, supportimage, supportmask, int(classidx), weight, scoretype

	def __len__(self):
		return len(self.files)

class FSSPairLoader(Dataset):
	def __init__(self, imagedir, maskdir, pairlistfile, image_size):
		self.imageTransform = compose(
			resizer(image_size),
			normalizer(np.array([.485, .456, .406]), np.array([.229, .224, .225])),
			tensorify)
		self.maskTransform = compose(
			mask_resizer(image_size),
			mask_tensorify
		)
		self.fsindex = FSSPairIndex(imagedir=imagedir, maskdir=maskdir, listfile=pairlistfile)

	def __getitem__(self, index):
		# get item paths
		qimage, qmask, simage, smask, classidx, weight, scoretype = self.fsindex[index]

		# read the stuff in and convert them to tensors
		qimage = torch.from_numpy(self.imageTransform(read_rgb(qimage))).float()
		qmask = torch.from_numpy(self.maskTransform(read_mask(qmask))).float()
		simage = torch.from_numpy(self.imageTransform(read_rgb(simage))).float()
		smask = torch.from_numpy(self.maskTransform(read_mask(smask))).float()

		return simage, smask, qimage, qmask, classidx, weight, scoretype

	def __len__(self):
		return len(self.fsindex.files)


def pair_iterator(loader, num_workers, batch_size, shuffle):
	return DataLoader(loader, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle)

def fixed_pair_iterator(imagedir, maskdir, image_size, listname, num_workers, batch_size):
	return pair_iterator(
		loader=FSSPairLoader(
			imagedir=imagedir,
			maskdir=maskdir,
			pairlistfile=listname,
			image_size=image_size),
		num_workers=num_workers,
		batch_size=batch_size,
		shuffle=False
	)


if __name__ == "__main__":
	iterator = fixed_pair_iterator( "/ssds/1/mayur/fss/voc/images", "/ssds/1/mayur/fss/voc/masks", 256, "split0_tier2.txt", 1, 4)
	for (simg, smask, qimg, qmask, cidx, wht, st) in iterator:
		print(simg.shape, smask.shape, qimg.shape, qmask.shape, cidx, wht, st)

