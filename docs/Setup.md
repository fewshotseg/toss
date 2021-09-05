## Setting up TOSS for evaluation

#### Clone the repo

	git clone https://github.com/feshotseg/toss

#### Ready the PASCAL 5<sup>i</sup> test data
Use the script src/pascal5i.sh as follows:
	
	source src/pascal5i.sh data


#### Ready the FSS-1000 dataset
Use the script as follows:
	source src/fss1k.sh data

#### Working with the splits
Each test split is a collection of pairs of mask names. It looks like:
	X_nnnn_nnnn	X_nnnn_nnnn

Where n is some digit, and X is the class number.


A loader for this can be written as follows:
```python

import numpy as np

# strip off the classname
def declass(filename):
	if filename.find("aug") != -1:
		return "_".join( filename.split("_")[2:] )
	return "_".join(filename.split("_")[1:])

# get the class name
def clsname(filename):
	return filename.split("_")[0]

# pair index
class FSSPairIndex:
	def __init__(self, imagedir, maskdir, listfile):
		self.imagedir = imagedir
		self.maskdir = maskdir
		self.files = np.genfromtxt(listfile, dtype=str)

	def __getitem__(self, index):
		classidx = clsname(self.files[index][0].strip())
		queryimage = os.path.join(self.imagedir, declass(self.files[index][0]) + ".jpg")
		querymask = os.path.join(self.maskdir, self.files[index][0] + ".png")
		supportimage = os.path.join(self.imagedir, declass(self.files[index][1]) + ".jpg")
		supportmask = os.path.join(self.maskdir, self.files[index][1] + ".png")
		return queryimage, querymask, supportimage, supportmask, int(classidx)

	def __len__(self):
		return len(self.files)

class FSSPairLoader(Dataset):
	def __init__(self, imagedir, maskdir, pairlistfile, image_size):
		self.fsindex = FSSPairIndex(imagedir=imagedir, maskdir=maskdir, listfile=pairlistfile)

	def __getitem__(self, index):
		# get item paths
		qimage, qmask, simage, smask, classidx = self.fsindex[index]

		# read the stuff in and convert them to tensors
		qimage = torch.from_numpy(self.imageTransform(read_rgb(qimage))).float()
		qmask = torch.from_numpy(self.maskTransform(read_mask(qmask))).float()
		simage = torch.from_numpy(self.imageTransform(read_rgb(simage))).float()
		smask = torch.from_numpy(self.maskTransform(read_mask(smask))).float()

		return simage, smask, qimage, qmask, classidx

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
```

#### Special instructions for Tier 2 files (data/tiers/suppcog/splitX\_tier2.txt)
These files include 2 extra values in each row.
	query\_image support\_image weight type

The weight specifies a weight used to mark the importance of this particular example in computing the SCS score as defined in the paper.
The type determines if it is a positive (1) or a negative (-1) kind of example. The mIou from all negative kind of examples are inverted (1-miou) for computing the
SCS score.

#### Special instructions for Tier 3 test file (data/tiers/general/general\_tier.txt)
These files are based on the FSS-1000 dataset. Therefore, the folder to use for testing it must be the path to the FSS-1000 files. data/fss1k

<!--

Alternatively, use the following procedure
1. Download the Berkeley Augmented PASCAL VOC images.
i. Using the script from [here](https://github.com/kazuto1011/deeplab-pytorch/blob/master/scripts/setup_voc12.sh) download and setup the VOC 2012 dataset.
ii. Download the binary masks from [here](https://github.com/icoz69/CaNet/raw/master/Binary_map_aug.zip) to the VOC2012 directory. The final folder structure should look like:
	|-toss
	|--- data
		  |--- VOCdevkit
		  		|-- VOC2012
				|---Annotations
				|---Binary\_map\_aug
				|---ImageSets
				|---JPEGImages
				|---SegmentationClass
				|---SegmentationObject

3. Run the src/organizepascal5i.py file with the path to the VOC2012 folder as input.
	python src/organizepascal5i.py ./toss/data/VOCdevkit/VOC2012
-->



