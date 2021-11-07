## Setting up TOSS for evaluation

#### 1. Clone the repo

	git clone https://github.com/feshotseg/toss

#### 2. Ready the PASCAL 5<sup>i</sup> test data
Use the script src/pascal5i.sh as follows:
	
	source src/pascal5i.sh data

#### 3. Ready the FSS-1000 dataset
 Download the FSS-1000 dataset from the authors' website: [here](https://drive.google.com/file/d/16TgqOeI_0P41Eh3jWQlxlRXG9KIqtMgI/view).
 Use the script as follows:
	source src/fss1000.sh path-to-the-fss1000-dataset

#### 4. Download the augmented tier-2 data
Download the augmented images and masks from [here](https://drive.google.com/file/d/12zq9R5WFBEtquryldjt9bJTyjdN0Jv96/view?usp=sharing).
	cd data/voc
	tar -zxf path-to-the-downloaded-aug.tar.gz

#### 5. Update your data loaders to load these files. 
##### Tier 1: Query complexity tests
The QC test files reside in ``data/tiers/attribute/splitX_q_EH_SAL.txt``. Here X is the PASCAL-5<sup>i</sup> split, EH is easy/hard, SAL is sal/nsal. 
Each split is a collection of pairs of mask names. It looks like:
```
	X_nnnn_nnnn	X_nnnn_nnnn
```
Where n is some digit, and X is the class number.

A loader for this can be written as follows:
```python

import numpy as np

# strip off the classname
def declass(filename):
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
		qimage = torch.from_numpy(self.image_transform(read_rgb(qimage))).float()
		qmask = torch.from_numpy(self.mask_transform(read_mask(qmask))).float()
		simage = torch.from_numpy(self.image_transform(read_rgb(simage))).float()
		smask = torch.from_numpy(self.mask_transform(read_mask(smask))).float()

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
Note that the `read\_rgb`, `read\_mask`, `image_transform` and `mask_transform` are assumed to functions written either with Numpy or PIL and are expected to produce a numpy array as their output for images and masks.


##### Tier 2: Support cognizance tiers (data/tiers/suppcog/splitX\_tier2.txt)
These files include 2 extra values in each row.
	query\_image support\_image weight type

[Here](https://github.com/fewshotseg/toss/blob/main/src/loader_tier2.py) is a sample loader for this tier.


##### Tier 3: Generalization tier (data/tiers/general/general\_tier.txt)
This a single file consisting of image pairs from the FSS-1000 dataset. Each row looks like:
```
	X_CNAME_Y  X_CNAME_Z
```
X is a class number, CNAME is the class-name, and Y and Z are file-numbers.
The class-number X should be ignored for the loader. 

#### 6. Setup the scores for reporting the tier-wise scores. 
`src/filescores2ics.py` for Tier-1 scores:
```python
	tsq = TestSetQCS()
    tsq.update(fold_number=0, part_type=TestSetQCS.PartType_EasySalient, mean_iou=0.5)
    tsq.update(fold_number=0, part_type=TestSetQCS.PartType_EasyNonSalient, mean_iou=0.35)
    tsq.update(fold_number=0, part_type=TestSetQCS.PartType_HardSalient, mean_iou=0.38)
    tsq.update(fold_number=0, part_type=TestSetQCS.PartType_HardNonSalient, mean_iou=0.13)

    print(tsq.fold_lca(0))
    print(tsq.fold_hca(0))
```

`src/filescores2scs.py` for Tier-2 scores:
```python
	scs = SCSScore(class_list=[0,1,2,3,4])
    def rand_mask(w, h):
        x = torch.randn((w, h))
        x[x>0.5] = 1
        x[x<=0.5] = 0
        return x

    scs.update( class_index=0,  case_weight_idx=-1, output=rand_mask(512, 512), label=rand_mask(512, 512))
    scs.update( class_index=1,  case_weight_idx=-1, output=rand_mask(512, 512), label=rand_mask(512, 512))
    scs.update( class_index=2,  case_weight_idx=-1, output=rand_mask(512, 512), label=rand_mask(512, 512))
    scs.update( class_index=3,  case_weight_idx=-1, output=rand_mask(512, 512), label=rand_mask(512, 512))
    scs.update( class_index=4,  case_weight_idx=-1, output=rand_mask(512, 512), label=rand_mask(512, 512))
    print(scs.meanIoU())
```

`src/filescores2gs.py` for Tier-3 scores
```python
	cm = ClasswiseMetrics()
    def rand_mask(w, h):
        x = torch.randn((w, h))
        x[x>0.5] = 1
        x[x<=0.5] = 0
        return x

    cm.update( class_index=0, output=rand_mask(512, 512), label=rand_mask(512, 512))
    cm.update( class_index=1, output=rand_mask(512, 512), label=rand_mask(512, 512))
    cm.update( class_index=2, output=rand_mask(512, 512), label=rand_mask(512, 512))
    cm.update( class_index=3, output=rand_mask(512, 512), label=rand_mask(512, 512))
    cm.update( class_index=4, output=rand_mask(512, 512), label=rand_mask(512, 512))
    print(cm.meanIoU())
```



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



