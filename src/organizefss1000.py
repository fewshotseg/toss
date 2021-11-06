import os
import shutil
import sys
from tqdm import tqdm

if len(sys.argv) < 2:
	print("Usage: python organizefss100.py fss100-folder")


# FSS-1000 dataset directory structure
ROOT_FSS1000 = sys.argv[1]

# Our FSS1K directory structure
fss1k = 'data/fss1k'
fss1k_images = 'data/fss1k/images'
fss1k_masks = 'data/fss1k/masks'


if not(os.path.isdir(fss1k)):
	os.mkdir(fss1k)

if not(os.path.isdir(fss1k_images)):
	os.mkdir(fss1k_images)

if not(os.path.isdir(fss1k_masks)):
	os.mkdir(fss1k_masks)

fss1k_classes = filter(lambda x: not(x.startswith(".")) and not(x.find(".") != -1), os.listdir(ROOT_FSS1000))

for class_name in tqdm(fss1k_classes):
	class_path = os.path.join(ROOT_FSS1000, class_name)
	class_files = os.listdir(class_path)

	for filename in class_files:

		file_num = filename.split('.')[0]
		new_filename = f'{class_name}_{file_num}'
		new_filename = new_filename.replace("'", "0").replace("-", "0")

		old_filepath = f'{ROOT_FSS1000}/{class_name}/{filename}'

		if filename.endswith('jpg'):
			new_filename += '.jpg'
			new_filepath = f'{fss1k_images}/{new_filename}'

		elif filename.endswith('png'):
			new_filename += '.png'
			new_filepath = f'{fss1k_masks}/{new_filename}'

		shutil.copy(old_filepath, new_filepath)
