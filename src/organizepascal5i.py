import os
import sys
import shutil
from tqdm import tqdm

def prepare_dir_structure(root_folder):
	for d in ["pascal5i", "pascal5i/masks"]:
		dpath = os.path.join(root_folder, d)
		if not os.path.isdir(dpath):
			os.mkdir(dpath)

def copy_images(root_folder, image_list):
	image_dest_dir = os.path.join(root_folder, "pascal5i/images")
	image_src_dir = os.path.join(root_folder, "JPEGImages")
	print("copying images")
	for _, image in tqdm(enumerate(image_list)):
		fname = image + ".jpg"
		shutil.copy( os.path.join(image_src_dir, fname), image_dest_dir )
	print("done copying images")

def copy_masks(root_folder): 
	mask_dest_dir = os.path.join(root_folder, "pascal5i/masks")
	print("copying class masks")
	image_list = set()
	for cls in tqdm(list(map(str, range(1, 21)))):
		for ttype in ["val", "train"]:
			mask_src_dir = os.path.join(root_folder, f"Binary_map_aug/{ttype}/{cls}")
			for f in filter(lambda x: x.endswith(".png"), os.listdir(mask_src_dir)):
				image_list.add( f.split(".")[0] )
				src_path = os.path.join(mask_src_dir, f)
				dest_path = os.path.join(mask_dest_dir, cls + "_" + f)
				shutil.copy(src_path, dest_path)
	return list(image_list)


def cleanup(root_folder):
	print("cleaning up")
	for f in ["JPEGImages", "Annotations", "Binary_map_aug", "ImageSets", "SegmentationClass", "SegmentationObject"]:
		print(f"Removing {root_folder}/{f}")
		ff = os.path.join(root_folder, f)
		if os.path.isdir(ff):
			shutil.rmtree(ff)
			if os.path.isdir(ff):
				os.rmdir(ff)
	shutil.move(os.path.join(root_folder, "pascal5i"), os.path.join(root_folder, "../../pascal5i"))
	shutil.rmtree(root_folder)

l = len(sys.argv)
if l == 2 or l == 3:
	root_folder = os.path.join(sys.argv[1], "VOCdevkit/VOC2012")
	if os.path.isdir(root_folder):
		prepare_dir_structure(root_folder)
		copy_images(root_folder, copy_masks(root_folder))
		if l == 3 and sys.argv[2] == "cleanup":
			cleanup(root_folder)
		print("Done.")
	else:
		print("Root folder needs to be the VOCdevkit folder")
else:
	print("invalid number of arguments")


